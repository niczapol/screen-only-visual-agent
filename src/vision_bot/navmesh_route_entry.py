from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from vision_bot.config import resource_path
from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.mmap_navmesh import MMapNavMesh, NavLocation, NavPath, NavPolygon, densify_path
from vision_bot.route_planner import MiningNode, ROUTE_ENTRY_WAYPOINT_SOURCE, RouteEntryPlan
from vision_bot.terrain_routing import (
    GRID_SPACING,
    TerrainCostConfig,
    TerrainCostMap,
    TerrainRaster,
    ZoneTransform,
    build_terrain_cost_map,
    build_terrain_raster,
    supercover_line,
)


@dataclass(frozen=True)
class NavMeshRouteEntrySettings:
    candidate_count: int = 32
    waypoint_spacing_yards: float = 25.0
    max_nav_snap_yards: float = 40.0
    max_nav_vertical_yards: float = 80.0
    max_expansions: int = 200_000
    min_clearance_cells: float = 0.0


@dataclass(frozen=True)
class NavMeshRouteEntryResult:
    entry_plan: RouteEntryPlan
    target_route_index: int
    world_length_yards: float
    corridor_polygon_count: int
    funnel_point_count: int
    waypoint_count: int
    candidates_considered: int
    candidates_with_ground: int
    candidates_with_path: int
    rejected_terrain_paths: int
    terrain_validation: dict[str, Any]
    load_seconds: float
    plan_seconds: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("entry_plan", None)
        result["target_route_sort_key"] = self.entry_plan.target_route_sort_key
        result["waypoints"] = [
            {
                "coord": node.coord,
                "x": coord_to_xy(node.coord)[0],
                "y": coord_to_xy(node.coord)[1],
                "source": node.source,
            }
            for node in self.entry_plan.waypoints
        ]
        return result


@dataclass(frozen=True)
class NavMeshGuidanceObservation:
    steering_coord: int
    strategic_target_coord: int
    status: str
    remaining_waypoints: int
    replanned: bool
    world_length_yards: float | None = None
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NavMeshRouteEntryPlanner:
    """Plan a one-shot, terrain-vetoed corridor from the live pose to a route loop."""

    def __init__(
        self,
        *,
        zone: ZoneTransform,
        raster: TerrainRaster,
        base_cost_map: TerrainCostMap,
        cost_map: TerrainCostMap,
        hazard_mask: np.ndarray,
        mesh: MMapNavMesh,
        settings: NavMeshRouteEntrySettings,
        zone_id: int,
        load_seconds: float = 0.0,
    ) -> None:
        self.zone = zone
        self.raster = raster
        self.base_cost_map = base_cost_map
        self.cost_map = cost_map
        self.hazard_mask = hazard_mask.astype(bool, copy=True)
        self.mesh = mesh
        self.settings = settings
        self.zone_id = int(zone_id)
        self.load_seconds = float(load_seconds)
        self._transition_cost = build_terrain_transition_cost(
            raster=raster,
            zone=zone,
            cost_map=cost_map,
            min_clearance=settings.min_clearance_cells,
        )
        self._hazard_egress_transition_cost = build_hazard_egress_transition_cost(
            raster=raster,
            zone=zone,
            base_cost_map=base_cost_map,
            hazard_mask=self.hazard_mask,
            min_clearance=settings.min_clearance_cells,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        zone_id: int,
    ) -> NavMeshRouteEntryPlanner:
        started_at = time.perf_counter()
        profile = navmesh_profile_for_zone(config, zone_id)
        manifest_path = resource_path(str(profile["manifest_path"]))
        adt_dir = resource_path(str(profile["adt_dir"]))
        mmap_dir = resource_path(str(profile["mmap_dir"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        zone = zone_from_manifest(manifest)
        raster = build_terrain_raster(adt_dir.glob("*.adt"))

        terrain_values = profile.get("terrain", {})
        terrain_config = TerrainCostConfig(
            soft_slope_degrees=float(terrain_values.get("soft_slope_degrees", 15.0)),
            hard_slope_degrees=float(terrain_values.get("hard_slope_degrees", 35.0)),
            slope_weight=float(terrain_values.get("slope_weight", 8.0)),
            roughness_weight=float(terrain_values.get("roughness_weight", 0.08)),
            clearance_cells=float(terrain_values.get("clearance_cells", 3.0)),
            clearance_weight=float(terrain_values.get("clearance_weight", 5.0)),
        )
        base_cost_map = build_terrain_cost_map(raster, terrain_config)
        cost_map = base_cost_map
        hazard_mask = np.zeros(base_cost_map.blocked.shape, dtype=bool)
        hazards_path = profile.get("hazards_path")
        if hazards_path:
            hazard_source = json.loads(
                resource_path(str(hazards_path)).read_text(encoding="utf-8")
            )
            hazard_mask = build_hazard_mask(
                base_cost_map.blocked.shape,
                raster=raster,
                zone=zone,
                hazards=hazard_source.get("hazards", []),
            )
            cost_map, _ = apply_hazard_exclusions(
                base_cost_map,
                raster=raster,
                zone=zone,
                hazards=hazard_source.get("hazards", []),
            )

        tiles = [(int(item["x"]), int(item["y"])) for item in manifest["tiles"]]
        mesh = MMapNavMesh.load_tiles(
            mmap_dir,
            map_id=int(profile.get("map_id", 1)),
            tiles=tiles,
        )
        settings = NavMeshRouteEntrySettings(
            candidate_count=max(1, int(profile.get("candidate_count", 32))),
            waypoint_spacing_yards=max(1.0, float(profile.get("waypoint_spacing_yards", 25.0))),
            max_nav_snap_yards=max(0.0, float(profile.get("max_nav_snap_yards", 40.0))),
            max_nav_vertical_yards=max(
                0.0, float(profile.get("max_nav_vertical_yards", 80.0))
            ),
            max_expansions=max(1, int(profile.get("max_expansions", 200_000))),
            min_clearance_cells=max(
                0.0, float(profile.get("min_clearance_cells", 0.0))
            ),
        )
        return cls(
            zone=zone,
            raster=raster,
            base_cost_map=base_cost_map,
            cost_map=cost_map,
            hazard_mask=hazard_mask,
            mesh=mesh,
            settings=settings,
            zone_id=zone_id,
            load_seconds=time.perf_counter() - started_at,
        )

    def plan(self, current_coord: int, route_loop: Sequence[int]) -> NavMeshRouteEntryResult:
        if not route_loop:
            raise ValueError("Cannot plan a navmesh route entry without a route loop")
        started_at = time.perf_counter()
        start_x, start_y = coord_to_xy(current_coord)
        start = self._locate_ui(start_x, start_y)
        if start is None:
            raise RuntimeError("Current position does not resolve to a ground navmesh polygon")
        component_id = self.mesh.component_id(start.polygon_id)
        hazard_egress = self.coord_is_hazard(current_coord)
        transition_cost = (
            self._hazard_egress_transition_cost
            if hazard_egress
            else self._transition_cost
        )
        candidate_indexes = sorted(
            range(len(route_loop)),
            key=lambda index: (
                math.dist((start_x, start_y), coord_to_xy(route_loop[index])),
                index,
            ),
        )[: self.settings.candidate_count]

        candidates_with_ground = 0
        candidates_with_path = 0
        rejected_terrain_paths = 0
        best: tuple[tuple[float, int], int, NavPath, dict[str, Any]] | None = None
        for route_index in candidate_indexes:
            target_x, target_y = coord_to_xy(route_loop[route_index])
            if self.ui_is_hazard(target_x, target_y):
                continue
            target = self._locate_ui(target_x, target_y)
            if target is None or self.mesh.component_id(target.polygon_id) != component_id:
                continue
            candidates_with_ground += 1
            path = self.mesh.find_path(
                start,
                target,
                max_expansions=self.settings.max_expansions,
                transition_cost=transition_cost,
            )
            if path is None:
                continue
            candidates_with_path += 1
            dense = densify_path(path.points, self.settings.waypoint_spacing_yards)
            validation = self._validate_path(dense, hazard_egress=hazard_egress)
            if not validation["walkable"]:
                rejected_terrain_paths += 1
                continue
            score = (path.world_length, route_index)
            if best is None or score < best[0]:
                best = score, route_index, path, validation

        if best is None:
            raise RuntimeError(
                "No terrain-validated navmesh path reaches a candidate route entry"
            )
        _, target_route_index, path, validation = best
        dense = densify_path(path.points, self.settings.waypoint_spacing_yards)
        entry_plan = route_entry_plan_from_recast_path(
            dense,
            zone=self.zone,
            zone_id=self.zone_id,
            target_route_index=target_route_index,
            current_coord=current_coord,
        )
        if not entry_plan.waypoints:
            entry_plan = RouteEntryPlan(
                waypoints=(
                    MiningNode(
                        zone_id=self.zone_id,
                        coord=route_loop[target_route_index],
                        ore_type="route_entry",
                        node_id=-4_000_000,
                        route_index=target_route_index,
                        route_t=0.0,
                        source=ROUTE_ENTRY_WAYPOINT_SOURCE,
                    ),
                ),
                target_route_sort_key=float(target_route_index),
            )
        return NavMeshRouteEntryResult(
            entry_plan=entry_plan,
            target_route_index=target_route_index,
            world_length_yards=path.world_length,
            corridor_polygon_count=len(path.polygon_ids),
            funnel_point_count=len(path.points),
            waypoint_count=len(entry_plan.waypoints),
            candidates_considered=len(candidate_indexes),
            candidates_with_ground=candidates_with_ground,
            candidates_with_path=candidates_with_path,
            rejected_terrain_paths=rejected_terrain_paths,
            terrain_validation=validation,
            load_seconds=self.load_seconds,
            plan_seconds=time.perf_counter() - started_at,
        )

    def plan_to_target(
        self,
        current_coord: int,
        target_coord: int,
        *,
        target_route_index: int,
    ) -> NavMeshRouteEntryResult:
        started_at = time.perf_counter()
        start = self._locate_coord(current_coord)
        target = self._locate_coord(target_coord)
        if start is None or target is None:
            raise RuntimeError("Route segment endpoint does not resolve to ground navmesh")
        if self.mesh.component_id(start.polygon_id) != self.mesh.component_id(target.polygon_id):
            raise RuntimeError("Route segment endpoints are in different navmesh components")
        hazard_egress = self.coord_is_hazard(current_coord)
        if self.coord_is_hazard(target_coord):
            raise RuntimeError("Route target is inside a configured hazard")
        path = self.mesh.find_path(
            start,
            target,
            max_expansions=self.settings.max_expansions,
            transition_cost=(
                self._hazard_egress_transition_cost
                if hazard_egress
                else self._transition_cost
            ),
        )
        if path is None:
            raise RuntimeError("No terrain-constrained navmesh path reaches route target")
        dense = densify_path(path.points, self.settings.waypoint_spacing_yards)
        validation = self._validate_path(dense, hazard_egress=hazard_egress)
        if not validation["walkable"]:
            raise RuntimeError("Navmesh path crosses a client-terrain or hazard veto")
        entry_plan = route_entry_plan_from_recast_path(
            dense,
            zone=self.zone,
            zone_id=self.zone_id,
            target_route_index=target_route_index,
            current_coord=current_coord,
        )
        return NavMeshRouteEntryResult(
            entry_plan=entry_plan,
            target_route_index=target_route_index,
            world_length_yards=path.world_length,
            corridor_polygon_count=len(path.polygon_ids),
            funnel_point_count=len(path.points),
            waypoint_count=len(entry_plan.waypoints),
            candidates_considered=1,
            candidates_with_ground=1,
            candidates_with_path=1,
            rejected_terrain_paths=0,
            terrain_validation=validation,
            load_seconds=self.load_seconds,
            plan_seconds=time.perf_counter() - started_at,
        )

    def _locate_ui(self, ui_x: float, ui_y: float) -> NavLocation | None:
        world_x, world_y = self.zone.ui_to_world(ui_x, ui_y)
        row, col = self.raster.ui_to_grid(ui_x, ui_y, self.zone)
        world_z = self.raster.sample_height(row, col)
        if world_z is None:
            return None
        return self.mesh.nearest_location(
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
            max_horizontal_distance=self.settings.max_nav_snap_yards,
            max_vertical_distance=self.settings.max_nav_vertical_yards,
        )

    def _locate_coord(self, coord: int) -> NavLocation | None:
        return self._locate_ui(*coord_to_xy(coord))

    def ui_is_hazard(self, ui_x: float, ui_y: float) -> bool:
        row, col = self.raster.ui_to_grid(ui_x, ui_y, self.zone)
        row_index, col_index = int(round(row)), int(round(col))
        rows, cols = self.hazard_mask.shape
        return bool(
            0 <= row_index < rows
            and 0 <= col_index < cols
            and self.hazard_mask[row_index, col_index]
        )

    def coord_is_hazard(self, coord: int) -> bool:
        return self.ui_is_hazard(*coord_to_xy(coord))

    def _validate_path(
        self,
        points: Sequence[tuple[float, float, float]],
        *,
        hazard_egress: bool,
    ) -> dict[str, Any]:
        if not hazard_egress:
            return validate_navmesh_path(
                points,
                raster=self.raster,
                zone=self.zone,
                cost_map=self.cost_map,
                min_clearance=self.settings.min_clearance_cells,
            )
        terrain = validate_navmesh_path(
            points,
            raster=self.raster,
            zone=self.zone,
            cost_map=self.base_cost_map,
            min_clearance=self.settings.min_clearance_cells,
        )
        egress = validate_hazard_egress_path(
            points,
            raster=self.raster,
            zone=self.zone,
            hazard_mask=self.hazard_mask,
        )
        return {
            **terrain,
            **egress,
            "walkable": bool(terrain["walkable"] and egress["hazard_egress_valid"]),
        }


class NavMeshRouteGuidance:
    """Cache a short navmesh corridor for the currently owned route target."""

    def __init__(
        self,
        planner: NavMeshRouteEntryPlanner,
        *,
        reached_distance: float = 0.08,
        pass_through_distance: float = 0.25,
        replan_distance: float = 0.55,
    ) -> None:
        self.planner = planner
        self.reached_distance = max(0.0, float(reached_distance))
        self.pass_through_distance = max(
            self.reached_distance,
            float(pass_through_distance),
        )
        self.replan_distance = max(self.reached_distance, float(replan_distance))
        self.target_coord: int | None = None
        self.target_route_index: int = 0
        self.queue: list[MiningNode] = []
        self.last_result: NavMeshRouteEntryResult | None = None
        self.last_error: str | None = None

    def observe(
        self,
        current_coord: int,
        strategic_target_coord: int,
        *,
        target_route_index: int,
        active: bool,
    ) -> NavMeshGuidanceObservation:
        if not active:
            return NavMeshGuidanceObservation(
                steering_coord=strategic_target_coord,
                strategic_target_coord=strategic_target_coord,
                status="suspended",
                remaining_waypoints=len(self.queue),
                replanned=False,
            )

        target_changed = self.target_coord != strategic_target_coord
        corridor_lost = bool(
            not target_changed
            and self.queue
            and min(
                math.dist(coord_to_xy(current_coord), coord_to_xy(node.coord))
                for node in self.queue
            )
            > self.replan_distance
        )
        # A failed segment plan is stable evidence for this strategic target.
        # Retrying the same expensive query every perception tick cannot change
        # the disconnected component or terrain veto; a new target will retry.
        replanned = target_changed or corridor_lost
        if replanned:
            self.target_coord = strategic_target_coord
            self.target_route_index = int(target_route_index)
            self.queue = []
            self.last_result = None
            self.last_error = None
            try:
                self.last_result = self.planner.plan_to_target(
                    current_coord,
                    strategic_target_coord,
                    target_route_index=self.target_route_index,
                )
                self.queue = list(self.last_result.entry_plan.waypoints)
            except (OSError, RuntimeError, ValueError) as exc:
                self.last_error = str(exc)

        if self.last_error is not None:
            return NavMeshGuidanceObservation(
                steering_coord=current_coord,
                strategic_target_coord=strategic_target_coord,
                status="blocked",
                remaining_waypoints=0,
                replanned=replanned,
                reason=self.last_error,
            )
        self._advance_passed_waypoints(current_coord)
        return NavMeshGuidanceObservation(
            steering_coord=(self.queue[0].coord if self.queue else strategic_target_coord),
            strategic_target_coord=strategic_target_coord,
            status="guided" if self.queue else "target_visible",
            remaining_waypoints=len(self.queue),
            replanned=replanned,
            world_length_yards=(
                self.last_result.world_length_yards if self.last_result is not None else None
            ),
        )

    def _advance_passed_waypoints(self, current_coord: int) -> None:
        current = coord_to_xy(current_coord)
        while self.queue:
            first = coord_to_xy(self.queue[0].coord)
            if math.dist(current, first) <= self.reached_distance:
                self.queue.pop(0)
                continue
            if len(self.queue) < 2 or math.dist(current, first) > self.pass_through_distance:
                return
            second = coord_to_xy(self.queue[1].coord)
            segment = (second[0] - first[0], second[1] - first[1])
            current_from_first = (current[0] - first[0], current[1] - first[1])
            if segment[0] * current_from_first[0] + segment[1] * current_from_first[1] <= 0.0:
                return
            self.queue.pop(0)


def navmesh_profile_for_zone(config: dict[str, Any], zone_id: int) -> dict[str, Any]:
    entry = config.get("route", {}).get("entry", {})
    profiles = entry.get("navmesh_profiles", {}) if isinstance(entry, dict) else {}
    profile = profiles.get(str(zone_id)) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise ValueError(f"No navmesh route-entry profile is configured for zone {zone_id}")
    required = ("manifest_path", "adt_dir", "mmap_dir")
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(f"Navmesh profile {zone_id} is missing: {', '.join(missing)}")
    return profile


def zone_from_manifest(manifest: dict[str, Any]) -> ZoneTransform:
    area = manifest["world_map_area"]
    return ZoneTransform(
        left=float(area["left"]),
        right=float(area["right"]),
        top=float(area["top"]),
        bottom=float(area["bottom"]),
    )


def route_entry_plan_from_recast_path(
    points: Sequence[tuple[float, float, float]],
    *,
    zone: ZoneTransform,
    zone_id: int,
    target_route_index: int,
    current_coord: int | None = None,
) -> RouteEntryPlan:
    coords: list[int] = []
    for recast_x, _height, recast_z in points:
        ui_x, ui_y = zone.world_to_ui(recast_z, recast_x)
        coord = xy_to_coord(ui_x, ui_y)
        if not coords or coords[-1] != coord:
            coords.append(coord)
    if current_coord is not None and coords and coords[0] == current_coord:
        coords.pop(0)
    nodes = tuple(
        MiningNode(
            zone_id=int(zone_id),
            coord=coord,
            ore_type="route_entry",
            node_id=-(4_000_000 + index),
            route_index=target_route_index,
            route_t=(index + 1) / (len(coords) + 1),
            source=ROUTE_ENTRY_WAYPOINT_SOURCE,
        )
        for index, coord in enumerate(coords)
    )
    return RouteEntryPlan(nodes, float(target_route_index))


def validate_navmesh_path(
    points: Sequence[tuple[float, float, float]],
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
    min_clearance: float,
) -> dict[str, Any]:
    grid_points: list[tuple[int, int]] = []
    for recast_x, _height, recast_z in points:
        ui_x, ui_y = zone.world_to_ui(recast_z, recast_x)
        row, col = raster.ui_to_grid(ui_x, ui_y, zone)
        grid_points.append((int(round(row)), int(round(col))))
    cells: list[tuple[int, int]] = []
    if len(grid_points) == 1:
        cells.append(grid_points[0])
    for first, second in zip(grid_points, grid_points[1:]):
        cells.extend(supercover_line(first, second))
    cells = list(dict.fromkeys(cells))

    blocked_cells: list[tuple[int, int]] = []
    slopes: list[float] = []
    clearances: list[float] = []
    rows, cols = cost_map.cost.shape
    for row, col in cells:
        if 0 <= row < rows and 0 <= col < cols:
            slope = float(cost_map.slope_degrees[row, col])
            clearance = float(cost_map.clearance[row, col])
            if math.isfinite(slope):
                slopes.append(slope)
            if math.isfinite(clearance):
                clearances.append(clearance)
        if not cost_map.is_walkable((row, col), min_clearance=min_clearance):
            blocked_cells.append((row, col))
    return {
        "walkable": not blocked_cells and bool(cells),
        "crossed_cell_count": len(cells),
        "blocked_cell_count": len(blocked_cells),
        "first_blocked_cell": list(blocked_cells[0]) if blocked_cells else None,
        "max_slope_degrees": round(max(slopes), 4) if slopes else None,
        "min_clearance_cells": round(min(clearances), 4) if clearances else None,
    }


def build_terrain_transition_cost(
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
    min_clearance: float,
):
    def transition_cost(
        first: NavPolygon,
        second: NavPolygon,
        portal: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float | None:
        portal_center = tuple(
            (float(portal[0][axis]) + float(portal[1][axis])) * 0.5
            for axis in range(3)
        )
        validation = validate_navmesh_path(
            (first.center, portal_center, second.center),
            raster=raster,
            zone=zone,
            cost_map=cost_map,
            min_clearance=min_clearance,
        )
        if not validation["walkable"]:
            return None
        base_cost = math.dist(
            (first.center[0], first.center[2]),
            (second.center[0], second.center[2]),
        )
        max_slope = float(validation["max_slope_degrees"] or 0.0)
        slope_fraction = max_slope / max(1.0, float(cost_map.config.hard_slope_degrees))
        return base_cost * (1.0 + slope_fraction * 0.25)

    return transition_cost


def build_hazard_egress_transition_cost(
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    base_cost_map: TerrainCostMap,
    hazard_mask: np.ndarray,
    min_clearance: float,
):
    """Allow an initial hazard prefix to exit, while forbidding every re-entry."""

    base_transition_cost = build_terrain_transition_cost(
        raster=raster,
        zone=zone,
        cost_map=base_cost_map,
        min_clearance=min_clearance,
    )

    def point_in_hazard(point: tuple[float, float, float]) -> bool:
        ui_x, ui_y = zone.world_to_ui(point[2], point[0])
        row, col = raster.ui_to_grid(ui_x, ui_y, zone)
        row_index, col_index = int(round(row)), int(round(col))
        rows, cols = hazard_mask.shape
        return bool(
            0 <= row_index < rows
            and 0 <= col_index < cols
            and hazard_mask[row_index, col_index]
        )

    def transition_cost(
        first: NavPolygon,
        second: NavPolygon,
        portal: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float | None:
        base_cost = base_transition_cost(first, second, portal)
        if base_cost is None:
            return None
        portal_center = tuple(
            (float(portal[0][axis]) + float(portal[1][axis])) * 0.5
            for axis in range(3)
        )
        first_in_hazard = point_in_hazard(first.center)
        portal_in_hazard = point_in_hazard(portal_center)
        second_in_hazard = point_in_hazard(second.center)
        if not first_in_hazard and (portal_in_hazard or second_in_hazard):
            return None
        return base_cost * (4.0 if first_in_hazard or second_in_hazard else 1.0)

    return transition_cost


def validate_hazard_egress_path(
    points: Sequence[tuple[float, float, float]],
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    hazard_mask: np.ndarray,
) -> dict[str, Any]:
    cells: list[tuple[int, int]] = []
    grid_points: list[tuple[int, int]] = []
    for recast_x, _height, recast_z in points:
        ui_x, ui_y = zone.world_to_ui(recast_z, recast_x)
        row, col = raster.ui_to_grid(ui_x, ui_y, zone)
        grid_points.append((int(round(row)), int(round(col))))
    if len(grid_points) == 1:
        cells.append(grid_points[0])
    for first, second in zip(grid_points, grid_points[1:]):
        segment_cells = supercover_line(first, second)
        if cells and segment_cells and cells[-1] == segment_cells[0]:
            segment_cells = segment_cells[1:]
        cells.extend(segment_cells)

    rows, cols = hazard_mask.shape
    occupancy = [
        bool(0 <= row < rows and 0 <= col < cols and hazard_mask[row, col])
        for row, col in cells
    ]
    start_in_hazard = bool(occupancy and occupancy[0])
    first_safe_index = next(
        (index for index, in_hazard in enumerate(occupancy) if not in_hazard),
        None,
    )
    exited_hazard = first_safe_index is not None
    reentered_hazard = bool(
        first_safe_index is not None and any(occupancy[first_safe_index + 1 :])
    )
    valid = bool(start_in_hazard and exited_hazard and not reentered_hazard)
    return {
        "hazard_egress": True,
        "hazard_egress_valid": valid,
        "hazard_start_inside": start_in_hazard,
        "hazard_exited": exited_hazard,
        "hazard_reentered": reentered_hazard,
        "hazard_cell_count": sum(occupancy),
        "hazard_first_safe_cell_index": first_safe_index,
    }


def build_hazard_mask(
    shape: tuple[int, ...],
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    hazards: Sequence[dict[str, Any]],
) -> np.ndarray:
    combined = np.zeros(shape[:2], dtype=np.uint8)
    rows, cols = np.indices(combined.shape)
    for hazard in hazards:
        mask = np.zeros(combined.shape, dtype=np.uint8)
        points = hazard.get("points")
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
            grid_points: list[tuple[int, int]] = []
            for point in points:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    ui_x, ui_y = float(point["x"]), float(point["y"])
                elif isinstance(point, Sequence) and len(point) >= 2:
                    ui_x, ui_y = float(point[0]), float(point[1])
                else:
                    continue
                row, col = raster.ui_to_grid(ui_x, ui_y, zone)
                grid_points.append((int(round(col)), int(round(row))))
            if len(grid_points) >= 3:
                cv2.fillPoly(mask, [np.asarray(grid_points, dtype=np.int32)], 1)
        elif all(key in hazard for key in ("x", "y", "radius_yards")):
            center_row, center_col = raster.ui_to_grid(
                float(hazard["x"]), float(hazard["y"]), zone
            )
            radius = max(0.0, float(hazard["radius_yards"]) / GRID_SPACING)
            mask[
                np.square(rows - center_row) + np.square(cols - center_col)
                <= radius * radius
            ] = 1
        margin = int(round(max(0.0, float(hazard.get("margin_yards", 0.0))) / GRID_SPACING))
        if margin > 0 and np.any(mask):
            size = margin * 2 + 1
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            )
        combined |= mask
    return combined.astype(bool)


def apply_hazard_exclusions(
    cost_map: TerrainCostMap,
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    hazards: Sequence[dict[str, Any]],
) -> tuple[TerrainCostMap, int]:
    blocked = cost_map.blocked.copy()
    blocked |= build_hazard_mask(
        blocked.shape,
        raster=raster,
        zone=zone,
        hazards=hazards,
    )
    added = int(np.count_nonzero(blocked & ~cost_map.blocked))
    if added == 0:
        return cost_map, 0
    clearance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3).astype(
        np.float32
    )
    config = cost_map.config
    if config.clearance_cells > 0.0:
        old_fraction = np.clip(
            (config.clearance_cells - cost_map.clearance) / config.clearance_cells,
            0.0,
            1.0,
        )
        new_fraction = np.clip(
            (config.clearance_cells - clearance) / config.clearance_cells,
            0.0,
            1.0,
        )
        cost = (
            cost_map.cost
            - config.clearance_weight * np.square(old_fraction)
            + config.clearance_weight * np.square(new_fraction)
        ).astype(np.float32)
    else:
        cost = cost_map.cost.copy()
    cost[blocked] = np.inf
    return (
        TerrainCostMap(
            cost=cost,
            slope_degrees=cost_map.slope_degrees.copy(),
            roughness=cost_map.roughness.copy(),
            clearance=clearance,
            blocked=blocked,
            config=config,
        ),
        added,
    )
