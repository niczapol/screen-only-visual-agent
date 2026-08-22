from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.coords import xy_to_coord
from vision_bot.mmap_navmesh import MMapNavMesh, NavLocation, NavPolygon, densify_path
from vision_bot.terrain_routing import (
    TerrainCostConfig,
    TerrainCostMap,
    TerrainRaster,
    ZoneTransform,
    build_terrain_cost_map,
    build_terrain_raster,
    supercover_line,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an anchor-component route from AzerothCore mmaps"
    )
    parser.add_argument("--input-route", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adt-dir", required=True)
    parser.add_argument("--mmap-dir", required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--anchor-x", type=float, required=True)
    parser.add_argument("--anchor-y", type=float, required=True)
    parser.add_argument("--output-route", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-overlay", required=True)
    parser.add_argument("--max-nav-snap", type=float, default=30.0)
    parser.add_argument("--max-nav-vertical", type=float, default=50.0)
    parser.add_argument("--max-waypoint-spacing", type=float, default=25.0)
    parser.add_argument("--min-clearance", type=float, default=1.0)
    parser.add_argument("--soft-slope-degrees", type=float, default=12.0)
    parser.add_argument("--hard-slope-degrees", type=float, default=32.0)
    parser.add_argument("--slope-weight", type=float, default=12.0)
    parser.add_argument("--roughness-weight", type=float, default=0.12)
    parser.add_argument("--clearance-cells", type=float, default=5.0)
    parser.add_argument("--clearance-weight", type=float, default=9.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.max_nav_snap < 0.0 or args.max_nav_vertical < 0.0:
        raise ValueError("Navmesh snap limits must be non-negative")
    if args.max_waypoint_spacing <= 0.0:
        raise ValueError("--max-waypoint-spacing must be positive")
    if args.min_clearance < 0.0:
        raise ValueError("--min-clearance must be non-negative")

    input_path = Path(args.input_route)
    manifest_path = Path(args.manifest)
    source_route = json.loads(input_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(Path(args.adt_dir).glob("*.adt"))
    terrain_config = TerrainCostConfig(
        soft_slope_degrees=float(args.soft_slope_degrees),
        hard_slope_degrees=float(args.hard_slope_degrees),
        slope_weight=float(args.slope_weight),
        roughness_weight=float(args.roughness_weight),
        clearance_cells=float(args.clearance_cells),
        clearance_weight=float(args.clearance_weight),
    )
    cost_map = build_terrain_cost_map(raster, terrain_config)
    transition_cost = build_terrain_transition_cost(
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        min_clearance=float(args.min_clearance),
    )
    tiles = [(int(item["x"]), int(item["y"])) for item in manifest["tiles"]]
    mesh = MMapNavMesh.load_tiles(
        args.mmap_dir,
        map_id=int(args.map_id),
        tiles=tiles,
    )

    anchor = locate_ui(
        mesh,
        raster,
        zone,
        float(args.anchor_x),
        float(args.anchor_y),
        max_nav_snap=float(args.max_nav_snap),
        max_nav_vertical=float(args.max_nav_vertical),
    )
    if anchor is None:
        raise RuntimeError("Anchor coordinate does not resolve to a ground navmesh polygon")
    anchor_component = mesh.component_id(anchor.polygon_id)

    accepted: list[tuple[dict[str, Any], NavLocation]] = []
    rejected: list[dict[str, Any]] = [dict(item) for item in source_route.get("rejected_nodes", [])]
    for node in source_route.get("route_nodes", []):
        location = locate_ui(
            mesh,
            raster,
            zone,
            float(node["x"]),
            float(node["y"]),
            max_nav_snap=float(args.max_nav_snap),
            max_nav_vertical=float(args.max_nav_vertical),
        )
        if location is None:
            rejected.append({**node, "reason": "navmesh_no_ground_approach"})
        elif mesh.component_id(location.polygon_id) != anchor_component:
            rejected.append({**node, "reason": "navmesh_outside_anchor_component"})
        else:
            accepted.append((dict(node), location))

    if not accepted:
        raise RuntimeError("No route nodes share the anchor ground component")

    accepted = nearest_neighbor_order(anchor, accepted)
    route_points: list[tuple[float, float, float]] = [anchor.point]
    current = anchor
    accepted_nodes: list[dict[str, Any]] = []
    segment_metrics: list[dict[str, Any]] = []
    segment_rejections: list[dict[str, Any]] = []
    for order, (node, location) in enumerate(accepted):
        path = mesh.find_path(current, location, transition_cost=transition_cost)
        if path is None:
            rejected.append({**node, "reason": "navmesh_no_client_terrain_constrained_path"})
            continue
        dense = densify_path(path.points, float(args.max_waypoint_spacing))
        terrain_validation = validate_navmesh_path(
            dense,
            raster=raster,
            zone=zone,
            cost_map=cost_map,
            min_clearance=float(args.min_clearance),
        )
        if not terrain_validation["walkable"]:
            rejection = {
                **node,
                "reason": "navmesh_constrained_path_crosses_client_terrain_blocker",
                "terrain_validation": terrain_validation,
            }
            rejected.append(rejection)
            segment_rejections.append(
                {
                    "target_node_id": node.get("node_id"),
                    "terrain_validation": terrain_validation,
                }
            )
            continue
        route_points.extend(dense[1:])
        node.update(
            {
                "route_order": len(accepted_nodes),
                "route_index": len(route_points) - 1,
                "route_t": 0.0,
                "distance_to_route": round(float(location.horizontal_distance), 4),
                "navmesh_polygon_id": int(location.polygon_id),
                "navmesh_approach_world": [round(value, 4) for value in location.point],
                "navmesh_approach_ui": [
                    round(value, 6)
                    for value in zone.world_to_ui(location.point[2], location.point[0])
                ],
            }
        )
        accepted_nodes.append(node)
        segment_metrics.append(
            {
                "index": order,
                "target_node_id": node.get("node_id"),
                "corridor_polygons": len(path.polygon_ids),
                "funnel_points": len(path.points),
                "world_length": round(path.world_length, 4),
                "graph_cost": round(path.graph_cost, 4),
                "terrain_validation": terrain_validation,
            }
        )
        current = location

    return_path = None
    return_validation = None
    route_status = "candidate"
    if not accepted_nodes:
        route_status = "rejected_no_validated_corridor"
        route_points = []
    else:
        return_path = mesh.find_path(current, anchor, transition_cost=transition_cost)
        if return_path is None:
            route_status = "rejected_no_return_path"
        else:
            dense_return = densify_path(return_path.points, float(args.max_waypoint_spacing))
            return_validation = validate_navmesh_path(
                dense_return,
                raster=raster,
                zone=zone,
                cost_map=cost_map,
                min_clearance=float(args.min_clearance),
            )
            if not return_validation["walkable"]:
                route_status = "rejected_constrained_return_crosses_client_terrain_blocker"
            else:
                route_points.extend(dense_return[1:])
                if _horizontal_distance(route_points[0], route_points[-1]) <= 0.01:
                    route_points[-1] = route_points[0]

    if route_status != "candidate":
        for node in accepted_nodes:
            rejected.append({**node, "reason": route_status})
        accepted_nodes = []
        route_points = []

    route_loop = []
    for index, point in enumerate(route_points):
        ui_x, ui_y = zone.world_to_ui(point[2], point[0])
        route_loop.append(
            {
                "index": index,
                "coord": xy_to_coord(ui_x, ui_y),
                "x": round(ui_x, 6),
                "y": round(ui_y, 6),
                "world_z": round(point[1], 4),
                "source": "azerothcore_mmap_funnel",
            }
        )

    metrics = {
        "map_id": int(args.map_id),
        "status": route_status,
        "terrain_cost_config": terrain_config.to_dict(),
        "min_clearance": float(args.min_clearance),
        "anchor_ui": [float(args.anchor_x), float(args.anchor_y)],
        "anchor_polygon_id": int(anchor.polygon_id),
        "anchor_component_id": int(anchor_component),
        "anchor_component_polygons": mesh.component_size(anchor.polygon_id),
        "loaded_ground_polygons": len(mesh.polygons),
        "accepted_node_count": len(accepted_nodes),
        "rejected_node_count": len(rejected),
        "route_waypoint_count": len(route_loop),
        "route_world_length": round(_polyline_length(route_points), 4),
        "segments": segment_metrics,
        "segment_rejections": segment_rejections,
        "return_segment": (
            {
                "corridor_polygons": len(return_path.polygon_ids),
                "funnel_points": len(return_path.points),
                "world_length": round(return_path.world_length, 4),
                "graph_cost": round(return_path.graph_cost, 4),
                "terrain_validation": return_validation,
            }
            if return_path is not None
            else None
        ),
        "source_paths": {
            "input_route": str(input_path),
            "manifest": str(manifest_path),
            "adt_dir": str(args.adt_dir),
            "mmap_dir": str(args.mmap_dir),
        },
    }
    result = {
        "schema_version": 3,
        "name": f"{source_route.get('name', 'route')}_mmap_anchor_cycle",
        "zone": source_route.get("zone", {}),
        "mode": "cyclic" if route_status == "candidate" else "blocked",
        "live_status": route_status,
        "source": {
            "route_name": source_route.get("name"),
            "route_data": str(input_path),
            "navigation": "AzerothCore mmap v20 ground polygons constrained by external client ADT",
        },
        "generation": {
            "bounds": source_route.get("generation", {}).get("bounds"),
            "anchor_ui": metrics["anchor_ui"],
            "route_waypoints": len(route_loop),
            "accepted_nodes": len(accepted_nodes),
            "rejected_nodes": len(rejected),
            "navmesh": metrics,
        },
        "route_loop": route_loop,
        "route_nodes": accepted_nodes,
        "rejected_nodes": rejected,
    }
    _write_json(result, args.output_route)
    _write_json(metrics, args.output_metrics)
    render_overlay(
        raster=raster,
        zone=zone,
        route_loop=route_loop,
        accepted_nodes=accepted_nodes,
        rejected_nodes=rejected,
        anchor_ui=(float(args.anchor_x), float(args.anchor_y)),
        output_path=args.output_overlay,
    )
    print(
        f"status={route_status} accepted={len(accepted_nodes)} rejected={len(rejected)} "
        f"waypoints={len(route_loop)} length={metrics['route_world_length']}"
    )


def zone_from_manifest(manifest: dict[str, Any]) -> ZoneTransform:
    area = manifest["world_map_area"]
    return ZoneTransform(
        left=float(area["left"]),
        right=float(area["right"]),
        top=float(area["top"]),
        bottom=float(area["bottom"]),
    )


def locate_ui(
    mesh: MMapNavMesh,
    raster: TerrainRaster,
    zone: Any,
    ui_x: float,
    ui_y: float,
    *,
    max_nav_snap: float,
    max_nav_vertical: float,
) -> NavLocation | None:
    world_x, world_y = zone.ui_to_world(ui_x, ui_y)
    row, col = raster.ui_to_grid(ui_x, ui_y, zone)
    world_z = raster.sample_height(row, col)
    if world_z is None:
        return None
    return mesh.nearest_location(
        world_x=world_x,
        world_y=world_y,
        world_z=world_z,
        max_horizontal_distance=max_nav_snap,
        max_vertical_distance=max_nav_vertical,
    )


def nearest_neighbor_order(
    start: NavLocation,
    nodes: list[tuple[dict[str, Any], NavLocation]],
) -> list[tuple[dict[str, Any], NavLocation]]:
    remaining = list(nodes)
    ordered: list[tuple[dict[str, Any], NavLocation]] = []
    current = start.point
    while remaining:
        selected = min(
            remaining,
            key=lambda item: (
                _horizontal_distance(current, item[1].point),
                int(item[0].get("node_id", 0)),
            ),
        )
        remaining.remove(selected)
        ordered.append(selected)
        current = selected[1].point
    return ordered


def validate_navmesh_path(
    points: Sequence[tuple[float, float, float]],
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
    min_clearance: float,
) -> dict[str, Any]:
    grid_points = []
    for recast_x, _height, recast_z in points:
        ui_x, ui_y = zone.world_to_ui(recast_z, recast_x)
        row, col = raster.ui_to_grid(ui_x, ui_y, zone)
        grid_points.append((int(round(row)), int(round(col))))
    return validate_grid_polyline(
        grid_points,
        cost_map=cost_map,
        min_clearance=min_clearance,
    )


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
        base_cost = _horizontal_distance(first.center, second.center)
        max_slope = float(validation["max_slope_degrees"] or 0.0)
        slope_fraction = max_slope / max(1.0, float(cost_map.config.hard_slope_degrees))
        return base_cost * (1.0 + slope_fraction * 0.25)

    return transition_cost


def validate_grid_polyline(
    points: Sequence[tuple[int, int]],
    *,
    cost_map: TerrainCostMap,
    min_clearance: float,
) -> dict[str, Any]:
    cells: list[tuple[int, int]] = []
    if len(points) == 1:
        cells.append(points[0])
    for first, second in zip(points, points[1:]):
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


def render_overlay(
    *,
    raster: TerrainRaster,
    zone: Any,
    route_loop: list[dict[str, Any]],
    accepted_nodes: list[dict[str, Any]],
    rejected_nodes: list[dict[str, Any]],
    anchor_ui: tuple[float, float],
    output_path: str | Path,
) -> None:
    heights = np.asarray(raster.heights, dtype=np.float32)
    valid = np.isfinite(heights)
    low, high = np.percentile(heights[valid], [2.0, 98.0])
    normalized = np.zeros_like(heights)
    normalized[valid] = np.clip((heights[valid] - low) / max(1.0e-6, high - low), 0.0, 1.0)
    image = cv2.cvtColor((normalized * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image[~valid] = (8, 8, 8)

    route_grid = [raster.ui_to_grid(float(item["x"]), float(item["y"]), zone) for item in route_loop]
    for first, second in zip(route_grid, route_grid[1:]):
        cv2.line(
            image,
            (int(round(first[1])), int(round(first[0]))),
            (int(round(second[1])), int(round(second[0]))),
            (255, 230, 0),
            4,
            cv2.LINE_AA,
        )
    for node in rejected_nodes:
        if "x" not in node or "y" not in node:
            continue
        row, col = raster.ui_to_grid(float(node["x"]), float(node["y"]), zone)
        if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
            cv2.drawMarker(image, (round(col), round(row)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
    for node in accepted_nodes:
        ui_x, ui_y = node["navmesh_approach_ui"]
        row, col = raster.ui_to_grid(float(ui_x), float(ui_y), zone)
        cv2.circle(image, (round(col), round(row)), 6, (0, 255, 255), -1, cv2.LINE_AA)
    row, col = raster.ui_to_grid(anchor_ui[0], anchor_ui[1], zone)
    cv2.drawMarker(image, (round(col), round(row)), (0, 255, 0), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    cv2.rectangle(image, (0, 0), (min(430, image.shape[1]), 88), (20, 20, 20), -1)
    cv2.putText(image, "Desolace navmesh anchor cycle", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "green=start  cyan=route  yellow=reachable ore", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(image, "magenta=outside current ground component", (12, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Failed to write overlay: {destination}")


def _polyline_length(points: Sequence[tuple[float, float, float]]) -> float:
    return sum(_horizontal_distance(first, second) for first, second in zip(points, points[1:]))


def _horizontal_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.dist((first[0], first[2]), (second[0], second[2]))


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
