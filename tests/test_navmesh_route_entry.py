from __future__ import annotations

import numpy as np
import pytest

from vision_bot.coords import xy_to_coord
from vision_bot.navmesh_route_entry import (
    NavMeshRouteEntryResult,
    NavMeshRouteGuidance,
    apply_hazard_exclusions,
    build_hazard_mask,
    navmesh_profile_for_zone,
    route_entry_plan_from_recast_path,
    validate_hazard_egress_path,
)
from vision_bot.route_planner import MiningNode, ROUTE_ENTRY_WAYPOINT_SOURCE, RouteEntryPlan
from vision_bot.terrain_routing import (
    TerrainCostConfig,
    TerrainCostMap,
    TerrainRaster,
    ZoneTransform,
)


def test_navmesh_profile_is_selected_by_route_zone() -> None:
    config = {
        "route": {
            "entry": {
                "navmesh_profiles": {
                    "162": {
                        "manifest_path": "manifest.json",
                        "adt_dir": "adt",
                        "mmap_dir": "mmaps",
                    }
                }
            }
        }
    }

    assert navmesh_profile_for_zone(config, 162)["adt_dir"] == "adt"
    with pytest.raises(ValueError, match="zone 102"):
        navmesh_profile_for_zone(config, 102)


def test_recast_path_becomes_ordered_route_entry_waypoints() -> None:
    zone = ZoneTransform(left=0.0, right=100.0, top=0.0, bottom=100.0)
    current = xy_to_coord(20.0, 10.0)

    plan = route_entry_plan_from_recast_path(
        ((20.0, 2.0, 10.0), (30.0, 2.0, 20.0), (40.0, 2.0, 30.0)),
        zone=zone,
        zone_id=162,
        target_route_index=275,
        current_coord=current,
    )

    assert plan.target_route_sort_key == 275.0
    assert [node.coord for node in plan.waypoints] == [
        xy_to_coord(30.0, 20.0),
        xy_to_coord(40.0, 30.0),
    ]
    assert all(node.source == ROUTE_ENTRY_WAYPOINT_SOURCE for node in plan.waypoints)


def test_hazard_polygon_becomes_a_hard_terrain_veto() -> None:
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=0,
        tile_min_y=0,
        tile_max_x=0,
        tile_max_y=0,
    )
    shape = raster.heights.shape
    config = TerrainCostConfig()
    cost_map = TerrainCostMap(
        cost=np.ones(shape, dtype=np.float32),
        slope_degrees=np.zeros(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        clearance=np.full(shape, 100.0, dtype=np.float32),
        blocked=np.zeros(shape, dtype=bool),
        config=config,
    )
    zone = ZoneTransform(
        left=17066.6666666666,
        right=16533.3333333333,
        top=17066.6666666666,
        bottom=16533.3333333333,
    )

    updated, added = apply_hazard_exclusions(
        cost_map,
        raster=raster,
        zone=zone,
        hazards=[{"points": [[40, 40], [60, 40], [60, 60], [40, 60]]}],
    )

    center = tuple(round(value) for value in raster.ui_to_grid(50.0, 50.0, zone))
    assert added > 0
    assert updated.blocked[center]
    assert not np.isfinite(updated.cost[center])


def test_hazard_egress_accepts_one_way_exit_and_rejects_reentry() -> None:
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=0,
        tile_min_y=0,
        tile_max_x=0,
        tile_max_y=0,
    )
    zone = ZoneTransform(
        left=17066.6666666666,
        right=16533.3333333333,
        top=17066.6666666666,
        bottom=16533.3333333333,
    )
    hazard_mask = build_hazard_mask(
        raster.heights.shape,
        raster=raster,
        zone=zone,
        hazards=[{"points": [[40, 40], [60, 40], [60, 60], [40, 60]]}],
    )

    def recast_point(ui_x: float, ui_y: float) -> tuple[float, float, float]:
        world_x, world_y = zone.ui_to_world(ui_x, ui_y)
        return world_y, 0.0, world_x

    exit_only = validate_hazard_egress_path(
        (recast_point(50, 50), recast_point(65, 50), recast_point(70, 50)),
        raster=raster,
        zone=zone,
        hazard_mask=hazard_mask,
    )
    reentry = validate_hazard_egress_path(
        (recast_point(50, 50), recast_point(65, 50), recast_point(50, 50)),
        raster=raster,
        zone=zone,
        hazard_mask=hazard_mask,
    )

    assert exit_only["hazard_egress_valid"] is True
    assert exit_only["hazard_exited"] is True
    assert exit_only["hazard_reentered"] is False
    assert reentry["hazard_egress_valid"] is False
    assert reentry["hazard_reentered"] is True


class _GuidancePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan_to_target(self, current_coord, target_coord, *, target_route_index):
        self.calls += 1
        plan = RouteEntryPlan(
            waypoints=(
                MiningNode(162, xy_to_coord(10.10, 10.00), "entry", -1),
                MiningNode(162, target_coord, "entry", -2),
            ),
            target_route_sort_key=float(target_route_index),
        )
        return NavMeshRouteEntryResult(
            entry_plan=plan,
            target_route_index=target_route_index,
            world_length_yards=20.0,
            corridor_polygon_count=3,
            funnel_point_count=2,
            waypoint_count=2,
            candidates_considered=1,
            candidates_with_ground=1,
            candidates_with_path=1,
            rejected_terrain_paths=0,
            terrain_validation={"walkable": True},
            load_seconds=0.0,
            plan_seconds=0.01,
        )


class _FailingGuidancePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan_to_target(self, current_coord, target_coord, *, target_route_index):
        self.calls += 1
        raise RuntimeError("Route segment endpoints are in different navmesh components")


def test_route_guidance_reuses_path_and_advances_local_waypoints() -> None:
    planner = _GuidancePlanner()
    guidance = NavMeshRouteGuidance(planner, reached_distance=0.08)
    target = xy_to_coord(10.30, 10.00)

    first = guidance.observe(
        xy_to_coord(10.00, 10.00),
        target,
        target_route_index=5,
        active=True,
    )
    second = guidance.observe(
        xy_to_coord(10.09, 10.00),
        target,
        target_route_index=5,
        active=True,
    )

    assert first.status == "guided"
    assert first.steering_coord == xy_to_coord(10.10, 10.00)
    assert second.steering_coord == target
    assert planner.calls == 1


def test_route_guidance_advances_a_close_waypoint_after_crossing_its_plane() -> None:
    planner = _GuidancePlanner()
    guidance = NavMeshRouteGuidance(
        planner,
        reached_distance=0.02,
        pass_through_distance=0.25,
    )
    target = xy_to_coord(10.30, 10.00)

    guidance.observe(
        xy_to_coord(10.00, 10.00),
        target,
        target_route_index=5,
        active=True,
    )
    passed = guidance.observe(
        xy_to_coord(10.12, 10.05),
        target,
        target_route_index=5,
        active=True,
    )

    assert passed.steering_coord == target
    assert passed.remaining_waypoints == 1
    assert planner.calls == 1


def test_route_guidance_does_not_replan_same_failed_target_every_tick() -> None:
    planner = _FailingGuidancePlanner()
    guidance = NavMeshRouteGuidance(planner)
    first_target = xy_to_coord(10.30, 10.00)
    second_target = xy_to_coord(10.60, 10.00)

    first = guidance.observe(
        xy_to_coord(10.00, 10.00),
        first_target,
        target_route_index=5,
        active=True,
    )
    repeated = guidance.observe(
        xy_to_coord(10.01, 10.00),
        first_target,
        target_route_index=5,
        active=True,
    )
    changed = guidance.observe(
        xy_to_coord(10.01, 10.00),
        second_target,
        target_route_index=6,
        active=True,
    )

    assert first.blocked and repeated.blocked and changed.blocked
    assert not repeated.replanned
    assert planner.calls == 2
