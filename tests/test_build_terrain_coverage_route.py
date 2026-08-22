from __future__ import annotations

import math

import numpy as np

from scripts.build_terrain_coverage_route import (
    apply_live_hazard_exclusions,
    build_coverage_node_access_plans,
    build_node_candidates,
    cheapest_insertion_order,
    cyclic_polyline_distance,
    greedy_coverage_points,
    node_indexes_in_mask,
    plan_cycle,
    point_to_segment_distance,
)
from vision_bot.terrain_routing import TerrainCostConfig, TerrainCostMap, TerrainRaster


def _cost_map(size: int = 12) -> TerrainCostMap:
    shape = (size, size)
    return TerrainCostMap(
        cost=np.ones(shape, dtype=np.float32),
        slope_degrees=np.zeros(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        clearance=np.full(shape, 10.0, dtype=np.float32),
        blocked=np.zeros(shape, dtype=bool),
        config=TerrainCostConfig(),
    )


def test_point_to_segment_distance_projects_inside_and_clamps() -> None:
    assert point_to_segment_distance((2.0, 3.0), (0.0, 0.0), (0.0, 6.0)) == 2.0
    assert math.isclose(
        point_to_segment_distance((0.0, 8.0), (0.0, 0.0), (0.0, 6.0)),
        2.0,
    )
    assert cyclic_polyline_distance(
        (1.0, 1.0),
        [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
    ) == 1.0


def test_build_candidates_and_greedy_cover_all_walkable_nodes() -> None:
    cost_map = _cost_map()
    node_points = [(2.0, 2.0), (2.0, 4.0), (9.0, 9.0)]
    candidates, unreachable = build_node_candidates(
        node_points=node_points,
        cost_map=cost_map,
        radius_cells=2.1,
        min_clearance_cells=0.0,
        sector_count=8,
    )
    selected, unresolved = greedy_coverage_points(
        candidates,
        initially_covered=set(),
        target_indexes={0, 1, 2},
        cost_map=cost_map,
    )

    assert unreachable == []
    assert unresolved == set()
    assert len(selected) == 2
    assert len(selected[0]["newly_covered_node_indexes"]) == 2


def test_live_hazard_exclusion_blocks_validated_circle_before_pathfinding() -> None:
    cost_map = _cost_map(size=12)
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=0,
        tile_min_y=0,
        tile_max_x=0,
        tile_max_y=0,
    )

    class _Zone:
        pass

    raster_ui_to_grid = lambda _x, _y, _zone: (6.0, 6.0)
    object.__setattr__(raster, "ui_to_grid", raster_ui_to_grid)
    updated, added = apply_live_hazard_exclusions(
        cost_map,
        raster=raster,
        zone=_Zone(),
        hazards=[{"x": 69.3, "y": 39.7, "radius_yards": 5.0}],
    )

    assert added > 0
    assert updated.blocked[6, 6]
    assert not updated.is_walkable((6, 6))


def test_live_hazard_exclusion_blocks_polygon_and_margin_before_pathfinding() -> None:
    cost_map = _cost_map(size=20)
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=0,
        tile_min_y=0,
        tile_max_x=0,
        tile_max_y=0,
    )

    class _Zone:
        pass

    object.__setattr__(raster, "ui_to_grid", lambda x, y, _zone: (y, x))
    updated, added = apply_live_hazard_exclusions(
        cost_map,
        raster=raster,
        zone=_Zone(),
        hazards=[
            {
                "points": [[6, 6], [12, 6], [12, 12], [6, 12]],
                "margin_yards": 5.0,
            }
        ],
    )

    assert added > 0
    assert updated.blocked[9, 9]
    assert updated.blocked[5, 9]
    assert not updated.blocked[2, 2]


def test_node_indexes_in_mask_excludes_only_spawn_points_inside_hazard() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    excluded = node_indexes_in_mask(
        [(1.0, 1.0), (4.4, 4.6), (7.1, 7.1), (9.0, 9.0)],
        mask,
    )

    assert excluded == {1, 2}


def test_cheapest_insertion_preserves_human_backbone_order() -> None:
    backbone = [
        {"grid": [0, 0], "source": "human_reference", "id": "a"},
        {"grid": [0, 10], "source": "human_reference", "id": "b"},
        {"grid": [10, 10], "source": "human_reference", "id": "c"},
    ]
    additions = [
        {"grid": [5, 10], "source": "terrain_coverage", "id": "x"},
        {"grid": [10, 5], "source": "terrain_coverage", "id": "y"},
    ]

    ordered = cheapest_insertion_order(backbone, additions)

    human_ids = [item["id"] for item in ordered if item["source"] == "human_reference"]
    assert human_ids == ["a", "b", "c"]
    assert {item["id"] for item in ordered} == {"a", "b", "c", "x", "y"}


def test_plan_cycle_returns_walkable_simplified_loop() -> None:
    cost_map = _cost_map()
    scan_points = [
        {"grid": [2, 2]},
        {"grid": [2, 9]},
        {"grid": [9, 9]},
        {"grid": [9, 2]},
    ]

    route, segments = plan_cycle(
        scan_points,
        cost_map=cost_map,
        max_waypoint_spacing_cells=4.0,
        min_clearance_cells=0.0,
    )

    assert len(segments) == 4
    assert route[0] == (2, 2)
    assert route[-1] != route[0]
    assert all(cost_map.is_walkable(point) for point in route)


def test_build_coverage_node_access_plans_emits_reversible_side_branch() -> None:
    cost_map = _cost_map(size=16)
    cost_map.blocked[4:12, 7] = True
    cost_map.cost[cost_map.blocked] = np.inf
    cost_map.clearance[cost_map.blocked] = 0.0
    route = [(3, 3), (3, 11), (13, 11), (13, 3)]
    nodes = [{"coord": 0, "ore_type": "Iron"}]

    class _Raster:
        @staticmethod
        def grid_to_ui(row, col, _zone):
            return float(col), float(row)

    metrics = build_coverage_node_access_plans(
        nodes=nodes,
        node_points=[(8.0, 10.0)],
        covered_indexes={0},
        route_points=route,
        raster=_Raster(),
        zone=None,
        cost_map=cost_map,
        access_radius_cells=3.0,
        min_clearance_cells=0.0,
        max_waypoint_spacing_cells=3.0,
        sector_count=8,
        max_options=2,
        max_extra_cells=3.0,
        distance_weight=1.0,
    )

    assert metrics["planned_node_count"] == 1
    plan = nodes[0]["terrain_access_plan"]
    assert plan["options"]
    primary = plan["options"][plan["primary_option_rank"]]
    assert primary["approach_distance_yards"] == min(
        option["approach_distance_yards"] for option in plan["options"]
    )
    for option in plan["options"]:
        inbound = option["inbound"]["waypoints"]
        returned = option["return"]["waypoints"]
        assert [item["coord"] for item in returned] == [
            item["coord"] for item in reversed(inbound)
        ]


def test_access_plan_finishes_moving_toward_a_blocked_node() -> None:
    cost_map = _cost_map(size=16)
    cost_map.blocked[8, 7] = True
    cost_map.cost[cost_map.blocked] = np.inf
    cost_map.clearance[cost_map.blocked] = 0.0
    nodes = [{"coord": 0, "ore_type": "Iron"}]

    class _Raster:
        @staticmethod
        def grid_to_ui(row, col, _zone):
            return float(col), float(row)

    metrics = build_coverage_node_access_plans(
        nodes=nodes,
        node_points=[(8.0, 7.0)],
        covered_indexes={0},
        route_points=[(3, 3), (3, 12), (12, 12), (12, 3)],
        raster=_Raster(),
        zone=None,
        cost_map=cost_map,
        access_radius_cells=3.0,
        min_clearance_cells=0.0,
        max_waypoint_spacing_cells=3.0,
        sector_count=8,
        max_options=3,
        max_extra_cells=3.0,
        distance_weight=1.0,
    )

    assert metrics["planned_node_count"] == 1
    assert metrics["facing_aligned_option_count"] > 0
    aligned = next(
        option
        for option in nodes[0]["terrain_access_plan"]["options"]
        if option["node_facing_aligned"]
    )
    inbound = aligned["inbound"]["waypoints"]
    assert len(inbound) >= 2
    previous = inbound[-2]["grid"]
    approach = inbound[-1]["grid"]
    movement = (approach[0] - previous[0], approach[1] - previous[1])
    toward_node = (8 - approach[0], 7 - approach[1])
    assert movement[0] * toward_node[0] + movement[1] * toward_node[1] > 0
