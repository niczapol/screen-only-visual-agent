from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts.build_topographic_route import (
    PlannedSegment,
    build_node_access_plans,
    build_pairwise_terrain_costs,
    build_metrics,
    build_anchor_entry_route,
    compose_output_route,
    cycle_cost,
    derive_overlay_title,
    optimize_cycle_order,
    optimize_ore_node_approaches,
    parse_args,
    plan_cycle,
    restrict_to_primary_component,
    restrict_to_anchor_component,
    restrict_to_ui_bounds,
    snap_route_nodes,
    find_ore_approach_candidates,
)
from vision_bot.terrain_routing import (
    TerrainCostConfig,
    TerrainCostMap,
    TerrainPath,
    TerrainRaster,
    ZoneTransform,
    line_is_walkable,
)


def _cost_map(
    blocked: np.ndarray,
    *,
    slope: np.ndarray | None = None,
    clearance: np.ndarray | None = None,
) -> TerrainCostMap:
    blocked = np.asarray(blocked, dtype=bool)
    cost = np.ones(blocked.shape, dtype=np.float32)
    cost[blocked] = np.inf
    return TerrainCostMap(
        cost=cost,
        slope_degrees=(
            np.zeros(blocked.shape, dtype=np.float32)
            if slope is None
            else np.asarray(slope, dtype=np.float32)
        ),
        roughness=np.zeros(blocked.shape, dtype=np.float32),
        clearance=(
            np.full(blocked.shape, 10.0, dtype=np.float32)
            if clearance is None
            else np.asarray(clearance, dtype=np.float32)
        ),
        blocked=blocked,
        config=TerrainCostConfig(),
    )


@pytest.mark.parametrize(
    ("source_route", "expected"),
    [
        (
            {"name": "fallback", "zone": {"name": "The Barrens"}},
            "The Barrens topographic route",
        ),
        (
            {"name": "fallback", "zone": {"name": "Desolace"}},
            "Desolace topographic route",
        ),
        (
            {"name": "Barrens safe cycle", "zone": {"name": "  "}},
            "Barrens safe cycle topographic route",
        ),
        ({}, "Zone topographic route"),
        (
            {"zone": {"name": "Feralas TOPOGRAPHIC ROUTE"}},
            "Feralas TOPOGRAPHIC ROUTE",
        ),
    ],
)
def test_derive_overlay_title(
    source_route: dict[str, object],
    expected: str,
) -> None:
    assert derive_overlay_title(source_route) == expected


def test_parse_args_accepts_zone_specific_terrain_cost_profile() -> None:
    args = parse_args(
        [
            "--soft-slope-degrees",
            "12",
            "--hard-slope-degrees",
            "32",
            "--slope-weight",
            "11",
            "--roughness-weight",
            "0.12",
            "--clearance-cells",
            "5",
            "--clearance-weight",
            "9",
        ]
    )

    assert args.soft_slope_degrees == 12.0
    assert args.hard_slope_degrees == 32.0
    assert args.slope_weight == 11.0
    assert args.roughness_weight == 0.12
    assert args.clearance_cells == 5.0
    assert args.clearance_weight == 9.0
    assert args.approach_access_options == 3


def test_cycle_cost_rejects_invalid_matrix_and_order() -> None:
    valid = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    assert cycle_cost([0, 1, 2], valid) == 6.0
    with pytest.raises(ValueError, match="square"):
        cycle_cost([0, 1], np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        cycle_cost([0, 1], np.array([[0.0, np.inf], [np.inf, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        cycle_cost([0, 1], np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="exact integer permutation"):
        cycle_cost([0, 0, 2], valid)
    with pytest.raises(ValueError, match="exact integer permutation"):
        cycle_cost([False, 1, 2], valid)


def test_optimize_cycle_order_is_deterministic_and_two_opt_improves() -> None:
    costs = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 10.0],
            [2.0, 2.0, 10.0, 0.0],
        ]
    )
    original = costs.copy()

    nearest_neighbor = optimize_cycle_order(costs, max_two_opt_iterations=0)
    optimized = optimize_cycle_order(costs, max_two_opt_iterations=10)

    assert nearest_neighbor == [0, 1, 2, 3]
    assert optimized == [0, 2, 1, 3]
    assert cycle_cost(optimized, costs) < cycle_cost(nearest_neighbor, costs)
    np.testing.assert_array_equal(costs, original)


def test_optimize_cycle_order_rejects_negative_iteration_limit() -> None:
    costs = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="must be non-negative"):
        optimize_cycle_order(costs, max_two_opt_iterations=-1)


def test_build_pairwise_terrain_costs_is_symmetric_for_walkable_nodes() -> None:
    cost_map = _cost_map(np.zeros((8, 8), dtype=bool))
    accepted_nodes = [
        {"terrain_approach_grid": [1, 1]},
        {"terrain_approach_grid": [1, 6]},
        {"terrain_approach_grid": [6, 4]},
    ]

    costs = build_pairwise_terrain_costs(cost_map, accepted_nodes)

    assert costs.shape == (3, 3)
    np.testing.assert_allclose(costs, costs.T)
    np.testing.assert_array_equal(np.diag(costs), np.zeros(3))
    assert np.all(costs[np.triu_indices(3, 1)] > 0.0)


def test_primary_component_matches_no_corner_cut_reachability() -> None:
    blocked = np.ones((7, 7), dtype=bool)
    blocked[0:3, 0:3] = False
    blocked[3:5, 3:5] = False
    source = _cost_map(blocked)
    original_cost = source.cost.copy()
    original_blocked = source.blocked.copy()

    restricted, primary_mask, primary_area = restrict_to_primary_component(source)

    assert primary_area == 9
    assert np.all(primary_mask[0:3, 0:3])
    assert np.all(restricted.blocked[3:5, 3:5])
    assert np.all(np.isinf(restricted.cost[3:5, 3:5]))
    np.testing.assert_array_equal(source.blocked, original_blocked)
    np.testing.assert_array_equal(source.cost, original_cost)


def test_primary_component_applies_minimum_clearance_before_labeling() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    clearance = np.full((5, 5), 3.0, dtype=np.float32)
    clearance[2, :] = 0.5

    restricted, _, area = restrict_to_primary_component(
        _cost_map(blocked, clearance=clearance),
        min_clearance=1.0,
    )

    assert area == 10
    assert np.all(restricted.blocked[2, :])
    assert np.count_nonzero(~restricted.blocked) == 10


def test_anchor_component_wins_over_larger_disconnected_component() -> None:
    blocked = np.ones((9, 9), dtype=bool)
    blocked[0:4, 0:4] = False
    blocked[6:9, 6:9] = False
    source = _cost_map(blocked)

    restricted, component_mask, area, snapped = restrict_to_anchor_component(
        source,
        (7, 7),
        max_snap_radius=2.0,
    )

    assert snapped == (7, 7)
    assert area == 9
    assert np.all(component_mask[6:9, 6:9])
    assert np.all(restricted.blocked[0:4, 0:4])
    assert np.count_nonzero(~restricted.blocked) == 9


def test_anchor_component_snaps_to_nearby_walkable_cell() -> None:
    blocked = np.ones((7, 7), dtype=bool)
    blocked[4:6, 4:6] = False

    _, _, area, snapped = restrict_to_anchor_component(
        _cost_map(blocked),
        (3, 3),
        max_snap_radius=2.0,
    )

    assert snapped == (4, 4)
    assert area == 4


def test_snap_route_nodes_uses_zone_transform_and_euclidean_radius() -> None:
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=34,
        tile_min_y=34,
        tile_max_x=34,
        tile_max_y=34,
    )
    zone = ZoneTransform(left=2622.0, right=-7510.0, top=1612.0, bottom=-5143.0)
    blocked = np.ones((129, 129), dtype=bool)
    blocked[20, 20] = False
    cost_map = _cost_map(blocked)
    accepted_ui = raster.grid_to_ui(20, 20, zone)
    rejected_ui = raster.grid_to_ui(30, 30, zone)
    nodes = [
        {"x": accepted_ui[0], "y": accepted_ui[1], "name": "accepted"},
        {"x": rejected_ui[0], "y": rejected_ui[1], "name": "rejected"},
    ]
    original = [dict(node) for node in nodes]

    accepted, rejected = snap_route_nodes(
        nodes=nodes,
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        max_snap_radius=3.0,
        min_clearance=0.0,
    )

    assert accepted[0]["terrain_approach_grid"] == [20, 20]
    assert accepted[0]["terrain_snap_cells"] == 0.0
    assert rejected[0]["reason"] == "terrain_no_primary_component_approach"
    assert nodes == original


def test_ore_approach_candidates_keep_distinct_walkable_sides():
    blocked = np.ones((9, 9), dtype=bool)
    blocked[4, 2:7] = False
    candidates = find_ore_approach_candidates(
        _cost_map(blocked),
        (4, 4),
        3.0,
        min_clearance=0.0,
        sector_count=8,
    )

    grids = {tuple(candidate["grid"]) for candidate in candidates}
    assert (4, 4) in grids
    assert any(col < 4 for row, col in grids if row == 4)
    assert any(col > 4 for row, col in grids if row == 4)


def test_approach_optimizer_can_choose_safer_side_over_nearest_cell():
    cost_map = _cost_map(np.zeros((11, 11), dtype=bool))
    cost_map.cost[5, 5] = 100.0
    nodes = [
        {
            "terrain_approach_grid": [5, 1],
            "terrain_approach_candidates": [
                {"grid": [5, 1], "snap_cells": 0.0, "local_score": 0.0}
            ],
        },
        {
            "terrain_approach_grid": [5, 5],
            "terrain_approach_candidates": [
                {"grid": [5, 5], "snap_cells": 0.0, "local_score": 0.0},
                {"grid": [3, 5], "snap_cells": 2.0, "local_score": 2.0},
            ],
        },
        {
            "terrain_approach_grid": [5, 9],
            "terrain_approach_candidates": [
                {"grid": [5, 9], "snap_cells": 0.0, "local_score": 0.0}
            ],
        },
    ]

    result = optimize_ore_node_approaches(
        cost_map,
        nodes,
        refinement_passes=2,
        approach_distance_weight=1.0,
        max_extra_cells=2.0,
    )

    assert nodes[1]["terrain_approach_grid"] == [3, 5]
    assert nodes[1]["terrain_selected_approach_index"] == 1
    assert result["changes"] >= 1


def test_approach_optimizer_rejects_candidate_beyond_extra_cell_limit():
    cost_map = _cost_map(np.zeros((11, 11), dtype=bool))
    cost_map.cost[5, 5] = 100.0
    nodes = [
        {
            "terrain_approach_grid": [5, 1],
            "terrain_approach_candidates": [
                {"grid": [5, 1], "snap_cells": 0.0, "local_score": 0.0}
            ],
        },
        {
            "terrain_approach_grid": [5, 5],
            "terrain_approach_candidates": [
                {"grid": [5, 5], "snap_cells": 0.0, "local_score": 0.0},
                {"grid": [2, 5], "snap_cells": 3.0, "local_score": 3.0},
            ],
        },
        {
            "terrain_approach_grid": [5, 9],
            "terrain_approach_candidates": [
                {"grid": [5, 9], "snap_cells": 0.0, "local_score": 0.0}
            ],
        },
    ]

    optimize_ore_node_approaches(
        cost_map,
        nodes,
        refinement_passes=2,
        approach_distance_weight=1.0,
        max_extra_cells=2.0,
    )

    assert nodes[1]["terrain_approach_grid"] == [5, 5]
    assert nodes[1]["terrain_selected_approach_index"] == 0


def test_node_access_plans_store_primary_fallback_return_and_resume_paths():
    blocked = np.zeros((129, 129), dtype=bool)
    cost_map = _cost_map(blocked)
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=32,
        tile_min_y=32,
        tile_max_x=32,
        tile_max_y=32,
    )
    zone = ZoneTransform(
        left=0.0,
        right=-533.3333333333,
        top=0.0,
        bottom=-533.3333333333,
    )
    nodes = [
        {
            "terrain_approach_grid": [64, 64],
            "terrain_selected_approach_index": 0,
            "terrain_approach_candidates": [
                {
                    "grid": [64, 64],
                    "sector": -1,
                    "bearing_degrees": 0.0,
                    "snap_cells": 0.0,
                    "local_score": 0.0,
                },
                {
                    "grid": [58, 64],
                    "sector": 6,
                    "bearing_degrees": 270.0,
                    "snap_cells": 6.0,
                    "local_score": 6.0,
                },
            ],
        }
    ]

    summary = build_node_access_plans(
        cost_map=cost_map,
        accepted_nodes=nodes,
        route_points=[(64, 40), (64, 64), (64, 88)],
        node_route_indexes=[1],
        raster=raster,
        zone=zone,
        max_waypoint_spacing=8.0,
        min_clearance=0.0,
        max_options=2,
        approach_distance_weight=1.0,
        max_extra_cells=6.0,
    )

    plan = nodes[0]["terrain_access_plan"]
    assert summary["fallback_option_count"] == 1
    assert plan["attachment_route_index"] == 0
    assert plan["resume_route_index"] == 2
    assert [option["primary"] for option in plan["options"]] == [True, False]
    for option in plan["options"]:
        inbound = option["inbound"]["waypoints"]
        return_path = option["return"]["waypoints"]
        resume = option["resume"]["waypoints"]
        assert inbound[0]["grid"] == [64, 40]
        assert inbound[-1]["grid"] == option["approach_grid"]
        assert return_path == list(reversed(inbound))
        assert resume[0]["grid"] == option["approach_grid"]
        assert resume[-1]["grid"] == [64, 88]


def test_ui_bounds_become_hard_cost_map_bounds() -> None:
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=34,
        tile_min_y=34,
        tile_max_x=34,
        tile_max_y=34,
    )
    zone = ZoneTransform(left=2622.0, right=-7510.0, top=1612.0, bottom=-5143.0)
    source = _cost_map(np.zeros((129, 129), dtype=bool))
    first_ui = raster.grid_to_ui(20, 30, zone)
    second_ui = raster.grid_to_ui(80, 90, zone)
    bounds = {
        "min_x": min(first_ui[0], second_ui[0]),
        "max_x": max(first_ui[0], second_ui[0]),
        "min_y": min(first_ui[1], second_ui[1]),
        "max_y": max(first_ui[1], second_ui[1]),
    }

    bounded, grid_bounds = restrict_to_ui_bounds(
        source,
        raster=raster,
        zone=zone,
        bounds=bounds,
        margin=0.0,
    )

    row_min, row_max, col_min, col_max = grid_bounds
    assert bounded.is_walkable(
        ((row_min + row_max) // 2, (col_min + col_max) // 2)
    )
    assert np.all(bounded.blocked[:row_min, :])
    assert np.all(np.isinf(bounded.cost[:row_min, :]))
    assert np.all(source.cost == 1.0)
    assert not np.any(source.blocked)


def test_plan_cycle_is_implicit_and_indexes_each_node_approach() -> None:
    cost_map = _cost_map(np.zeros((20, 20), dtype=bool))
    accepted_nodes = [
        {"terrain_approach_grid": [2, 2]},
        {"terrain_approach_grid": [2, 15]},
        {"terrain_approach_grid": [15, 10]},
    ]

    segments, points, point_segments, route_indexes = plan_cycle(
        cost_map=cost_map,
        accepted_nodes=accepted_nodes,
        max_waypoint_spacing=8.0,
        min_clearance=0.0,
    )

    assert len(segments) == 3
    assert points[-1] != points[0]
    assert len(points) == len(point_segments)
    assert len(route_indexes) == len(accepted_nodes)
    for node, route_index in zip(accepted_nodes, route_indexes):
        assert points[route_index] == tuple(node["terrain_approach_grid"])
    for start, end in zip(points, points[1:] + points[:1]):
        assert line_is_walkable(cost_map, start, end)


def test_plan_cycle_rejects_disconnected_nodes() -> None:
    blocked = np.zeros((20, 20), dtype=bool)
    blocked[10, :] = True
    accepted_nodes = [
        {"terrain_approach_grid": [5, 5]},
        {"terrain_approach_grid": [15, 15]},
    ]

    with pytest.raises(ValueError, match="No terrain path"):
        plan_cycle(
            cost_map=_cost_map(blocked),
            accepted_nodes=accepted_nodes,
            max_waypoint_spacing=8.0,
            min_clearance=0.0,
        )


def test_anchor_entry_uses_walkable_waypoints_and_ends_on_cycle() -> None:
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=34,
        tile_min_y=34,
        tile_max_x=34,
        tile_max_y=34,
    )
    zone = ZoneTransform(left=2622.0, right=-7510.0, top=1612.0, bottom=-5143.0)
    cost_map = _cost_map(np.zeros((129, 129), dtype=bool))
    route_points = [(40, 50), (60, 80), (80, 50)]

    entry, points = build_anchor_entry_route(
        anchor_point=(10, 10),
        route_points=route_points,
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        max_waypoint_spacing=12.0,
        min_clearance=0.0,
    )

    assert points[0] == (10, 10)
    assert points[-1] == route_points[entry["target_route_index"]]
    assert entry["waypoint_count"] == len(points)
    assert all(
        line_is_walkable(cost_map, start, end)
        for start, end in zip(points, points[1:])
    )


def test_compose_output_preserves_source_metadata_and_rejections() -> None:
    source_route = {
        "schema_version": 1,
        "name": "source-cycle",
        "zone": {"id": 5, "name": "The Barrens"},
        "source": {"route_data": "fixture"},
        "generation": {"ordering": "fixture"},
        "route_nodes": [],
        "route_loop": [],
        "rejected_nodes": [{"reason": "existing"}],
    }
    accepted_nodes = [{"route_index": -1, "x": 50.0, "y": 30.0}]
    terrain_rejected = [{"reason": "terrain"}]
    route_loop = [{"index": 0, "x": 50.0, "y": 30.0}]
    metrics = {"detour_ratio": 1.0}

    output = compose_output_route(
        source_route=source_route,
        accepted_nodes=accepted_nodes,
        terrain_rejected=terrain_rejected,
        node_route_indexes=[0],
        route_loop=route_loop,
        metrics=metrics,
    )

    assert output["schema_version"] == 2
    assert output["name"] == source_route["name"]
    assert output["zone"] == source_route["zone"]
    assert output["source"] == source_route["source"]
    assert [item["reason"] for item in output["rejected_nodes"]] == [
        "existing",
        "terrain",
    ]
    assert output["generation"]["ordering"] == "fixture"
    assert output["generation"]["accepted_nodes"] == 1
    assert output["generation"]["rejected_nodes"] == 2
    assert output["generation"]["topographic"] == metrics


def test_build_metrics_samples_every_crossed_route_cell() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    slope = np.zeros((5, 5), dtype=np.float32)
    slope[1, 2] = 12.0
    clearance = np.full((5, 5), 3.0, dtype=np.float32)
    clearance[1, 2] = 0.5
    cost_map = _cost_map(blocked, slope=slope, clearance=clearance)
    segments = [
        PlannedSegment(
            index=0,
            start_node_index=0,
            end_node_index=1,
            raw_path=TerrainPath(points=[(1, 1), (1, 2), (1, 3)], total_cost=2.0),
            simplified_points=[(1, 1), (1, 3)],
            search_mode="fixture",
        ),
        PlannedSegment(
            index=1,
            start_node_index=1,
            end_node_index=0,
            raw_path=TerrainPath(points=[(1, 3), (1, 2), (1, 1)], total_cost=2.0),
            simplified_points=[(1, 3), (1, 1)],
            search_mode="fixture",
        ),
    ]

    metrics = build_metrics(
        segments=segments,
        route_points=[(1, 1), (1, 3)],
        cost_map=cost_map,
        primary_component_area=25,
        accepted_count=2,
        terrain_rejected_count=0,
        source_paths={"fixture": "fixture"},
    )

    assert metrics["max_route_slope_degrees"] == 12.0
    assert metrics["min_route_clearance_cells"] == 0.5
    assert metrics["route_crossed_cell_count"] >= 4
    assert metrics["raw_point_count"] == 6
    assert metrics["route_waypoint_count"] == 2
    json.dumps(metrics, allow_nan=False)
    for value in metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
