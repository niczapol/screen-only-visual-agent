from __future__ import annotations

import numpy as np

from vision_bot.mmap_navmesh import NavLocation
from vision_bot.terrain_routing import TerrainCostConfig, TerrainCostMap
from scripts.build_mmap_route import nearest_neighbor_order, parse_args, validate_grid_polyline


def test_parse_args_accepts_reproducible_required_inputs() -> None:
    args = parse_args(
        [
            "--input-route", "route.json",
            "--manifest", "manifest.json",
            "--adt-dir", "adt",
            "--mmap-dir", "mmaps",
            "--map-id", "1",
            "--anchor-x", "38.36",
            "--anchor-y", "59.71",
            "--output-route", "out.json",
            "--output-metrics", "metrics.json",
            "--output-overlay", "overlay.png",
        ]
    )

    assert args.anchor_x == 38.36
    assert args.map_id == 1


def test_nearest_neighbor_order_is_deterministic() -> None:
    start = NavLocation(0, (0.0, 0.0, 0.0), 0.0, 0.0)
    nodes = [
        ({"node_id": 2}, NavLocation(2, (10.0, 0.0, 0.0), 0.0, 0.0)),
        ({"node_id": 1}, NavLocation(1, (3.0, 0.0, 0.0), 0.0, 0.0)),
    ]

    assert [item[0]["node_id"] for item in nearest_neighbor_order(start, nodes)] == [1, 2]


def test_grid_polyline_validation_rejects_client_terrain_blocker() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    clearance = np.full((5, 5), 4.0, dtype=np.float32)
    clearance[2, 2] = 0.0
    cost = np.ones((5, 5), dtype=np.float32)
    cost[blocked] = np.inf
    cost_map = TerrainCostMap(
        cost=cost,
        slope_degrees=np.where(blocked, 60.0, 3.0).astype(np.float32),
        roughness=np.zeros((5, 5), dtype=np.float32),
        clearance=clearance,
        blocked=blocked,
        config=TerrainCostConfig(),
    )

    result = validate_grid_polyline([(2, 0), (2, 4)], cost_map=cost_map, min_clearance=1.0)

    assert not result["walkable"]
    assert result["blocked_cell_count"] == 1
    assert result["first_blocked_cell"] == [2, 2]
    assert result["max_slope_degrees"] == 60.0


def test_grid_polyline_validation_accepts_clear_supercover() -> None:
    cost_map = TerrainCostMap(
        cost=np.ones((5, 5), dtype=np.float32),
        slope_degrees=np.full((5, 5), 3.0, dtype=np.float32),
        roughness=np.zeros((5, 5), dtype=np.float32),
        clearance=np.full((5, 5), 4.0, dtype=np.float32),
        blocked=np.zeros((5, 5), dtype=bool),
        config=TerrainCostConfig(),
    )

    result = validate_grid_polyline([(0, 0), (4, 4)], cost_map=cost_map, min_clearance=1.0)

    assert result["walkable"]
    assert result["blocked_cell_count"] == 0
