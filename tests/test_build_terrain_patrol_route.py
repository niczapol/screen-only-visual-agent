from __future__ import annotations

import numpy as np

from scripts.build_terrain_patrol_route import farthest_terrain_path
from vision_bot.terrain_routing import TerrainCostConfig, TerrainCostMap


def test_farthest_terrain_path_stays_in_walkable_component() -> None:
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[:, 4] = True
    cost = np.ones((7, 7), dtype=np.float32)
    cost[blocked] = np.inf
    cost_map = TerrainCostMap(
        cost=cost,
        slope_degrees=np.zeros((7, 7), dtype=np.float32),
        roughness=np.zeros((7, 7), dtype=np.float32),
        clearance=np.where(blocked, 0.0, 3.0).astype(np.float32),
        blocked=blocked,
        config=TerrainCostConfig(),
    )

    path, reachable = farthest_terrain_path(
        cost_map,
        (3, 1),
        max_radius_cells=10,
        min_clearance=1.0,
    )

    assert path is not None
    assert reachable == 28
    assert all(col < 4 for _, col in path.points)
    assert path.points[0] == (3, 1)
