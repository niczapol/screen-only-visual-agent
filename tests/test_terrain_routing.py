from __future__ import annotations

import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_bot.terrain_routing import (
    TILE_GRID_CELLS,
    TerrainCostConfig,
    TerrainCostMap,
    TerrainRaster,
    ZoneTransform,
    astar,
    build_terrain_cost_map,
    build_terrain_raster,
    line_is_walkable,
    load_adt_v9,
    nearest_walkable,
    path_length,
    simplify_path,
)


def _write_synthetic_adt(path: Path, heights: np.ndarray) -> None:
    assert heights.shape == (129, 129)
    chunks: list[bytes] = []
    for iy in range(16):
        for ix in range(16):
            base_height = float(heights[iy * 8, ix * 8])
            relative = np.zeros(145, dtype="<f4")
            for local_row in range(9):
                row = iy * 8 + local_row
                col = ix * 8
                relative[local_row * 17 : local_row * 17 + 9] = (
                    heights[row, col : col + 9] - base_height
                )

            chunk = bytearray(724)
            chunk[0:4] = b"KNCM"
            struct.pack_into("<I", chunk, 4, len(chunk) - 8)
            struct.pack_into("<II", chunk, 12, ix, iy)
            struct.pack_into("<I", chunk, 28, 136)
            struct.pack_into("<f", chunk, 120, base_height)
            chunk[136:140] = b"TVCM"
            struct.pack_into("<I", chunk, 140, 145 * 4)
            chunk[144:724] = relative.tobytes()
            chunks.append(bytes(chunk))
    path.write_bytes(b"".join(chunks))


def _cost_map(blocked: np.ndarray, *, cost: np.ndarray | None = None) -> TerrainCostMap:
    blocked = np.asarray(blocked, dtype=bool)
    terrain_cost = (
        np.ones(blocked.shape, dtype=np.float32)
        if cost is None
        else np.asarray(cost, dtype=np.float32).copy()
    )
    terrain_cost[blocked] = np.inf
    clearance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3)
    return TerrainCostMap(
        cost=terrain_cost,
        slope_degrees=np.zeros(blocked.shape, dtype=np.float32),
        roughness=np.zeros(blocked.shape, dtype=np.float32),
        clearance=clearance,
        blocked=blocked,
        config=TerrainCostConfig(),
    )


def test_load_adt_v9_reconstructs_absolute_v9_heights(tmp_path: Path) -> None:
    row, col = np.mgrid[0:129, 0:129]
    expected = (73.25 + row * 0.2 + col * 0.1).astype(np.float32)
    path = tmp_path / "Kalimdor_34_34.adt"
    _write_synthetic_adt(path, expected)

    actual = load_adt_v9(path)

    assert actual.shape == (129, 129)
    np.testing.assert_allclose(actual, expected, atol=1.0e-4)


def test_load_adt_v9_rejects_incomplete_tile(tmp_path: Path) -> None:
    path = tmp_path / "Kalimdor_34_34.adt"
    path.write_bytes(b"KNCM" + struct.pack("<I", 128) + bytes(128))

    with pytest.raises(ValueError, match="Expected complete|Invalid MCVT"):
        load_adt_v9(path)


def test_build_terrain_raster_stitches_shared_edge(tmp_path: Path) -> None:
    row, col = np.mgrid[0:129, 0:257]
    global_heights = (10.0 + row * 0.03 + col * 0.05).astype(np.float32)
    first_path = tmp_path / "Kalimdor_34_34.adt"
    second_path = tmp_path / "Kalimdor_35_34.adt"
    _write_synthetic_adt(first_path, global_heights[:, :129])
    _write_synthetic_adt(second_path, global_heights[:, 128:])

    raster = build_terrain_raster([second_path, first_path])

    assert raster.heights.shape == (129, 257)
    assert (raster.tile_min_x, raster.tile_max_x) == (34, 35)
    np.testing.assert_allclose(raster.heights, global_heights, atol=1.0e-4)


def test_zone_ui_grid_round_trip() -> None:
    raster = TerrainRaster(
        heights=np.zeros((1025, 1281), dtype=np.float32),
        tile_min_x=31,
        tile_min_y=33,
        tile_max_x=40,
        tile_max_y=40,
    )
    zone = ZoneTransform(left=2622.0, right=-7510.0, top=1612.0, bottom=-5143.0)

    for ui_point in ((50.0, 30.0), (57.25, 50.16), (12.0, 88.0)):
        row, col = raster.ui_to_grid(*ui_point, zone)
        actual = raster.grid_to_ui(row, col, zone)
        assert actual == pytest.approx(ui_point, abs=1.0e-9)


def test_zone_transform_uses_wow_map_axis_orientation() -> None:
    zone = ZoneTransform(left=4233.0, right=-262.0, top=452.0, bottom=-2545.0)

    world = zone.ui_to_world(38.36, 59.71)

    assert world == pytest.approx((-1337.5087, 2508.718))
    assert zone.world_to_ui(*world) == pytest.approx((38.36, 59.71))


def test_build_cost_map_blocks_steep_cliff_but_keeps_flat_ground() -> None:
    heights = np.zeros((129, 129), dtype=np.float32)
    heights[:, 65:] = 100.0
    raster = TerrainRaster(
        heights=heights,
        tile_min_x=34,
        tile_min_y=34,
        tile_max_x=34,
        tile_max_y=34,
    )

    cost_map = build_terrain_cost_map(
        raster,
        TerrainCostConfig(soft_slope_degrees=10.0, hard_slope_degrees=30.0),
    )

    assert cost_map.is_walkable((64, 20))
    assert cost_map.is_walkable((64, 100))
    assert cost_map.blocked[64, 64]
    assert cost_map.blocked[64, 65]


def test_astar_uses_optimal_diagonal_on_flat_map() -> None:
    cost_map = _cost_map(np.zeros((6, 6), dtype=bool))

    path = astar(cost_map, (0, 0), (5, 5))

    assert path is not None
    assert path.points == [(index, index) for index in range(6)]
    assert path.total_cost == pytest.approx(5.0 * math.sqrt(2.0))
    assert path_length(path.points, spacing=1.0) == pytest.approx(5.0 * math.sqrt(2.0))


def test_astar_returns_none_for_solid_wall() -> None:
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[3, :] = True

    assert astar(_cost_map(blocked), (1, 1), (5, 5)) is None


def test_astar_routes_through_wall_gap() -> None:
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[3, :] = True
    blocked[3, 3] = False

    path = astar(_cost_map(blocked), (1, 1), (5, 5))

    assert path is not None
    assert (3, 3) in path.points


def test_astar_does_not_cut_blocked_diagonal_corner() -> None:
    blocked = np.zeros((2, 2), dtype=bool)
    blocked[0, 1] = True
    blocked[1, 0] = True

    assert astar(_cost_map(blocked), (0, 0), (1, 1)) is None


def test_nearest_walkable_is_deterministic_and_honors_radius() -> None:
    blocked = np.ones((5, 5), dtype=bool)
    blocked[1, 2] = False
    blocked[2, 1] = False
    cost_map = _cost_map(blocked)

    assert nearest_walkable(cost_map, (2, 2), 1.0) == (1, 2)
    assert nearest_walkable(cost_map, (4, 4), 1.0) is None


def test_simplify_path_preserves_turn_around_obstacle() -> None:
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[3, 3] = True
    cost_map = _cost_map(blocked)
    points = [(3, 1), (2, 2), (2, 3), (2, 4), (3, 5)]

    simplified = simplify_path(cost_map, points, max_spacing_cells=10.0)

    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
    assert len(simplified) >= 3
    assert not line_is_walkable(cost_map, points[0], points[-1])
