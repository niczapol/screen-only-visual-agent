import numpy as np

from scripts.render_topographic_route_map import _render_terrain, crop_grid_bounds
from vision_bot.terrain_routing import TerrainRaster, ZoneTransform


def test_crop_grid_bounds_maps_ui_rectangle_inside_raster():
    raster = TerrainRaster(
        heights=np.zeros((129, 129), dtype=np.float32),
        tile_min_x=32,
        tile_min_y=32,
        tile_max_x=32,
        tile_max_y=32,
    )
    zone = ZoneTransform(left=0.0, right=-533.33333, top=0.0, bottom=-533.33333)

    row0, row1, col0, col1 = crop_grid_bounds(raster, zone, (10.0, 20.0, 30.0, 40.0))

    assert 0 <= row0 < row1 <= 129
    assert 0 <= col0 < col1 <= 129


def test_render_terrain_marks_hard_slope_differently_from_flat_ground():
    heights = np.tile(np.arange(12, dtype=np.float32), (12, 1))
    slopes = np.zeros((12, 12), dtype=np.float32)
    slopes[:, 7:] = 40.0

    rendered = _render_terrain(
        heights,
        slopes,
        soft_slope=15.0,
        hard_slope=32.0,
        contour_interval=100.0,
    )

    flat_mean = rendered[:, 1:5].mean(axis=(0, 1))
    hard_mean = rendered[:, 8:11].mean(axis=(0, 1))
    assert hard_mean[2] > flat_mean[2]
