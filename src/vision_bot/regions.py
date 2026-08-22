from __future__ import annotations

from typing import Any

import numpy as np


def resolve_region(
    region_config: dict[str, Any],
    frame_shape: tuple[int, ...],
    config: dict[str, Any] | None = None,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    cfg = config or {}
    screen_cfg = cfg.get("screen", {})

    reference_width = int(screen_cfg.get("reference_width", frame_width))
    reference_height = int(screen_cfg.get("reference_height", frame_height))
    scale_regions = bool(screen_cfg.get("scale_regions", True))

    if reference_width <= 0 or reference_height <= 0:
        reference_width = frame_width
        reference_height = frame_height

    scale_x = frame_width / reference_width if scale_regions else 1.0
    scale_y = frame_height / reference_height if scale_regions else 1.0

    x = int(round(float(region_config.get("x", 0)) * scale_x))
    y = int(round(float(region_config.get("y", 0)) * scale_y))
    width = int(round(float(region_config.get("width", frame_width)) * scale_x))
    height = int(round(float(region_config.get("height", frame_height)) * scale_y))

    x = max(0, min(x, frame_width))
    y = max(0, min(y, frame_height))
    width = max(0, min(width, frame_width - x))
    height = max(0, min(height, frame_height - y))
    return x, y, width, height


def crop_region(frame: np.ndarray, region_config: dict[str, Any], config: dict[str, Any] | None = None) -> np.ndarray:
    x, y, width, height = resolve_region(region_config, frame.shape, config)
    return frame[y : y + height, x : x + width]
