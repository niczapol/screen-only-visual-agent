from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision_bot.regions import resolve_region


@dataclass(frozen=True)
class RuntimeTelemetry:
    coord: int
    x: float
    y: float
    heading_degrees: float
    protocol_version: int = 1
    frame_sequence: int | None = None
    event_sequence: int | None = None
    status_bits: int = 0
    checksum_valid: bool = True


RUNTIME_STATUS_COMBAT = 1 << 0
RUNTIME_STATUS_OUTGOING_HIT = 1 << 1
RUNTIME_STATUS_WRONG_FACING = 1 << 2
RUNTIME_STATUS_OUT_OF_RANGE = 1 << 3
RUNTIME_STATUS_TARGET_IS_ATTACKER = 1 << 4
RUNTIME_STATUS_LOOT_OPENED = 1 << 5
RUNTIME_STATUS_LOOT_PENDING = 1 << 6
RUNTIME_STATUS_MOUNTED = 1 << 7


@dataclass(frozen=True)
class MinimapOreTooltipTelemetry:
    ore_id: int
    ore_type: str


MINIMAP_ORE_TYPES: dict[int, str] = {
    1: "Copper",
    2: "Tin",
    3: "Silver",
    4: "Iron",
    5: "Gold",
    6: "Mithril",
    7: "Truesilver",
    8: "Small Thorium",
    9: "Rich Thorium",
    10: "Fel Iron",
    11: "Adamantite",
    12: "Rich Adamantite",
    13: "Khorium",
    14: "Nethercite",
    15: "Cobalt",
    16: "Rich Cobalt",
    17: "Saronite",
    18: "Rich Saronite",
    19: "Titanium",
}


def read_runtime_telemetry(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> RuntimeTelemetry | None:
    if frame.size == 0:
        return None
    cfg = config or {}
    marker_cfg = cfg.get("runtime_markers", {})
    telemetry_cfg = marker_cfg.get("telemetry", {})
    if not bool(marker_cfg.get("enabled", False)) or not bool(
        telemetry_cfg.get("enabled", False)
    ):
        return None

    region_cfg = telemetry_cfg.get(
        "region", {"x": 700, "y": 45, "width": 1160, "height": 50}
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[y : y + height, x : x + width]
    if roi.size == 0:
        return None

    left = _sentinel_center(roi, "magenta")
    right = _sentinel_center(roi, "cyan")
    if left is None or right is None or right[0] <= left[0]:
        return None
    scale = (right[0] - left[0]) / 302.0
    if not 0.5 <= scale <= 3.0 or abs(right[1] - left[1]) > max(3.0, scale * 3.0):
        return None

    row = int(round((left[1] + right[1]) * 0.5))
    threshold = int(telemetry_cfg.get("bit_threshold", 150))
    values: list[int] = []
    for index in range(72):
        column = int(round(left[0] + (9.0 + index * 4.0) * scale))
        if index < 3:
            channel = 2
        elif index < 11:
            channel = 2
        elif index < 25:
            channel = 2
        elif index < 39:
            channel = 1
        elif index < 48:
            channel = 0
        elif index < 56:
            channel = 2
        elif index < 64:
            channel = 1
        else:
            channel = 2
        values.append(
            int(_sample_channel(roi, column, row, channel) >= threshold)
        )
    protocol_version = _decode_bits(values[:3])
    frame_sequence = _decode_bits(values[3:11])
    x_value = _decode_bits(values[11:25])
    y_value = _decode_bits(values[25:39])
    heading_value = _decode_bits(values[39:48])
    event_sequence = _decode_bits(values[48:56])
    status_bits = _decode_bits(values[56:64])
    checksum = _decode_bits(values[64:72])
    expected_version = int(cfg.get("v09", {}).get("protocol_version", 2))
    if protocol_version != expected_version:
        return None
    expected_checksum = _runtime_telemetry_checksum(
        protocol_version,
        frame_sequence,
        x_value,
        y_value,
        heading_value,
        event_sequence,
        status_bits,
    )
    if checksum != expected_checksum:
        return None
    if not (0 <= x_value <= 10000 and 0 <= y_value <= 10000):
        return None
    if x_value == 0 and y_value == 0:
        return None

    x_percent = x_value / 100.0
    y_percent = y_value / 100.0
    coord = int(f"{x_value:04d}{y_value:04d}00")
    return RuntimeTelemetry(
        coord=coord,
        x=x_percent,
        y=y_percent,
        heading_degrees=(heading_value / 511.0) * 360.0,
        protocol_version=protocol_version,
        frame_sequence=frame_sequence,
        event_sequence=event_sequence,
        status_bits=status_bits,
        checksum_valid=True,
    )


def _runtime_telemetry_checksum(
    protocol_version: int,
    frame_sequence: int,
    x_value: int,
    y_value: int,
    heading_value: int,
    event_sequence: int,
    status_bits: int,
) -> int:
    return (
        protocol_version
        + frame_sequence
        + event_sequence
        + status_bits
        + x_value % 256
        + (x_value // 256) % 256
        + y_value % 256
        + (y_value // 256) % 256
        + heading_value % 256
        + (heading_value // 256) % 256
    ) % 256


def read_minimap_ore_tooltip_telemetry(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> MinimapOreTooltipTelemetry | None:
    """Decode the addon's visible, minimap-only ore tooltip strip.

    Both sentinels, the explicit minimap-source bit and a known ore id are
    required.  The reader therefore fails closed on a stale/partial marker.
    """
    if frame.size == 0:
        return None
    cfg = config or {}
    marker_cfg = cfg.get("runtime_markers", {})
    tooltip_cfg = marker_cfg.get("minimap_ore_tooltip", {})
    if not bool(marker_cfg.get("enabled", False)) or not bool(
        tooltip_cfg.get("enabled", False)
    ):
        return None

    region_cfg = tooltip_cfg.get(
        "region", {"x": 1120, "y": 92, "width": 320, "height": 76}
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[y : y + height, x : x + width]
    if roi.size == 0:
        return None

    left = _sentinel_center(roi, "yellow")
    right = _sentinel_center(roi, "blue")
    if left is None or right is None or right[0] <= left[0]:
        return None
    scale = (right[0] - left[0]) / 52.0
    if not 0.5 <= scale <= 3.0 or abs(right[1] - left[1]) > max(3.0, scale * 3.0):
        return None

    row = int(round((left[1] + right[1]) * 0.5))
    threshold = int(tooltip_cfg.get("bit_threshold", 150))
    values: list[int] = []
    for index in range(9):
        column = int(round(left[0] + (9.0 + index * 4.0) * scale))
        channel = 2 if index < 8 else 1
        values.append(int(_sample_channel(roi, column, row, channel) >= threshold))

    ore_id = _decode_bits(values[:8])
    source_is_minimap = bool(values[8])
    ore_type = MINIMAP_ORE_TYPES.get(ore_id)
    if not source_is_minimap or ore_type is None:
        return None
    return MinimapOreTooltipTelemetry(ore_id=ore_id, ore_type=ore_type)


def _sentinel_center(roi: np.ndarray, kind: str) -> tuple[float, float] | None:
    blue = roi[:, :, 0].astype(np.int16)
    green = roi[:, :, 1].astype(np.int16)
    red = roi[:, :, 2].astype(np.int16)
    if kind == "magenta":
        mask = (red >= 180) & (blue >= 180) & (green <= 100)
    elif kind == "cyan":
        mask = (blue >= 180) & (green >= 180) & (red <= 100)
    elif kind == "yellow":
        mask = (red >= 180) & (green >= 180) & (blue <= 100)
    elif kind == "blue":
        mask = (blue >= 180) & (green <= 100) & (red <= 100)
    else:
        return None
    ys, xs = np.where(mask)
    if xs.size < 8:
        return None
    return float(np.median(xs)), float(np.median(ys))


def _sample_channel(
    roi: np.ndarray,
    x: int,
    y: int,
    channel: int,
) -> float:
    x1 = max(0, x - 1)
    x2 = min(roi.shape[1], x + 2)
    y1 = max(0, y - 1)
    y2 = min(roi.shape[0], y + 2)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float(np.median(roi[y1:y2, x1:x2, channel]))


def _decode_bits(bits: list[int]) -> int:
    return sum((1 << index) for index, value in enumerate(bits) if value)


def detect_mounted_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="mounted_region",
        default_region={"x": 1560, "y": 0, "width": 130, "height": 64},
        lower_key="mounted_hsv_lower",
        default_lower=[100, 210, 210],
        upper_key="mounted_hsv_upper",
        default_upper=[115, 255, 255],
        min_pixels_key="mounted_min_pixels",
        default_min_pixels=700,
    )


def detect_armor_critical_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="armor_critical_region",
        default_region={"x": 2300, "y": 310, "width": 250, "height": 60},
        lower_key="armor_critical_hsv_lower",
        default_lower=[0, 210, 210],
        upper_key="armor_critical_hsv_upper",
        default_upper=[6, 255, 255],
        min_pixels_key="armor_critical_min_pixels",
        default_min_pixels=700,
    )


def detect_forbidden_subzone_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="forbidden_subzone_region",
        default_region={"x": 1660, "y": 0, "width": 170, "height": 64},
        lower_key="forbidden_subzone_hsv_lower",
        default_lower=[20, 210, 210],
        upper_key="forbidden_subzone_hsv_upper",
        default_upper=[35, 255, 255],
        min_pixels_key="forbidden_subzone_min_pixels",
        default_min_pixels=700,
    )


def detect_dead_hostile_target_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="dead_hostile_target_region",
        default_region={"x": 1740, "y": 0, "width": 190, "height": 64},
        lower_key="dead_hostile_target_hsv_lower",
        default_lower=[50, 190, 190],
        upper_key="dead_hostile_target_hsv_upper",
        default_upper=[85, 255, 255],
        min_pixels_key="dead_hostile_target_min_pixels",
        default_min_pixels=700,
    )


def detect_loot_opened_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="loot_opened_region",
        default_region={"x": 2140, "y": 0, "width": 120, "height": 64},
        lower_key="loot_opened_hsv_lower",
        default_lower=[0, 0, 225],
        upper_key="loot_opened_hsv_upper",
        default_upper=[179, 45, 255],
        min_pixels_key="loot_opened_min_pixels",
        default_min_pixels=700,
    )


def detect_combat_loot_pending_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="combat_loot_pending_region",
        default_region={"x": 870, "y": 0, "width": 130, "height": 64},
        lower_key="combat_loot_pending_hsv_lower",
        default_lower=[132, 175, 210],
        upper_key="combat_loot_pending_hsv_upper",
        default_upper=[142, 255, 255],
        min_pixels_key="combat_loot_pending_min_pixels",
        default_min_pixels=700,
    )


def detect_spirit_healer_target_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="spirit_healer_target_region",
        default_region={"x": 1760, "y": 0, "width": 500, "height": 64},
        lower_key="spirit_healer_target_hsv_lower",
        default_lower=[160, 185, 210],
        upper_key="spirit_healer_target_hsv_upper",
        default_upper=[172, 255, 255],
        min_pixels_key="spirit_healer_target_min_pixels",
        default_min_pixels=700,
    )


def detect_spirit_healer_dialog_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    return _detect_marker(
        frame,
        config,
        region_key="spirit_healer_dialog_region",
        default_region={"x": 1760, "y": 0, "width": 500, "height": 64},
        lower_key="spirit_healer_dialog_hsv_lower",
        default_lower=[68, 135, 210],
        upper_key="spirit_healer_dialog_hsv_upper",
        default_upper=[80, 225, 255],
        min_pixels_key="spirit_healer_dialog_min_pixels",
        default_min_pixels=700,
    )


def _detect_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None,
    *,
    region_key: str,
    default_region: dict[str, int],
    lower_key: str,
    default_lower: list[int],
    upper_key: str,
    default_upper: list[int],
    min_pixels_key: str,
    default_min_pixels: int,
) -> bool:
    if frame.size == 0:
        return False
    cfg = config or {}
    marker_cfg = cfg.get("runtime_markers", {})
    if not bool(marker_cfg.get("enabled", False)):
        return False
    x, y, width, height = resolve_region(marker_cfg.get(region_key, default_region), frame.shape, cfg)
    roi = frame[y : y + height, x : x + width]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(marker_cfg.get(lower_key, default_lower), dtype=np.uint8),
        np.array(marker_cfg.get(upper_key, default_upper), dtype=np.uint8),
    )
    return int(cv2.countNonZero(mask)) >= int(
        marker_cfg.get(min_pixels_key, default_min_pixels)
    )
