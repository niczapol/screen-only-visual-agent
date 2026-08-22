from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision_bot.regions import resolve_region
from vision_bot.screen_objects import BoundingBox


@dataclass(frozen=True)
class ThreatDetection:
    reason: str
    bbox: BoundingBox
    score: float
    turn_key: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "score": self.score,
            "turn_key": self.turn_key,
            "source": self.source,
        }


def detect_hostile_threat(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
    *,
    fallback_turn_key: str = "D",
) -> ThreatDetection | None:
    cfg = config or {}
    avoid_cfg = _avoidance_cfg(cfg)
    if not bool(avoid_cfg.get("enabled", True)):
        return None

    target_threat = _detect_hostile_target_frame(frame, cfg, avoid_cfg, fallback_turn_key)
    if target_threat is not None:
        return target_threat

    nameplate_threat = _detect_central_red_nameplate(frame, cfg, avoid_cfg)
    if nameplate_threat is not None:
        return nameplate_threat

    if bool(avoid_cfg.get("combat_warning_enabled", True)):
        return _detect_combat_warning(frame, cfg, avoid_cfg, fallback_turn_key)
    return None


def _detect_hostile_target_frame(
    frame: np.ndarray,
    config: dict[str, Any],
    avoid_cfg: dict[str, Any],
    fallback_turn_key: str,
) -> ThreatDetection | None:
    if not bool(avoid_cfg.get("target_frame_enabled", True)):
        return None

    region_cfg = avoid_cfg.get(
        "target_frame_region",
        {"x": 230, "y": 0, "width": 420, "height": 150},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    mask = _red_mask(roi, avoid_cfg)
    detections = _red_bar_boxes(
        mask,
        offset=(roi_x, roi_y),
        min_area=int(avoid_cfg.get("target_frame_min_area", 90)),
        max_area=int(avoid_cfg.get("target_frame_max_area", 10000)),
        min_width=int(avoid_cfg.get("target_frame_min_width", 28)),
        max_width=int(avoid_cfg.get("target_frame_max_width", 380)),
        min_height=int(avoid_cfg.get("target_frame_min_height", 4)),
        max_height=int(avoid_cfg.get("target_frame_max_height", 36)),
        min_aspect=float(avoid_cfg.get("target_frame_min_aspect", 2.4)),
        max_aspect=float(avoid_cfg.get("target_frame_max_aspect", 80.0)),
        min_fill_ratio=float(avoid_cfg.get("target_frame_min_fill_ratio", 0.55)),
    )
    if not detections:
        return None

    bbox = max(detections, key=lambda item: item.area())
    return ThreatDetection(
        reason="hostile_target_frame",
        bbox=bbox,
        score=float(avoid_cfg.get("target_frame_score", 0.95)),
        turn_key=_valid_turn_key(fallback_turn_key),
        source="top_left_hostile_target_ui",
    )


def _detect_central_red_nameplate(
    frame: np.ndarray,
    config: dict[str, Any],
    avoid_cfg: dict[str, Any],
) -> ThreatDetection | None:
    region_cfg = avoid_cfg.get(
        "central_threat_region",
        {"x": 520, "y": 180, "width": 1320, "height": 650},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    mask = _red_mask(roi, avoid_cfg)
    detections = _red_bar_boxes(
        mask,
        offset=(roi_x, roi_y),
        min_area=int(avoid_cfg.get("nameplate_min_area", 80)),
        max_area=int(avoid_cfg.get("nameplate_max_area", 6000)),
        min_width=int(avoid_cfg.get("nameplate_min_width", 24)),
        max_width=int(avoid_cfg.get("nameplate_max_width", 320)),
        min_height=int(avoid_cfg.get("nameplate_min_height", 4)),
        max_height=int(avoid_cfg.get("nameplate_max_height", 28)),
        min_aspect=float(avoid_cfg.get("nameplate_min_aspect", 2.6)),
        max_aspect=float(avoid_cfg.get("nameplate_max_aspect", 60.0)),
        min_fill_ratio=float(avoid_cfg.get("nameplate_min_fill_ratio", 0.60)),
    )
    detections = [bbox for bbox in detections if not _is_ignored_ui_bbox(bbox, frame.shape, avoid_cfg)]
    if not detections:
        return None

    frame_center_x = frame.shape[1] / 2.0
    bbox = min(detections, key=lambda item: abs((item.x + item.width / 2.0) - frame_center_x))
    turn_key = "D" if bbox.x + bbox.width / 2.0 < frame_center_x else "A"
    return ThreatDetection(
        reason="aggressive_nameplate",
        bbox=bbox,
        score=float(avoid_cfg.get("nameplate_score", 0.85)),
        turn_key=turn_key,
        source="central_red_nameplate",
    )


def _detect_combat_warning(
    frame: np.ndarray,
    config: dict[str, Any],
    avoid_cfg: dict[str, Any],
    fallback_turn_key: str,
) -> ThreatDetection | None:
    region_cfg = avoid_cfg.get(
        "combat_warning_region",
        {"x": 820, "y": 420, "width": 920, "height": 360},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    mask = _red_mask(roi, avoid_cfg)
    area = int(np.count_nonzero(mask))
    if area < int(avoid_cfg.get("combat_warning_min_red_area", 180)):
        return None

    boxes = _red_bar_boxes(
        mask,
        offset=(roi_x, roi_y),
        min_area=int(avoid_cfg.get("combat_warning_component_min_area", 500)),
        max_area=int(avoid_cfg.get("combat_warning_component_max_area", 3000)),
        min_width=int(avoid_cfg.get("combat_warning_min_width", 8)),
        max_width=int(avoid_cfg.get("combat_warning_max_width", 260)),
        min_height=int(avoid_cfg.get("combat_warning_min_height", 6)),
        max_height=int(avoid_cfg.get("combat_warning_max_height", 80)),
        min_aspect=float(avoid_cfg.get("combat_warning_min_aspect", 0.4)),
        max_aspect=float(avoid_cfg.get("combat_warning_max_aspect", 12.0)),
        min_fill_ratio=float(avoid_cfg.get("combat_warning_min_fill_ratio", 0.45)),
    )
    if not boxes:
        return None

    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return ThreatDetection(
        reason="combat_warning",
        bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
        score=float(avoid_cfg.get("combat_warning_score", 0.70)),
        turn_key=_valid_turn_key(fallback_turn_key),
        source="center_red_combat_warning",
    )


def _red_mask(frame: np.ndarray, avoid_cfg: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = avoid_cfg.get(
        "red_hsv_ranges",
        [
            [[0, 90, 90], [10, 255, 255]],
            [[170, 90, 90], [179, 255, 255]],
        ],
    )
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
        )
    close_kernel = int(avoid_cfg.get("red_close_kernel", 5))
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _red_bar_boxes(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    min_area: int,
    max_area: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    min_aspect: float,
    max_aspect: float,
    min_fill_ratio: float = 0.0,
) -> list[BoundingBox]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    offset_x, offset_y = offset
    boxes: list[BoundingBox] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        aspect = width / float(height)
        fill_ratio = area / float(width * height)
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and min_aspect <= aspect <= max_aspect
            and fill_ratio >= min_fill_ratio
        ):
            continue
        boxes.append(BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)))
    return boxes


def _is_ignored_ui_bbox(
    bbox: BoundingBox,
    frame_shape: tuple[int, ...],
    avoid_cfg: dict[str, Any],
) -> bool:
    frame_height, frame_width = frame_shape[:2]
    center_x = bbox.x + bbox.width / 2.0
    center_y = bbox.y + bbox.height / 2.0
    if center_y >= frame_height * float(avoid_cfg.get("ignore_bottom_ui_fraction", 0.72)):
        return True
    if center_x >= frame_width * float(avoid_cfg.get("ignore_right_ui_fraction", 0.78)):
        return True
    if (
        center_x <= frame_width * float(avoid_cfg.get("ignore_left_ui_fraction", 0.18))
        and center_y <= frame_height * float(avoid_cfg.get("ignore_top_left_ui_fraction", 0.18))
    ):
        return True
    return False


def _valid_turn_key(value: str) -> str:
    return value if value in {"A", "D"} else "D"


def _avoidance_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("safety", {}).get("hostile_avoidance", {})
