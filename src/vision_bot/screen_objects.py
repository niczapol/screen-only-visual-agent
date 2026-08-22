from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.recognition import recognize_ore_points
from vision_bot.regions import resolve_region


class ScreenObjectClass(str, Enum):
    AGGRESSIVE_MOB = "aggressive_mob"
    PEACEFUL_MOB = "peaceful_mob"
    ORE_VEIN = "ore_vein"
    MINIMAP_ORE_ICON = "minimap_ore_icon"
    TERRAIN_OBSTACLE = "terrain_obstacle"
    TERRAIN_EDGE = "terrain_edge"
    MOVEMENT_BLOCKER = "movement_blocker"
    WALKABLE_CORRIDOR = "walkable_corridor"


DEFAULT_CLASS_NAMES = [item.value for item in ScreenObjectClass]
_YOLO_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clipped(self, frame_shape: tuple[int, ...]) -> "BoundingBox":
        frame_height, frame_width = frame_shape[:2]
        x = max(0, min(self.x, frame_width))
        y = max(0, min(self.y, frame_height))
        x2 = max(x, min(self.x2, frame_width))
        y2 = max(y, min(self.y2, frame_height))
        return BoundingBox(x, y, x2 - x, y2 - y)


@dataclass(frozen=True)
class ScreenDetection:
    label: str
    bbox: BoundingBox
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "score": self.score,
            "source": self.source,
        }


def detect_screen_objects(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    screen_cfg = _screen_cfg(cfg)
    model_cfg = screen_cfg.get("model", {})
    detections: list[ScreenDetection] = []
    model_mode = str(model_cfg.get("mode", "combined"))
    if bool(model_cfg.get("enabled", False)) and model_mode != "bootstrap_only":
        detections.extend(detect_screen_objects_ml(frame, cfg))
        if model_mode == "model_only":
            return sorted(
                deduplicate_detections(
                    detections,
                    iou_threshold=float(screen_cfg.get("nms_iou_threshold", 0.35)),
                ),
                key=lambda detection: detection.score,
                reverse=True,
            )

    detections.extend(detect_nameplate_mobs(frame, cfg))
    detections.extend(detect_minimap_ore_icons(frame, cfg))
    detections.extend(detect_terrain_candidates(frame, cfg))
    detections.extend(detect_walkability_candidates(frame, cfg))
    return sorted(
        deduplicate_detections(detections, iou_threshold=float(screen_cfg.get("nms_iou_threshold", 0.35))),
        key=lambda detection: detection.score,
        reverse=True,
    )


def detect_screen_objects_ml(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    model_cfg = _screen_cfg(cfg).get("model", {})
    model_path_value = model_cfg.get("path")
    if not model_path_value:
        return []

    model_path = Path(str(model_path_value))
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    model = _load_yolo_model(model_path)
    predict_args: dict[str, Any] = {
        "source": frame,
        "imgsz": max(64, int(model_cfg.get("imgsz", 416))),
        "conf": max(0.0, min(1.0, float(model_cfg.get("conf", 0.25)))),
        "verbose": False,
        "max_det": max(1, int(model_cfg.get("max_det", 100))),
    }
    if model_cfg.get("device", None) is not None:
        predict_args["device"] = model_cfg["device"]
    results = model.predict(**predict_args)
    detections = yolo_results_to_detections(results, frame.shape)
    return _filter_by_class_min_scores(detections, model_cfg.get("class_min_scores", {}))


def yolo_results_to_detections(
    results: Any,
    frame_shape: tuple[int, ...],
    class_names: list[str] | None = None,
) -> list[ScreenDetection]:
    result_items = results if isinstance(results, list | tuple) else [results]
    detections: list[ScreenDetection] = []
    for result in result_items:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        conf = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        cls = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
        names = class_names or getattr(result, "names", None) or DEFAULT_CLASS_NAMES
        for index, coords in enumerate(xyxy.reshape(-1, 4)):
            if index >= len(conf) or index >= len(cls):
                continue
            x1, y1, x2, y2 = [float(value) for value in coords]
            bbox = BoundingBox(
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ).clipped(frame_shape)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            detections.append(
                ScreenDetection(
                    _label_from_class_index(int(cls[index]), names),
                    bbox,
                    float(conf[index]),
                    "yolo_model",
                )
            )
    return detections


def detect_nameplate_mobs(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    screen_cfg = _screen_cfg(cfg)
    nameplate_cfg = screen_cfg.get("nameplates", {})
    roi_x, roi_y, roi_width, roi_height = _world_region(frame, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return []

    aggressive = _detections_from_colored_bars(
        roi,
        label=ScreenObjectClass.AGGRESSIVE_MOB.value,
        source="red_nameplate_or_hpbar",
        hsv_ranges=[
            ([0, 80, 80], [10, 255, 255]),
            ([170, 80, 80], [179, 255, 255]),
        ],
        offset=(roi_x, roi_y),
        config=nameplate_cfg,
    )
    peaceful = _detections_from_colored_bars(
        roi,
        label=ScreenObjectClass.PEACEFUL_MOB.value,
        source="yellow_nameplate_or_hpbar",
        hsv_ranges=[([16, 70, 90], [40, 255, 255])],
        offset=(roi_x, roi_y),
        config=nameplate_cfg,
    )
    return aggressive + peaceful


def detect_minimap_ore_icons(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    screen_cfg = _screen_cfg(cfg)
    icon_cfg = screen_cfg.get("minimap_ore_icon", {})
    if not bool(icon_cfg.get("enabled", True)):
        return []

    minimap_cfg = cfg.get("minimap", {})
    if not minimap_cfg:
        return []

    x, y, width, height = resolve_region(minimap_cfg, frame.shape, cfg)
    minimap = frame[y : y + height, x : x + width]
    if minimap.size == 0:
        return []

    point_radius = int(icon_cfg.get("bbox_radius", 7))
    detections: list[ScreenDetection] = []
    for point_x, point_y in recognize_ore_points(minimap, cfg):
        bbox = BoundingBox(
            x + point_x - point_radius,
            y + point_y - point_radius,
            point_radius * 2,
            point_radius * 2,
        ).clipped(frame.shape)
        detections.append(
            ScreenDetection(
                ScreenObjectClass.MINIMAP_ORE_ICON.value,
                bbox,
                float(icon_cfg.get("score", 0.90)),
                "minimap_ore_detector",
            )
        )
    return detections


def detect_terrain_candidates(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    terrain_cfg = _screen_cfg(cfg).get("terrain", {})
    if not bool(terrain_cfg.get("enabled", True)):
        return []

    roi_x, roi_y, roi_width, roi_height = _navigation_region(frame, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        int(terrain_cfg.get("canny_low", 45)),
        int(terrain_cfg.get("canny_high", 120)),
    )
    kernel_size = int(terrain_cfg.get("close_kernel", 7))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    obstacle_detections = _detections_from_mask(
        edges,
        label=ScreenObjectClass.TERRAIN_OBSTACLE.value,
        source="terrain_edge_bootstrap",
        offset=(roi_x, roi_y),
        min_area=int(terrain_cfg.get("min_area", 1200)),
        max_area=int(terrain_cfg.get("max_area", 160000)),
        min_width=int(terrain_cfg.get("min_width", 45)),
        max_width=int(terrain_cfg.get("max_width", 900)),
        min_height=int(terrain_cfg.get("min_height", 35)),
        max_height=int(terrain_cfg.get("max_height", 600)),
        min_aspect=float(terrain_cfg.get("min_aspect", 0.20)),
        max_aspect=float(terrain_cfg.get("max_aspect", 8.0)),
        score=float(terrain_cfg.get("score", 0.25)),
    )
    return obstacle_detections[: int(terrain_cfg.get("max_candidates", 8))]


def detect_walkability_candidates(frame: np.ndarray, config: dict[str, Any] | None = None) -> list[ScreenDetection]:
    cfg = config or {}
    screen_cfg = _screen_cfg(cfg)
    walk_cfg = screen_cfg.get("walkability", {})
    if not bool(walk_cfg.get("enabled", True)):
        return []

    roi_x, roi_y, roi_width, roi_height = _navigation_region(frame, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return []

    height, width = roi.shape[:2]
    lower_fraction = max(0.30, min(0.85, float(walk_cfg.get("lower_fraction", 0.52))))
    lower_y = int(height * lower_fraction)
    analysis_bottom_fraction = max(
        lower_fraction + 0.05,
        min(0.98, float(walk_cfg.get("analysis_bottom_fraction", 0.84))),
    )
    analysis_bottom_y = int(height * analysis_bottom_fraction)
    if lower_y >= analysis_bottom_y - 8:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        int(walk_cfg.get("canny_low", 35)),
        int(walk_cfg.get("canny_high", 105)),
    )
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    vertical_edges = (np.abs(sobel_x) > int(walk_cfg.get("vertical_edge_threshold", 35))).astype(np.uint8)
    valid_mask = np.ones(edges.shape, dtype=bool)
    if bool(walk_cfg.get("self_ignore_enabled", True)):
        ignore_width = int(width * max(0.0, min(0.8, float(walk_cfg.get("self_ignore_x_fraction", 0.18)))))
        ignore_top = int(height * max(lower_fraction, min(0.95, float(walk_cfg.get("self_ignore_y_fraction", 0.64)))))
        ignore_x1 = max(0, width // 2 - ignore_width // 2)
        ignore_x2 = min(width, width // 2 + ignore_width // 2)
        valid_mask[ignore_top:height, ignore_x1:ignore_x2] = False

    columns = max(3, int(walk_cfg.get("columns", 9)))
    columns = min(columns, max(3, width // 24))
    column_width = width / float(columns)
    risks: list[float] = []
    boxes: list[BoundingBox] = []
    for index in range(columns):
        x1 = int(round(index * column_width))
        x2 = int(round((index + 1) * column_width))
        segment_edges = edges[lower_y:analysis_bottom_y, x1:x2]
        segment_vertical = vertical_edges[lower_y:analysis_bottom_y, x1:x2]
        segment_gray = gray[lower_y:analysis_bottom_y, x1:x2]
        segment_valid = valid_mask[lower_y:analysis_bottom_y, x1:x2]
        area = int(np.count_nonzero(segment_valid))
        if area <= 0:
            risks.append(0.0)
            boxes.append(BoundingBox(roi_x + x1, roi_y + lower_y, max(1, x2 - x1), analysis_bottom_y - lower_y))
            continue

        edge_density = float(np.count_nonzero(segment_edges[segment_valid])) / float(area)
        vertical_density = float(np.count_nonzero(segment_vertical[segment_valid])) / float(area)
        texture_score = min(1.0, float(np.std(segment_gray[segment_valid])) / 58.0)
        bottom_start = min(analysis_bottom_y - 1, int(height * 0.76))
        bottom_edges = edges[bottom_start:analysis_bottom_y, x1:x2]
        bottom_valid = valid_mask[bottom_start:analysis_bottom_y, x1:x2]
        bottom_area = max(1, int(np.count_nonzero(bottom_valid)))
        bottom_density = float(np.count_nonzero(bottom_edges[bottom_valid])) / float(bottom_area)

        risk = (
            edge_density * float(walk_cfg.get("edge_weight", 1.5))
            + vertical_density * float(walk_cfg.get("vertical_weight", 2.8))
            + texture_score * float(walk_cfg.get("texture_weight", 0.05))
            + bottom_density * float(walk_cfg.get("bottom_edge_weight", 2.0))
        )
        risk = max(0.0, min(1.0, risk))
        risks.append(risk)
        boxes.append(BoundingBox(roi_x + x1, roi_y + lower_y, max(1, x2 - x1), analysis_bottom_y - lower_y))

    risks = _smooth_scores(risks)
    blocker_threshold = max(0.05, min(0.95, float(walk_cfg.get("blocker_threshold", 0.48))))
    free_threshold = max(0.02, min(blocker_threshold, float(walk_cfg.get("free_threshold", 0.33))))
    detections: list[ScreenDetection] = []

    for group in _contiguous_groups([index for index, risk in enumerate(risks) if risk >= blocker_threshold]):
        if not group:
            continue
        first = group[0]
        last = group[-1]
        first_box = boxes[first]
        last_box = boxes[last]
        risk = max(risks[index] for index in group)
        detections.append(
            ScreenDetection(
                ScreenObjectClass.MOVEMENT_BLOCKER.value,
                BoundingBox(first_box.x, first_box.y, last_box.x2 - first_box.x, first_box.height),
                risk,
                "walkability_horizon_bootstrap",
            )
        )

    free_groups = _contiguous_groups([index for index, risk in enumerate(risks) if risk <= free_threshold])
    center = (columns - 1) / 2.0
    best_group = _choose_best_free_group(free_groups, risks, center)
    if best_group is not None:
        first = best_group[0]
        last = best_group[-1]
        first_box = boxes[first]
        last_box = boxes[last]
        mean_risk = float(np.mean([risks[index] for index in best_group]))
        detections.append(
            ScreenDetection(
                ScreenObjectClass.WALKABLE_CORRIDOR.value,
                BoundingBox(first_box.x, first_box.y, last_box.x2 - first_box.x, first_box.height),
                1.0 - mean_risk,
                "walkability_horizon_bootstrap",
            )
        )

    edge_detection = _terrain_edge_from_rows(edges, roi_x, roi_y, walk_cfg)
    if edge_detection is not None:
        detections.append(edge_detection)

    return detections


def draw_detections(frame: np.ndarray, detections: list[ScreenDetection]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        color = _label_color(detection.label)
        bbox = detection.bbox.clipped(frame.shape)
        cv2.rectangle(output, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        text = f"{detection.label} {detection.score:.2f}"
        cv2.putText(
            output,
            text,
            (bbox.x, max(16, bbox.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def detections_to_yolo_lines(
    detections: list[ScreenDetection],
    frame_shape: tuple[int, ...],
    class_names: list[str] | None = None,
    min_score: float = 0.0,
) -> list[str]:
    names = class_names or DEFAULT_CLASS_NAMES
    frame_height, frame_width = frame_shape[:2]
    lines: list[str] = []
    for detection in detections:
        if detection.score < min_score or detection.label not in names:
            continue
        bbox = detection.bbox.clipped(frame_shape)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        class_index = names.index(detection.label)
        center_x = (bbox.x + bbox.width / 2.0) / frame_width
        center_y = (bbox.y + bbox.height / 2.0) / frame_height
        norm_width = bbox.width / frame_width
        norm_height = bbox.height / frame_height
        lines.append(
            f"{class_index} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
        )
    return lines


def deduplicate_detections(
    detections: list[ScreenDetection],
    iou_threshold: float = 0.35,
) -> list[ScreenDetection]:
    kept: list[ScreenDetection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(
            detection.label != existing.label or _iou(detection.bbox, existing.bbox) < iou_threshold
            for existing in kept
        ):
            kept.append(detection)
    return kept


def _filter_by_class_min_scores(
    detections: list[ScreenDetection],
    class_min_scores: dict[str, Any],
) -> list[ScreenDetection]:
    if not class_min_scores:
        return detections
    filtered: list[ScreenDetection] = []
    for detection in detections:
        min_score = float(class_min_scores.get(detection.label, 0.0))
        if detection.score >= min_score:
            filtered.append(detection)
    return filtered


def _detections_from_colored_bars(
    frame: np.ndarray,
    *,
    label: str,
    source: str,
    hsv_ranges: list[tuple[list[int], list[int]]] | list[list[list[int]]],
    offset: tuple[int, int],
    config: dict[str, Any],
) -> list[ScreenDetection]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
        )

    close_kernel = int(config.get("close_kernel", 5))
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return _detections_from_mask(
        mask,
        label=label,
        source=source,
        offset=offset,
        min_area=int(config.get("min_area", 80)),
        max_area=int(config.get("max_area", 9000)),
        min_width=int(config.get("min_width", 20)),
        max_width=int(config.get("max_width", 300)),
        min_height=int(config.get("min_height", 4)),
        max_height=int(config.get("max_height", 42)),
        min_aspect=float(config.get("min_aspect", 3.0)),
        max_aspect=float(config.get("max_aspect", 45.0)),
        score=float(config.get("score", 0.70)),
    )


def _detections_from_mask(
    mask: np.ndarray,
    *,
    label: str,
    source: str,
    offset: tuple[int, int],
    min_area: int,
    max_area: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    min_aspect: float,
    max_aspect: float,
    score: float,
) -> list[ScreenDetection]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    offset_x, offset_y = offset
    detections: list[ScreenDetection] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        aspect = width / float(height)
        if not (
            min_area <= area <= max_area
            and min_width <= width <= max_width
            and min_height <= height <= max_height
            and min_aspect <= aspect <= max_aspect
        ):
            continue
        detections.append(
            ScreenDetection(
                label=label,
                bbox=BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)),
                score=score,
                source=source,
            )
        )
    return detections


def _smooth_scores(scores: list[float]) -> list[float]:
    if len(scores) < 3:
        return scores
    smoothed: list[float] = []
    for index, score in enumerate(scores):
        values = [score]
        if index > 0:
            values.append(scores[index - 1])
        if index < len(scores) - 1:
            values.append(scores[index + 1])
        smoothed.append(sum(values) / float(len(values)))
    return smoothed


def _contiguous_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups: list[list[int]] = []
    current = [indices[0]]
    for index in indices[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)
    return groups


def _choose_best_free_group(
    groups: list[list[int]],
    risks: list[float],
    center: float,
) -> list[int] | None:
    if not groups:
        return None
    return min(
        groups,
        key=lambda group: (
            abs((group[0] + group[-1]) / 2.0 - center),
            float(np.mean([risks[index] for index in group])),
            -len(group),
        ),
    )


def _terrain_edge_from_rows(
    edges: np.ndarray,
    offset_x: int,
    offset_y: int,
    config: dict[str, Any],
) -> ScreenDetection | None:
    height, width = edges.shape[:2]
    if height <= 12 or width <= 12:
        return None
    top = int(height * max(0.05, min(0.75, float(config.get("horizon_min_fraction", 0.18)))))
    bottom = int(height * max(0.20, min(0.95, float(config.get("horizon_max_fraction", 0.72)))))
    if bottom <= top:
        return None
    band = edges[top:bottom, :]
    row_density = np.count_nonzero(band, axis=1) / float(width)
    row_index = int(np.argmax(row_density))
    density = float(row_density[row_index])
    min_density = float(config.get("horizon_min_density", 0.08))
    if density < min_density:
        return None
    y = top + row_index
    line_height = max(4, int(config.get("horizon_box_height", 10)))
    return ScreenDetection(
        ScreenObjectClass.TERRAIN_EDGE.value,
        BoundingBox(offset_x, offset_y + max(0, y - line_height // 2), width, line_height),
        min(1.0, density / max(min_density, 0.001)),
        "walkability_horizon_bootstrap",
    )


def _world_region(frame: np.ndarray, config: dict[str, Any]) -> tuple[int, int, int, int]:
    screen_cfg = _screen_cfg(config)
    default = {
        "x": 0,
        "y": 40,
        "width": int(frame.shape[1] * 0.84),
        "height": int(frame.shape[0] * 0.72),
    }
    return resolve_region(screen_cfg.get("world_region", default), frame.shape, config)


def _navigation_region(frame: np.ndarray, config: dict[str, Any]) -> tuple[int, int, int, int]:
    screen_cfg = _screen_cfg(config)
    default = {
        "x": int(frame.shape[1] * 0.20),
        "y": int(frame.shape[0] * 0.22),
        "width": int(frame.shape[1] * 0.55),
        "height": int(frame.shape[0] * 0.48),
    }
    return resolve_region(screen_cfg.get("navigation_region", default), frame.shape, config)


def _screen_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("screen_objects", {})


def _load_yolo_model(model_path: Path) -> Any:
    cache_key = str(model_path)
    if cache_key not in _YOLO_MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ML screen-object detection requires the optional 'ultralytics' package."
            ) from exc
        _YOLO_MODEL_CACHE[cache_key] = YOLO(cache_key)
    return _YOLO_MODEL_CACHE[cache_key]


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _label_from_class_index(class_index: int, names: Any) -> str:
    if isinstance(names, dict):
        return str(names.get(class_index, _default_label_from_index(class_index)))
    if isinstance(names, list | tuple) and 0 <= class_index < len(names):
        return str(names[class_index])
    return _default_label_from_index(class_index)


def _default_label_from_index(class_index: int) -> str:
    if 0 <= class_index < len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES[class_index]
    return f"class_{class_index}"


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area() + b.area() - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def _label_color(label: str) -> tuple[int, int, int]:
    colors = {
        ScreenObjectClass.AGGRESSIVE_MOB.value: (0, 0, 255),
        ScreenObjectClass.PEACEFUL_MOB.value: (0, 220, 220),
        ScreenObjectClass.ORE_VEIN.value: (0, 255, 120),
        ScreenObjectClass.MINIMAP_ORE_ICON.value: (0, 255, 255),
        ScreenObjectClass.TERRAIN_OBSTACLE.value: (255, 128, 0),
        ScreenObjectClass.TERRAIN_EDGE.value: (255, 0, 255),
        ScreenObjectClass.MOVEMENT_BLOCKER.value: (0, 120, 255),
        ScreenObjectClass.WALKABLE_CORRIDOR.value: (80, 255, 80),
    }
    return colors.get(label, (255, 255, 255))
