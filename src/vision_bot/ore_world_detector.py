from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vision_bot.config import runtime_state_path


_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class OreWorldDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass(frozen=True)
class OreWorldDetectorResult:
    detections: tuple[OreWorldDetection, ...] = ()
    elapsed_ms: float = 0.0
    reason: str = "disabled"


class OreWorldDetector:
    """Optional one-shot ore candidate provider for the authoritative hover scan."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.config = config
        self.detector_config = config.get("mining", {}).get("ore_world_detector", {})
        self.enabled = bool(self.detector_config.get("enabled", False))
        self._uses_default_loader = model_loader is None
        self.model_loader = model_loader or _load_ultralytics_model
        self._model: Any | None = None
        self._load_error: str | None = None

    def load(self) -> OreWorldDetectorResult:
        """Load the configured model without running inference.

        Route-live preflight uses this in the *same interpreter* that would run
        the controller.  That prevents a missing optional ML dependency from
        silently degrading the authoritative hover scan to blind grid probes.
        """
        if not self.enabled:
            return OreWorldDetectorResult(reason="disabled")
        started = time.perf_counter()
        model = self._get_model()
        return OreWorldDetectorResult(
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            reason="model_ready" if model is not None else self._load_error or "model_unavailable",
        )

    def detect(self, frame: np.ndarray) -> OreWorldDetectorResult:
        if not self.enabled:
            return OreWorldDetectorResult(reason="disabled")

        started = time.perf_counter()
        model = self._get_model()
        if model is None:
            return OreWorldDetectorResult(
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                reason=self._load_error or "model_unavailable",
            )

        try:
            predict_args: dict[str, Any] = {
                "source": frame,
                "imgsz": max(320, int(self.detector_config.get("imgsz", 640))),
                "conf": _bounded_float(self.detector_config.get("confidence", 0.35)),
                "iou": _bounded_float(self.detector_config.get("model_iou_threshold", 0.45)),
                "max_det": max(1, int(self.detector_config.get("model_max_detections", 12))),
                "verbose": False,
            }
            if self.detector_config.get("device") is not None:
                predict_args["device"] = self.detector_config["device"]
            results = model.predict(**predict_args)
            detections = parse_ore_detections(results, frame.shape)
            detections = filter_ore_detections_by_geometry(
                detections,
                frame.shape,
                min_area_fraction=max(
                    0.0,
                    float(self.detector_config.get("min_area_fraction", 0.0005)),
                ),
                max_area_fraction=min(
                    1.0,
                    float(self.detector_config.get("max_area_fraction", 0.05)),
                ),
            )
            detections = filter_ore_detections_by_self_region(
                detections,
                frame.shape,
                enabled=bool(
                    self.detector_config.get("self_exclusion_enabled", True)
                ),
                center_x_fraction=float(
                    self.detector_config.get("self_exclusion_center_x_fraction", 0.50)
                ),
                center_y_fraction=float(
                    self.detector_config.get("self_exclusion_center_y_fraction", 0.62)
                ),
                width_fraction=float(
                    self.detector_config.get("self_exclusion_width_fraction", 0.14)
                ),
                height_fraction=float(
                    self.detector_config.get("self_exclusion_height_fraction", 0.26)
                ),
            )
            detections = deduplicate_ore_detections(
                detections,
                iou_threshold=_bounded_float(
                    self.detector_config.get("dedupe_iou_threshold", 0.45)
                ),
            )
            max_candidates = max(1, int(self.detector_config.get("max_candidates", 3)))
            return OreWorldDetectorResult(
                tuple(detections[:max_candidates]),
                round((time.perf_counter() - started) * 1000.0, 3),
                "ok" if detections else "no_detection",
            )
        except Exception as exc:  # Fail closed; hover scanning remains available.
            return OreWorldDetectorResult(
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                reason=f"inference_error:{type(exc).__name__}",
            )

    def warm_up(self, frame: np.ndarray) -> OreWorldDetectorResult:
        if not bool(self.detector_config.get("prewarm", True)):
            return OreWorldDetectorResult(reason="prewarm_disabled")
        return self.detect(frame)

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        value = self.detector_config.get("model_path")
        if not value:
            self._load_error = "model_path_missing"
            return None
        path = runtime_state_path(str(value)).resolve()
        if not path.is_file():
            self._load_error = "model_file_missing"
            return None
        cache_key = str(path)
        if self._uses_default_loader and cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return self._model
        try:
            self._model = self.model_loader(path)
            if self._uses_default_loader:
                _MODEL_CACHE[cache_key] = self._model
        except Exception as exc:
            message = "_".join(str(exc).strip().split())
            suffix = f":{message[:160]}" if message else ""
            self._load_error = f"model_load_error:{type(exc).__name__}{suffix}"
            return None
        return self._model


def parse_ore_detections(
    results: Any,
    frame_shape: tuple[int, ...],
) -> list[OreWorldDetection]:
    items = results if isinstance(results, list | tuple) else [results]
    frame_height, frame_width = frame_shape[:2]
    detections: list[OreWorldDetection] = []
    for result in items:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
        confidence = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        for index, coords in enumerate(xyxy):
            if index >= len(confidence):
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width, x2))
            y2 = max(0, min(frame_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                OreWorldDetection(x1, y1, x2, y2, float(confidence[index]))
            )
    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def deduplicate_ore_detections(
    detections: list[OreWorldDetection],
    *,
    iou_threshold: float,
) -> list[OreWorldDetection]:
    kept: list[OreWorldDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_intersection_over_union(detection, accepted) < iou_threshold for accepted in kept):
            kept.append(detection)
    return kept


def filter_ore_detections_by_geometry(
    detections: list[OreWorldDetection],
    frame_shape: tuple[int, ...],
    *,
    min_area_fraction: float,
    max_area_fraction: float,
) -> list[OreWorldDetection]:
    frame_area = max(1, int(frame_shape[0]) * int(frame_shape[1]))
    return [
        detection
        for detection in detections
        if min_area_fraction
        <= ((detection.x2 - detection.x1) * (detection.y2 - detection.y1)) / frame_area
        <= max_area_fraction
    ]


def filter_ore_detections_by_self_region(
    detections: list[OreWorldDetection],
    frame_shape: tuple[int, ...],
    *,
    enabled: bool,
    center_x_fraction: float,
    center_y_fraction: float,
    width_fraction: float,
    height_fraction: float,
) -> list[OreWorldDetection]:
    """Suppress candidate boxes centered on the player's visible body.

    A detector candidate never authorizes a click, but probing the character
    first wastes the short world-search window.  The exclusion is deliberately
    limited to the lower central body band; ore in front of the character and
    the common mid-screen hillside placements remain eligible.
    """
    if not enabled:
        return detections
    height, width = frame_shape[:2]
    center_x = width * max(0.0, min(1.0, center_x_fraction))
    center_y = height * max(0.0, min(1.0, center_y_fraction))
    half_width = width * max(0.0, min(1.0, width_fraction)) / 2.0
    half_height = height * max(0.0, min(1.0, height_fraction)) / 2.0
    x1, x2 = center_x - half_width, center_x + half_width
    y1, y2 = center_y - half_height, center_y + half_height
    return [
        detection
        for detection in detections
        if not (
            x1 <= detection.center[0] <= x2
            and y1 <= detection.center[1] <= y2
        )
    ]


def build_ore_detector_probe_points(
    result: OreWorldDetectorResult,
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    detector_cfg = config.get("mining", {}).get("ore_world_detector", {})
    fractions = detector_cfg.get("probe_offset_fractions", [0.0, -0.20, 0.20])
    height, width = frame_shape[:2]
    points: list[tuple[int, int]] = []
    for detection in result.detections:
        center_x, center_y = detection.center
        box_width = max(1, detection.x2 - detection.x1)
        for fraction in fractions:
            x = int(round(center_x + box_width * float(fraction)))
            point = (max(0, min(width - 1, x)), max(0, min(height - 1, center_y)))
            if point not in points:
                points.append(point)
    return points


def _intersection_over_union(first: OreWorldDetection, second: OreWorldDetection) -> float:
    intersection_width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def _bounded_float(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _load_ultralytics_model(path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ore detection requires the optional ultralytics package") from exc
    return YOLO(str(path))
