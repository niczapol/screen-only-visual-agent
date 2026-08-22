from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from vision_bot.screen_objects import ScreenDetection


LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "aggressive_mob": (40, 40, 235),
    "peaceful_mob": (0, 220, 220),
    "ore_vein": (80, 230, 80),
    "minimap_ore_icon": (240, 80, 240),
    "terrain_obstacle": (255, 140, 0),
    "terrain_edge": (0, 165, 255),
    "movement_blocker": (0, 120, 255),
    "walkable_corridor": (80, 255, 80),
}

LABEL_DESCRIPTIONS: dict[str, str] = {
    "aggressive_mob": "hostile mob / red bar",
    "peaceful_mob": "peaceful mob / yellow bar",
    "ore_vein": "visible world ore",
    "minimap_ore_icon": "ore icon on minimap",
    "terrain_obstacle": "terrain / obstacle",
    "terrain_edge": "terrain edge",
    "movement_blocker": "local movement blocker",
    "walkable_corridor": "local free corridor",
}


def draw_detection_report(
    frame: np.ndarray,
    detections: Iterable[ScreenDetection],
    *,
    title: str = "Screen object check",
    model_path: str | Path | None = None,
) -> np.ndarray:
    detection_list = list(detections)
    annotated = frame.copy()
    for detection in detection_list:
        _draw_detection(annotated, detection)

    panel_width = 520
    panel = np.full((frame.shape[0], panel_width, 3), 28, dtype=np.uint8)
    _draw_panel(panel, detection_list, title=title, model_path=model_path)
    return np.hstack([annotated, panel])


def summarize_detections(detections: Iterable[ScreenDetection]) -> dict[str, int]:
    counts = Counter(detection.label for detection in detections)
    return dict(sorted(counts.items()))


def _draw_detection(output: np.ndarray, detection: ScreenDetection) -> None:
    color = LABEL_COLORS.get(detection.label, (255, 255, 255))
    bbox = detection.bbox.clipped(output.shape)
    cv2.rectangle(output, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 3)
    label = f"{detection.label} {detection.score:.2f}"
    _draw_label(output, label, bbox.x, max(22, bbox.y - 8), color)


def _draw_label(output: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, max(0, output.shape[1] - text_width - 8)))
    y = max(text_height + 8, min(y, output.shape[0] - baseline - 4))
    cv2.rectangle(
        output,
        (x, y - text_height - 7),
        (x + text_width + 8, y + baseline + 5),
        (18, 18, 18),
        -1,
    )
    cv2.rectangle(
        output,
        (x, y - text_height - 7),
        (x + text_width + 8, y + baseline + 5),
        color,
        1,
    )
    cv2.putText(output, text, (x + 4, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_panel(
    panel: np.ndarray,
    detections: list[ScreenDetection],
    *,
    title: str,
    model_path: str | Path | None,
) -> None:
    y = 44
    _panel_text(panel, title, 24, y, scale=0.82, color=(240, 240, 240), thickness=2)
    y += 42
    _panel_text(panel, f"Detections: {len(detections)}", 24, y, scale=0.68, color=(225, 225, 225))
    y += 30
    if model_path is not None:
        path_text = _short_model_path(model_path)
        _panel_text(panel, f"Model: {path_text}", 24, y, scale=0.48, color=(190, 190, 190))
        y += 34

    y += 8
    _panel_text(panel, "Counts", 24, y, scale=0.68, color=(240, 240, 240), thickness=2)
    y += 34
    counts = summarize_detections(detections)
    if not counts:
        _panel_text(panel, "No objects detected", 24, y, scale=0.58, color=(180, 180, 180))
        y += 30
    else:
        for label, count in counts.items():
            color = LABEL_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(panel, (24, y - 15), (44, y + 5), color, -1)
            _panel_text(panel, f"{label}: {count}", 56, y, scale=0.58, color=(225, 225, 225))
            y += 30

    y += 18
    _panel_text(panel, "Legend", 24, y, scale=0.68, color=(240, 240, 240), thickness=2)
    y += 34
    for label, description in LABEL_DESCRIPTIONS.items():
        color = LABEL_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(panel, (24, y - 15), (44, y + 5), color, -1)
        _panel_text(panel, label, 56, y, scale=0.54, color=(230, 230, 230))
        y += 24
        _panel_text(panel, description, 56, y, scale=0.45, color=(170, 170, 170))
        y += 31


def _panel_text(
    panel: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _short_model_path(model_path: str | Path) -> str:
    path = Path(model_path)
    parts = path.parts
    if len(parts) <= 4:
        return str(path)
    return str(Path(*parts[-4:]))
