from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from vision_bot.capture import ScreenCapture
from vision_bot.screen_objects import (
    DEFAULT_CLASS_NAMES,
    ScreenDetection,
    detect_screen_objects,
    detections_to_yolo_lines,
    draw_detections,
)


@dataclass(frozen=True)
class VisionSample:
    stem: str
    image_path: Path
    label_path: Path | None
    overlay_path: Path | None
    detections: list[ScreenDetection]


@dataclass(frozen=True)
class VisionCollectionSummary:
    output_dir: Path
    frame_count: int
    labeled_count: int
    overlay_count: int
    detection_count: int


def collect_vision_dataset(
    config: dict[str, Any],
    *,
    count: int,
    interval: float,
    output_dir: str | Path,
    auto_label: bool = True,
    save_overlays: bool = True,
) -> VisionCollectionSummary:
    capture = ScreenCapture(config)
    capture.find_window()
    capture.activate_window()
    time.sleep(0.5)

    dataset_dir = Path(output_dir)
    prepare_vision_dataset(dataset_dir)

    samples: list[VisionSample] = []
    for index in range(max(1, count)):
        frame = capture.capture_client_region()
        detections = detect_screen_objects(frame, config)
        sample = save_vision_sample(
            frame,
            detections,
            dataset_dir,
            auto_label=auto_label,
            save_overlay=save_overlays,
            sample_index=index,
            config=config,
        )
        samples.append(sample)
        if index < count - 1:
            time.sleep(max(0.0, interval))

    return VisionCollectionSummary(
        output_dir=dataset_dir,
        frame_count=len(samples),
        labeled_count=sum(1 for sample in samples if sample.label_path is not None),
        overlay_count=sum(1 for sample in samples if sample.overlay_path is not None),
        detection_count=sum(len(sample.detections) for sample in samples),
    )


def save_vision_sample(
    frame: np.ndarray,
    detections: list[ScreenDetection],
    output_dir: str | Path,
    *,
    auto_label: bool,
    save_overlay: bool,
    sample_index: int,
    config: dict[str, Any] | None = None,
) -> VisionSample:
    dataset_dir = Path(output_dir)
    prepare_vision_dataset(dataset_dir)
    stem = _sample_stem(sample_index)

    raw_path = dataset_dir / "images" / "raw" / f"{stem}.png"
    cv2.imwrite(str(raw_path), frame)

    label_path: Path | None = None
    if auto_label:
        bootstrap_image_path = dataset_dir / "images" / "bootstrap" / f"{stem}.png"
        label_path = dataset_dir / "labels" / "bootstrap" / f"{stem}.txt"
        cv2.imwrite(str(bootstrap_image_path), frame)
        min_score = float((config or {}).get("vision_dataset", {}).get("label_min_score", 0.0))
        label_path.write_text(
            "\n".join(detections_to_yolo_lines(detections, frame.shape, DEFAULT_CLASS_NAMES, min_score)),
            encoding="utf-8",
        )

    overlay_path: Path | None = None
    if save_overlay:
        overlay_path = dataset_dir / "overlays" / f"{stem}.png"
        cv2.imwrite(str(overlay_path), draw_detections(frame, detections))

    _append_metadata(
        dataset_dir / "metadata.jsonl",
        {
            "stem": stem,
            "image": _relative_posix(raw_path, dataset_dir),
            "label": _relative_posix(label_path, dataset_dir) if label_path is not None else None,
            "overlay": _relative_posix(overlay_path, dataset_dir) if overlay_path is not None else None,
            "detections": [detection.to_dict() for detection in detections],
        },
    )

    return VisionSample(stem, raw_path, label_path, overlay_path, detections)


def prepare_vision_dataset(output_dir: str | Path, class_names: list[str] | None = None) -> Path:
    dataset_dir = Path(output_dir)
    for relative in [
        "images/raw",
        "images/bootstrap",
        "labels/bootstrap",
        "overlays",
    ]:
        (dataset_dir / relative).mkdir(parents=True, exist_ok=True)

    names = class_names or DEFAULT_CLASS_NAMES
    (dataset_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    write_yolo_dataset_yaml(dataset_dir, names)
    return dataset_dir


def write_yolo_dataset_yaml(output_dir: str | Path, class_names: list[str] | None = None) -> Path:
    dataset_dir = Path(output_dir)
    names = class_names or DEFAULT_CLASS_NAMES
    data = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": "images/bootstrap",
        "val": "images/bootstrap",
        "names": {index: name for index, name in enumerate(names)},
    }
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def _append_metadata(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _sample_stem(sample_index: int) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1.0) * 1000)
    return f"{timestamp}_{millis:03d}_{sample_index:04d}"


def _relative_posix(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
