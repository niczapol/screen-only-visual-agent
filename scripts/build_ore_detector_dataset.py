from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_DEMO_EVENTS = (
    136.49,
    209.77,
    267.47,
    327.13,
    567.82,
    626.95,
    658.12,
    689.16,
    726.20,
    777.38,
    822.16,
    870.53,
    964.48,
)
DEFAULT_EVENT_OFFSETS = (-2.0, -1.0, 0.0, 0.25)


@dataclass(frozen=True)
class SourceImage:
    image_id: str
    path: Path
    source_kind: str
    source_url: str | None = None
    ore_type: str | None = None
    expected_positive: bool | None = None
    cursor_hint: tuple[int, int] | None = None


@dataclass(frozen=True)
class OreBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def clipped(self, width: int, height: int) -> "OreBox | None":
        x1 = max(0, min(width - 1, self.x1))
        y1 = max(0, min(height - 1, self.y1))
        x2 = max(x1 + 1, min(width, self.x2))
        y2 = max(y1 + 1, min(height, self.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return OreBox(x1, y1, x2, y2, max(0.0, min(1.0, self.confidence)))


def load_reference_sources(manifest_path: Path) -> list[SourceImage]:
    sources: list[SourceImage] = []
    for row in _read_jsonl(manifest_path):
        path = Path(str(row["local_path"]))
        image_id = _image_id(path, str(row.get("sha256", "")))
        sources.append(
            SourceImage(
                image_id=image_id,
                path=path,
                source_kind=str(row.get("source_name", "web_reference")),
                source_url=str(row.get("image_url")) if row.get("image_url") else None,
                ore_type=str(row.get("ore_type")) if row.get("ore_type") else None,
                expected_positive=True,
            )
        )
    return sources


def extract_demo_event_candidates(
    recording_dir: Path,
    output_dir: Path,
    *,
    event_offsets: Iterable[float] = DEFAULT_DEMO_EVENTS,
    sample_offsets: Iterable[float] = DEFAULT_EVENT_OFFSETS,
    input_size: tuple[int, int] = (2048, 1152),
) -> list[SourceImage]:
    manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    started_at = float(manifest["started_at"])
    timeline = list(_read_jsonl(recording_dir / "window_capture_timeline.jsonl"))
    input_rows = list(_read_jsonl(recording_dir / "input_timeline.jsonl"))
    if not timeline:
        raise ValueError(f"No capture timeline rows in {recording_dir}")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    video_cache: dict[int, cv2.VideoCapture] = {}
    sources: list[SourceImage] = []
    try:
        for event in event_offsets:
            click_row = find_last_stationary_right_click(
                input_rows,
                started_at=started_at,
                event_offset=float(event),
            )
            anchor_timestamp = (
                float(click_row["timestamp"])
                if click_row is not None
                else started_at + float(event) - 3.5
            )
            raw_cursor_hint = (
                tuple(int(value) for value in click_row["cursor_client"])
                if click_row is not None
                else None
            )
            for relative in sample_offsets:
                target_timestamp = anchor_timestamp + float(relative)
                row = min(
                    timeline,
                    key=lambda item: abs(float(item["timestamp"]) - target_timestamp),
                )
                segment_index = int(row["segment_index"])
                segment_frame = int(row["segment_frame_index"])
                video_path = recording_dir / f"demonstration_{segment_index:04d}.mp4"
                capture = video_cache.get(segment_index)
                if capture is None:
                    capture = cv2.VideoCapture(str(video_path))
                    if not capture.isOpened():
                        raise RuntimeError(f"Cannot open {video_path}")
                    video_cache[segment_index] = capture
                capture.set(cv2.CAP_PROP_POS_FRAMES, segment_frame)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot read frame {segment_frame} from {video_path}")
                cursor_hint = (
                    scale_point(
                        raw_cursor_hint,
                        source_size=input_size,
                        destination_size=(frame.shape[1], frame.shape[0]),
                    )
                    if raw_cursor_hint is not None
                    else None
                )

                stem = f"demo_event_{event:07.2f}_plus_{relative:04.1f}".replace(".", "p")
                path = image_dir / f"{stem}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                sources.append(
                    SourceImage(
                        image_id=stem,
                        path=path,
                        source_kind="human_tanaris_demo",
                        source_url=None,
                        ore_type=None,
                        expected_positive=True,
                        cursor_hint=cursor_hint,
                    )
                )
    finally:
        for capture in video_cache.values():
            capture.release()

    _write_source_manifest(output_dir / "manifest.jsonl", sources)
    return sources


def scale_point(
    point: tuple[int, int],
    *,
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    return (
        max(0, min(destination_width - 1, round(point[0] / source_width * destination_width))),
        max(0, min(destination_height - 1, round(point[1] / source_height * destination_height))),
    )


def find_last_stationary_right_click(
    rows: Iterable[dict[str, Any]],
    *,
    started_at: float,
    event_offset: float,
    lookback_seconds: float = 8.0,
) -> dict[str, Any] | None:
    window_start = started_at + event_offset - lookback_seconds
    window_end = started_at + event_offset - 0.5
    previous_right = False
    candidates: list[dict[str, Any]] = []
    for row in rows:
        timestamp = float(row["timestamp"])
        if timestamp < window_start:
            previous_right = "mouse_right" in row.get("mouse_down", [])
            continue
        if timestamp > window_end:
            break
        right = "mouse_right" in row.get("mouse_down", [])
        keys = set(row.get("keys_down", []))
        if right and not previous_right and "W" not in keys and "A" not in keys and "D" not in keys:
            candidates.append(row)
        previous_right = right
    return candidates[-1] if candidates else None


def add_local_positive_sources(paths: Iterable[Path], output_dir: Path) -> list[SourceImage]:
    return add_local_sources(
        paths,
        output_dir,
        prefix="live_positive",
        source_kind="reviewed_live_positive",
        expected_positive=True,
    )


def add_local_negative_sources(paths: Iterable[Path], output_dir: Path) -> list[SourceImage]:
    return add_local_sources(
        paths,
        output_dir,
        prefix="live_negative",
        source_kind="reviewed_live_negative",
        expected_positive=False,
    )


def add_local_sources(
    paths: Iterable[Path],
    output_dir: Path,
    *,
    prefix: str,
    source_kind: str,
    expected_positive: bool,
) -> list[SourceImage]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    existing_rows = list(_read_jsonl(manifest_path))
    sources = [
        SourceImage(
            image_id=str(row["image_id"]),
            path=Path(str(row["path"])),
            source_kind=str(row["source_kind"]),
            source_url=row.get("source_url"),
            ore_type=row.get("ore_type"),
            expected_positive=row.get("expected_positive"),
            cursor_hint=(
                tuple(int(value) for value in row["cursor_hint"])
                if row.get("cursor_hint") is not None
                else None
            ),
        )
        for row in existing_rows
    ]
    existing_digests = {
        source.image_id.rsplit("_", 1)[-1]
        for source in sources
    }
    next_index = len(sources)
    for source_path in paths:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable local image: {source_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
        if digest in existing_digests:
            continue
        destination = image_dir / f"{prefix}_{next_index:03d}_{digest}.png"
        if not destination.exists():
            shutil.copy2(source_path, destination)
        sources.append(
            SourceImage(
                image_id=destination.stem,
                path=destination,
                source_kind=source_kind,
                expected_positive=expected_positive,
            )
        )
        existing_digests.add(digest)
        next_index += 1
    _write_source_manifest(manifest_path, sources)
    return sources


def annotate_with_qwen(
    sources: Iterable[SourceImage],
    output_path: Path,
    *,
    model: str,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {row["image_id"]: row for row in _read_jsonl(output_path)}
    results: list[dict[str, Any]] = []
    for source in sources:
        cached = existing.get(source.image_id)
        if cached is not None and not cached.get("error"):
            results.append(cached)
            continue
        image = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
        if image is None:
            result = _annotation_error(source, "unreadable_image")
        else:
            result = _request_qwen_annotation(
                source,
                image,
                model=model,
                endpoint=endpoint,
                timeout_seconds=timeout_seconds,
            )
        _append_jsonl(output_path, result)
        results.append(result)
    return results


def parse_qwen_boxes(content: str, width: int, height: int) -> list[OreBox]:
    payload = json.loads(extract_json_object(content))
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("objects must be a list")
    boxes: list[OreBox] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        values = item.get("bbox_1000")
        if not isinstance(values, list) or len(values) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in values]
        box = OreBox(
            int(round(x1 / 1000.0 * width)),
            int(round(y1 / 1000.0 * height)),
            int(round(x2 / 1000.0 * width)),
            int(round(y2 / 1000.0 * height)),
            float(item.get("confidence", 0.0)),
        ).clipped(width, height)
        if box is None:
            continue
        area_fraction = ((box.x2 - box.x1) * (box.y2 - box.y1)) / float(width * height)
        if 0.0002 <= area_fraction <= 0.92:
            boxes.append(box)
    return boxes


def extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response contains no JSON object")
    return stripped[start : end + 1]


def render_annotation_sheets(
    sources: Iterable[SourceImage],
    annotations: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    columns: int = 4,
    rows: int = 4,
    tile_width: int = 420,
    tile_height: int = 280,
) -> list[Path]:
    source_map = {source.image_id: source for source in sources}
    items = [row for row in annotations if row.get("image_id") in source_map]
    output_dir.mkdir(parents=True, exist_ok=True)
    page_size = max(1, columns * rows)
    outputs: list[Path] = []
    for page_index in range(0, len(items), page_size):
        page = items[page_index : page_index + page_size]
        sheet = np.full((rows * tile_height, columns * tile_width, 3), 28, dtype=np.uint8)
        for local_index, annotation in enumerate(page):
            source = source_map[str(annotation["image_id"])]
            image = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            for box_data in annotation.get("boxes", []):
                cv2.rectangle(
                    image,
                    (int(box_data["x1"]), int(box_data["y1"])),
                    (int(box_data["x2"]), int(box_data["y2"])),
                    (0, 255, 0),
                    max(2, image.shape[1] // 700),
                )
            thumb = _fit_image(image, tile_width, tile_height - 42)
            row_index = local_index // columns
            col_index = local_index % columns
            x = col_index * tile_width + (tile_width - thumb.shape[1]) // 2
            y = row_index * tile_height + 34
            sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
            status = "ERR" if annotation.get("error") else f"box={len(annotation.get('boxes', []))}"
            title = f"{source.image_id[:34]} {status}"
            cv2.putText(
                sheet,
                title,
                (col_index * tile_width + 8, row_index * tile_height + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"annotations_{page_index // page_size:03d}.jpg"
        cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs.append(output_path)
    return outputs


def build_yolo_dataset(
    sources: Iterable[SourceImage],
    annotations: Iterable[dict[str, Any]],
    review_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_map = {source.image_id: source for source in sources}
    annotation_map = {str(row["image_id"]): row for row in annotations}
    review = json.loads(review_path.read_text(encoding="utf-8"))
    samples = review.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("review file must contain a non-empty samples list")

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {
        split: {"images": 0, "positive": 0, "negative": 0, "boxes": 0}
        for split in ("train", "val", "test")
    }
    reviewed_rows: list[dict[str, Any]] = []
    for sample in samples:
        image_id = str(sample["image_id"])
        split = str(sample["split"])
        if split not in counts:
            raise ValueError(f"unsupported split {split!r} for {image_id}")
        source = source_map.get(image_id)
        if source is None:
            raise KeyError(f"review source not found: {image_id}")
        image = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"review image is unreadable: {source.path}")

        negative = bool(sample.get("negative", False))
        boxes: list[OreBox] = []
        if not negative:
            annotation = annotation_map.get(image_id)
            if annotation is None:
                raise KeyError(f"annotation not found for positive sample: {image_id}")
            candidate_boxes = [OreBox(**box) for box in annotation.get("boxes", [])]
            indices = sample.get("box_indices")
            if indices is not None:
                candidate_boxes = [candidate_boxes[int(index)] for index in indices]
            if sample.get("merge_boxes") and candidate_boxes:
                candidate_boxes = [
                    OreBox(
                        min(box.x1 for box in candidate_boxes),
                        min(box.y1 for box in candidate_boxes),
                        max(box.x2 for box in candidate_boxes),
                        max(box.y2 for box in candidate_boxes),
                        max(box.confidence for box in candidate_boxes),
                    )
                ]
            boxes = candidate_boxes
            if not boxes:
                raise ValueError(f"positive sample has no accepted boxes: {image_id}")

        destination_image = output_dir / "images" / split / f"{image_id}.jpg"
        cv2.imwrite(str(destination_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        height, width = image.shape[:2]
        label_lines = [_yolo_label(box, width, height) for box in boxes]
        (output_dir / "labels" / split / f"{image_id}.txt").write_text(
            "".join(label_lines),
            encoding="ascii",
        )
        counts[split]["images"] += 1
        counts[split]["negative" if negative else "positive"] += 1
        counts[split]["boxes"] += len(boxes)
        reviewed_rows.append(
            {
                **_source_dict(source),
                "split": split,
                "negative": negative,
                "boxes": [box.__dict__ for box in boxes],
            }
        )

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: ore_vein",
                "",
            ]
        ),
        encoding="ascii",
    )
    _write_jsonl(output_dir / "reviewed_manifest.jsonl", reviewed_rows)
    summary = {
        "review_file": review_path.as_posix(),
        "dataset_yaml": dataset_yaml.as_posix(),
        "counts": counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _yolo_label(box: OreBox, width: int, height: int) -> str:
    center_x = ((box.x1 + box.x2) / 2.0) / width
    center_y = ((box.y1 + box.y2) / 2.0) / height
    box_width = (box.x2 - box.x1) / width
    box_height = (box.y2 - box.y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _request_qwen_annotation(
    source: SourceImage,
    image: np.ndarray,
    *,
    model: str,
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    annotation_image = image
    crop_origin = (0, 0)
    annotation_hint = source.cursor_hint
    if source.cursor_hint is not None:
        annotation_image, crop_origin = crop_around_point(
            image,
            source.cursor_hint,
            crop_width=1000,
            crop_height=800,
        )
        annotation_hint = (
            int(
                round(
                    (source.cursor_hint[0] - crop_origin[0])
                    / annotation_image.shape[1]
                    * 1000
                )
            ),
            int(
                round(
                    (source.cursor_hint[1] - crop_origin[1])
                    / annotation_image.shape[0]
                    * 1000
                )
            ),
        )
    prompt = (
        "Locate every visible mineable World of Warcraft ore vein or mineral deposit in the full image. "
        "Exclude characters, UI, minimap markers, cursors, item icons and ordinary rocks. "
        "Return exactly one enclosing box for each complete physical deposit: include its rock base and all "
        "attached crystals, never emit separate boxes for individual crystals on the same deposit. "
        "The image may contain zero, one or several deposits. Think briefly, then return strict JSON only "
        "in content: {\"objects\":[{\"bbox_1000\":[x1,y1,x2,y2],\"confidence\":0.0}]}. "
        "Coordinates are integers 0..1000 relative to the full image."
    )
    if source.expected_positive is True:
        prompt += " This source is known to contain at least one ore deposit; do not return empty without locating it."
    if annotation_hint is not None:
        prompt += (
            f" A human mining click occurred near normalized 0..1000 coordinate {annotation_hint}; the deposit "
            "contains or immediately touches that point. Use it as a spatial hint and still box the complete "
            "visible deposit."
        )
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [_base64_image_array(annotation_image)],
            }
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 700,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        message = response_payload.get("message", {})
        content = str(message.get("content", ""))
        if not content.strip():
            content = str(message.get("thinking", ""))
        crop_boxes = parse_qwen_boxes(
            content,
            annotation_image.shape[1],
            annotation_image.shape[0],
        )
        boxes = [
            OreBox(
                box.x1 + crop_origin[0],
                box.y1 + crop_origin[1],
                box.x2 + crop_origin[0],
                box.y2 + crop_origin[1],
                box.confidence,
            )
            for box in crop_boxes
        ]
        return {
            **_source_dict(source),
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "boxes": [box.__dict__ for box in boxes],
            "annotation_crop": {
                "x": crop_origin[0],
                "y": crop_origin[1],
                "width": annotation_image.shape[1],
                "height": annotation_image.shape[0],
            },
            "review_status": "candidate",
            "error": None,
        }
    except Exception as exc:
        return {
            **_annotation_error(source, f"{type(exc).__name__}:{exc}"),
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }


def crop_around_point(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    crop_width: int,
    crop_height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    actual_width = min(width, crop_width)
    actual_height = min(height, crop_height)
    x = max(0, min(width - actual_width, int(point[0]) - actual_width // 2))
    y = max(0, min(height - actual_height, int(point[1]) - actual_height // 2))
    return image[y : y + actual_height, x : x + actual_width], (x, y)


def _base64_image_array(image: np.ndarray) -> str:
    import base64

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("cannot encode annotation image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _annotation_error(source: SourceImage, error: str) -> dict[str, Any]:
    return {
        **_source_dict(source),
        "boxes": [],
        "review_status": "error",
        "error": error,
    }


def _source_dict(source: SourceImage) -> dict[str, Any]:
    return {
        "image_id": source.image_id,
        "path": source.path.as_posix(),
        "source_kind": source.source_kind,
        "source_url": source.source_url,
        "ore_type": source.ore_type,
        "expected_positive": source.expected_positive,
        "cursor_hint": list(source.cursor_hint) if source.cursor_hint is not None else None,
    }


def _write_source_manifest(path: Path, sources: Iterable[SourceImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for source in sources:
            handle.write(json.dumps(_source_dict(source), ensure_ascii=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _image_id(path: Path, digest: str = "") -> str:
    suffix = digest[:12] if digest else hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}_{suffix}"


def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height))
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reviewed ore-detector bootstrap data")
    parser.add_argument("--root", type=Path, default=Path("data/ore_world_detector_v1"))
    parser.add_argument(
        "--references-manifest",
        type=Path,
        default=Path("data/ore_world_detector_v1/references/manifest.jsonl"),
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/human_demo_tanaris_20260802_114252"),
    )
    parser.add_argument("--extract-demo", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--qwen-model", default="qwen3-vl:4b")
    parser.add_argument("--local-positive", action="append", type=Path, default=[])
    parser.add_argument("--local-negative", action="append", type=Path, default=[])
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--build-yolo", action="store_true")
    args = parser.parse_args()

    sources = load_reference_sources(args.references_manifest)
    demo_manifest = args.root / "demo_candidates" / "manifest.jsonl"
    if args.extract_demo:
        extract_demo_event_candidates(args.demo_dir, args.root / "demo_candidates")
    if demo_manifest.exists():
        sources.extend(
            SourceImage(
                image_id=str(row["image_id"]),
                path=Path(str(row["path"])),
                source_kind=str(row["source_kind"]),
                source_url=row.get("source_url"),
                ore_type=row.get("ore_type"),
                expected_positive=row.get("expected_positive"),
                cursor_hint=(
                    tuple(int(value) for value in row["cursor_hint"])
                    if row.get("cursor_hint") is not None
                    else None
                ),
            )
            for row in _read_jsonl(demo_manifest)
        )
    local_manifest = args.root / "local_positives" / "manifest.jsonl"
    if args.local_positive:
        add_local_positive_sources(args.local_positive, args.root / "local_positives")
    if local_manifest.exists():
        sources.extend(
            SourceImage(
                image_id=str(row["image_id"]),
                path=Path(str(row["path"])),
                source_kind=str(row["source_kind"]),
                source_url=row.get("source_url"),
                ore_type=row.get("ore_type"),
                expected_positive=row.get("expected_positive"),
                cursor_hint=(
                    tuple(int(value) for value in row["cursor_hint"])
                    if row.get("cursor_hint") is not None
                    else None
                ),
            )
            for row in _read_jsonl(local_manifest)
        )

    negative_manifest = args.root / "local_negatives" / "manifest.jsonl"
    if args.local_negative:
        add_local_negative_sources(args.local_negative, args.root / "local_negatives")
    if negative_manifest.exists():
        sources.extend(
            SourceImage(
                image_id=str(row["image_id"]),
                path=Path(str(row["path"])),
                source_kind=str(row["source_kind"]),
                source_url=row.get("source_url"),
                ore_type=row.get("ore_type"),
                expected_positive=row.get("expected_positive"),
                cursor_hint=(
                    tuple(int(value) for value in row["cursor_hint"])
                    if row.get("cursor_hint") is not None
                    else None
                ),
            )
            for row in _read_jsonl(negative_manifest)
        )

    annotations_path = args.root / "qwen_candidate_annotations.jsonl"
    annotations = (
        annotate_with_qwen(sources, annotations_path, model=args.qwen_model)
        if args.annotate
        else list(_read_jsonl(annotations_path))
    )
    outputs = (
        render_annotation_sheets(sources, annotations, args.root / "annotation_sheets")
        if args.render and annotations
        else []
    )
    dataset_summary = None
    if args.build_yolo:
        if args.review_file is None:
            parser.error("--build-yolo requires --review-file")
        dataset_summary = build_yolo_dataset(
            sources,
            annotations,
            args.review_file,
            args.root / "yolo_dataset",
        )
    print(f"Sources: {len(sources)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Sheets: {len(outputs)}")
    if dataset_summary is not None:
        print(f"YOLO: {json.dumps(dataset_summary['counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
