"""Build a review video from a saved live-route probe.

The script is read-only for probe directories: it reads frames/*.png and
metadata.jsonl, overlays compact route/combat/death telemetry, then writes a
new video file for manual review.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_FPS = 6.0
DEFAULT_MAX_WIDTH = 1600
VIDEO_CODECS = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
)


@dataclass(frozen=True)
class VideoTarget:
    path: Path
    fourcc: str


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    frames_dir = input_dir / "frames"
    metadata_path = input_dir / "metadata.jsonl"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"no PNG frames found in {frames_dir}")

    metadata_by_frame, metadata_by_index = load_metadata(metadata_path)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"could not read first frame: {frame_paths[0]}")

    video_size, scale = output_size(first, args.max_width)
    writer, actual_output = open_writer(output_path, video_size, args.fps)
    written = 0
    try:
        for fallback_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"warning: skipping unreadable frame: {frame_path}")
                continue
            metadata = metadata_for_frame(input_dir, frame_path, fallback_index, metadata_by_frame, metadata_by_index)
            frame = overlay_frame(frame, metadata, fallback_index)
            if scale != 1.0:
                frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError("no frames were written")

    print(f"wrote {written} frames to {actual_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a live-route probe review video.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Probe directory containing frames/ and metadata.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Preferred output video path, usually .mp4")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"Output frames per second, default {DEFAULT_FPS}")
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help=f"Resize wide frames to this width while preserving aspect ratio; 0 keeps original size",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_frame: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"warning: skipping invalid metadata line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                continue
            frame = row.get("frame")
            if isinstance(frame, str) and frame:
                by_frame[normalize_frame_key(frame)] = row
            index = row.get("index")
            if isinstance(index, int):
                by_index[index] = row
    return by_frame, by_index


def normalize_frame_key(value: str | Path) -> str:
    return Path(value).as_posix().replace("\\", "/").lstrip("./")


def metadata_for_frame(
    input_dir: Path,
    frame_path: Path,
    fallback_index: int,
    metadata_by_frame: dict[str, dict[str, Any]],
    metadata_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    try:
        relative = frame_path.relative_to(input_dir)
        key = normalize_frame_key(relative)
    except ValueError:
        key = normalize_frame_key(frame_path.name)
    return metadata_by_frame.get(key) or metadata_by_index.get(fallback_index) or {"index": fallback_index}


def output_size(frame: np.ndarray, max_width: int) -> tuple[tuple[int, int], float]:
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return (width, height), 1.0
    scale = max_width / float(width)
    return (max_width, max(1, int(round(height * scale)))), scale


def open_writer(output_path: Path, size: tuple[int, int], fps: float) -> tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = preferred_video_targets(output_path)
    for target in attempts:
        writer = cv2.VideoWriter(
            str(target.path),
            cv2.VideoWriter_fourcc(*target.fourcc),
            max(0.1, fps),
            size,
        )
        if writer.isOpened():
            return writer, target.path
        writer.release()
    tried = ", ".join(f"{target.fourcc}:{target.path.suffix}" for target in attempts)
    raise RuntimeError(f"could not open VideoWriter; tried {tried}")


def preferred_video_targets(output_path: Path) -> list[VideoTarget]:
    suffix = output_path.suffix.lower()
    targets: list[VideoTarget] = []
    if suffix == ".mp4":
        targets.extend(VideoTarget(output_path, codec) for codec, codec_suffix in VIDEO_CODECS if codec_suffix == ".mp4")
        avi_path = output_path.with_suffix(".avi")
        targets.extend(VideoTarget(avi_path, codec) for codec, codec_suffix in VIDEO_CODECS if codec_suffix == ".avi")
        return targets
    if suffix == ".avi":
        targets.extend(VideoTarget(output_path, codec) for codec, codec_suffix in VIDEO_CODECS if codec_suffix == ".avi")
        mp4_path = output_path.with_suffix(".mp4")
        targets.extend(VideoTarget(mp4_path, codec) for codec, codec_suffix in VIDEO_CODECS if codec_suffix == ".mp4")
        return targets
    targets.extend(VideoTarget(output_path.with_suffix(codec_suffix), codec) for codec, codec_suffix in VIDEO_CODECS)
    return targets


def overlay_frame(frame: np.ndarray, metadata: dict[str, Any], fallback_index: int) -> np.ndarray:
    output = frame.copy()
    draw_bboxes(output, metadata)

    lines = overlay_lines(metadata, fallback_index)
    draw_text_panel(output, lines)
    return output


def draw_bboxes(frame: np.ndarray, metadata: dict[str, Any]) -> None:
    threat = metadata.get("threat")
    if isinstance(threat, dict):
        draw_bbox(frame, threat.get("bbox"), (0, 128, 255), "threat")
    combat = metadata.get("combat")
    if isinstance(combat, dict):
        draw_bbox(frame, combat.get("bbox"), (255, 64, 255), "combat")


def draw_bbox(frame: np.ndarray, bbox: Any, color: tuple[int, int, int], label: str) -> None:
    if not isinstance(bbox, dict):
        return
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return
    if width <= 0 or height <= 0:
        return
    x2 = min(frame.shape[1] - 1, x + width)
    y2 = min(frame.shape[0] - 1, y + height)
    x = max(0, x)
    y = max(0, y)
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    cv2.putText(frame, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def overlay_lines(metadata: dict[str, Any], fallback_index: int) -> list[str]:
    combat = dict_or_empty(metadata.get("combat"))
    threat = dict_or_empty(metadata.get("threat"))
    death_recovery = dict_or_empty(metadata.get("death_recovery"))
    game_state = dict_or_empty(metadata.get("game_state"))

    index = metadata.get("index", fallback_index)
    action = metadata.get("action") or "-"
    coord = format_coord(metadata.get("coord"))
    target = format_coord(metadata.get("target_coord"))
    distance = format_float(metadata.get("distance"))
    progress = format_float(metadata.get("distance_progress"))
    threat_text = threat.get("reason") or threat.get("source") or "-"
    threat_turn = threat.get("turn_key") or "-"
    combat_text = (
        f"active={flag(combat.get('active'))} target={flag(combat.get('target_present'))} "
        f"nameplate={flag(combat.get('nameplate_visible'))} turn={combat.get('face_turn_key') or '-'}"
    )
    death_text = (
        f"modal={flag(game_state.get('death_or_blocking_modal'))} ghost={flag(game_state.get('ghost_visual'))} "
        f"recovery={death_recovery.get('action') or '-'}"
    )
    return [
        f"idx={index} action={action}",
        f"coord={coord} target={target} dist={distance} progress={progress}",
        f"combat {combat_text}",
        f"death {death_text}",
        f"threat={threat_text} threat_turn={threat_turn}",
    ]


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def format_coord(value: Any) -> str:
    if not isinstance(value, int):
        return "-"
    coord = str(value).rjust(10, "0")
    try:
        x = int(coord[:4]) / 100.0
        y = int(coord[4:8]) / 100.0
    except ValueError:
        return str(value)
    return f"{x:.2f},{y:.2f}"


def format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def flag(value: Any) -> str:
    return "1" if value is True else "0"


def draw_text_panel(frame: np.ndarray, lines: Iterable[str]) -> None:
    lines = list(lines)
    margin = 18
    line_height = 27
    panel_width = min(frame.shape[1] - margin * 2, 760)
    panel_height = margin + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 12 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    y = 42
    for line in lines:
        cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        y += line_height


if __name__ == "__main__":
    raise SystemExit(main())
