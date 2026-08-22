"""Recover a timestamped coordinate trajectory from a passive WoW demonstration."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from vision_bot.config import load_config
from vision_bot.position import _parse_coordinates
from vision_bot.regions import crop_region


@dataclass(frozen=True)
class CoordinateCandidate:
    x: float
    y: float
    penalty: float
    source: str
    repaired: bool = False


@dataclass
class FrameSample:
    sample_index: int
    timestamp: float
    wall_offset: float
    capture_offset: float
    segment_index: int
    segment_frame_index: int
    coordinate_crop: np.ndarray
    ocr_texts: dict[str, str]
    candidates: list[CoordinateCandidate]


@dataclass
class _TrajectoryState:
    cost: float
    last_candidate: CoordinateCandidate | None
    last_timestamp: float | None
    parent: "_TrajectoryState | None"
    choice: CoordinateCandidate | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=0.6)
    parser.add_argument("--max-interpolation-gap", type=float, default=8.0)
    parser.add_argument("--min-x", type=float, default=0.0)
    parser.add_argument("--max-x", type=float, default=100.0)
    parser.add_argument("--min-y", type=float, default=0.0)
    parser.add_argument("--max-y", type=float, default=100.0)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-summary")
    return parser.parse_args()


def coordinate_candidates_from_text(
    text: str,
    *,
    source: str,
    source_penalty: float = 0.0,
    bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0),
) -> list[CoordinateCandidate]:
    candidates: dict[tuple[float, float], CoordinateCandidate] = {}

    parsed = _parse_coordinates(text)
    if parsed is not None:
        digits = f"{parsed:010d}"
        x = int(digits[:4]) / 100.0
        y = int(digits[4:8]) / 100.0
        _add_candidate(
            candidates,
            x,
            y,
            source_penalty,
            source,
            False,
            bounds,
        )

    digits_only = "".join(re.findall(r"\d", text))
    if len(digits_only) == 8:
        _add_digit_candidate(
            candidates,
            digits_only,
            source_penalty + 0.15,
            source + ":digits",
            False,
            bounds,
        )
    elif len(digits_only) == 7:
        for index in range(8):
            for digit in "0123456789":
                repaired = digits_only[:index] + digit + digits_only[index:]
                _add_digit_candidate(
                    candidates,
                    repaired,
                    source_penalty + 3.0,
                    source + ":insert_digit",
                    True,
                    bounds,
                )
    elif len(digits_only) == 9:
        for index in range(9):
            repaired = digits_only[:index] + digits_only[index + 1 :]
            _add_digit_candidate(
                candidates,
                repaired,
                source_penalty + 2.5,
                source + ":delete_digit",
                True,
                bounds,
            )
    elif len(digits_only) > 9:
        for start in range(len(digits_only) - 7):
            _add_digit_candidate(
                candidates,
                digits_only[start : start + 8],
                source_penalty + 3.5,
                source + ":digit_window",
                True,
                bounds,
            )

    return sorted(candidates.values(), key=lambda item: (item.penalty, item.x, item.y))


def select_continuous_candidates(
    observations: list[tuple[float, list[CoordinateCandidate]]],
    *,
    max_speed: float,
    skip_penalty: float = 4.0,
    beam_width: int = 96,
) -> list[CoordinateCandidate | None]:
    if not observations:
        return []

    states = [_TrajectoryState(0.0, None, None, None, None)]
    for timestamp, candidates in observations:
        next_states: list[_TrajectoryState] = []
        for state in states:
            next_states.append(
                _TrajectoryState(
                    state.cost + skip_penalty,
                    state.last_candidate,
                    state.last_timestamp,
                    state,
                    None,
                )
            )

        for candidate in candidates:
            ranked: list[tuple[float, _TrajectoryState]] = []
            for state in states:
                transition = 0.0
                if state.last_candidate is None or state.last_timestamp is None:
                    transition = 0.25
                else:
                    dt = max(0.05, timestamp - state.last_timestamp)
                    distance = math.hypot(
                        candidate.x - state.last_candidate.x,
                        candidate.y - state.last_candidate.y,
                    )
                    speed = distance / dt
                    transition = 0.25 * (speed / max_speed) ** 2
                    if speed > max_speed:
                        transition += 100.0 + (speed - max_speed) * 40.0
                ranked.append((state.cost + transition, state))
            best_cost, best_parent = min(ranked, key=lambda item: item[0])
            next_states.append(
                _TrajectoryState(
                    best_cost + candidate.penalty,
                    candidate,
                    timestamp,
                    best_parent,
                    candidate,
                )
            )

        states = sorted(next_states, key=lambda state: state.cost)[: max(2, beam_width)]

    state = min(states, key=lambda item: item.cost)
    reversed_choices: list[CoordinateCandidate | None] = []
    while state.parent is not None:
        reversed_choices.append(state.choice)
        state = state.parent
    reversed_choices.reverse()
    if len(reversed_choices) != len(observations):
        raise RuntimeError("Trajectory backtracking did not cover every observation")
    return reversed_choices


def interpolate_trajectory(
    timestamps: list[float],
    selected: list[CoordinateCandidate | None],
    *,
    max_gap: float,
) -> list[tuple[float, float, bool] | None]:
    output: list[tuple[float, float, bool] | None] = [None] * len(selected)
    known = [index for index, candidate in enumerate(selected) if candidate is not None]
    for index in known:
        candidate = selected[index]
        assert candidate is not None
        output[index] = (candidate.x, candidate.y, False)
    for left, right in zip(known, known[1:]):
        gap = timestamps[right] - timestamps[left]
        if gap <= 0.0 or gap > max_gap:
            continue
        left_candidate = selected[left]
        right_candidate = selected[right]
        assert left_candidate is not None and right_candidate is not None
        for index in range(left + 1, right):
            ratio = (timestamps[index] - timestamps[left]) / gap
            output[index] = (
                left_candidate.x + (right_candidate.x - left_candidate.x) * ratio,
                left_candidate.y + (right_candidate.y - left_candidate.y) * ratio,
                True,
            )
    return output


def main() -> None:
    args = parse_args()
    recording_dir = Path(args.recording_dir)
    config = load_config(args.config)
    analysis_dir = recording_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else analysis_dir / "trajectory.jsonl"
    output_summary = Path(args.output_summary) if args.output_summary else analysis_dir / "trajectory_summary.json"

    samples = sample_recording(
        recording_dir,
        config,
        sample_fps=max(0.1, float(args.sample_fps)),
    )
    if not samples:
        raise RuntimeError("No video frames were sampled")

    tesseract_cmd = str(config.get("position_ocr", {}).get("tesseract_cmd", "tesseract"))
    texts_by_variant = batch_ocr_coordinate_crops(
        [sample.coordinate_crop for sample in samples],
        config,
        tesseract_cmd=tesseract_cmd,
    )
    bounds = (float(args.min_x), float(args.max_x), float(args.min_y), float(args.max_y))
    for sample_index, sample in enumerate(samples):
        texts = {variant: pages[sample_index] for variant, pages in texts_by_variant.items()}
        sample.ocr_texts = texts
        merged: dict[tuple[float, float], CoordinateCandidate] = {}
        for source, text in texts.items():
            source_penalty = 0.0 if source == "gray" else 0.15
            for candidate in coordinate_candidates_from_text(
                text,
                source=source,
                source_penalty=source_penalty,
                bounds=bounds,
            ):
                key = (candidate.x, candidate.y)
                if key not in merged or candidate.penalty < merged[key].penalty:
                    merged[key] = candidate
        sample.candidates = sorted(merged.values(), key=lambda item: (item.penalty, item.x, item.y))

    observations = [(sample.capture_offset, sample.candidates) for sample in samples]
    selected = select_continuous_candidates(observations, max_speed=max(0.1, float(args.max_speed)))
    trajectory = interpolate_trajectory(
        [sample.capture_offset for sample in samples],
        selected,
        max_gap=max(0.0, float(args.max_interpolation_gap)),
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for sample, candidate, point in zip(samples, selected, trajectory):
            row: dict[str, Any] = {
                "sample_index": sample.sample_index,
                "timestamp": sample.timestamp,
                "wall_offset": round(sample.wall_offset, 4),
                "capture_offset": round(sample.capture_offset, 4),
                "segment_index": sample.segment_index,
                "segment_frame_index": sample.segment_frame_index,
                "ocr_texts": sample.ocr_texts,
                "candidate_count": len(sample.candidates),
                "selected_source": candidate.source if candidate else None,
                "selected_repaired": candidate.repaired if candidate else None,
                "interpolated": point[2] if point else None,
                "x": round(point[0], 4) if point else None,
                "y": round(point[1], 4) if point else None,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = trajectory_summary(samples, selected, trajectory)
    summary.update(
        {
            "recording_dir": str(recording_dir),
            "output_jsonl": str(output_jsonl),
            "sample_fps": float(args.sample_fps),
            "max_speed": float(args.max_speed),
            "bounds": {
                "min_x": bounds[0],
                "max_x": bounds[1],
                "min_y": bounds[2],
                "max_y": bounds[3],
            },
        }
    )
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def sample_recording(
    recording_dir: Path,
    config: dict[str, Any],
    *,
    sample_fps: float,
) -> list[FrameSample]:
    timeline_rows = [
        json.loads(line)
        for line in (recording_dir / "window_capture_timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not timeline_rows:
        return []
    started_at = float(json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))["started_at"])
    timeline_by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in timeline_rows:
        timeline_by_segment.setdefault(int(row["segment_index"]), []).append(row)

    summary = json.loads((recording_dir / "window_capture_summary.json").read_text(encoding="utf-8"))
    source_fps = float(summary.get("fps", 10.0))
    stride = max(1, int(round(source_fps / sample_fps)))
    samples: list[FrameSample] = []
    for video_path in sorted(recording_dir.glob("demonstration_*.mp4")):
        segment_index = int(video_path.stem.rsplit("_", 1)[1])
        segment_timeline = timeline_by_segment.get(segment_index, [])
        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0 and frame_index < len(segment_timeline):
                timeline = segment_timeline[frame_index]
                samples.append(
                    FrameSample(
                        sample_index=len(samples),
                        timestamp=float(timeline["timestamp"]),
                        wall_offset=float(timeline["timestamp"]) - started_at,
                        capture_offset=len(samples) / sample_fps,
                        segment_index=segment_index,
                        segment_frame_index=frame_index,
                        coordinate_crop=crop_region(frame, config.get("player_position", {}), config),
                        ocr_texts={},
                        candidates=[],
                    )
                )
            frame_index += 1
        capture.release()
    return samples


def batch_ocr_coordinate_crops(
    crops: list[np.ndarray],
    config: dict[str, Any],
    *,
    tesseract_cmd: str,
) -> dict[str, list[str]]:
    if not crops:
        return {"gray": [], "yellow": []}
    with tempfile.TemporaryDirectory(prefix="wow_coord_ocr_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        variant_paths: dict[str, list[Path]] = {"gray": [], "yellow": []}
        for index, crop in enumerate(crops):
            variants = _coordinate_ocr_variants(crop, config)
            for variant, image in variants.items():
                path = temp_dir / f"{variant}_{index:05d}.png"
                cv2.imwrite(str(path), image)
                variant_paths[variant].append(path)

        output: dict[str, list[str]] = {}
        for variant, paths in variant_paths.items():
            image_list = temp_dir / f"{variant}_images.txt"
            image_list.write_text("\n".join(str(path.resolve()) for path in paths), encoding="utf-8")
            command = [
                tesseract_cmd,
                str(image_list),
                "stdout",
                "--oem",
                "3",
                "--psm",
                "7",
                "-c",
                "tessedit_char_whitelist=0123456789.,",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            pages = completed.stdout.split("\x0c")
            if len(pages) < len(paths):
                pages.extend([""] * (len(paths) - len(pages)))
            output[variant] = [page.strip() for page in pages[: len(paths)]]
        return output


def trajectory_summary(
    samples: list[FrameSample],
    selected: list[CoordinateCandidate | None],
    trajectory: list[tuple[float, float, bool] | None],
) -> dict[str, Any]:
    valid = [(index, point) for index, point in enumerate(trajectory) if point is not None]
    speeds: list[float] = []
    distance = 0.0
    for (left_index, left), (right_index, right) in zip(valid, valid[1:]):
        dt = samples[right_index].capture_offset - samples[left_index].capture_offset
        segment_distance = math.hypot(right[0] - left[0], right[1] - left[1])
        distance += segment_distance
        if dt > 0.0:
            speeds.append(segment_distance / dt)
    direct_count = sum(point is not None and not point[2] for point in trajectory)
    interpolated_count = sum(point is not None and point[2] for point in trajectory)
    repaired_count = sum(candidate is not None and candidate.repaired for candidate in selected)
    xs = [point[0] for _index, point in valid]
    ys = [point[1] for _index, point in valid]
    return {
        "samples": len(samples),
        "samples_with_ocr_candidates": sum(bool(sample.candidates) for sample in samples),
        "direct_points": direct_count,
        "interpolated_points": interpolated_count,
        "missing_points": len(samples) - len(valid),
        "selected_repaired_points": repaired_count,
        "path_distance_coord": round(distance, 4),
        "speed_p50_coord_per_second": round(_percentile(speeds, 50), 4),
        "speed_p95_coord_per_second": round(_percentile(speeds, 95), 4),
        "trajectory_bounds": {
            "min_x": round(min(xs), 4) if xs else None,
            "max_x": round(max(xs), 4) if xs else None,
            "min_y": round(min(ys), 4) if ys else None,
            "max_y": round(max(ys), 4) if ys else None,
        },
    }


def _coordinate_ocr_variants(crop: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = crop.shape[:2]
    inner = crop[
        max(0, int(round(height * 0.18))) : min(height, int(round(height * 0.78))),
        max(0, int(round(width * 0.05))) : min(width, int(round(width * 0.95))),
    ]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    ocr_cfg = config.get("position_ocr", {})
    yellow_mask = cv2.inRange(
        hsv,
        np.array(ocr_cfg.get("fallback_hsv_lower", [14, 50, 120]), dtype=np.uint8),
        np.array(ocr_cfg.get("fallback_hsv_upper", [55, 255, 255]), dtype=np.uint8),
    )
    output: dict[str, np.ndarray] = {}
    for name, image in (("gray", gray_mask), ("yellow", yellow_mask)):
        bordered = cv2.copyMakeBorder(image, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
        output[name] = cv2.resize(bordered, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    return output


def _add_digit_candidate(
    candidates: dict[tuple[float, float], CoordinateCandidate],
    digits: str,
    penalty: float,
    source: str,
    repaired: bool,
    bounds: tuple[float, float, float, float],
) -> None:
    if len(digits) != 8:
        return
    _add_candidate(
        candidates,
        int(digits[:4]) / 100.0,
        int(digits[4:]) / 100.0,
        penalty,
        source,
        repaired,
        bounds,
    )


def _add_candidate(
    candidates: dict[tuple[float, float], CoordinateCandidate],
    x: float,
    y: float,
    penalty: float,
    source: str,
    repaired: bool,
    bounds: tuple[float, float, float, float],
) -> None:
    min_x, max_x, min_y, max_y = bounds
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        return
    key = (round(x, 2), round(y, 2))
    candidate = CoordinateCandidate(key[0], key[1], float(penalty), source, repaired)
    existing = candidates.get(key)
    if existing is None or candidate.penalty < existing.penalty:
        candidates[key] = candidate


def _percentile(values: Iterable[float], percentile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    index = int(round((len(items) - 1) * percentile / 100.0))
    return items[index]


if __name__ == "__main__":
    main()
