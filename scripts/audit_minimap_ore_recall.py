from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from vision_bot.config import load_config
from vision_bot.recognition import recognize_bright_ore_template_points
from vision_bot.regions import crop_region


@dataclass(frozen=True)
class Candidate:
    run: Path
    index: int
    frame: Path
    reasons: tuple[str, ...]
    score: int
    current_points: tuple[tuple[int, int], ...]
    relaxed_points: tuple[tuple[int, int], ...]
    component_points: tuple[tuple[int, int], ...]


def _runtime_points(row: dict[str, Any]) -> list[tuple[int, int]]:
    mining = row.get("mining") or {}
    points: list[tuple[int, int]] = []
    for key in ("bright_points", "dark_points"):
        for value in mining.get(key) or []:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                points.append((int(value[0]), int(value[1])))
    return points


def _strong_reasons(row: dict[str, Any]) -> list[str]:
    mining = row.get("mining") or {}
    probe = row.get("minimap_tooltip_probe") or {}
    reasons: list[str] = []
    if row.get("minimap_ore_tooltip"):
        reasons.append("visible_tooltip_marker")
    if probe.get("confirmed_track_id") is not None:
        reasons.append("tooltip_confirmed_track")
    if mining.get("minimap_tooltip_confirmed") or mining.get("confirmed_ore_type"):
        reasons.append("mining_tooltip_confirmed")
    if mining.get("target_marker_id") is not None:
        reasons.append("retained_marker_track")
    return reasons


def _nearby_runtime_detection(rows: list[dict[str, Any]], index: int, radius: int) -> bool:
    before = any(_runtime_points(rows[pos]) for pos in range(max(0, index - radius), index))
    after = any(
        _runtime_points(rows[pos])
        for pos in range(index + 1, min(len(rows), index + radius + 1))
    )
    return before and after


def _component_points(minimap: np.ndarray, *, relaxed: bool) -> list[tuple[int, int]]:
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 70, 70] if relaxed else [18, 130, 125], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    scale = min(minimap.shape[1] / 293.0, minimap.shape[0] / 278.0)
    center_x = minimap.shape[1] * 0.50
    center_y = minimap.shape[0] * 0.48
    sensor_radius = min(minimap.shape[:2]) * 0.42
    points: list[tuple[int, int]] = []
    for component in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        if not max(2, int(round(5 * scale))) <= area <= max(12, int(round(180 * scale * scale))):
            continue
        aspect = width / max(1, height)
        if not 0.52 <= aspect <= 1.85:
            continue
        if width > max(5, int(round(22 * scale))) or height > max(5, int(round(22 * scale))):
            continue
        point_x, point_y = [int(round(value)) for value in centroids[component]]
        if (point_x - center_x) ** 2 + (point_y - center_y) ** 2 > sensor_radius**2:
            continue
        points.append((point_x, point_y))
    return points


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def audit_run(
    run: Path,
    config: dict[str, Any],
    *,
    temporal_radius: int,
    scan_all: bool,
) -> tuple[list[Candidate], dict[str, int]]:
    metadata_path = run / "metadata.jsonl"
    rows = _read_rows(metadata_path)
    relaxed_config = copy.deepcopy(config)
    relaxed_recognition = relaxed_config.setdefault("recognition", {})
    relaxed_recognition["bright_template_threshold"] = min(
        0.50, float(relaxed_recognition.get("bright_template_threshold", 0.60))
    )
    relaxed_recognition["bright_template_scale_offsets"] = [0.88, 0.94, 1.0, 1.06, 1.12]
    relaxed_recognition["bright_icon_min_component_aspect"] = 0.52
    relaxed_recognition["bright_icon_max_component_aspect"] = 1.85

    candidates: list[Candidate] = []
    scanned = 0
    excluded_recognized = 0
    missing_frames = 0
    excluded_tooltip_occluded = 0
    for position, row in enumerate(rows):
        if _runtime_points(row):
            excluded_recognized += 1
            continue
        probe = row.get("minimap_tooltip_probe") or {}
        if row.get("minimap_ore_tooltip") or bool(probe.get("active")):
            # The native tooltip and hover cursor cover the marker. These rows
            # are useful transaction evidence but invalid icon-recall labels.
            excluded_tooltip_occluded += 1
            continue
        reasons = _strong_reasons(row)
        temporal_sandwich = _nearby_runtime_detection(rows, position, temporal_radius)
        if temporal_sandwich:
            reasons.append("temporal_sandwich")
        actionable_reasons = [reason for reason in reasons if reason != "retained_marker_track"]
        previous_reasons = (
            set(_strong_reasons(rows[position - 1])) if position > 0 else set()
        )
        evidence_transition = bool(set(actionable_reasons) - previous_reasons)
        if evidence_transition:
            reasons.append("evidence_transition")
        if not scan_all and not temporal_sandwich and not evidence_transition:
            continue
        relative_frame = row.get("frame")
        if not relative_frame:
            continue
        frame_path = run / str(relative_frame)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            missing_frames += 1
            continue
        scanned += 1
        minimap = crop_region(frame, config["minimap"], config)
        component_points = _component_points(minimap, relaxed=True)
        if not component_points and not reasons:
            continue
        current_points = recognize_bright_ore_template_points(minimap, config)
        relaxed_points = (
            current_points
            if current_points
            else recognize_bright_ore_template_points(minimap, relaxed_config)
        )
        if current_points:
            reasons.append("current_detector_recovers_runtime_miss")
        elif relaxed_points:
            reasons.append("relaxed_template_candidate")
        elif component_points:
            reasons.append("yellow_component_candidate")
        if not reasons:
            continue

        weights = {
            "visible_tooltip_marker": 6,
            "tooltip_confirmed_track": 5,
            "mining_tooltip_confirmed": 5,
            "temporal_sandwich": 4,
            "retained_marker_track": 2,
            "current_detector_recovers_runtime_miss": 4,
            "relaxed_template_candidate": 2,
            "yellow_component_candidate": 1,
            "evidence_transition": 1,
        }
        score = sum(weights.get(reason, 0) for reason in set(reasons))
        # A raw yellow component alone is a proposal, not ore evidence. Keep it
        # out of the review pack unless another independent signal exists.
        if score <= 1:
            continue
        candidates.append(
            Candidate(
                run=run,
                index=int(row.get("index", position)),
                frame=frame_path,
                reasons=tuple(dict.fromkeys(reasons)),
                score=score,
                current_points=tuple(current_points),
                relaxed_points=tuple(relaxed_points),
                component_points=tuple(component_points),
            )
        )
    return candidates, {
        "rows": len(rows),
        "excluded_runtime_recognized": excluded_recognized,
        "frames_scanned": scanned,
        "missing_frames": missing_frames,
        "excluded_tooltip_occluded": excluded_tooltip_occluded,
    }


def _render_review_sheet(
    candidates: Iterable[Candidate],
    config: dict[str, Any],
    output: Path,
    *,
    limit: int,
) -> int:
    tiles: list[np.ndarray] = []
    for candidate in list(candidates)[:limit]:
        frame = cv2.imread(str(candidate.frame), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        minimap = crop_region(frame, config["minimap"], config).copy()
        for point in candidate.component_points:
            cv2.circle(minimap, point, 7, (0, 0, 255), 1)
        for point in candidate.relaxed_points:
            cv2.circle(minimap, point, 10, (0, 165, 255), 2)
        for point in candidate.current_points:
            cv2.circle(minimap, point, 13, (0, 255, 0), 2)
        label = np.zeros((54, minimap.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            label,
            f"{candidate.run.name} #{candidate.index} score={candidate.score}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            label,
            ",".join(candidate.reasons)[:72],
            (4, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        tile = np.vstack((label, minimap))
        tiles.append(tile)
    if not tiles:
        return 0
    columns = 4
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_height
        x = (index % columns) * tile_width
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return len(tiles)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a review pack of minimap ore-icon false-negative candidates"
    )
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, default=Path("data/minimap_ore_recall_v0822"))
    parser.add_argument("--temporal-radius", type=int, default=3)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument("--scan-all", action="store_true")
    args = parser.parse_args()

    config = load_config()
    runs = args.runs or sorted(
        path.parent for path in Path("data").glob("live_v08*/metadata.jsonl")
    )
    all_candidates: list[Candidate] = []
    run_stats: dict[str, dict[str, int]] = {}
    for run in runs:
        metadata_path = run / "metadata.jsonl"
        if not metadata_path.is_file():
            continue
        candidates, stats = audit_run(
            run,
            config,
            temporal_radius=max(1, args.temporal_radius),
            scan_all=args.scan_all,
        )
        all_candidates.extend(candidates)
        run_stats[run.name] = stats | {"candidates": len(candidates)}

    all_candidates.sort(key=lambda item: (-item.score, item.run.name, item.index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy": {
            "runtime_recognized_frames_excluded": True,
            "candidate_is_not_ground_truth": True,
            "manual_review_required_before_training": True,
        },
        "runs": run_stats,
        "candidate_count": len(all_candidates),
        "candidates": [
            {
                "run": item.run.as_posix(),
                "index": item.index,
                "frame": item.frame.as_posix(),
                "score": item.score,
                "reasons": list(item.reasons),
                "current_points": item.current_points,
                "relaxed_points": item.relaxed_points,
                "component_points": item.component_points,
            }
            for item in all_candidates
        ],
    }
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    rendered = _render_review_sheet(
        all_candidates,
        config,
        args.output_dir / "candidate_contact_sheet.png",
        limit=max(1, args.review_limit),
    )
    summary = {
        "runs": len(run_stats),
        "candidates": len(all_candidates),
        "rendered": rendered,
        "output_dir": args.output_dir.as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
