from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import cv2

from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy
from vision_bot.heading_navigation import (
    HeadingNavigationController,
    heading_to_coord_degrees,
    signed_heading_error_degrees,
)
from vision_bot.runtime_markers import read_runtime_telemetry


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_heading(
    run_dir: Path,
    row: dict[str, Any],
    config: dict[str, Any],
) -> float | None:
    frame_name = row.get("frame")
    if not frame_name:
        return None
    frame = cv2.imread(str(run_dir / str(frame_name)))
    if frame is None:
        return None
    telemetry = read_runtime_telemetry(frame, config)
    return None if telemetry is None else telemetry.heading_degrees


def _distance(a: int, b: int) -> float:
    ax, ay = coord_to_xy(a)
    bx, by = coord_to_xy(b)
    return math.hypot(bx - ax, by - ay)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _turn_reversals(values: list[str | None]) -> int:
    reversals = 0
    previous: str | None = None
    for value in values:
        if value not in {"A", "D"}:
            continue
        if previous is not None and value != previous:
            reversals += 1
        previous = value
    return reversals


def evaluate_run(
    run_dir: Path,
    config: dict[str, Any],
    *,
    stride: int,
    min_distance: float,
    max_motion_samples: int,
    max_turn_samples: int,
) -> dict[str, Any]:
    rows = _load_rows(run_dir / "metadata.jsonl")
    sampled: list[tuple[dict[str, Any], float]] = []
    for row in rows[:: max(1, stride)]:
        coord = row.get("coord")
        if coord is None:
            continue
        heading = _read_heading(run_dir, row, config)
        if heading is not None:
            sampled.append((row, heading))

    motion_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for previous, current in zip(rows, rows[1:]):
        previous_coord = previous.get("coord")
        current_coord = current.get("coord")
        if previous_coord is None or current_coord is None:
            continue
        if _distance(int(previous_coord), int(current_coord)) < min_distance:
            continue
        movement = previous.get("movement") or {}
        if movement.get("held_key") != "W":
            continue
        if (
            previous.get("local_turn_key") is not None
            or movement.get("turn_key") is not None
        ):
            continue
        if previous.get("action") not in {"continue_forward", "vector_forward"}:
            continue
        motion_pairs.append((previous, current))

    if len(motion_pairs) > max_motion_samples:
        step = len(motion_pairs) / max_motion_samples
        motion_pairs = [motion_pairs[int(index * step)] for index in range(max_motion_samples)]

    motion_errors: list[float] = []
    heading_cache = {int(row["index"]): heading for row, heading in sampled}
    for previous, current in motion_pairs:
        previous_coord = int(previous["coord"])
        current_coord = int(current["coord"])
        previous_heading = heading_cache.get(int(previous.get("index", -1)))
        if previous_heading is None:
            previous_heading = _read_heading(run_dir, previous, config)
        if previous_heading is None:
            continue
        desired = heading_to_coord_degrees(previous_coord, current_coord)
        if desired is None:
            continue
        motion_errors.append(
            abs(signed_heading_error_degrees(previous_heading, desired))
        )

    turn_deltas: dict[str, list[float]] = {"A": [], "D": []}
    for row, following in zip(rows, rows[1:]):
        key = row.get("local_turn_key") or (row.get("movement") or {}).get("turn_key")
        if key not in turn_deltas or len(turn_deltas[key]) >= max_turn_samples:
            continue
        row_index = int(row.get("index", -1))
        following_index = int(following.get("index", -1))
        start = heading_cache.get(row_index)
        if start is None:
            start = _read_heading(run_dir, row, config)
            if start is not None:
                heading_cache[row_index] = start
        end = heading_cache.get(following_index)
        if end is None:
            end = _read_heading(run_dir, following, config)
            if end is not None:
                heading_cache[following_index] = end
        if start is not None and end is not None:
            turn_deltas[key].append(signed_heading_error_degrees(start, end))

    heading_cfg = config.get("movement", {}).get("heading_control", {})
    positive_error_turn_key = str(
        heading_cfg.get("positive_error_turn_key", "A")
    ).upper()
    if positive_error_turn_key not in {"A", "D"}:
        positive_error_turn_key = "A"
    shadow_controller = HeadingNavigationController(
        turn_engage_degrees=float(heading_cfg.get("turn_engage_degrees", 9.0)),
        turn_release_degrees=float(heading_cfg.get("turn_release_degrees", 4.0)),
        pivot_degrees=float(heading_cfg.get("pivot_degrees", 100.0)),
        positive_error_turn_key=positive_error_turn_key,
        turn_rate_degrees_per_second=float(
            heading_cfg.get("turn_rate_degrees_per_second", 110.0)
        ),
        min_turn_pulse_seconds=float(
            heading_cfg.get("min_turn_pulse_seconds", 0.06)
        ),
        max_turn_pulse_seconds=float(
            heading_cfg.get("max_turn_pulse_seconds", 0.35)
        ),
        opposite_turn_lock_seconds=float(
            heading_cfg.get("opposite_turn_lock_seconds", 2.50)
        ),
        opposite_turn_engage_degrees=float(
            heading_cfg.get("opposite_turn_engage_degrees", 28.0)
        ),
    )
    old_turns: list[str | None] = []
    new_turns: list[str | None] = []
    shadow_errors: list[float] = []
    shadow_disagreements = 0
    shadow_braking_rows = 0
    old_wrong_direction = 0
    old_unnecessary_turn = 0
    old_missed_correction = 0
    route_actions = {
        "continue_forward",
        "vector_forward",
        "course_correct_and_forward",
        "turn_and_forward",
        "turn_in_place_and_forward",
    }
    for row, heading in sampled:
        if row.get("action") not in route_actions:
            continue
        movement = row.get("movement") or {}
        current_coord = row.get("coord")
        target_coord = (
            row.get("steering_target_coord")
            or movement.get("target_coord")
            or row.get("target_coord")
        )
        if current_coord is None or target_coord is None:
            continue
        command = shadow_controller.plan(
            current_coord=int(current_coord),
            target_coord=int(target_coord),
            heading_degrees=heading,
            now=float(row.get("timestamp", row.get("index", 0))),
        )
        old_turn = row.get("local_turn_key") or movement.get("turn_key")
        old_turn = old_turn if old_turn in {"A", "D"} else None
        old_turns.append(old_turn)
        new_turns.append(command.turn_key)
        if command.heading_error_degrees is not None:
            error = float(command.heading_error_degrees)
            shadow_errors.append(abs(error))
            expected_turn = "A" if error > 0.0 else "D"
            if old_turn is not None and abs(error) > 4.0 and old_turn != expected_turn:
                old_wrong_direction += 1
            if old_turn is not None and abs(error) <= 4.0:
                old_unnecessary_turn += 1
            if old_turn is None and abs(error) >= 9.0:
                old_missed_correction += 1
        shadow_disagreements += int(old_turn != command.turn_key)
        shadow_braking_rows += int(command.braking)

    return {
        "run_dir": str(run_dir),
        "metadata_rows": len(rows),
        "sampled_heading_rows": len(sampled),
        "motion_segments": len(motion_errors),
        "motion_heading_absolute_error_degrees": {
            "median": statistics.median(motion_errors) if motion_errors else None,
            "p75": _percentile(motion_errors, 0.75),
            "p90": _percentile(motion_errors, 0.90),
            "within_30_degrees_fraction": (
                sum(value <= 30.0 for value in motion_errors) / len(motion_errors)
                if motion_errors
                else None
            ),
        },
        "observed_turn_heading_delta_degrees": {
            key: {
                "samples": len(values),
                "median": statistics.median(values) if values else None,
            }
            for key, values in turn_deltas.items()
        },
        "shadow_controller": {
            "eligible_rows": len(new_turns),
            "command_disagreement_fraction": (
                shadow_disagreements / len(new_turns) if new_turns else None
            ),
            "old_turn_reversals": _turn_reversals(old_turns),
            "new_turn_reversals": _turn_reversals(new_turns),
            "old_turn_rows": sum(value is not None for value in old_turns),
            "new_turn_rows": sum(value is not None for value in new_turns),
            "new_pivot_braking_rows": shadow_braking_rows,
            "absolute_heading_error_degrees": {
                "median": statistics.median(shadow_errors) if shadow_errors else None,
                "p75": _percentile(shadow_errors, 0.75),
            },
            "recorded_wrong_direction_turns": old_wrong_direction,
            "recorded_unnecessary_turns": old_unnecessary_turn,
            "recorded_missed_corrections": old_missed_correction,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate visible player heading against a saved route run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--max-motion-samples", type=int, default=250)
    parser.add_argument("--max-turn-samples", type=int, default=80)
    args = parser.parse_args()

    report = evaluate_run(
        args.run_dir,
        load_config(args.config),
        stride=args.stride,
        min_distance=args.min_distance,
        max_motion_samples=args.max_motion_samples,
        max_turn_samples=args.max_turn_samples,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
