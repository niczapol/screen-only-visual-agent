"""Analyze a recorded live route probe without controlling the game client.

The report combines route telemetry with the raw segmented window recording.
It is intentionally read-only for the source run directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from vision_bot.coords import coord_to_xy
from vision_bot.route_database import project_to_loop
from vision_bot.route_following import DirectedRouteFollower
from vision_bot.terrain_routing import ZoneTransform, build_terrain_raster


OVERVIEW_INTERVAL_SECONDS = 10.0
CONTACT_SHEET_COLUMNS = 3
CONTACT_TILE_SIZE = (512, 288)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a saved live route run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
    parser.add_argument("--terrain-overlay", type=Path)
    parser.add_argument("--terrain-manifest", type=Path)
    parser.add_argument("--terrain-adt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overview-interval", type=float, default=OVERVIEW_INTERVAL_SECONDS)
    parser.add_argument(
        "--dense-ranges",
        default="",
        help="Comma-separated metadata index ranges, for example 680:720,940:1010.",
    )
    parser.add_argument("--dense-step", type=int, default=2)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_route_coords(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    route_loop = payload.get("route_loop") if isinstance(payload, dict) else None
    if not isinstance(route_loop, list):
        raise ValueError(f"Route has no route_loop: {path}")
    coords = [int(item["coord"]) for item in route_loop if isinstance(item, dict) and "coord" in item]
    if len(coords) < 2:
        raise ValueError(f"Route loop is too short: {path}")
    return coords


def fresh_coordinate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("coord_fresh") is True and isinstance(row.get("coord"), int)
    ]


def project_trajectory(
    rows: Sequence[dict[str, Any]], route_coords: list[int]
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    previous_progress: float | None = None
    loop_size = float(len(route_coords))
    for row in fresh_coordinate_rows(rows):
        coord = int(row["coord"])
        projection = project_to_loop(coord, route_coords)
        progress = float(projection.sort_key)
        signed_delta = None
        if previous_progress is not None:
            raw = progress - previous_progress
            signed_delta = ((raw + loop_size / 2.0) % loop_size) - loop_size / 2.0
        x, y = coord_to_xy(coord)
        projected.append(
            {
                "index": int(row.get("index", len(projected))),
                "timestamp": row.get("timestamp"),
                "coord": coord,
                "x": x,
                "y": y,
                "route_progress": progress,
                "route_distance": float(projection.distance),
                "signed_route_delta": signed_delta,
                "action": row.get("action"),
                "target_coord": row.get("target_coord"),
            }
        )
        previous_progress = progress
    return projected


def replay_directed_route_following(
    rows: Sequence[dict[str, Any]],
    route_coords: list[int],
    *,
    lookahead_distance: float = 0.85,
    corridor_radius: float = 1.50,
    backward_tolerance: float = 0.25,
    max_forward_advance: float = 24.0,
    pass_tolerance: float = 0.05,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    follower = DirectedRouteFollower(
        route_coords,
        lookahead_distance=lookahead_distance,
        corridor_radius=corridor_radius,
        backward_tolerance=backward_tolerance,
        max_forward_advance=max_forward_advance,
        pass_tolerance=pass_tolerance,
    )
    replay: list[dict[str, Any]] = []
    for row in fresh_coordinate_rows(rows):
        observation = follower.observe(int(row["coord"]))
        replay.append(
            {
                "index": int(row.get("index", len(replay))),
                "coord": int(row["coord"]),
                "action": row.get("action"),
                **observation.to_dict(),
            }
        )

    progress = [float(item["progress"]) for item in replay]
    accepted_deltas = [
        float(item["progress_delta"])
        for item in replay
        if item["projection_accepted"] is True
    ]
    cross_track = [float(item["cross_track_distance"]) for item in replay]
    progress_span = max(progress) - min(progress) if progress else 0.0
    net_progress = progress[-1] - progress[0] if progress else 0.0
    unique_progress_bins = {
        int(math.floor(item["progress"])) % len(route_coords)
        for item in replay
        if item["projection_accepted"] is True
    }
    summary = {
        "observations": len(replay),
        "accepted_projections": sum(
            item["projection_accepted"] is True for item in replay
        ),
        "rejected_projections": sum(
            item["projection_accepted"] is False for item in replay
        ),
        "off_corridor_observations": sum(item["on_corridor"] is False for item in replay),
        "progress_regressions": sum(
            current + 1e-9 < previous
            for previous, current in zip(progress, progress[1:])
        ),
        "net_progress": round(net_progress, 6) if progress else 0.0,
        "route_progress_span": round(progress_span, 6),
        "route_progress_coverage_fraction": round(
            min(1.0, progress_span / float(len(route_coords))),
            6,
        ),
        "route_progress_bins_visited": len(unique_progress_bins),
        "route_progress_bin_fraction": round(
            len(unique_progress_bins) / float(len(route_coords)),
            6,
        ),
        "completed_laps": int(max(0.0, net_progress) // float(len(route_coords))),
        "max_accepted_progress_delta": round(max(accepted_deltas, default=0.0), 6),
        "median_cross_track_distance": (
            round(float(np.median(cross_track)), 6) if cross_track else 0.0
        ),
    }
    return summary, replay


def analyze_rows(
    rows: Sequence[dict[str, Any]], projected: Sequence[dict[str, Any]], route_size: int
) -> dict[str, Any]:
    actions = Counter(str(row.get("action") or "") for row in rows)
    mining_outcome_events: list[dict[str, Any]] = []
    previous_mining_outcome_count = 0
    x_events: list[dict[str, Any]] = []
    stale_after_fresh = 0
    for index, row in enumerate(rows):
        outcomes = row.get("mining_outcomes")
        if isinstance(outcomes, int) and outcomes > previous_mining_outcome_count:
            mining_outcome_events.append(
                {
                    "index": row.get("index"),
                    "count": outcomes,
                    "action": row.get("action"),
                    "coord": row.get("coord"),
                }
            )
            previous_mining_outcome_count = outcomes
        keys: list[str] = []
        for value in ((row.get("combat") or {}).get("keys_tapped"), row.get("combat_keys_tapped")):
            if isinstance(value, list):
                keys.extend(str(key).upper() for key in value)
        if "X" in keys:
            combat = row.get("combat") if isinstance(row.get("combat"), dict) else {}
            x_events.append(
                {
                    "index": row.get("index"),
                    "combat_active": combat.get("active"),
                    "combat_marker_visible": combat.get("combat_marker_visible"),
                    "target_is_attacker": combat.get("target_is_attacker"),
                    "action": row.get("action"),
                }
            )
        if index > 0 and rows[index - 1].get("coord_fresh") is True and row.get("coord_fresh") is False:
            previous = rows[index - 1].get("coord")
            current = row.get("coord")
            if isinstance(previous, int) and isinstance(current, int) and previous != current:
                stale_after_fresh += 1

    deltas = [
        float(item["signed_route_delta"])
        for item in projected
        if isinstance(item.get("signed_route_delta"), (int, float))
    ]
    backward = [delta for delta in deltas if delta < -0.35]
    forward = [delta for delta in deltas if delta > 0.35]
    large_jumps = [
        item
        for item in projected
        if isinstance(item.get("signed_route_delta"), (int, float))
        and abs(float(item["signed_route_delta"])) > max(8.0, route_size * 0.08)
    ]
    mining_action_count = sum(count for action, count in actions.items() if action.startswith("mining"))
    combat_action_count = sum(count for action, count in actions.items() if action.startswith("combat"))
    recovery_action_count = sum(
        count for action, count in actions.items() if action.startswith("recover_")
    )
    mouse_turns: list[dict[str, Any]] = []
    seen_mouse_sequences: set[int] = set()
    for row in rows:
        steering = row.get("mouse_steering")
        if not isinstance(steering, dict):
            continue
        sequence = steering.get("sequence")
        if not isinstance(sequence, int) or sequence in seen_mouse_sequences:
            continue
        seen_mouse_sequences.add(sequence)
        mouse_turns.append(steering)
    mouse_keys = [
        str(item.get("turn_key"))
        for item in mouse_turns
        if item.get("turn_key") in {"A", "D"}
    ]
    elapsed_seconds = (
        float(rows[-1].get("timestamp", 0.0)) - float(rows[0].get("timestamp", 0.0))
        if rows
        else 0.0
    )
    return {
        "rows": len(rows),
        "fresh_coordinate_rows": len(projected),
        "elapsed_seconds": elapsed_seconds,
        "action_counts": dict(actions.most_common()),
        "mining_action_rows": mining_action_count,
        "combat_action_rows": combat_action_count,
        "recovery_action_rows": recovery_action_count,
        "mining_outcome_events": mining_outcome_events,
        "mining_successes": sum(
            1 for item in mining_outcome_events if "mining_verified" in str(item.get("action"))
        ),
        "mining_failures": sum(
            1 for item in mining_outcome_events if "mining_failed" in str(item.get("action"))
        ),
        "x_events": x_events,
        "x_without_confirmed_combat": sum(
            1
            for item in x_events
            if not (
                item.get("combat_active") is True
                and item.get("combat_marker_visible") is True
                and item.get("target_is_attacker") is True
            )
        ),
        "stale_coordinate_regressions_after_fresh_read": stale_after_fresh,
        "forward_projection_steps": len(forward),
        "backward_projection_steps": len(backward),
        "net_signed_route_progress": round(sum(deltas), 6),
        "large_projection_jumps": large_jumps,
        "mouse_steering": {
            "turns": len(mouse_turns),
            "turns_per_minute": (
                round(len(mouse_turns) * 60.0 / elapsed_seconds, 3)
                if elapsed_seconds > 0.0
                else 0.0
            ),
            "direction_reversals": sum(
                current != previous
                for previous, current in zip(mouse_keys, mouse_keys[1:])
            ),
            "contexts": dict(
                Counter(str(item.get("context") or "") for item in mouse_turns)
            ),
            "total_drag_seconds": round(
                sum(float(item.get("duration") or 0.0) for item in mouse_turns),
                3,
            ),
        },
    }


def video_segments(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("window_capture_*.mp4"))


def read_video_frame(path: Path, frame_index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def sample_video_overview(run_dir: Path, interval_seconds: float) -> list[tuple[np.ndarray, str]]:
    samples: list[tuple[np.ndarray, str]] = []
    elapsed = 0.0
    interval = max(1.0, interval_seconds)
    next_sample = 0.0
    for segment_index, path in enumerate(video_segments(run_dir)):
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0.0 or count <= 0:
                continue
            duration = count / fps
        finally:
            capture.release()
        while next_sample < elapsed + duration:
            local_seconds = max(0.0, next_sample - elapsed)
            frame = read_video_frame(path, int(round(local_seconds * fps)))
            if frame is not None:
                samples.append((frame, f"t={next_sample:06.1f}s seg={segment_index:02d}"))
            next_sample += interval
        elapsed += duration
    return samples


def timeline_frame_lookup(
    run_dir: Path, rows: Sequence[dict[str, Any]], indexes: Iterable[int]
) -> list[tuple[np.ndarray, str]]:
    timeline = read_jsonl(run_dir / "window_capture_timeline.jsonl")
    timeline_timestamps = np.asarray([float(item["timestamp"]) for item in timeline], dtype=np.float64)
    by_index = {int(row.get("index", -1)): row for row in rows}
    samples: list[tuple[np.ndarray, str]] = []
    for index in indexes:
        row = by_index.get(index)
        if row is None or not timeline:
            continue
        timestamp = float(row.get("timestamp", 0.0))
        position = int(np.searchsorted(timeline_timestamps, timestamp))
        position = min(max(position, 0), len(timeline) - 1)
        if position > 0 and abs(timeline_timestamps[position - 1] - timestamp) < abs(
            timeline_timestamps[position] - timestamp
        ):
            position -= 1
        item = timeline[position]
        segment_index = int(item["segment_index"])
        segment_frame_index = int(item["segment_frame_index"])
        path = run_dir / f"window_capture_{segment_index:04d}.mp4"
        frame = read_video_frame(path, segment_frame_index)
        if frame is None:
            continue
        samples.append((frame, f"idx={index} {str(row.get('action') or '-')[:48]}"))
    return samples


def annotate_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, CONTACT_TILE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (CONTACT_TILE_SIZE[0], 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_contact_sheets(
    samples: Sequence[tuple[np.ndarray, str]], output_dir: Path, stem: str, per_sheet: int = 12
) -> list[Path]:
    paths: list[Path] = []
    for sheet_index in range(0, len(samples), per_sheet):
        selected = samples[sheet_index : sheet_index + per_sheet]
        tiles = [annotate_tile(frame, label) for frame, label in selected]
        row_count = math.ceil(len(tiles) / CONTACT_SHEET_COLUMNS)
        blank = np.zeros((CONTACT_TILE_SIZE[1], CONTACT_TILE_SIZE[0], 3), dtype=np.uint8)
        while len(tiles) < row_count * CONTACT_SHEET_COLUMNS:
            tiles.append(blank.copy())
        image = np.vstack(
            [
                np.hstack(tiles[offset : offset + CONTACT_SHEET_COLUMNS])
                for offset in range(0, len(tiles), CONTACT_SHEET_COLUMNS)
            ]
        )
        path = output_dir / f"{stem}_{sheet_index // per_sheet:02d}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write contact sheet: {path}")
        paths.append(path)
    return paths


def parse_index_ranges(value: str, *, max_index: int, step: int) -> list[int]:
    indexes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
            start = max(0, int(left))
            end = min(max_index, int(right))
            indexes.extend(range(start, end + 1, max(1, step)))
        else:
            indexes.append(min(max(0, int(token)), max_index))
    return list(dict.fromkeys(indexes))


def render_trajectory_overlay(
    *,
    projected: Sequence[dict[str, Any]],
    route_coords: Sequence[int],
    background_path: Path,
    manifest_path: Path,
    adt_dir: Path,
    output_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    area = manifest["world_map_area"]
    zone = ZoneTransform(
        left=float(area["left"]),
        right=float(area["right"]),
        top=float(area["top"]),
        bottom=float(area["bottom"]),
    )
    raster = build_terrain_raster(sorted(adt_dir.glob("*.adt")))
    heights = np.asarray(raster.heights, dtype=np.float32)
    valid = np.isfinite(heights)
    normalized = np.zeros(heights.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(heights[valid], [2.0, 98.0])
        normalized[valid] = np.clip(
            (heights[valid] - low) / max(1.0, float(high - low)) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    image = cv2.applyColorMap(normalized, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    image[~valid] = (12, 12, 12)

    def pixel(coord: int) -> tuple[int, int]:
        x, y = coord_to_xy(coord)
        row, col = raster.ui_to_grid(x, y, zone)
        return int(round(col)), int(round(row))

    route_points = np.asarray([pixel(coord) for coord in route_coords], dtype=np.int32)
    if len(route_points) > 1:
        cv2.polylines(image, [route_points], True, (225, 225, 225), 1, cv2.LINE_AA)
    trajectory_points = [pixel(int(item["coord"])) for item in projected]
    for index in range(1, len(trajectory_points)):
        ratio = index / max(1, len(trajectory_points) - 1)
        color = (int(round(255 * (1.0 - ratio))), int(round(220 * ratio)), 255)
        cv2.line(image, trajectory_points[index - 1], trajectory_points[index], color, 3, cv2.LINE_AA)
    if trajectory_points:
        cv2.circle(image, trajectory_points[0], 7, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, trajectory_points[-1], 7, (0, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(image, (12, 12), (510, 116), (8, 8, 8), -1)
    legend = [
        ("planned route", (225, 225, 225)),
        ("actual path: blue -> magenta -> yellow", (255, 80, 255)),
        ("start", (255, 255, 0)),
        ("finish", (0, 255, 255)),
    ]
    for index, (label, color) in enumerate(legend):
        y = 34 + index * 22
        cv2.line(image, (28, y), (52, y), color, 3, cv2.LINE_AA)
        cv2.putText(image, label, (62, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write trajectory overlay: {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    route_path = args.route.resolve()
    output_dir = (args.output_dir or (run_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(run_dir / "metadata.jsonl")
    route_coords = load_route_coords(route_path)
    projected = project_trajectory(rows, route_coords)
    summary = analyze_rows(rows, projected, len(route_coords))
    directed_summary, directed_replay = replay_directed_route_following(rows, route_coords)
    summary["directed_route_following"] = directed_summary
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output_dir / "projected_trajectory.jsonl", projected)
    write_jsonl(output_dir / "directed_route_following.jsonl", directed_replay)

    overview_samples = sample_video_overview(run_dir, args.overview_interval)
    overview_paths = write_contact_sheets(overview_samples, output_dir, "video_overview")
    dense_paths: list[Path] = []
    if args.dense_ranges:
        indexes = parse_index_ranges(
            args.dense_ranges,
            max_index=max(0, len(rows) - 1),
            step=max(1, args.dense_step),
        )
        dense_samples = timeline_frame_lookup(run_dir, rows, indexes)
        dense_paths = write_contact_sheets(dense_samples, output_dir, "video_dense")

    trajectory_overlay_path: Path | None = None
    if args.terrain_overlay and args.terrain_manifest and args.terrain_adt_dir:
        trajectory_overlay_path = output_dir / "trajectory_topographic_overlay.png"
        render_trajectory_overlay(
            projected=projected,
            route_coords=route_coords,
            background_path=args.terrain_overlay.resolve(),
            manifest_path=args.terrain_manifest.resolve(),
            adt_dir=args.terrain_adt_dir.resolve(),
            output_path=trajectory_overlay_path,
        )

    output = {
        **summary,
        "overview_contact_sheets": [str(path) for path in overview_paths],
        "dense_contact_sheets": [str(path) for path in dense_paths],
        "trajectory_overlay": str(trajectory_overlay_path) if trajectory_overlay_path else None,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
