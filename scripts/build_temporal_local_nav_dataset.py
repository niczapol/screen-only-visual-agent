"""Build strict temporal traversability labels from saved route-live telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from vision_bot.coords import coord_to_xy


RECOVERY_PREFIXES = ("recover_",)
EXCLUDED_ACTION_PREFIXES = (
    "combat_",
    "death_",
    "mining_",
    "route_mount_",
    "mount_",
)
DEFAULT_MANUAL_EXCLUSIONS = {
    "live_tanaris_v3_full_20260803_001634": ((685, 1339),),
    "live_route_desolace_sartheris_run1_20260731_225722": ((232, 234),),
    "live_route_probe_20260729_ratchet_route_video_v1": ((44, 44),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data")
    parser.add_argument(
        "--output-dir",
        default="data/local_navigation_temporal_v0_3_20260803",
    )
    parser.add_argument("--max-passable-per-run", type=int, default=24)
    parser.add_argument("--contact-sheet-size", type=int, default=12)
    parser.add_argument(
        "--review-decisions",
        help="Optional JSON file containing reviewed sample rejections.",
    )
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discover_route_metadata(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(source_root.glob("live*/metadata.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = next((line for line in handle if line.strip()), "")
            row = json.loads(first) if first else {}
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row.get("movement"), dict) and isinstance(row.get("navigation"), dict):
            paths.append(path)
    return paths


def build_dataset(
    metadata_paths: Iterable[Path],
    output_dir: Path,
    *,
    max_passable_per_run: int = 24,
    make_contact_sheets: bool = True,
    contact_sheet_size: int = 12,
    review_decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        rows = read_jsonl(metadata_path)
        run_id = metadata_path.parent.name
        run_samples = extract_run_samples(
            rows,
            metadata_path,
            max_passable=max_passable_per_run,
            manual_exclusions=DEFAULT_MANUAL_EXCLUSIONS.get(run_id, ()),
        )
        samples.extend(run_samples)
        run_summaries.append(
            {
                "run_id": run_id,
                "rows": len(rows),
                "samples": len(run_samples),
                "labels": dict(Counter(sample["label"] for sample in run_samples)),
                "zone": infer_zone(run_id, rows),
            }
        )

    review = apply_review_decisions(samples, review_decisions)
    samples = review["samples"]

    split_by_run = assign_run_splits(samples)
    for sample in samples:
        sample["split"] = split_by_run.get(sample["run_id"], "train")

    metadata_output = output_dir / "metadata.jsonl"
    with metadata_output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

    label_counts = Counter(sample["label"] for sample in samples)
    split_counts = Counter(sample["split"] for sample in samples)
    split_label_counts = {
        split: dict(Counter(sample["label"] for sample in samples if sample["split"] == split))
        for split in ("train", "val", "test")
    }
    blocked_runs = {
        split: len({sample["run_id"] for sample in samples if sample["split"] == split and sample["label"] == "blocked"})
        for split in ("train", "val", "test")
    }
    blocked_zones = sorted({sample["zone"] for sample in samples if sample["label"] == "blocked"})
    gate = {
        "min_blocked_samples": 80,
        "min_blocked_samples_per_val_test": 15,
        "min_blocked_runs_per_val_test": 2,
        "min_blocked_zones": 3,
    }
    gate_passed = (
        label_counts.get("blocked", 0) >= gate["min_blocked_samples"]
        and all(split_label_counts[split].get("blocked", 0) >= gate["min_blocked_samples_per_val_test"] for split in ("val", "test"))
        and all(blocked_runs[split] >= gate["min_blocked_runs_per_val_test"] for split in ("val", "test"))
        and len(blocked_zones) >= gate["min_blocked_zones"]
    )
    summary = {
        "metadata": str(metadata_output.resolve()),
        "source_metadata_count": len(run_summaries),
        "sample_count": len(samples),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "split_label_counts": split_label_counts,
        "blocked_run_counts": blocked_runs,
        "blocked_zones": blocked_zones,
        "training_gate": gate,
        "training_gate_passed": gate_passed,
        "review": {
            "configured_rejections": review["configured_rejections"],
            "applied_rejections": review["applied_rejections"],
            "unmatched_rejections": review["unmatched_rejections"],
            "rejected_label_counts": review["rejected_label_counts"],
            "rejected_reason_counts": review["rejected_reason_counts"],
        },
        "runs": run_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if make_contact_sheets:
        for label in ("blocked", "passable"):
            label_samples = [sample for sample in samples if sample["label"] == label]
            review_limit = len(label_samples) if label == "blocked" else max(1, contact_sheet_size) * 6
            if len(label_samples) > review_limit:
                review_indices = np.linspace(0, len(label_samples) - 1, review_limit, dtype=int)
                label_samples = [label_samples[int(index)] for index in review_indices]
            for page, start in enumerate(range(0, len(label_samples), max(1, contact_sheet_size))):
                write_contact_sheet(
                    label_samples[start : start + contact_sheet_size],
                    output_dir / f"{label}_contact_{page:02d}.png",
                )
    return summary


def load_review_decisions(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review decisions must be a JSON object")
    return payload


def apply_review_decisions(
    samples: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
) -> dict[str, Any]:
    entries = decisions.get("reject", []) if isinstance(decisions, dict) else []
    rejected_keys: dict[tuple[str, int, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        run_id = str(entry.get("run_id") or "")
        row_index = integer(entry.get("row_index"))
        label = str(entry.get("label") or "")
        if not run_id or row_index is None or label not in {"blocked", "passable"}:
            continue
        rejected_keys[(run_id, row_index, label)] = str(entry.get("reason") or "manual_review")

    retained: list[dict[str, Any]] = []
    applied: set[tuple[str, int, str]] = set()
    rejected_labels: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    for sample in samples:
        key = (str(sample["run_id"]), int(sample["row_index"]), str(sample["label"]))
        reason = rejected_keys.get(key)
        if reason is None:
            retained.append(sample)
            continue
        applied.add(key)
        rejected_labels[sample["label"]] += 1
        rejected_reasons[reason] += 1

    return {
        "samples": retained,
        "configured_rejections": len(rejected_keys),
        "applied_rejections": len(applied),
        "unmatched_rejections": len(set(rejected_keys) - applied),
        "rejected_label_counts": dict(rejected_labels),
        "rejected_reason_counts": dict(rejected_reasons),
    }


def extract_run_samples(
    rows: list[dict[str, Any]],
    metadata_path: Path,
    *,
    max_passable: int,
    manual_exclusions: tuple[tuple[int, int], ...] = (),
) -> list[dict[str, Any]]:
    run_id = metadata_path.parent.name
    zone = infer_zone(run_id, rows)
    blocked: list[dict[str, Any]] = []
    last_blocked_timestamp = -math.inf
    last_blocked_coord: int | None = None

    for position, row in enumerate(rows):
        if not is_recovery_row(row) or is_contaminated(row, manual_exclusions):
            continue
        timestamp = number(row.get("timestamp"), default=float(position))
        coord = integer(row.get("coord"))
        if (
            timestamp - last_blocked_timestamp < 20.0
            and coord is not None
            and last_blocked_coord is not None
            and coord_distance(coord, last_blocked_coord) < 0.35
        ):
            continue

        prior = clean_window(rows, position, lookback=10, manual_exclusions=manual_exclusions)
        target_coord = row.get("target_coord")
        if target_coord is not None:
            prior = [candidate for candidate in prior if candidate.get("target_coord") == target_coord]
        no_progress = [candidate for candidate in prior if is_no_progress(candidate)]
        no_progress_count = len(no_progress) + int(is_no_progress(row))
        if no_progress_count < 2:
            continue
        attempted = attempted_sector(row)
        sector = navigation_sector(row, attempted)
        if sector is not None:
            obstacle_score = number(sector.get("obstacle_score"))
            passable_score = number(sector.get("passable_score"))
            if obstacle_score < 0.07 and passable_score > 0.82:
                continue
        frames = temporal_frames(rows, position, metadata_path.parent, manual_exclusions=manual_exclusions)
        if not frames:
            continue
        recovery_outcome, outcome_progress = recovery_outcome_after(
            rows,
            position,
            manual_exclusions=manual_exclusions,
        )
        observation = row
        blocked.append(
            make_sample(
                run_id=run_id,
                zone=zone,
                label="blocked",
                row=observation,
                frames=frames,
                reason="temporal_no_progress_before_recovery",
                confidence=0.94 if len(no_progress) >= 2 else 0.86,
                evidence={
                    "no_progress_rows": no_progress_count,
                    "recovery_outcome": recovery_outcome,
                    "outcome_progress": round(outcome_progress, 6),
                },
            )
        )
        last_blocked_timestamp = timestamp
        last_blocked_coord = coord

    passable_candidates: list[dict[str, Any]] = []
    last_passable_timestamp = -math.inf
    last_passable_coord: int | None = None
    for position, row in enumerate(rows):
        if is_contaminated(row, manual_exclusions) or not is_passable_progress(row):
            continue
        timestamp = number(row.get("timestamp"), default=float(position))
        coord = integer(row.get("coord"))
        if timestamp - last_passable_timestamp < 8.0:
            continue
        if coord is not None and last_passable_coord is not None and coord_distance(coord, last_passable_coord) < 0.45:
            continue
        frames = temporal_frames(rows, position, metadata_path.parent, manual_exclusions=manual_exclusions)
        if len(frames) < 2:
            continue
        passable_candidates.append(
            make_sample(
                run_id=run_id,
                zone=zone,
                label="passable",
                row=row,
                frames=frames,
                reason="sustained_coordinate_and_target_progress",
                confidence=0.95,
                evidence={
                    "coord_delta": number(row.get("coord_delta")),
                    "distance_progress": number(row.get("distance_progress")),
                },
            )
        )
        last_passable_timestamp = timestamp
        last_passable_coord = coord

    if len(passable_candidates) > max_passable:
        indices = np.linspace(0, len(passable_candidates) - 1, max_passable, dtype=int)
        passable_candidates = [passable_candidates[int(index)] for index in indices]
    return blocked + passable_candidates


def make_sample(
    *,
    run_id: str,
    zone: str,
    label: str,
    row: dict[str, Any],
    frames: list[Path],
    reason: str,
    confidence: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    sector = attempted_sector(row)
    source_frame = frames[-1]
    token = f"{run_id}:{row.get('index')}:{label}:{source_frame}".encode("utf-8")
    return {
        "sample_id": hashlib.sha1(token).hexdigest()[:16],
        "run_id": run_id,
        "episode_id": f"{run_id}:local_nav",
        "zone": zone,
        "row_index": row.get("index"),
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "attempted_sector": sector,
        "source_frame": str(source_frame.resolve()),
        "temporal_frames": [str(path.resolve()) for path in frames],
        "action": row.get("action"),
        "coord": row.get("coord"),
        "navigation": row.get("navigation"),
        "movement": row.get("movement"),
        "evidence": evidence,
    }


def attempted_sector(row: dict[str, Any]) -> str:
    movement = row.get("movement") if isinstance(row.get("movement"), dict) else {}
    key = str(row.get("local_turn_key") or movement.get("turn_key") or "").upper()
    if key == "A":
        return "slight_left"
    if key == "D":
        return "slight_right"
    return "center"


def navigation_sector(row: dict[str, Any], name: str) -> dict[str, Any] | None:
    navigation = row.get("navigation") if isinstance(row.get("navigation"), dict) else {}
    for sector in navigation.get("sectors") or []:
        if sector.get("name") == name:
            return sector
    return None


def is_recovery_row(row: dict[str, Any]) -> bool:
    action = str(row.get("action") or "")
    movement = row.get("movement") if isinstance(row.get("movement"), dict) else {}
    movement_action = str(movement.get("action") or "")
    return action.startswith(RECOVERY_PREFIXES) or movement_action.startswith(RECOVERY_PREFIXES)


def is_no_progress(row: dict[str, Any]) -> bool:
    if not bool(row.get("coord_fresh")):
        return False
    coord_delta = number(row.get("coord_delta"), default=math.inf)
    distance_progress = number(row.get("distance_progress"), default=math.inf)
    return abs(coord_delta) <= 0.018 and distance_progress <= 0.018


def is_passable_progress(row: dict[str, Any]) -> bool:
    if not bool(row.get("coord_fresh")):
        return False
    movement = row.get("movement") if isinstance(row.get("movement"), dict) else {}
    if str(movement.get("held_key") or "").upper() != "W":
        return False
    coord_delta = number(row.get("coord_delta"), default=-math.inf)
    distance_progress = number(row.get("distance_progress"), default=-math.inf)
    return 0.035 <= coord_delta <= 2.0 and distance_progress >= 0.02


def is_contaminated(
    row: dict[str, Any],
    manual_exclusions: tuple[tuple[int, int], ...] = (),
) -> bool:
    index = integer(row.get("index"))
    if index is not None and any(start <= index <= end for start, end in manual_exclusions):
        return True
    if row.get("foreground_ok") is False or row.get("death_recovery") is not None or row.get("threat") is not None:
        return True
    action = str(row.get("action") or "")
    if action.startswith(EXCLUDED_ACTION_PREFIXES):
        return True
    game_state = row.get("game_state") if isinstance(row.get("game_state"), dict) else {}
    if game_state.get("death_or_blocking_modal") or game_state.get("dismissable_modal_click"):
        return True
    combat = row.get("combat") if isinstance(row.get("combat"), dict) else {}
    if any(
        combat.get(key)
        for key in (
            "active",
            "combat_marker_visible",
            "target_is_attacker",
            "hostile_target_hint",
            "nameplate_visible",
            "outgoing_damage_visible",
            "facing_error_visible",
        )
    ):
        return True
    mining = row.get("mining") if isinstance(row.get("mining"), dict) else {}
    if str(mining.get("phase") or "idle") != "idle" or mining.get("candidate") is not None:
        return True
    return False


def clean_window(
    rows: list[dict[str, Any]],
    position: int,
    *,
    lookback: int,
    manual_exclusions: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    start = max(0, position - lookback)
    return [row for row in rows[start:position] if not is_contaminated(row, manual_exclusions)]


def temporal_frames(
    rows: list[dict[str, Any]],
    position: int,
    base_dir: Path,
    *,
    manual_exclusions: tuple[tuple[int, int], ...],
    count: int = 3,
) -> list[Path]:
    paths: list[Path] = []
    for row in reversed(rows[max(0, position - 12) : position + 1]):
        if is_contaminated(row, manual_exclusions):
            continue
        raw = row.get("frame")
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = base_dir / path
        if path.exists() and path not in paths:
            paths.append(path)
        if len(paths) >= count:
            break
    return list(reversed(paths))


def recovery_outcome_after(
    rows: list[dict[str, Any]],
    position: int,
    *,
    manual_exclusions: tuple[tuple[int, int], ...],
) -> tuple[str, float]:
    progress = 0.0
    fresh_rows = 0
    for row in rows[position + 1 : position + 31]:
        if is_contaminated(row, manual_exclusions):
            break
        if bool(row.get("coord_fresh")):
            fresh_rows += 1
            progress += max(0.0, number(row.get("distance_progress"), default=0.0))
        if progress >= 0.08:
            return "recovered", progress
    return ("failed" if fresh_rows >= 2 else "unknown"), progress


def assign_run_splits(samples: list[dict[str, Any]]) -> dict[str, str]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_run[sample["run_id"]].append(sample)
    runs = sorted(
        by_run,
        key=lambda run: (
            -sum(sample["label"] == "blocked" for sample in by_run[run]),
            hashlib.sha1(run.encode("utf-8")).hexdigest(),
        ),
    )
    assignment: dict[str, str] = {}
    blocked_totals = {"train": 0, "val": 0, "test": 0}
    run_totals = {"train": 0, "val": 0, "test": 0}
    for run in runs:
        blocked = sum(sample["label"] == "blocked" for sample in by_run[run])
        split = min(
            ("test", "val", "train"),
            key=lambda candidate: (
                blocked_totals[candidate] / {"train": 0.60, "val": 0.20, "test": 0.20}[candidate],
                run_totals[candidate] / {"train": 0.60, "val": 0.20, "test": 0.20}[candidate],
            ),
        )
        assignment[run] = split
        blocked_totals[split] += blocked
        run_totals[split] += len(by_run[run])
    return assignment


def infer_zone(run_id: str, rows: list[dict[str, Any]]) -> str:
    lowered = run_id.lower()
    for zone in ("tanaris", "desolace", "barrens", "durotar", "ratchet", "crossroads"):
        if zone in lowered:
            return zone
    zone_ids = Counter(
        integer(row.get("target_zone_id"))
        for row in rows
        if integer(row.get("target_zone_id")) is not None
    )
    return f"zone_{zone_ids.most_common(1)[0][0]}" if zone_ids else "unknown"


def coord_distance(first: int, second: int) -> float:
    first_x, first_y = coord_to_xy(first)
    second_x, second_y = coord_to_xy(second)
    return math.hypot(first_x - second_x, first_y - second_y)


def write_contact_sheet(samples: list[dict[str, Any]], output_path: Path) -> None:
    if not samples:
        return
    tiles: list[np.ndarray] = []
    for sample in samples:
        frames: list[np.ndarray] = []
        source_paths = sample.get("temporal_frames") or [sample.get("source_frame")]
        for raw in source_paths[-3:]:
            image = cv2.imread(str(raw), cv2.IMREAD_COLOR)
            if image is None:
                continue
            frames.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        if not frames:
            continue
        while len(frames) < 3:
            frames.insert(0, frames[0].copy())
        tile = np.hstack(frames[-3:])
        color = (30, 30, 220) if sample["label"] == "blocked" else (30, 190, 30)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (0, 0, 0), -1)
        title = (
            f"{sample['run_id']} row={sample['row_index']} {sample['label']} "
            f"{sample['attempted_sector']}"
        )
        cv2.putText(tile, title[:115], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = 2
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank.copy())
    grid = np.vstack([np.hstack(tiles[index : index + cols]) for index in range(0, len(tiles), cols)])
    cv2.imwrite(str(output_path), grid)


def number(value: Any, *, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def integer(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary = build_dataset(
        discover_route_metadata(source_root),
        output_dir,
        max_passable_per_run=max(0, args.max_passable_per_run),
        make_contact_sheets=not args.no_contact_sheets,
        contact_sheet_size=max(1, args.contact_sheet_size),
        review_decisions=load_review_decisions(
            Path(args.review_decisions).resolve() if args.review_decisions else None
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
