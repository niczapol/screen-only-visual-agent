import json
from pathlib import Path

import cv2
import numpy as np

from scripts.build_temporal_local_nav_dataset import (
    apply_review_decisions,
    assign_run_splits,
    build_dataset,
    extract_run_samples,
    is_contaminated,
)


def test_extracts_temporal_blocked_and_passable_samples(tmp_path: Path):
    run = tmp_path / "live_tanaris_fixture"
    frames = run / "frames"
    frames.mkdir(parents=True)
    rows = []
    for index in range(8):
        frame_path = frames / f"{index:04d}.png"
        cv2.imwrite(str(frame_path), np.full((120, 200, 3), 40 + index, dtype=np.uint8))
        rows.append(_row(index, frame_path.name, coord_delta=0.08, distance_progress=0.06))
    rows[3] = _row(3, "0003.png", coord_delta=0.0, distance_progress=0.0)
    rows[4] = _row(4, "0004.png", coord_delta=0.0, distance_progress=0.0, action="recover_jump")
    metadata = run / "metadata.jsonl"
    _write_rows(metadata, rows)

    samples = extract_run_samples(rows, metadata, max_passable=4)

    assert any(sample["label"] == "blocked" for sample in samples)
    blocked = next(sample for sample in samples if sample["label"] == "blocked")
    assert len(blocked["temporal_frames"]) == 3
    assert blocked["evidence"]["no_progress_rows"] >= 2
    assert any(sample["label"] == "passable" for sample in samples)


def test_combat_mining_and_manual_ranges_are_excluded():
    combat = _row(1, "0001.png", coord_delta=0.0, distance_progress=0.0)
    combat["combat"] = {"active": True}
    mining = _row(2, "0002.png", coord_delta=0.0, distance_progress=0.0)
    mining["mining"] = {"phase": "intercept", "candidate": 1}
    manual = _row(5, "0005.png", coord_delta=0.0, distance_progress=0.0)

    assert is_contaminated(combat)
    assert is_contaminated(mining)
    assert is_contaminated(manual, ((4, 6),))


def test_build_dataset_splits_by_whole_run(tmp_path: Path):
    paths = []
    for run_index in range(6):
        run = tmp_path / f"live_zone_{run_index}"
        frames = run / "frames"
        frames.mkdir(parents=True)
        rows = []
        for index in range(5):
            frame_path = frames / f"{index:04d}.png"
            cv2.imwrite(str(frame_path), np.full((120, 200, 3), 50 + run_index, dtype=np.uint8))
            rows.append(_row(index, frame_path.name, coord_delta=0.0, distance_progress=0.0))
        rows[4]["action"] = "recover_jump"
        metadata = run / "metadata.jsonl"
        _write_rows(metadata, rows)
        paths.append(metadata)

    output = tmp_path / "dataset"
    summary = build_dataset(paths, output, max_passable_per_run=0, make_contact_sheets=False)
    samples = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]

    assert summary["label_counts"] == {"blocked": 6}
    assert all(len({sample["split"] for sample in samples if sample["run_id"] == run}) == 1 for run in {sample["run_id"] for sample in samples})


def test_run_split_assignment_is_deterministic():
    samples = [
        {"run_id": f"run_{index}", "label": "blocked" if index < 4 else "passable"}
        for index in range(8)
    ]
    assert assign_run_splits(samples) == assign_run_splits(list(reversed(samples)))


def test_review_decisions_reject_exact_run_row_and_label():
    samples = [
        {"run_id": "run_a", "row_index": 7, "label": "blocked"},
        {"run_id": "run_a", "row_index": 7, "label": "passable"},
        {"run_id": "run_b", "row_index": 7, "label": "blocked"},
    ]
    result = apply_review_decisions(
        samples,
        {
            "reject": [
                {
                    "run_id": "run_a",
                    "row_index": 7,
                    "label": "blocked",
                    "reason": "actor_occlusion",
                }
            ]
        },
    )

    assert result["samples"] == samples[1:]
    assert result["applied_rejections"] == 1
    assert result["unmatched_rejections"] == 0
    assert result["rejected_label_counts"] == {"blocked": 1}
    assert result["rejected_reason_counts"] == {"actor_occlusion": 1}


def _row(index: int, frame_name: str, *, coord_delta: float, distance_progress: float, action: str = "continue_forward"):
    return {
        "index": index,
        "timestamp": float(index * 10),
        "foreground_ok": True,
        "frame": f"frames/{frame_name}",
        "coord": 5000500000 + index * 100000,
        "coord_fresh": True,
        "coord_delta": coord_delta,
        "distance_progress": distance_progress,
        "action": action,
        "combat": {},
        "game_state": {},
        "mining": {"phase": "idle", "candidate": None},
        "movement": {"action": action, "held_key": "W", "turn_key": None},
        "navigation": {
            "sectors": [
                {"name": "center", "bbox": {"x": 70, "y": 30, "width": 60, "height": 80}},
                {"name": "slight_left", "bbox": {"x": 30, "y": 30, "width": 40, "height": 80}},
                {"name": "slight_right", "bbox": {"x": 130, "y": 30, "width": 40, "height": 80}},
            ]
        },
    }


def _write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
