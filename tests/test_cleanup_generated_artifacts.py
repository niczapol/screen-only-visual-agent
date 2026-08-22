from pathlib import Path

import pytest

from scripts.cleanup_generated_artifacts import (
    CleanupPlan,
    apply_cleanup_plan,
    build_cleanup_plan,
    is_within,
)


def test_cleanup_plan_selects_only_allowlisted_generated_artifacts(tmp_path: Path):
    live = tmp_path / "data" / "live_example"
    reports = live / "reports"
    reports.mkdir(parents=True)
    (reports / "0001.png").write_bytes(b"report")
    (live / "start_report.png").write_bytes(b"start")
    (live / "frames").mkdir()
    (live / "frames" / "0001.png").write_bytes(b"raw")
    overlays = tmp_path / "data" / "vision_dataset" / "overlays"
    overlays.mkdir(parents=True)
    (overlays / "0001.png").write_bytes(b"overlay")
    (tmp_path / "data" / "vision_dataset" / "images").mkdir()
    raw = tmp_path / "data" / "vision_dataset" / "images" / "raw.png"
    raw.write_bytes(b"training")

    plan = build_cleanup_plan(tmp_path)
    apply_cleanup_plan(plan, tmp_path)

    assert not reports.exists()
    assert not (live / "start_report.png").exists()
    assert not overlays.exists()
    assert (live / "frames" / "0001.png").read_bytes() == b"raw"
    assert raw.read_bytes() == b"training"


def test_cleanup_path_guard_rejects_parent(tmp_path: Path):
    assert is_within(tmp_path / "data", tmp_path)
    assert not is_within(tmp_path.parent, tmp_path)
    with pytest.raises(ValueError):
        apply_cleanup_plan(
            CleanupPlan((tmp_path.parent,), (), 0, 0),
            tmp_path,
        )


def test_reviewed_live_cleanup_removes_video_but_preserves_raw_frames(tmp_path: Path):
    live = tmp_path / "data" / "live_example"
    frames = live / "frames"
    frames.mkdir(parents=True)
    raw = frames / "0001.png"
    raw.write_bytes(b"raw")
    video = live / "window_capture_0000.mp4"
    video.write_bytes(b"video")

    plan = build_cleanup_plan(tmp_path, include_reviewed_live_media=True)
    apply_cleanup_plan(plan, tmp_path)

    assert raw.read_bytes() == b"raw"
    assert not video.exists()
