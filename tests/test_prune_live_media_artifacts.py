from pathlib import Path

from scripts.prune_live_media_artifacts import apply_plan, build_plan


def test_prune_live_media_deletes_old_media_but_keeps_telemetry(tmp_path: Path):
    old_live = tmp_path / "data" / "live_old_run"
    frames = old_live / "frames"
    frames.mkdir(parents=True)
    old_png = frames / "0001.png"
    old_mp4 = old_live / "window_capture_0000.mp4"
    telemetry = old_live / "metadata.jsonl"
    old_png.write_bytes(b"png")
    old_mp4.write_bytes(b"mp4")
    telemetry.write_text("{}", encoding="utf-8")

    plan = build_plan(tmp_path, keep_live_names=set())
    apply_plan(plan, tmp_path)

    assert not old_png.exists()
    assert not old_mp4.exists()
    assert telemetry.read_text(encoding="utf-8") == "{}"


def test_prune_live_media_preserves_retained_live_and_datasets(tmp_path: Path):
    retained = tmp_path / "data" / "live_v080_direct_node_mining_20260808_152018"
    retained.mkdir(parents=True)
    retained_mp4 = retained / "window_capture_0000.mp4"
    retained_mp4.write_bytes(b"keep")

    dataset = tmp_path / "data" / "vision_dataset" / "images"
    dataset.mkdir(parents=True)
    dataset_png = dataset / "train.png"
    dataset_png.write_bytes(b"training")

    plan = build_plan(
        tmp_path,
        keep_live_names={"live_v080_direct_node_mining_20260808_152018"},
    )
    apply_plan(plan, tmp_path)

    assert retained_mp4.read_bytes() == b"keep"
    assert dataset_png.read_bytes() == b"training"


def test_prune_live_media_includes_debug_live_media(tmp_path: Path):
    debug_live = tmp_path / "debug_output" / "live_smoke"
    debug_live.mkdir(parents=True)
    debug_video = debug_live / "capture.avi"
    debug_note = debug_live / "notes.md"
    debug_video.write_bytes(b"avi")
    debug_note.write_text("keep", encoding="utf-8")

    plan = build_plan(tmp_path, keep_live_names=set())
    apply_plan(plan, tmp_path)

    assert not debug_video.exists()
    assert debug_note.read_text(encoding="utf-8") == "keep"
