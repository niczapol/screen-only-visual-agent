import json

import cv2
import numpy as np

from vision_bot.local_navigation import (
    SECTOR_NAMES,
    analyze_local_navigation,
    collect_local_navigation_sources,
    collect_route_probe_metadata_sources,
    draw_local_navigation_report,
    prepare_local_navigation_dataset,
    prepare_local_navigation_outcome_dataset,
    prepare_local_navigation_review_pack,
)


def _config() -> dict:
    return {
        "screen": {"reference_width": 500, "reference_height": 300, "scale_regions": False},
        "local_navigation": {
            "region": {"x": 50, "y": 30, "width": 400, "height": 220},
            "analysis_top_fraction": 0.25,
            "analysis_bottom_fraction": 0.90,
            "edge_weight": 1.6,
            "vertical_weight": 2.8,
            "bottom_edge_weight": 1.8,
            "texture_weight": 0.05,
            "self_ignore": {
                "enabled": True,
                "center_x_fraction": 0.50,
                "bottom_fraction": 0.98,
                "width_fraction": 0.20,
                "height_fraction": 0.25,
            },
        },
    }


def test_analyze_local_navigation_builds_stable_sectors():
    frame = np.full((300, 500, 3), 110, dtype=np.uint8)

    navigation = analyze_local_navigation(frame, _config())

    assert [sector.name for sector in navigation.sectors] == list(SECTOR_NAMES)
    assert navigation.region.x == 50
    assert navigation.best_sector == "center"
    assert navigation.center_blocked is False
    assert navigation.recommended_turn is None


def test_analyze_local_navigation_marks_center_obstacle():
    frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    rng = np.random.default_rng(42)
    frame[95:205, 210:290] = rng.integers(10, 245, size=(110, 80, 3), dtype=np.uint8)
    cv2.rectangle(frame, (210, 95), (290, 205), (15, 15, 15), 3)
    cv2.line(frame, (250, 95), (250, 205), (245, 245, 245), 3)

    navigation = analyze_local_navigation(frame, _config())
    center = navigation.center_sector()

    assert center is not None
    assert center.obstacle_score > 0.52
    assert navigation.center_blocked is True
    assert navigation.best_sector in {"left", "slight_left", "slight_right", "right"}
    assert navigation.recommended_turn in {"A", "D"}


def test_self_ignore_region_does_not_create_center_blocker():
    frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    # Simulate the character/mount in the lower center. It should be ignored.
    frame[195:250, 215:285] = 20
    cv2.rectangle(frame, (215, 195), (285, 250), (245, 245, 245), 2)

    navigation = analyze_local_navigation(frame, _config())
    center = navigation.center_sector()

    assert center is not None
    assert navigation.self_ignore_region is not None
    assert center.obstacle_score < 0.52
    assert navigation.center_blocked is False


def test_draw_local_navigation_report_adds_panel():
    frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    navigation = analyze_local_navigation(frame, _config())

    report = draw_local_navigation_report(frame, navigation)

    assert report.shape[0] == frame.shape[0]
    assert report.shape[1] > frame.shape[1]
    assert np.count_nonzero(report) > np.count_nonzero(frame)


def test_prepare_local_navigation_dataset_writes_metadata(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    image_path = source_dir / "frame.png"
    cv2.imwrite(str(image_path), frame)

    output_dir = tmp_path / "local_nav"
    summary = prepare_local_navigation_dataset([image_path], output_dir, _config())

    assert summary.frame_count == 1
    assert summary.report_count == 1
    assert summary.copied_image_count == 1
    assert (output_dir / "metadata.jsonl").exists()
    assert list((output_dir / "images" / "raw").glob("*.png"))
    assert list((output_dir / "reports").glob("*.png"))


def test_prepare_local_navigation_review_pack_writes_edge_case_reports(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    clear_frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    blocked_frame = np.full((300, 500, 3), 110, dtype=np.uint8)
    rng = np.random.default_rng(7)
    blocked_frame[95:205, 210:290] = rng.integers(10, 245, size=(110, 80, 3), dtype=np.uint8)
    cv2.rectangle(blocked_frame, (210, 95), (290, 205), (15, 15, 15), 3)
    clear_path = source_dir / "clear.png"
    blocked_path = source_dir / "blocked.png"
    cv2.imwrite(str(clear_path), clear_frame)
    cv2.imwrite(str(blocked_path), blocked_frame)

    dataset_dir = tmp_path / "dataset"
    prepare_local_navigation_dataset([clear_path, blocked_path], dataset_dir, _config(), copy_images=False)
    review_dir = tmp_path / "review"
    summary = prepare_local_navigation_review_pack(dataset_dir / "metadata.jsonl", review_dir, _config())

    assert summary.report_count >= 1
    assert summary.item_count == summary.report_count
    assert (review_dir / "metadata.jsonl").exists()
    assert list((review_dir / "reports").glob("*.png"))


def test_collect_local_navigation_sources_deduplicates_images(tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))

    paths = collect_local_navigation_sources([tmp_path, tmp_path])

    assert paths == [image_path]


def test_prepare_local_navigation_outcome_dataset_labels_attempted_sectors(tmp_path):
    episode_dir = tmp_path / "episode_a"
    _write_route_probe_episode(
        episode_dir,
        [
            _route_row(0, coord_delta=0.05, distance_progress=0.04),
            _route_row(1, action="recover_jump", movement={"action": "recover_jump", "held_key": None}),
            _route_row(2, coord_delta=None, distance_progress=None, visual_motion_delta=15.0),
            _route_row(3, local_turn_key="A", coord_delta=0.04, distance_progress=0.04),
            _route_row(4, movement={"action": "course_correct_and_forward", "held_key": "W", "turn_key": "D"}, coord_delta=0.04, distance_progress=0.04),
        ],
    )

    output_dir = tmp_path / "outcome"
    summary = prepare_local_navigation_outcome_dataset(
        [episode_dir / "metadata.jsonl"],
        output_dir,
        _config(),
        copy_images=False,
        save_reports=False,
        min_coord_delta=0.03,
        min_distance_progress=0.03,
        min_visual_motion_delta=3.0,
        min_visual_stuck_samples=2,
    )
    rows = _read_jsonl(output_dir / "metadata.jsonl")

    assert summary.sample_count == 5
    assert summary.label_counts == {"passable": 3, "blocked": 1, "unknown": 1}
    assert [row["attempted_sector"] for row in rows] == [
        "center",
        "center",
        "center",
        "slight_left",
        "slight_right",
    ]
    assert [row["label"] for row in rows] == ["passable", "blocked", "unknown", "passable", "passable"]


def test_prepare_local_navigation_outcome_dataset_skips_combat_death_and_threat_rows(tmp_path):
    episode_dir = tmp_path / "episode_b"
    _write_route_probe_episode(
        episode_dir,
        [
            _route_row(0, combat={"active": True}, coord_delta=0.10, distance_progress=0.10),
            _route_row(1, death_recovery={"action": "death_recovery_search_spirit_healer"}, coord_delta=0.10),
            _route_row(2, threat={"reason": "aggressive_nameplate"}, coord_delta=0.10),
            _route_row(3, game_state={"death_or_blocking_modal": True}, coord_delta=0.10),
        ],
    )

    output_dir = tmp_path / "outcome"
    summary = prepare_local_navigation_outcome_dataset(
        [episode_dir / "metadata.jsonl"],
        output_dir,
        _config(),
        copy_images=False,
        save_reports=False,
    )

    assert summary.sample_count == 0
    assert summary.skipped_count == 4
    assert (output_dir / "episode_splits.json").exists()


def test_prepare_local_navigation_outcome_dataset_keeps_episode_splits_exclusive(tmp_path):
    metadata_paths = []
    for index in range(5):
        episode_dir = tmp_path / f"episode_{index}"
        _write_route_probe_episode(
            episode_dir,
            [_route_row(0, coord_delta=0.05 + index * 0.01, distance_progress=0.04)],
        )
        metadata_paths.append(episode_dir / "metadata.jsonl")

    output_dir = tmp_path / "outcome"
    summary = prepare_local_navigation_outcome_dataset(
        metadata_paths,
        output_dir,
        _config(),
        copy_images=False,
        save_reports=False,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    splits = json.loads((output_dir / "episode_splits.json").read_text(encoding="utf-8"))["splits"]
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert summary.episode_count == 5
    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_collect_route_probe_metadata_sources_scans_directories(tmp_path):
    first = tmp_path / "live_route_probe_a"
    second = tmp_path / "live_route_probe_b" / "nested"
    _write_route_probe_episode(first, [_route_row(0)])
    _write_route_probe_episode(second, [_route_row(0)])

    paths = collect_route_probe_metadata_sources([tmp_path])

    assert paths == [first / "metadata.jsonl", second / "metadata.jsonl"]


def _route_row(
    index: int,
    *,
    action: str = "vector_forward",
    local_turn_key: str | None = None,
    movement: dict | None = None,
    coord_delta: float | None = 0.04,
    distance_progress: float | None = 0.04,
    visual_motion_delta: float | None = 8.0,
    visual_stuck_samples: int = 0,
    combat: dict | None = None,
    death_recovery: dict | None = None,
    threat: dict | None = None,
    game_state: dict | None = None,
) -> dict:
    return {
        "index": index,
        "frame": f"frames/{index:04d}.png",
        "action": action,
        "local_turn_key": local_turn_key,
        "coord": 5000500000 + index,
        "coord_fresh": coord_delta is not None or distance_progress is not None,
        "coord_delta": coord_delta,
        "distance_progress": distance_progress,
        "visual_motion_delta": visual_motion_delta,
        "visual_stuck_samples": visual_stuck_samples,
        "combat": combat or {"active": False},
        "death_recovery": death_recovery,
        "threat": threat,
        "game_state": game_state or {"death_or_blocking_modal": False},
        "movement": movement
        if movement is not None
        else {
            "action": "forward_motion",
            "held_key": "W",
            "turn_key": None,
        },
        "navigation": {},
    }


def _write_route_probe_episode(path, rows):
    frames_dir = path / "frames"
    frames_dir.mkdir(parents=True)
    for row in rows:
        frame = np.full((300, 500, 3), 110 + int(row["index"]) % 20, dtype=np.uint8)
        cv2.imwrite(str(path / row["frame"]), frame)
    with (path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
