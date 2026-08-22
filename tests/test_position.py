import cv2
import numpy as np
import pytest
from pathlib import Path
import sys
import types

from vision_bot.config import load_config
from vision_bot.position import (
    _normalize_coordinate_chars,
    _parse_coordinates,
    _to_coord_id,
    read_player_pose_observation,
    read_player_position,
    read_player_position_candidates,
    _extract_tesseract_variant_texts,
)
from vision_bot.position_filter import choose_best_coord_candidate
from vision_bot.runtime_markers import RuntimeTelemetry


def test_to_coord_id_formats_coordinate_correctly():
    assert _to_coord_id(31.23, 45.67) == 3123456700
    assert _to_coord_id(0.0, 0.0) == 0
    assert _to_coord_id(100.0, 99.99) == 10000999900


def test_parse_coordinates_handles_wow_coordinate_text():
    assert _parse_coordinates("37.52, 52.12") == 3752521200
    assert _parse_coordinates("37.52 52.12") == 3752521200
    assert _parse_coordinates("53402,3604") is None


def test_normalize_coordinate_chars_recovers_missing_decimal_dot():
    chars = list("5402.8604")

    assert _normalize_coordinate_chars(chars) == "54.02,86.04"


def test_read_player_position_from_calibration_screenshot():
    if not Path("screen.png").exists():
        pytest.skip("private calibration screenshot is not distributed")
    frame = cv2.imread("screen.png", cv2.IMREAD_COLOR)
    assert frame is not None

    coord = read_player_position(frame, load_config("config.yaml"))

    assert coord == 3752521200


def test_read_player_position_from_live_probe_frame_when_available():
    frame_path = Path("debug_output/live_after_probe/window_frame.png")
    if not frame_path.exists():
        return

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None

    coord = read_player_position(frame, load_config("config.yaml"))

    assert coord == 5396860300


def test_read_player_position_from_live_mouse_probe_frame_when_available():
    frame_path = Path("debug_output/live_after_mouse_probe/window_frame.png")
    if not frame_path.exists():
        return

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None

    config = load_config("config.yaml")
    candidates = read_player_position_candidates(frame, config)
    coord = choose_best_coord_candidate(3716528700, candidates, max_jump=1.5)

    assert coord == 3722530600


def test_read_player_position_from_live_dark_confirm_frame_when_available():
    frame_path = Path("debug_output/live_after_dark_confirm/window_frame.png")
    if not frame_path.exists():
        return

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None

    coord = read_player_position(frame, load_config("config.yaml"))

    assert coord == 3683551700


def test_read_player_position_from_current_barrens_frame_when_available():
    frame_path = Path("debug_output/pre_live_modal_ignored_20260731_1650.png")
    if not frame_path.exists():
        return

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None

    coord = read_player_position(frame, load_config("config.yaml"))

    assert coord == 5333597400


def test_read_player_position_candidates_uses_configured_fallback_regions(monkeypatch):
    primary = {"x": 10, "y": 20, "width": 30, "height": 40}
    fallback = {"x": 50, "y": 60, "width": 70, "height": 80}
    config = {"player_position": {**primary, "fallback_regions": [fallback]}}
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_calls = []

    def fake_crop_region(_frame, region_cfg, _config):
        crop_calls.append(region_cfg)
        marker = len(crop_calls)
        return np.full((1, 1), marker, dtype=np.uint8)

    def fake_extract_coordinate_candidates(region, _config):
        if int(region[0, 0]) == 1:
            return []
        return [5396860300, 5396860300]

    monkeypatch.setattr("vision_bot.position.crop_region", fake_crop_region)
    monkeypatch.setattr(
        "vision_bot.position._extract_coordinate_candidates",
        fake_extract_coordinate_candidates,
    )

    assert read_player_position_candidates(frame, config) == [5396860300]
    assert crop_calls == [config["player_position"], fallback]


def test_read_player_position_candidates_direct_region_skips_fallback_regions(monkeypatch):
    config = {
        "player_position": {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "fallback_regions": [{"x": 50, "y": 60, "width": 70, "height": 80}],
        }
    }
    frame = np.full((1, 1), 7, dtype=np.uint8)

    def fail_crop_region(*_args, **_kwargs):
        raise AssertionError("direct_region must not crop configured regions")

    def fake_extract_coordinate_candidates(region, _config):
        assert int(region[0, 0]) == 7
        return [3752521200]

    monkeypatch.setattr("vision_bot.position.crop_region", fail_crop_region)
    monkeypatch.setattr(
        "vision_bot.position._extract_coordinate_candidates",
        fake_extract_coordinate_candidates,
    )

    assert read_player_position_candidates(frame, config, direct_region=True) == [3752521200]


def test_visible_runtime_telemetry_has_priority_over_ocr(monkeypatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "vision_bot.position.read_runtime_telemetry",
        lambda _frame, _config: RuntimeTelemetry(5388287400, 53.88, 28.74, 90.0),
    )
    monkeypatch.setattr(
        "vision_bot.position.crop_region",
        lambda *_args: (_ for _ in ()).throw(AssertionError("OCR must not run")),
    )

    assert read_player_position_candidates(frame, {}) == [5388287400]


def test_player_pose_observation_preserves_visible_heading(monkeypatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "vision_bot.position.read_runtime_telemetry",
        lambda _frame, _config: RuntimeTelemetry(5388287400, 53.88, 28.74, 271.5),
    )

    pose = read_player_pose_observation(frame, {})

    assert pose.coord == 5388287400
    assert pose.coord_candidates == (5388287400,)
    assert pose.heading_degrees == 271.5
    assert pose.source == "runtime_telemetry"


def test_player_pose_observation_marks_ocr_heading_unavailable(monkeypatch):
    frame = np.full((1, 1), 7, dtype=np.uint8)
    monkeypatch.setattr(
        "vision_bot.position._extract_coordinate_candidates",
        lambda _region, _config: [3752521200],
    )

    pose = read_player_pose_observation(frame, {}, direct_region=True)

    assert pose.coord == 3752521200
    assert pose.heading_degrees is None
    assert pose.source == "direct_ocr"


def test_position_ocr_fast_path_skips_tesseract_when_template_is_valid(monkeypatch):
    config = {"position_ocr": {"template_first_fast_path": True}}
    region = np.zeros((24, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "vision_bot.position._prepare_coordinate_mask",
        lambda _region, _config: np.zeros((12, 80), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "vision_bot.position._extract_text_with_templates",
        lambda _mask: "40.85, 57.98",
    )
    monkeypatch.setattr(
        "vision_bot.position._extract_tesseract_variant_texts",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Tesseract must not run")),
    )

    assert read_player_position_candidates(region, config, direct_region=True) == [4085579800]


def test_position_ocr_can_stop_after_primary_valid_region(monkeypatch):
    config = {
        "player_position": {
            "x": 0,
            "y": 0,
            "width": 10,
            "height": 10,
            "fallback_regions": [{"x": 20, "y": 20, "width": 10, "height": 10}],
        },
        "position_ocr": {"stop_after_first_valid_region": True},
    }
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(
        "vision_bot.position.crop_region",
        lambda _frame, region_cfg, _config: calls.append(region_cfg) or np.ones((1, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "vision_bot.position._extract_coordinate_candidates",
        lambda _region, _config: [4085579800],
    )

    assert read_player_position_candidates(frame, config) == [4085579800]
    assert calls == [config["player_position"]]


def test_fast_grayscale_tesseract_returns_before_legacy_variants(monkeypatch):
    calls = []
    fake_tesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        image_to_string=lambda _image, config: calls.append(config) or "40.85,57.98",
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)
    region = np.zeros((30, 120, 3), dtype=np.uint8)
    mask = np.zeros((30, 120), dtype=np.uint8)

    texts = _extract_tesseract_variant_texts(
        region,
        mask,
        {"position_ocr": {"fast_grayscale_first": True, "scale": 2}},
    )

    assert texts == ["40.85,57.98"]
    assert len(calls) == 1
    assert "--psm 7" in calls[0]
