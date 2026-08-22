from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vision_bot.config import _resolve_config_path, load_config
from vision_bot.capture import crop_region
from vision_bot.recognition import (
    _bright_icon_shape_is_compact,
    recognize_bright_ore_template_points,
    recognize_dark_ore_points,
    recognize_ore_point_classes,
    recognize_ore_points,
    save_debug_artifacts,
)


FIXTURES = ROOT / "tests" / "fixtures" / "minimap_ore"


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
minimap:
  x: 10
  y: 20
  width: 100
  height: 80
""".strip()
    )

    config = load_config(config_path)

    assert config["minimap"]["x"] == 10
    assert config["minimap"]["width"] == 100


def test_resolve_config_path_uses_existing_requested_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("debug:\n  enabled: true\n")

    assert _resolve_config_path(config_path) == config_path


def test_recognize_yellow_ore_points_on_saved_png(tmp_path):
    image_path = tmp_path / "minimap.png"
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 30), (50, 60), (40, 40, 40), -1)
    cv2.circle(image, (35, 45), 6, (0, 255, 255), -1)
    cv2.circle(image, (110, 90), 5, (0, 255, 255), -1)
    cv2.circle(image, (170, 20), 7, (0, 255, 255), -1)
    cv2.imwrite(str(image_path), image)

    points = recognize_ore_points(image_path)

    assert len(points) == 3
    assert any(abs(point[0] - 35) < 3 and abs(point[1] - 45) < 3 for point in points)
    assert any(abs(point[0] - 110) < 3 and abs(point[1] - 90) < 3 for point in points)
    assert any(abs(point[0] - 170) < 3 and abs(point[1] - 20) < 3 for point in points)


def test_recognize_dark_ore_points_on_saved_png(tmp_path):
    image_path = tmp_path / "dark_minimap.png"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    hsv = np.zeros((120, 120, 3), dtype=np.uint8)
    hsv[:, :] = (0, 0, 0)
    cv2.circle(hsv, (60, 60), 5, (28, 180, 80), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(image_path), image)

    points = recognize_dark_ore_points(image_path)

    assert points == [(60, 60)]


def test_recognize_dark_ore_points_ignores_minimap_edges(tmp_path):
    image_path = tmp_path / "dark_edge_minimap.png"
    hsv = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(hsv, (116, 116), 5, (28, 180, 80), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(image_path), image)

    points = recognize_dark_ore_points(image_path)

    assert points == []


def test_bright_template_replay_detects_live_ore_then_clears_after_gather():
    config = load_config(ROOT / "config.yaml")
    if not (FIXTURES / "tanaris_live_ore_01.png").exists():
        pytest.skip("private replay images are not distributed")

    first = recognize_bright_ore_template_points(
        FIXTURES / "tanaris_live_ore_01.png",
        config,
    )
    second = recognize_bright_ore_template_points(
        FIXTURES / "tanaris_live_ore_02.png",
        config,
    )
    cleared = recognize_bright_ore_template_points(
        FIXTURES / "tanaris_after_gather.png",
        config,
    )

    assert any(abs(x - 83) <= 2 and abs(y - 104) <= 2 for x, y in first)
    assert any(abs(x - 88) <= 2 and abs(y - 93) <= 2 for x, y in second)
    assert cleared == []


def test_bright_icon_shape_filter_accepts_ore_and_rejects_tall_quest_glyph():
    config = load_config(ROOT / "config.yaml")
    rec_cfg = config["recognition"]
    if not (ROOT / "assets" / "minimap_ore_bright_reference.png").exists():
        pytest.skip("client-derived icon fixture is not distributed")
    ore = cv2.imread(str(ROOT / "assets" / "minimap_ore_bright_reference.png"))
    assert ore is not None
    quest = np.zeros((31, 31, 3), dtype=np.uint8)
    cv2.rectangle(quest, (13, 5), (17, 18), (0, 230, 255), -1)
    cv2.circle(quest, (15, 25), 2, (0, 230, 255), -1)

    assert _bright_icon_shape_is_compact(
        ore,
        (ore.shape[1] // 2, ore.shape[0] // 2),
        rec_cfg,
    )
    assert not _bright_icon_shape_is_compact(quest, (15, 15), rec_cfg)


def test_template_icons_are_split_by_local_yellow_luminance(monkeypatch):
    hsv = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(hsv, (35, 60), 6, (28, 210, 200), -1)
    cv2.circle(hsv, (85, 60), 6, (28, 210, 155), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    monkeypatch.setattr(
        "vision_bot.recognition.recognize_bright_ore_template_points",
        lambda _image, _config: [(35, 60), (85, 60)],
    )
    config = {
        "recognition": {
            "bright_template_enabled": True,
            "bright_template_fallback_hsv": False,
            "bright_icon_luminance_radius": 7,
            "bright_icon_min_yellow_pixels": 20,
            "bright_icon_min_yellow_value": 175,
            "dark_icon": {"template_classification_enabled": True},
        }
    }

    bright, dark = recognize_ore_point_classes(image, config)

    assert bright == [(35, 60)]
    assert dark == [(85, 60)]


def test_tooltip_authority_can_treat_dim_template_match_as_live_candidate(monkeypatch):
    hsv = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(hsv, (60, 60), 6, (28, 210, 145), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    monkeypatch.setattr(
        "vision_bot.recognition.recognize_bright_ore_template_points",
        lambda _image, _config: [(60, 60)],
    )
    config = {
        "recognition": {
            "bright_template_enabled": True,
            "bright_template_fallback_hsv": False,
            "dark_icon": {"template_classification_enabled": False},
        }
    }

    bright, dark = recognize_ore_point_classes(image, config)

    assert bright == [(60, 60)]
    assert dark == []


def test_surface_icon_with_dark_background_is_classified_by_yellow_median():
    config = load_config("config.yaml")
    if not Path("screen.png").exists():
        pytest.skip("private calibration screenshot is not distributed")
    frame = cv2.imread("screen.png", cv2.IMREAD_COLOR)
    assert frame is not None
    minimap = crop_region(frame, config["minimap"], config)

    bright, dark = recognize_ore_point_classes(minimap, config)

    assert bright == [(157, 115)]
    assert dark == []


def test_save_debug_artifacts_writes_files(tmp_path):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(image, (40, 40), 6, (0, 255, 255), -1)

    output_dir = tmp_path / "debug"
    output_paths = save_debug_artifacts(image, [(40, 40)], output_dir)

    assert output_dir.exists()
    assert output_paths["original"].exists()
    assert output_paths["processed"].exists()
    assert output_paths["mask"].exists()
