import cv2
import pytest
from pathlib import Path

from vision_bot.capture import ScreenCapture
from vision_bot.config import load_config
from vision_bot.recognition import recognize_ore_points


def test_calibrated_minimap_crop_and_ore_detection_from_screenshot():
    config = load_config("config.yaml")
    if not Path("screen.png").exists():
        pytest.skip("private calibration screenshot is not distributed")
    frame = cv2.imread("screen.png", cv2.IMREAD_COLOR)
    assert frame is not None

    minimap = ScreenCapture(config).crop_minimap(frame, config)
    points = recognize_ore_points(minimap, config)

    assert minimap.shape[:2] == (275, 291)
    assert points == [(157, 115)]
