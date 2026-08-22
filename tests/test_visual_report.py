import numpy as np

from vision_bot.screen_objects import BoundingBox, ScreenDetection
from vision_bot.visual_report import draw_detection_report, summarize_detections


def test_draw_detection_report_adds_legend_panel():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = [
        ScreenDetection("terrain_obstacle", BoundingBox(10, 20, 40, 30), 0.9, "test"),
        ScreenDetection("terrain_obstacle", BoundingBox(80, 20, 30, 25), 0.8, "test"),
        ScreenDetection("aggressive_mob", BoundingBox(120, 50, 20, 10), 0.7, "test"),
    ]

    report = draw_detection_report(frame, detections, model_path="runs/vision/model.pt")

    assert report.shape[0] == frame.shape[0]
    assert report.shape[1] > frame.shape[1]
    assert summarize_detections(detections) == {
        "aggressive_mob": 1,
        "terrain_obstacle": 2,
    }
