import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_ore_detector_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_ore_detector_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_qwen_boxes_scales_and_filters_invalid_boxes():
    boxes = MODULE.parse_qwen_boxes(
        """{
          "objects": [
            {"bbox_1000": [100, 200, 500, 800], "confidence": 0.9},
            {"bbox_1000": [300, 300, 300, 400], "confidence": 0.7},
            {"bbox_1000": [-50, -50, 1050, 1050], "confidence": 1.2}
          ]
        }""",
        2000,
        1000,
    )

    assert len(boxes) == 1
    assert boxes[0] == MODULE.OreBox(200, 200, 1000, 800, 0.9)


def test_parse_qwen_boxes_accepts_empty_objects():
    assert MODULE.parse_qwen_boxes('{"objects": []}', 640, 480) == []


def test_parse_qwen_boxes_accepts_fenced_json():
    boxes = MODULE.parse_qwen_boxes(
        'analysis before result\n```json\n{"objects":[{"bbox_1000":[10,20,30,40],"confidence":0.8}]}\n```',
        1000,
        1000,
    )
    assert boxes == [MODULE.OreBox(10, 20, 30, 40, 0.8)]


def test_find_last_stationary_right_click_uses_last_candidate():
    rows = [
        {"timestamp": 95.0, "keys_down": ["W"], "mouse_down": ["mouse_right"], "cursor_client": [1, 2]},
        {"timestamp": 96.0, "keys_down": [], "mouse_down": [], "cursor_client": [3, 4]},
        {"timestamp": 96.2, "keys_down": [], "mouse_down": ["mouse_right"], "cursor_client": [5, 6]},
        {"timestamp": 96.3, "keys_down": [], "mouse_down": [], "cursor_client": [5, 6]},
        {"timestamp": 98.0, "keys_down": [], "mouse_down": ["mouse_right"], "cursor_client": [7, 8]},
    ]
    match = MODULE.find_last_stationary_right_click(rows, started_at=0.0, event_offset=100.0)
    assert match is not None
    assert match["cursor_client"] == [7, 8]


def test_crop_around_point_stays_inside_image():
    import numpy as np

    image = np.zeros((600, 800, 3), dtype=np.uint8)
    crop, origin = MODULE.crop_around_point(
        image,
        (780, 590),
        crop_width=400,
        crop_height=300,
    )
    assert crop.shape == (300, 400, 3)
    assert origin == (400, 300)


def test_scale_point_matches_recording_resize():
    assert MODULE.scale_point(
        (1024, 576),
        source_size=(2048, 1152),
        destination_size=(1600, 900),
    ) == (800, 450)


def test_add_local_sources_appends_and_deduplicates_manifest(tmp_path):
    import cv2
    import numpy as np

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.ones((20, 20, 3), dtype=np.uint8))
    output_dir = tmp_path / "sources"

    initial = MODULE.add_local_negative_sources([first], output_dir)
    appended = MODULE.add_local_negative_sources([first, second], output_dir)

    assert len(initial) == 1
    assert len(appended) == 2
    assert appended[0].image_id == initial[0].image_id
    assert appended[1].image_id.startswith("live_negative_001_")
    assert len((output_dir / "manifest.jsonl").read_text().splitlines()) == 2


def test_build_yolo_dataset_merges_boxes_and_writes_negative_label(tmp_path):
    import json
    import cv2
    import numpy as np

    positive_path = tmp_path / "positive.png"
    negative_path = tmp_path / "negative.png"
    cv2.imwrite(str(positive_path), np.zeros((100, 200, 3), dtype=np.uint8))
    cv2.imwrite(str(negative_path), np.zeros((100, 200, 3), dtype=np.uint8))
    sources = [
        MODULE.SourceImage("positive", positive_path, "test", expected_positive=True),
        MODULE.SourceImage("negative", negative_path, "test", expected_positive=False),
    ]
    annotations = [
        {
            "image_id": "positive",
            "boxes": [
                {"x1": 10, "y1": 20, "x2": 50, "y2": 60, "confidence": 0.8},
                {"x1": 40, "y1": 30, "x2": 90, "y2": 80, "confidence": 0.9},
            ],
        }
    ]
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"image_id": "positive", "split": "train", "merge_boxes": True},
                    {"image_id": "negative", "split": "val", "negative": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = MODULE.build_yolo_dataset(
        sources,
        annotations,
        review_path,
        tmp_path / "dataset",
    )

    label = (tmp_path / "dataset/labels/train/positive.txt").read_text()
    assert label == "0 0.25000000 0.50000000 0.40000000 0.60000000\n"
    assert (tmp_path / "dataset/labels/val/negative.txt").read_text() == ""
    assert summary["counts"]["train"]["boxes"] == 1
