import numpy as np

from vision_bot.screen_objects import BoundingBox, ScreenDetection
from vision_bot.vision_dataset import save_vision_sample


def test_save_vision_sample_writes_yolo_dataset_files(tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = [
        ScreenDetection("aggressive_mob", BoundingBox(50, 20, 40, 20), 0.9, "test"),
    ]

    sample = save_vision_sample(
        frame,
        detections,
        tmp_path,
        auto_label=True,
        save_overlay=True,
        sample_index=0,
        config={"vision_dataset": {"label_min_score": 0.25}},
    )

    assert sample.image_path.exists()
    assert sample.label_path is not None
    assert sample.label_path.exists()
    assert sample.overlay_path is not None
    assert sample.overlay_path.exists()
    assert (tmp_path / "classes.txt").exists()
    assert (tmp_path / "dataset.yaml").exists()
    assert (tmp_path / "metadata.jsonl").exists()
    assert sample.label_path.read_text(encoding="utf-8").startswith("0 ")
