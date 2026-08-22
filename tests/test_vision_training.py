import numpy as np
import cv2
import sys
import types

from vision_bot.vision_training import prepare_yolo_training_split, train_yolo_model


def test_prepare_yolo_training_split_creates_train_val_dirs(tmp_path):
    image_dir = tmp_path / "images" / "bootstrap"
    label_dir = tmp_path / "labels" / "bootstrap"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(5):
        image_path = image_dir / f"sample_{index}.png"
        cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
        (label_dir / f"sample_{index}.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.250000\n",
            encoding="utf-8",
        )

    summary = prepare_yolo_training_split(tmp_path, val_fraction=0.4, seed=1)

    assert summary.train_count == 3
    assert summary.val_count == 2
    assert summary.skipped_without_labels == 0
    assert summary.yaml_path.exists()
    assert len(list((tmp_path / "images" / "train").glob("*.png"))) == 3
    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 2
    assert len(list((tmp_path / "labels" / "train").glob("*.txt"))) == 3
    assert len(list((tmp_path / "labels" / "val").glob("*.txt"))) == 2
    yaml_text = summary.yaml_path.read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text


def test_train_yolo_model_reports_actual_save_dir(tmp_path, monkeypatch):
    actual_run_dir = tmp_path / "actual_project" / "actual_run"
    weights_dir = actual_run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"
    best_weights.write_bytes(b"best")
    last_weights.write_bytes(b"last")

    class FakeTrainResult:
        save_dir = actual_run_dir

    class FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kwargs):
            return FakeTrainResult()

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    summary = train_yolo_model(
        tmp_path / "dataset.yaml",
        project=tmp_path / "configured_project",
        name="configured_run",
    )

    assert summary.run_dir == actual_run_dir.resolve()
    assert summary.weights_dir == weights_dir.resolve()
    assert summary.best_weights == best_weights.resolve()
    assert summary.last_weights == last_weights.resolve()
