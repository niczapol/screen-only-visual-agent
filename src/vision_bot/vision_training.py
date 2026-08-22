from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vision_bot.screen_objects import DEFAULT_CLASS_NAMES


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class TrainingSplitSummary:
    dataset_dir: Path
    yaml_path: Path
    train_count: int
    val_count: int
    skipped_without_labels: int


@dataclass(frozen=True)
class TrainingRunSummary:
    project_dir: Path
    run_name: str
    run_dir: Path
    weights_dir: Path | None
    best_weights: Path | None
    last_weights: Path | None


@dataclass(frozen=True)
class PredictionRunSummary:
    model_path: Path
    source_path: Path
    run_dir: Path | None
    result_count: int
    detection_count: int


def prepare_yolo_training_split(
    dataset_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 1337,
    class_names: list[str] | None = None,
) -> TrainingSplitSummary:
    root = Path(dataset_dir)
    bootstrap_images = root / "images" / "bootstrap"
    bootstrap_labels = root / "labels" / "bootstrap"
    if not bootstrap_images.exists():
        raise FileNotFoundError(f"Missing bootstrap image directory: {bootstrap_images}")
    if not bootstrap_labels.exists():
        raise FileNotFoundError(f"Missing bootstrap label directory: {bootstrap_labels}")

    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for image_path in sorted(bootstrap_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = bootstrap_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue
        pairs.append((image_path, label_path))

    if not pairs:
        raise ValueError(f"No image/label pairs found in {bootstrap_images}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * max(0.0, min(0.9, val_fraction)))))
    if len(pairs) > 1:
        val_count = min(val_count, len(pairs) - 1)
    else:
        val_count = 0

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for relative in ["images/train", "images/val", "labels/train", "labels/val"]:
        target_dir = root / relative
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    _copy_pairs(train_pairs, root / "images" / "train", root / "labels" / "train")
    _copy_pairs(val_pairs, root / "images" / "val", root / "labels" / "val")
    yaml_path = write_training_dataset_yaml(root, class_names or DEFAULT_CLASS_NAMES)
    return TrainingSplitSummary(root, yaml_path, len(train_pairs), len(val_pairs), skipped)


def write_training_dataset_yaml(dataset_dir: str | Path, class_names: list[str] | None = None) -> Path:
    root = Path(dataset_dir)
    names = class_names or DEFAULT_CLASS_NAMES
    data = {
        "path": str(root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(names)},
    }
    output = root / "dataset_train.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def train_yolo_model(
    dataset_yaml: str | Path,
    *,
    model: str = "yolov8n.pt",
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 4,
    device: str | int | None = None,
    workers: int = 0,
    project: str | Path = "runs/vision",
    name: str = "screen_objects_bootstrap",
) -> TrainingRunSummary:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the optional 'ultralytics' package. "
            "Install it in a compatible Python environment before training."
        ) from exc

    project_dir = Path(project).resolve()
    model_instance = YOLO(model)
    train_args: dict[str, Any] = {
        "data": str(Path(dataset_yaml)),
        "epochs": max(1, epochs),
        "imgsz": max(64, imgsz),
        "batch": batch,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "workers": max(0, workers),
    }
    if device is not None:
        train_args["device"] = device

    train_result = model_instance.train(**train_args)
    run_dir = _resolve_run_dir(train_result, project_dir, name)
    weights_dir = run_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"
    return TrainingRunSummary(
        run_dir.parent,
        run_dir.name,
        run_dir,
        weights_dir if weights_dir.exists() else None,
        best_weights if best_weights.exists() else None,
        last_weights if last_weights.exists() else None,
    )


def predict_yolo_model(
    model_path: str | Path,
    source: str | Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
    project: str | Path = "runs/vision_eval",
    name: str = "screen_objects_prediction",
) -> PredictionRunSummary:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Prediction requires the optional 'ultralytics' package. "
            "Install it in a compatible Python environment before running inference."
        ) from exc

    model = Path(model_path)
    input_source = Path(source)
    project_dir = Path(project).resolve()
    model_instance = YOLO(str(model))
    predict_args: dict[str, Any] = {
        "source": str(input_source),
        "imgsz": max(64, imgsz),
        "conf": max(0.0, min(1.0, conf)),
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "save": True,
    }
    if device is not None:
        predict_args["device"] = device

    results = model_instance.predict(**predict_args)
    result_items = list(results) if isinstance(results, list | tuple) else [results]
    save_dir = _extract_save_dir(result_items) or _extract_save_dir(results)
    detection_count = sum(_result_box_count(item) for item in result_items)
    return PredictionRunSummary(
        model.resolve(),
        input_source.resolve(),
        save_dir,
        len(result_items),
        detection_count,
    )


def _copy_pairs(pairs: list[tuple[Path, Path]], image_dir: Path, label_dir: Path) -> None:
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def _resolve_run_dir(train_result: Any, project_dir: Path, run_name: str) -> Path:
    save_dir = _extract_save_dir(train_result)
    if save_dir is not None:
        return save_dir

    candidates = [
        project_dir / run_name,
        Path.cwd() / "runs" / "detect" / _relative_or_name(project_dir) / run_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (project_dir / run_name).resolve()


def _extract_save_dir(result: Any) -> Path | None:
    if result is None:
        return None
    save_dir = getattr(result, "save_dir", None)
    if save_dir is not None:
        return Path(save_dir).resolve()
    if isinstance(result, list | tuple):
        for item in result:
            save_dir = _extract_save_dir(item)
            if save_dir is not None:
                return save_dir
    return None


def _relative_or_name(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return Path(path.name)


def _result_box_count(result: Any) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    try:
        return len(boxes)
    except TypeError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and train the screen-object detector")
    parser.add_argument("--dataset-dir", default="data/vision_dataset")
    parser.add_argument("--prepare", action="store_true", help="Prepare train/val split")
    parser.add_argument("--train", action="store_true", help="Train YOLO model after preparing split")
    parser.add_argument("--predict", action="store_true", help="Run YOLO inference and save prediction images")
    parser.add_argument("--predict-source", default=None, help="Image/video/source path for prediction")
    parser.add_argument("--predict-model", default=None, help="Model weights for prediction")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/vision")
    parser.add_argument("--name", default="screen_objects_bootstrap")
    args = parser.parse_args()

    if args.prepare or args.train:
        summary = prepare_yolo_training_split(
            args.dataset_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(f"Dataset: {summary.dataset_dir}")
        print(f"Training YAML: {summary.yaml_path}")
        print(f"Train images: {summary.train_count}")
        print(f"Val images: {summary.val_count}")
        print(f"Skipped without labels: {summary.skipped_without_labels}")

        if args.train:
            run = train_yolo_model(
                summary.yaml_path,
                model=args.model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=args.name,
            )
            print(f"Training run: {run.run_dir}")
            print(f"Best weights: {run.best_weights}")
            print(f"Last weights: {run.last_weights}")

    if args.predict:
        if args.predict_source is None:
            parser.error("--predict-source is required with --predict")
        summary = predict_yolo_model(
            args.predict_model or args.model,
            args.predict_source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        print(f"Prediction source: {summary.source_path}")
        print(f"Prediction run: {summary.run_dir}")
        print(f"Prediction images: {summary.result_count}")
        print(f"Detections: {summary.detection_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
