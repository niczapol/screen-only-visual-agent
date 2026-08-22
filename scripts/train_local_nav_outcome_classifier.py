"""Train a small sector-crop classifier for local navigation outcomes.

This is an offline experiment: it reads saved route-probe frames and labels, and
does not interact with the live game client.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = ["blocked", "passable", "unknown"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train local navigation outcome classifier on attempted-sector crops."
    )
    parser.add_argument(
        "--dataset",
        default="data/local_navigation_outcome_dataset_v0_2/metadata.jsonl",
        help="Path to outcome dataset metadata.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/local_navigation/outcome_sector_cnn_v0_2",
        help="Directory for model checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--labels",
        default="blocked,passable,unknown",
        help="Comma-separated labels to train/evaluate, in output-index order.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. auto prefers CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Windows-safe default is 0; increase only after a clean baseline run.",
    )
    return parser.parse_args()


def configure_labels(raw_labels: str) -> None:
    labels = [item.strip() for item in raw_labels.split(",") if item.strip()]
    if len(labels) < 2:
        raise ValueError("--labels must contain at least two labels.")
    if len(labels) != len(set(labels)):
        raise ValueError(f"--labels contains duplicates: {raw_labels}")

    global LABELS, LABEL_TO_INDEX
    LABELS = labels
    LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def find_sector_bbox(row: dict[str, Any]) -> dict[str, int] | None:
    attempted_sector = row.get("attempted_sector")
    navigation = row.get("navigation") or {}
    for sector in navigation.get("sectors") or []:
        if sector.get("name") == attempted_sector:
            bbox = sector.get("bbox") or {}
            required = ("x", "y", "width", "height")
            if all(key in bbox for key in required):
                return {key: int(bbox[key]) for key in required}
    return None


def resolve_image_path(raw_path: str, project_root: Path, dataset_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate

    dataset_candidate = dataset_dir / path
    if dataset_candidate.exists():
        return dataset_candidate

    return project_candidate


def load_samples(dataset_path: Path, project_root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    dataset_dir = dataset_path.parent

    for row in read_jsonl(dataset_path):
        label = row.get("label")
        if label not in LABEL_TO_INDEX:
            skipped["bad_label"] += 1
            continue

        split = row.get("split")
        if split not in {"train", "val", "test"}:
            skipped["bad_split"] += 1
            continue

        raw_frame = row.get("source_frame") or row.get("image")
        if not raw_frame:
            skipped["missing_frame"] += 1
            continue

        image_path = resolve_image_path(str(raw_frame), project_root, dataset_dir)
        if not image_path.exists():
            skipped["frame_not_found"] += 1
            continue

        bbox = find_sector_bbox(row)
        if bbox is None:
            skipped["missing_sector_bbox"] += 1
            continue

        samples.append(
            {
                "image_path": image_path,
                "label": label,
                "split": split,
                "bbox": bbox,
                "episode_id": row.get("episode_id"),
                "row_index": row.get("row_index"),
                "attempted_sector": row.get("attempted_sector"),
            }
        )

    return samples, skipped


class SectorCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[dict[str, Any]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.mean = np.array(DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample['image_path']}")

        height, width = image.shape[:2]
        bbox = sample["bbox"]
        x1 = max(0, min(width - 1, bbox["x"]))
        y1 = max(0, min(height - 1, bbox["y"]))
        x2 = max(x1 + 1, min(width, bbox["x"] + bbox["width"]))
        y2 = max(y1 + 1, min(height, bbox["y"] + bbox["height"]))
        crop = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = crop.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = (tensor - self.mean) / self.std
        label = LABEL_TO_INDEX[sample["label"]]
        return torch.from_numpy(tensor), torch.tensor(label, dtype=torch.long)


class SmallSectorCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def confusion_matrix(predictions: list[int], targets: list[int]) -> list[list[int]]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, Any]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "support": actual,
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    matrix = confusion_matrix(predictions, targets)
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = round(running_loss / total, 6) if total else 0.0
    metrics["confusion"] = matrix
    return metrics


def make_loaders(
    samples_by_split: dict[str, list[dict[str, Any]]],
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {}
    for split, samples in samples_by_split.items():
        shuffle = split == "train"
        dataset = SectorCropDataset(samples, image_size=image_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def compute_class_weights(train_samples: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(sample["label"] for sample in train_samples)
    total = sum(counts.values())
    weights = []
    for label in LABELS:
        count = counts.get(label, 0)
        weights.append(total / (len(LABELS) * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def write_epoch_results(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "train_loss": item["train"]["loss"],
                    "train_accuracy": item["train"]["accuracy"],
                    "val_loss": item["val"]["loss"],
                    "val_accuracy": item["val"]["accuracy"],
                    "test_loss": item.get("test", {}).get("loss", ""),
                    "test_accuracy": item.get("test", {}).get("accuracy", ""),
                }
            )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    model.train()
    predictions: list[int] = []
    targets: list[int] = []
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    matrix = confusion_matrix(predictions, targets)
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = round(running_loss / total, 6) if total else 0.0
    metrics["confusion"] = matrix
    return metrics


def main() -> None:
    args = parse_args()
    configure_labels(args.labels)
    seed_everything(args.seed)

    project_root = Path.cwd()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_path = dataset_path.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    samples, skipped = load_samples(dataset_path, project_root)
    samples_by_split = {
        split: [sample for sample in samples if sample["split"] == split]
        for split in ("train", "val", "test")
    }

    if not samples_by_split["train"]:
        raise RuntimeError("No train samples found.")
    if not samples_by_split["val"]:
        raise RuntimeError("No val samples found.")

    loaders = make_loaders(
        samples_by_split=samples_by_split,
        batch_size=args.batch_size,
        image_size=args.imgsz,
        num_workers=args.num_workers,
    )

    model = SmallSectorCNN(num_classes=len(LABELS)).to(device)
    class_weights = compute_class_weights(samples_by_split["train"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], device, criterion, optimizer)
        val_metrics = evaluate(model, loaders["val"], device, criterion)
        test_metrics = (
            evaluate(model, loaders["test"], device, criterion)
            if samples_by_split["test"]
            else {"loss": 0.0, "accuracy": 0.0, "confusion": []}
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_map": LABEL_TO_INDEX,
                    "image_size": args.imgsz,
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "args": vars(args),
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"test_acc={test_metrics['accuracy']:.3f}"
        )

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    final_metrics = {
        split: evaluate(model, loaders[split], device, criterion)
        for split in loaders
    }
    label_counts = {
        split: dict(Counter(sample["label"] for sample in split_samples))
        for split, split_samples in samples_by_split.items()
    }
    metrics = {
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "labels": LABELS,
        "label_to_index": LABEL_TO_INDEX,
        "sample_counts": {split: len(split_samples) for split, split_samples in samples_by_split.items()},
        "label_counts": label_counts,
        "skipped": dict(skipped),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final": final_metrics,
        "history": history,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_map.json").write_text(
        json.dumps(LABEL_TO_INDEX, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_epoch_results(output_dir / "results.csv", history)
    print(json.dumps({key: metrics[key] for key in ("best_epoch", "best_val_accuracy", "final")}, indent=2))


if __name__ == "__main__":
    main()
