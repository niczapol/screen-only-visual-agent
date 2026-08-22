from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


POSITIVE_SAMPLES = {
    285: (1590, 620, 1680, 720),
    288: (1810, 675, 1900, 795),
    291: (2045, 715, 2185, 895),
    294: (2220, 735, 2395, 945),
    297: (2260, 735, 2420, 945),
}

RUN4_NEGATIVE_SAMPLES = (80, 83, 86, 89, 92, 95, 98, 312, 318, 324, 330, 336, 342, 348)
RUN3_NEGATIVE_SAMPLES = (83, 86)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _copy_base_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing dataset: {destination}. Remove it explicitly first."
        )
    for split in ("train", "val", "test"):
        image_output = destination / "images" / split
        label_output = destination / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / "images" / split).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                shutil.copy2(path, image_output / path.name)
        for path in sorted((source / "labels" / split).glob("*.txt")):
            shutil.copy2(path, label_output / path.name)


def _yolo_line(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def _square_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    size: int = 640,
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int, int, int]]:
    width, height = image.size
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    crop_width = min(size, width)
    crop_height = min(size, height)
    crop_x1 = max(0, min(width - crop_width, center_x - crop_width // 2))
    crop_y1 = max(0, min(height - crop_height, center_y - crop_height // 2))
    crop_box = (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)
    relative_box = (
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1,
    )
    return image.crop(crop_box), relative_box, crop_box


def _add_positive_samples(
    run_dir: Path,
    dataset: Path,
    review_dir: Path,
    *,
    repeats: int = 1,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
    review_dir.mkdir(parents=True, exist_ok=True)
    for index, box in POSITIVE_SAMPLES.items():
        source = run_dir / "frames" / f"{index:04d}.png"
        image = Image.open(source).convert("RGB")
        width, height = image.size
        crop, crop_box_relative, crop_source_box = _square_crop(image, box)
        base_stem = f"v0814_run4_mithril_{index:04d}"

        review = image.copy()
        draw = ImageDraw.Draw(review)
        draw.rectangle(box, outline=(0, 255, 0), width=5)
        draw.text((box[0], max(0, box[1] - 24)), f"Mithril {index}", fill=(0, 255, 0))
        review.save(review_dir / f"{base_stem}_review.jpg", quality=92)

        for repeat in range(max(1, repeats)):
            suffix = "" if repeat == 0 else f"_rep{repeat}"
            stem = f"{base_stem}{suffix}"
            destination = dataset / "images" / "train" / f"{stem}.png"
            shutil.copy2(source, destination)
            (dataset / "labels" / "train" / f"{stem}.txt").write_text(
                _yolo_line(box, width, height),
                encoding="utf-8",
            )
            crop_stem = f"{stem}_crop"
            crop_destination = dataset / "images" / "train" / f"{crop_stem}.png"
            crop.save(crop_destination)
            (dataset / "labels" / "train" / f"{crop_stem}.txt").write_text(
                _yolo_line(crop_box_relative, crop.width, crop.height),
                encoding="utf-8",
            )

            for sample_stem, sample_path, sample_box, source_crop in (
                (stem, destination, box, None),
                (crop_stem, crop_destination, crop_box_relative, crop_source_box),
            ):
                manifest_rows.append(
                    {
                        "image_id": sample_stem,
                        "path": sample_path.as_posix(),
                        "source_kind": "reviewed_v0814_centered_mithril_positive",
                        "source_run": run_dir.as_posix(),
                        "source_index": index,
                        "source_repeat": repeat,
                        "ore_type": "Mithril Deposit",
                        "expected_positive": True,
                        "split": "train",
                        "negative": False,
                        "boxes": [
                            {
                                "x1": sample_box[0],
                                "y1": sample_box[1],
                                "x2": sample_box[2],
                                "y2": sample_box[3],
                                "confidence": 1.0,
                            }
                        ],
                        "source_crop": source_crop,
                    }
                )
    return manifest_rows


def _add_negative_samples(
    source_run: Path,
    indexes: tuple[int, ...],
    dataset: Path,
    *,
    prefix: str,
    repeats: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in indexes:
        source = source_run / "frames" / f"{index:04d}.png"
        if not source.is_file():
            continue
        for repeat in range(max(1, repeats)):
            suffix = "" if repeat == 0 else f"_rep{repeat}"
            stem = f"{prefix}_{index:04d}{suffix}"
            destination = dataset / "images" / "train" / f"{stem}.png"
            shutil.copy2(source, destination)
            (dataset / "labels" / "train" / f"{stem}.txt").write_text(
                "", encoding="utf-8"
            )
            rows.append(
                {
                    "image_id": stem,
                    "path": destination.as_posix(),
                    "source_kind": "reviewed_v0814_hard_negative",
                    "source_run": source_run.as_posix(),
                    "source_index": index,
                    "source_repeat": repeat,
                    "ore_type": None,
                    "expected_positive": False,
                    "split": "train",
                    "negative": True,
                    "boxes": [],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reviewed v0.8.14 ore dataset")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/ore_world_detector_v0813/yolo_dataset"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ore_world_detector_v0814/yolo_dataset"),
    )
    parser.add_argument(
        "--run4",
        type=Path,
        default=Path("data/live_v0814_mining_run4b_20260812_032931"),
    )
    parser.add_argument(
        "--run3",
        type=Path,
        default=Path("data/live_v0814_mining_run3_20260812_030612"),
    )
    parser.add_argument("--positive-repeats", type=int, default=1)
    parser.add_argument("--negative-repeats", type=int, default=1)
    args = parser.parse_args()

    _copy_base_dataset(args.base, args.output)
    review_dir = args.output.parent / "label_review"
    positive_repeats = max(1, args.positive_repeats)
    negative_repeats = max(1, args.negative_repeats)
    new_rows = _add_positive_samples(
        args.run4,
        args.output,
        review_dir,
        repeats=positive_repeats,
    )
    new_rows.extend(
        _add_negative_samples(
            args.run4,
            RUN4_NEGATIVE_SAMPLES,
            args.output,
            prefix="v0814_run4_negative",
            repeats=negative_repeats,
        )
    )
    new_rows.extend(
        _add_negative_samples(
            args.run3,
            RUN3_NEGATIVE_SAMPLES,
            args.output,
            prefix="v0814_run3_false_positive",
            repeats=negative_repeats,
        )
    )

    base_manifest = args.base / "reviewed_manifest.jsonl"
    manifest_output = args.output / "reviewed_manifest.jsonl"
    base_text = base_manifest.read_text(encoding="utf-8").rstrip()
    additions = "\n".join(json.dumps(row, ensure_ascii=False) for row in new_rows)
    manifest_output.write_text(f"{base_text}\n{additions}\n", encoding="utf-8")
    dataset_yaml = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ore_vein\n"
    )
    (args.output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    counts = {
        split: len(
            [
                path
                for path in (args.output / "images" / split).iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        for split in ("train", "val", "test")
    }
    summary = {
        "base": args.base.as_posix(),
        "output": args.output.as_posix(),
        "image_counts": counts,
        "new_positive_images": len(POSITIVE_SAMPLES) * 2 * positive_repeats,
        "new_negative_images": (
            len(new_rows) - len(POSITIVE_SAMPLES) * 2 * positive_repeats
        ),
        "positive_repeats": positive_repeats,
        "negative_repeats": negative_repeats,
        "positive_source_indexes": sorted(POSITIVE_SAMPLES),
        "run4_negative_indexes": list(RUN4_NEGATIVE_SAMPLES),
        "run3_negative_indexes": list(RUN3_NEGATIVE_SAMPLES),
    }
    (args.output.parent / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
