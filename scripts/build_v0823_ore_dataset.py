from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


SAMPLES = {
    # First post-F7 frame is less zoomed; subsequent frames use the settled
    # camera scale observed throughout the failed hover scan.
    63: (550, 580, 925, 1050, "Small Thorium Vein"),
    66: (120, 480, 730, 1220, "Small Thorium Vein"),
    80: (120, 480, 730, 1220, "Small Thorium Vein"),
    100: (120, 480, 730, 1220, "Small Thorium Vein"),
    # The live Mithril is genuinely clipped by the left screen edge after the
    # deterministic camera reset. Edge-truncated examples are intentional.
    # Label the exposed crystal-bearing portion, not the top-left unit-frame UI
    # that occludes the rest of the deposit.
    198: (90, 350, 450, 1120, "Mithril Deposit"),
    212: (90, 350, 450, 1120, "Mithril Deposit"),
    230: (90, 350, 450, 1120, "Mithril Deposit"),
    251: (90, 350, 450, 1120, "Mithril Deposit"),
    272: (90, 350, 450, 1120, "Mithril Deposit"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _yolo_line(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"0 {((x1 + x2) / 2) / width:.8f} {((y1 + y2) / 2) / height:.8f} "
        f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n"
    )


def _crop_around_box(
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
    crop_source = (
        crop_x1,
        crop_y1,
        crop_x1 + crop_width,
        crop_y1 + crop_height,
    )
    clipped = (
        max(x1, crop_source[0]),
        max(y1, crop_source[1]),
        min(x2, crop_source[2]),
        min(y2, crop_source[3]),
    )
    relative = (
        clipped[0] - crop_source[0],
        clipped[1] - crop_source[1],
        clipped[2] - crop_source[0],
        clipped[3] - crop_source[1],
    )
    return image.crop(crop_source), relative, crop_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reviewed v0.8.23 live-ore dataset")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/ore_world_detector_v0814c/yolo_dataset"),
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("data/live_v0823_run5_20260816_170214"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ore_world_detector_v0823b/yolo_dataset"),
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {args.output}")
    shutil.copytree(args.base, args.output)
    for cache in args.output.rglob("*.cache"):
        cache.unlink()

    review_dir = args.output.parent / "label_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "reviewed_manifest.jsonl"
    additions: list[dict[str, object]] = []
    repeats = max(1, int(args.repeats))

    for index, values in SAMPLES.items():
        x1, y1, x2, y2, ore_type = values
        box = (x1, y1, x2, y2)
        source = args.run / "frames" / f"{index:04d}.png"
        image = Image.open(source).convert("RGB")
        width, height = image.size
        crop, crop_box, crop_source = _crop_around_box(image, box)

        review = image.copy()
        draw = ImageDraw.Draw(review)
        draw.rectangle(box, outline=(0, 255, 0), width=5)
        draw.text((max(0, x1), max(0, y1 + 8)), f"{ore_type} {index}", fill=(0, 255, 0))
        review.save(review_dir / f"v0823_run5_{index:04d}_review.jpg", quality=92)

        for repeat in range(repeats):
            suffix = "" if repeat == 0 else f"_rep{repeat}"
            for crop_kind, sample, sample_box, source_crop in (
                ("full", image, box, None),
                ("crop", crop, crop_box, crop_source),
            ):
                stem = f"v0823_run5_{index:04d}_{crop_kind}{suffix}"
                image_path = args.output / "images" / "train" / f"{stem}.png"
                label_path = args.output / "labels" / "train" / f"{stem}.txt"
                sample.save(image_path)
                label_path.write_text(
                    _yolo_line(sample_box, sample.width, sample.height),
                    encoding="utf-8",
                )
                additions.append(
                    {
                        "image_id": stem,
                        "path": image_path.as_posix(),
                        "source_kind": "reviewed_v0823_live_close_ore_positive",
                        "source_run": args.run.as_posix(),
                        "source_index": index,
                        "source_repeat": repeat,
                        "ore_type": ore_type,
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

    base_manifest = manifest_path.read_text(encoding="utf-8").rstrip()
    manifest_path.write_text(
        base_manifest
        + "\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in additions)
        + "\n",
        encoding="utf-8",
    )
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: ore_vein\n",
        encoding="utf-8",
    )
    summary = {
        "base": args.base.as_posix(),
        "output": args.output.as_posix(),
        "source_run": args.run.as_posix(),
        "repeats": repeats,
        "new_positive_images": len(additions),
        "source_indexes": sorted(SAMPLES),
        "image_counts": {
            split: len(
                [
                    path
                    for path in (args.output / "images" / split).iterdir()
                    if path.suffix.lower() in IMAGE_SUFFIXES
                ]
            )
            for split in ("train", "val", "test")
        },
    }
    (args.output.parent / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
