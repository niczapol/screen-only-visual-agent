from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def _serializable_metrics(metrics: Any) -> dict[str, float]:
    values = getattr(metrics, "results_dict", {}) or {}
    return {
        str(key): round(float(value), 6)
        for key, value in values.items()
        if isinstance(value, int | float)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the reviewed one-class world ore detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/ore_world_detector_v1/yolo_dataset/dataset.yaml"),
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conservative-finetune",
        action="store_true",
        help="Freeze the backbone and use low-variance augmentation with a small learning rate",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/ore_detector"))
    parser.add_argument("--name", default="ore_world_v1")
    parser.add_argument(
        "--runtime-model",
        type=Path,
        default=Path("data/models/ore_world_v1.pt"),
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    started = time.monotonic()
    train_options: dict[str, Any] = dict(
        data=str(args.data),
        epochs=max(1, args.epochs),
        imgsz=max(320, args.imgsz),
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        seed=1337,
        deterministic=True,
        patience=35,
        cache="disk",
        plots=True,
        close_mosaic=10,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.45,
        degrees=4.0,
        translate=0.15,
        scale=0.55,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    if args.conservative_finetune:
        train_options.update(
            optimizer="AdamW",
            lr0=0.0002,
            lrf=0.1,
            warmup_epochs=1.0,
            freeze=10,
            patience=max(10, min(20, args.epochs)),
            close_mosaic=0,
            hsv_h=0.01,
            hsv_s=0.25,
            hsv_v=0.20,
            degrees=2.0,
            translate=0.05,
            scale=0.15,
            mosaic=0.0,
            mixup=0.0,
            erasing=0.0,
        )
    result = model.train(**train_options)
    run_dir = Path(getattr(result, "save_dir", args.project / args.name)).resolve()
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training did not produce {best_weights}")

    best_model = YOLO(str(best_weights))
    validation = best_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        project=str(args.project.resolve()),
        name=f"{args.name}_val",
    )
    test = best_model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        project=str(args.project.resolve()),
        name=f"{args.name}_test",
    )

    args.runtime_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.runtime_model)
    summary = {
        "base_model": args.model,
        "data": args.data.as_posix(),
        "run_dir": run_dir.as_posix(),
        "runtime_model": args.runtime_model.as_posix(),
        "epochs_requested": args.epochs,
        "conservative_finetune": args.conservative_finetune,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "validation": _serializable_metrics(validation),
        "test": _serializable_metrics(test),
        "validation_speed_ms": getattr(validation, "speed", {}),
        "test_speed_ms": getattr(test, "speed", {}),
    }
    summary_path = run_dir / "ore_detector_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
