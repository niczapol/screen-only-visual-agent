from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2

from vision_bot.config import load_config
from vision_bot.ore_world_detector import OreWorldDetector


RUN4 = Path("data/live_v0814_mining_run4b_20260812_032931/frames")
RUN3 = Path("data/live_v0814_mining_run3_20260812_030612/frames")
RUN0813_GOLD = Path("data/live_v0813_run1_20260811_003638/frames")
MANUAL_MITHRIL = Path("data/ore_world_detector_v0813/manual_mithril_sequence")
USER_SCREENSHOT = Path(
    "data/external_evaluation/"
    "codex-clipboard-163a5502-4b1b-4b88-b35e-08f65164a011.png"
)


def _frames(root: Path, indexes: tuple[int, ...]) -> list[Path]:
    return [root / f"{index:04d}.png" for index in indexes]


def evaluation_groups() -> dict[str, tuple[bool, list[Path]]]:
    groups: dict[str, tuple[bool, list[Path]]] = {
        "new_run4_mithril": (
            True,
            _frames(RUN4, (285, 288, 291, 294, 297)),
        ),
        "retained_manual_mithril": (
            True,
            sorted(MANUAL_MITHRIL.glob("frame_*.png")),
        ),
        "retained_run0813_gold": (
            True,
            _frames(RUN0813_GOLD, (338, 341, 344, 347, 350, 353, 356, 359, 362)),
        ),
        "run4_hard_negatives": (
            False,
            _frames(
                RUN4,
                (80, 83, 86, 89, 92, 95, 98, 312, 318, 324, 330, 336, 342, 348),
            ),
        ),
        "run3_false_positive_negatives": (
            False,
            _frames(RUN3, (83, 86)),
        ),
    }
    if USER_SCREENSHOT.is_file():
        groups["user_close_mithril_screenshot"] = (True, [USER_SCREENSHOT])
    return groups


def evaluate_model(model_path: Path) -> dict[str, object]:
    config = copy.deepcopy(load_config())
    detector_config = config["mining"]["ore_world_detector"]
    detector_config["enabled"] = True
    detector_config["model_path"] = str(model_path)
    detector = OreWorldDetector(config)

    group_results: dict[str, object] = {}
    for name, (expected_positive, paths) in evaluation_groups().items():
        items: list[dict[str, object]] = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                items.append({"path": path.as_posix(), "reason": "missing_frame"})
                continue
            result = detector.detect(frame)
            items.append(
                {
                    "path": path.as_posix(),
                    "reason": result.reason,
                    "detections": len(result.detections),
                    "max_confidence": round(
                        max((item.confidence for item in result.detections), default=0.0),
                        6,
                    ),
                }
            )
        detected = sum(bool(item.get("detections")) for item in items)
        total = len(items)
        group_results[name] = {
            "expected_positive": expected_positive,
            "detected": detected,
            "total": total,
            "correct": detected if expected_positive else total - detected,
            "items": items,
        }
    return {"model": model_path.as_posix(), "groups": group_results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare v0.8.14 ore model candidates")
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {"models": [evaluate_model(path) for path in args.models]}
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
