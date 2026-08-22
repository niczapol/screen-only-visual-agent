from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import ScreenCapture
from .config import load_config
from .recognition import recognize_ore_points, save_debug_artifacts


def run_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    capture = ScreenCapture(config)
    capture.find_window()
    frame = capture.capture_client_region()
    minimap = capture.crop_minimap(frame, config)
    points = recognize_ore_points(minimap, config)

    output_dir = Path(config.get("debug", {}).get("output_dir", "debug_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_debug_artifacts(minimap, points, output_dir, config)
    cv2.imwrite(str(output_dir / "minimap.png"), minimap)

    if config.get("debug", {}).get("enabled", True):
        debug_image = cv2.imread(str(artifacts["debug"]), cv2.IMREAD_COLOR)
        if debug_image is not None:
            try:
                if cv2.getWindowProperty("vision_debug", cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow("vision_debug")
            except cv2.error:
                pass
            cv2.namedWindow("vision_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("vision_debug", debug_image)
            cv2.waitKey(1)

    return {"points": points, "frame_shape": minimap.shape, "config": config}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the vision bot pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
