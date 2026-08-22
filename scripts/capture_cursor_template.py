from __future__ import annotations

import argparse
import json

from vision_bot.cursor_classifier import save_cursor_calibration_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture the currently visible Win32 cursor for later human review."
    )
    parser.add_argument("label", choices=("mine", "loot", "neutral"))
    parser.add_argument("--output-dir", default="artifacts/cursor_calibration")
    args = parser.parse_args()
    print(
        json.dumps(
            save_cursor_calibration_sample(args.output_dir, label=args.label),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
