from __future__ import annotations

import argparse

from vision_bot.cursor_classifier import promote_cursor_calibration_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote one visually reviewed cursor sample into the runtime manifest."
    )
    parser.add_argument("sample", help="Path to the reviewed calibration JSON")
    parser.add_argument(
        "--manifest",
        default="data/cursor_templates/cursor_templates.json",
    )
    parser.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="Required acknowledgement that the PNG was manually checked",
    )
    args = parser.parse_args()
    if not args.confirm_reviewed:
        parser.error("--confirm-reviewed is required")
    added = promote_cursor_calibration_sample(args.sample, args.manifest)
    print("promoted" if added else "already_present")


if __name__ == "__main__":
    main()
