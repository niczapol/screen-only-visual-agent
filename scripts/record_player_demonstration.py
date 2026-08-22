"""Passively record a human WoW route without sending any game input."""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any

from vision_bot.capture import ScreenCapture
from vision_bot.config import load_config
from vision_bot.window_video import WindowVideoRecorder


MOUSE_CODES = {
    "mouse_left": 0x01,
    "mouse_right": 0x02,
    "mouse_middle": 0x04,
    "mouse_x1": 0x05,
    "mouse_x2": 0x06,
}
SPECIAL_VK_NAMES = {
    0x08: "BACKSPACE",
    0x09: "TAB",
    0x0D: "ENTER",
    0x10: "SHIFT",
    0x11: "CTRL",
    0x12: "ALT",
    0x13: "PAUSE",
    0x14: "CAPS_LOCK",
    0x1B: "ESC",
    0x20: "SPACE",
    0x21: "PAGE_UP",
    0x22: "PAGE_DOWN",
    0x23: "END",
    0x24: "HOME",
    0x25: "LEFT",
    0x26: "UP",
    0x27: "RIGHT",
    0x28: "DOWN",
    0x2C: "PRINT_SCREEN",
    0x2D: "INSERT",
    0x2E: "DELETE",
    0x5B: "LEFT_WIN",
    0x5C: "RIGHT_WIN",
    0x5D: "APPS",
    0x90: "NUM_LOCK",
    0x91: "SCROLL_LOCK",
    0xA0: "LEFT_SHIFT",
    0xA1: "RIGHT_SHIFT",
    0xA2: "LEFT_CTRL",
    0xA3: "RIGHT_CTRL",
    0xA4: "LEFT_ALT",
    0xA5: "RIGHT_ALT",
}


def virtual_key_name(code: int) -> str:
    if code in SPECIAL_VK_NAMES:
        return SPECIAL_VK_NAMES[code]
    if ord("0") <= code <= ord("9") or ord("A") <= code <= ord("Z"):
        return chr(code)
    if 0x60 <= code <= 0x69:
        return f"NUMPAD_{code - 0x60}"
    if 0x70 <= code <= 0x87:
        return f"F{code - 0x6F}"
    return f"VK_{code:02X}"


ALL_KEY_CODES = {
    virtual_key_name(code): code
    for code in range(1, 256)
    if code not in MOUSE_CODES.values()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; zero records until Ctrl+C or stop.request.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--heartbeat", type=float, default=0.25)
    return parser.parse_args()


def build_recording_config(
    config: dict[str, Any],
    *,
    fps: float,
    max_width: int,
    segment_seconds: float,
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    video = output.setdefault("training_capture", {}).setdefault("route_live", {}).setdefault("video", {})
    video.update(
        {
            "enabled": True,
            "fps": max(1.0, float(fps)),
            "max_width": max(0, int(max_width)),
            "segment_seconds": max(5.0, float(segment_seconds)),
            "filename": "demonstration.mp4",
        }
    )
    return output


def sample_input_state(
    hwnd: int,
    key_codes: dict[str, int] | None = None,
) -> dict[str, Any]:
    user32 = ctypes.windll.user32
    tracked_keys = key_codes or ALL_KEY_CODES
    keys_down = [name for name, code in tracked_keys.items() if user32.GetAsyncKeyState(code) & 0x8000]
    mouse_down = [name for name, code in MOUSE_CODES.items() if user32.GetAsyncKeyState(code) & 0x8000]
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))

    try:
        import win32gui

        client_x, client_y = win32gui.ScreenToClient(hwnd, (point.x, point.y))
        foreground = int(win32gui.GetForegroundWindow()) == int(hwnd)
    except Exception:
        client_x, client_y = point.x, point.y
        foreground = False

    return {
        "keys_down": keys_down,
        "mouse_down": mouse_down,
        "cursor_client": [int(client_x), int(client_y)],
        "foreground": foreground,
    }


def state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(state.get("keys_down") or ()),
        tuple(state.get("mouse_down") or ()),
        tuple(state.get("cursor_client") or ()),
        bool(state.get("foreground")),
    )


def main() -> None:
    args = parse_args()
    config = build_recording_config(
        load_config(args.config),
        fps=args.fps,
        max_width=args.max_width,
        segment_seconds=args.segment_seconds,
    )
    capture = ScreenCapture(config)
    hwnd = capture.find_window()
    if hwnd is None:
        raise RuntimeError("World of Warcraft window was not found; passive recorder will not capture the desktop.")

    output_dir = Path(args.output_dir) if args.output_dir else Path("data") / (
        "human_demonstration_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    stop_path = output_dir / "stop.request"
    timeline_path = output_dir / "input_timeline.jsonl"
    manifest_path = output_dir / "manifest.json"
    recorder = WindowVideoRecorder(config, target_hwnd=int(hwnd), output_dir=output_dir)

    started_at = time.time()
    deadline = started_at + args.duration if args.duration > 0 else None
    last_signature: tuple[Any, ...] | None = None
    last_write_at = 0.0
    input_rows = 0
    recorder.start()
    try:
        with timeline_path.open("w", encoding="utf-8") as timeline:
            while deadline is None or time.time() < deadline:
                if stop_path.exists():
                    break
                state = sample_input_state(int(hwnd))
                now = time.time()
                signature = state_signature(state)
                if signature != last_signature or now - last_write_at >= max(0.05, args.heartbeat):
                    timeline.write(json.dumps({"timestamp": now, **state}) + "\n")
                    timeline.flush()
                    input_rows += 1
                    last_signature = signature
                    last_write_at = now
                time.sleep(max(0.005, args.poll_interval))
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        stop_path.unlink(missing_ok=True)

    manifest = {
        "mode": "passive_human_demonstration",
        "sends_input": False,
        "window_handle": int(hwnd),
        "started_at": started_at,
        "finished_at": time.time(),
        "input_timeline": str(timeline_path),
        "input_rows": input_rows,
        "video_summary": str(recorder.summary_path),
        "tracked_keys": list(ALL_KEY_CODES),
        "stop_file": str(stop_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
