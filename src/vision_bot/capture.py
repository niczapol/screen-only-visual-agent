from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Any

import cv2
import numpy as np
from mss import MSS

from .regions import crop_region


class ScreenCapture:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.sct = MSS()
        self._monitor = None
        self.window_handle = None

    def find_window(self) -> Any:
        try:
            import win32gui
            import win32process
        except ImportError:
            return None

        window_config = self.config.get("window", {})
        process_names = _normalize_match_values(window_config.get("process_name", "wow.exe"))
        title_matches = _normalize_match_values(window_config.get("title_contains", "World of Warcraft"))

        candidates: list[dict[str, Any]] = []

        def callback(hwnd, _extra):
            if not win32gui.IsWindowVisible(hwnd):
                return True

            title = win32gui.GetWindowText(hwnd).lower()
            if title_matches and not any(title_match in title for title_match in title_matches):
                return True

            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                return True

            process_name_actual = self._get_process_name(pid).lower()
            process_rank = _process_match_rank(process_name_actual, process_names)
            if process_names and process_rank is None:
                return True

            candidates.append(
                {
                    "hwnd": hwnd,
                    "pid": pid,
                    "process_rank": process_rank or 0,
                    "requires_elevation": _target_requires_elevation(pid),
                    "area": _window_area(win32gui, hwnd),
                }
            )

            return True

        try:
            win32gui.EnumWindows(callback, None)
        except Exception:
            if self.window_handle is None:
                raise
        else:
            if candidates:
                candidates.sort(
                    key=lambda item: (
                        item["process_rank"],
                        1 if item["requires_elevation"] else 0,
                        -item["area"],
                    )
                )
                selected_handle = candidates[0]["hwnd"]
                if selected_handle != self.window_handle:
                    self._monitor = None
                self.window_handle = selected_handle
            else:
                self.window_handle = None
                self._monitor = None
        return self.window_handle

    def capture_client_region(self, region: dict[str, int] | None = None) -> np.ndarray:
        if region is None:
            self.ensure_capture_target()
        monitor = region or self._resolve_monitor_region()
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def ensure_capture_target(self) -> None:
        capture_config = self.config.get("capture", {})
        if not bool(capture_config.get("fail_closed", bool(self.config.get("window")))):
            return
        if self.window_handle is None:
            raise RuntimeError("Game window is not selected; refusing desktop fallback capture")

        try:
            import win32gui
        except ImportError as exc:
            raise RuntimeError("win32gui is required to validate the game capture target") from exc

        hwnd = self.window_handle
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            self.window_handle = None
            self._monitor = None
            raise RuntimeError("Selected game window is no longer available")
        if hasattr(win32gui, "IsIconic") and win32gui.IsIconic(hwnd):
            raise RuntimeError("Selected game window is minimized; refusing desktop capture")

        title_matches = _normalize_match_values(
            self.config.get("window", {}).get("title_contains", "World of Warcraft")
        )
        title = win32gui.GetWindowText(hwnd).lower()
        if title_matches and not any(title_match in title for title_match in title_matches):
            self.window_handle = None
            self._monitor = None
            raise RuntimeError("Selected window no longer matches the configured game title")

        if bool(capture_config.get("require_foreground", True)):
            foreground = win32gui.GetForegroundWindow()
            if foreground != hwnd:
                raise RuntimeError(
                    "Game window is not foreground; refusing an occluded desktop capture"
                )

    def client_to_screen_point(self, x: int, y: int) -> tuple[int, int]:
        monitor = self._resolve_monitor_region()
        return int(monitor.get("left", 0)) + int(x), int(monitor.get("top", 0)) + int(y)

    def activate_window(self) -> bool:
        if self.window_handle is None:
            return False
        try:
            import win32gui
            import win32con
        except ImportError:
            return False

        try:
            if win32gui.GetForegroundWindow() == self.window_handle:
                return True
            win32gui.ShowWindow(self.window_handle, win32con.SW_RESTORE)
            time.sleep(0.10)
            win32gui.SetForegroundWindow(self.window_handle)
            time.sleep(0.20)
            if win32gui.GetForegroundWindow() == self.window_handle:
                return True
        except Exception:
            pass
        return _attach_and_activate_window(self.window_handle)

    def _resolve_monitor_region(self) -> dict[str, int]:
        if self._monitor is not None:
            return self._monitor

        if self.window_handle is not None:
            try:
                import win32gui
            except ImportError:
                return self._default_monitor_region()

            left, top, right, bottom = win32gui.GetWindowRect(self.window_handle)
            width = max(1, right - left)
            height = max(1, bottom - top)
            self._monitor = {"left": left, "top": top, "width": width, "height": height}
            return self._monitor

        if bool(self.config.get("capture", {}).get("fail_closed", bool(self.config.get("window")))):
            raise RuntimeError("Game window is not selected; refusing desktop fallback capture")
        return self._default_monitor_region()

    def _default_monitor_region(self) -> dict[str, int]:
        self._monitor = {
            "left": 0,
            "top": 0,
            "width": 1920,
            "height": 1080,
        }
        return self._monitor

    def _get_process_name(self, pid: int) -> str:
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            return ""

        if not output:
            return ""
        image_name = output.split(",", 1)[0].strip('"')
        return image_name

    def crop_minimap(self, frame: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
        cfg = config or self.config
        minimap_cfg = cfg.get("minimap", {})
        return crop_region(frame, minimap_cfg, cfg)

    def save_frame(self, frame: np.ndarray, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return path


def _normalize_match_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).lower() for item in value if str(item)]
    return [str(value).lower()]


def _attach_and_activate_window(hwnd: int) -> bool:
    """Use Win32 thread attachment when foreground-lock blocks SetForegroundWindow."""
    try:
        import ctypes
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError:
        return False

    attached_threads: list[int] = []
    current_thread = int(win32api.GetCurrentThreadId())
    try:
        foreground = int(win32gui.GetForegroundWindow())
        foreground_thread = (
            int(win32process.GetWindowThreadProcessId(foreground)[0]) if foreground else 0
        )
        target_thread = int(win32process.GetWindowThreadProcessId(hwnd)[0])
        for thread_id in (foreground_thread, target_thread):
            if thread_id and thread_id != current_thread and thread_id not in attached_threads:
                if ctypes.windll.user32.AttachThreadInput(current_thread, thread_id, True):
                    attached_threads.append(thread_id)

        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.BringWindowToTop(hwnd)
        try:
            win32gui.SetActiveWindow(hwnd)
        except Exception:
            pass
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        return False
    finally:
        for thread_id in reversed(attached_threads):
            try:
                ctypes.windll.user32.AttachThreadInput(current_thread, thread_id, False)
            except Exception:
                pass

    time.sleep(0.20)
    try:
        return int(win32gui.GetForegroundWindow()) == int(hwnd)
    except Exception:
        return False


def _process_match_rank(process_name_actual: str, process_names: list[str]) -> int | None:
    if not process_names:
        return 0
    for index, process_name in enumerate(process_names):
        if process_name and (
            process_name_actual == process_name or process_name_actual.endswith(process_name)
        ):
            return index
    return None


def _target_requires_elevation(pid: int) -> bool:
    try:
        from .windows_security import target_requires_elevation

        return bool(target_requires_elevation(pid))
    except Exception:
        return False


def _window_area(win32gui: Any, hwnd: Any) -> int:
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    except Exception:
        return 0
    return max(0, int(right) - int(left)) * max(0, int(bottom) - int(top))
