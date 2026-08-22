import sys
import types

import pytest

from vision_bot.capture import ScreenCapture, _normalize_match_values


def test_normalize_match_values_accepts_string_and_list():
    assert _normalize_match_values("World of Warcraft Retail") == ["world of warcraft retail"]
    assert _normalize_match_values(["GameClient.exe", "wow.exe"]) == ["gameclient.exe", "wow.exe"]


def test_find_window_prefers_process_name_order(monkeypatch):
    capture = _capture_without_mss(
        {
            "window": {
                "title_contains": "LegacyClient",
                "process_name": ["GameClientPreferred.exe", "GameClient.exe"],
            }
        }
    )
    _install_fake_windows_modules(
        monkeypatch,
        [
            {"hwnd": 100, "pid": 10, "title": "LegacyClient", "visible": True, "rect": (0, 0, 2560, 1440)},
            {"hwnd": 200, "pid": 20, "title": "LegacyClient", "visible": True, "rect": (0, 0, 2560, 1440)},
        ],
        {10: "GameClient.exe", 20: "GameClientPreferred.exe"},
    )

    assert capture.find_window() == 200


def test_find_window_prefers_non_elevated_candidate_with_same_process_rank(monkeypatch):
    capture = _capture_without_mss(
        {
            "window": {
                "title_contains": "LegacyClient",
                "process_name": ["GameClient.exe"],
            }
        }
    )
    _install_fake_windows_modules(
        monkeypatch,
        [
            {"hwnd": 100, "pid": 10, "title": "LegacyClient", "visible": True, "rect": (0, 0, 2560, 1440)},
            {"hwnd": 200, "pid": 20, "title": "LegacyClient", "visible": True, "rect": (0, 0, 2560, 1440)},
        ],
        {10: "GameClient.exe", 20: "GameClient.exe"},
        elevated_pids={10},
    )

    assert capture.find_window() == 200


def test_capture_target_rejects_desktop_fallback():
    capture = _capture_without_mss(
        {
            "window": {"title_contains": "LegacyClient"},
            "capture": {"fail_closed": True},
        }
    )

    with pytest.raises(RuntimeError, match="not selected"):
        capture.ensure_capture_target()


def test_capture_target_rejects_occluding_foreground_window(monkeypatch):
    capture = _capture_without_mss(
        {
            "window": {"title_contains": "LegacyClient"},
            "capture": {"fail_closed": True, "require_foreground": True},
        }
    )
    capture.window_handle = 100
    _install_fake_windows_modules(
        monkeypatch,
        [{"hwnd": 100, "pid": 10, "title": "LegacyClient", "visible": True, "rect": (0, 0, 1920, 1080)}],
        {10: "GameClient.exe"},
        foreground_hwnd=999,
    )

    with pytest.raises(RuntimeError, match="not foreground"):
        capture.ensure_capture_target()


def test_capture_target_accepts_matching_foreground_window(monkeypatch):
    capture = _capture_without_mss(
        {
            "window": {"title_contains": "LegacyClient"},
            "capture": {"fail_closed": True, "require_foreground": True},
        }
    )
    capture.window_handle = 100
    _install_fake_windows_modules(
        monkeypatch,
        [{"hwnd": 100, "pid": 10, "title": "LegacyClient", "visible": True, "rect": (0, 0, 1920, 1080)}],
        {10: "GameClient.exe"},
        foreground_hwnd=100,
    )

    capture.ensure_capture_target()


def test_activate_window_uses_thread_attachment_fallback(monkeypatch):
    capture = _capture_without_mss({})
    capture.window_handle = 100
    fake_win32gui = types.SimpleNamespace(
        GetForegroundWindow=lambda: 999,
        ShowWindow=lambda *_args: None,
        SetForegroundWindow=lambda *_args: None,
    )
    fake_win32con = types.SimpleNamespace(SW_RESTORE=9)
    monkeypatch.setitem(sys.modules, "win32gui", fake_win32gui)
    monkeypatch.setitem(sys.modules, "win32con", fake_win32con)
    called = []
    monkeypatch.setattr(
        "vision_bot.capture._attach_and_activate_window",
        lambda hwnd: called.append(hwnd) is None,
    )

    assert capture.activate_window()
    assert called == [100]


def _capture_without_mss(config):
    capture = ScreenCapture.__new__(ScreenCapture)
    capture.config = config
    capture.sct = None
    capture._monitor = None
    capture.window_handle = None
    return capture


def _install_fake_windows_modules(
    monkeypatch,
    windows,
    process_names,
    elevated_pids=None,
    foreground_hwnd=None,
):
    elevated_pids = elevated_pids or set()
    windows_by_handle = {window["hwnd"]: window for window in windows}

    fake_win32gui = types.SimpleNamespace()
    fake_win32process = types.SimpleNamespace()

    def enum_windows(callback, extra):
        for window in windows:
            if callback(window["hwnd"], extra) is False:
                break

    fake_win32gui.EnumWindows = enum_windows
    fake_win32gui.IsWindowVisible = lambda hwnd: windows_by_handle[hwnd]["visible"]
    fake_win32gui.IsWindow = lambda hwnd: hwnd in windows_by_handle
    fake_win32gui.IsIconic = lambda _hwnd: False
    fake_win32gui.GetWindowText = lambda hwnd: windows_by_handle[hwnd]["title"]
    fake_win32gui.GetWindowRect = lambda hwnd: windows_by_handle[hwnd]["rect"]
    fake_win32gui.GetForegroundWindow = lambda: foreground_hwnd
    fake_win32process.GetWindowThreadProcessId = lambda hwnd: (
        1,
        windows_by_handle[hwnd]["pid"],
    )

    monkeypatch.setitem(sys.modules, "win32gui", fake_win32gui)
    monkeypatch.setitem(sys.modules, "win32process", fake_win32process)
    monkeypatch.setattr(
        ScreenCapture,
        "_get_process_name",
        lambda _self, pid: process_names[pid],
    )
    monkeypatch.setattr(
        "vision_bot.capture._target_requires_elevation",
        lambda pid: pid in elevated_pids,
    )
