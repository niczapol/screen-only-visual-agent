from __future__ import annotations

import ctypes
import math
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Literal

from vision_bot.coords import coord_to_xy
from vision_bot.heading_navigation import HeadingNavigationCommand


INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
ULONG_PTR = wintypes.WPARAM

KEY_CODES = {
    **{chr(code): code for code in range(ord("A"), ord("Z") + 1)},
    **{str(digit): ord(str(digit)) for digit in range(10)},
    "TAB": 0x09,
    "ENTER": 0x0D,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "F6": 0x75,
    "F7": 0x76,
    "F8": 0x77,
    "F9": 0x78,
    "SHIFT": 0x10,
    "SPACE": 0x20,
    "/": 0xBF,
}

MOVEMENT_KEYS = ("W", "A", "S", "D")


@dataclass(frozen=True)
class MovementDiagnostics:
    action: str = "idle"
    distance_to_target: float | None = None
    progress: float | None = None
    distance_progress: float | None = None
    alignment: float | None = None
    turn_key: str | None = None
    held_key: str | None = None
    stuck_samples: int = 0
    forward_vector: tuple[float, float] | None = None
    target_coord: int | None = None
    coord_delta: float | None = None
    movement_delta: float | None = None
    detour_phase: str | None = None
    detour_turn_key: str | None = None
    detour_duration: float | None = None
    held_keys: tuple[str, ...] = ()
    desired_heading_degrees: float | None = None
    heading_error_degrees: float | None = None
    heading_braking: bool = False


@dataclass(frozen=True)
class ObstacleRecovery:
    action: str
    turn_key: Literal["A", "D"] | None


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


def _send_input(inputs: list[INPUT]) -> int:
    if not inputs:
        return 0
    input_array = (INPUT * len(inputs))(*inputs)
    return ctypes.windll.user32.SendInput(len(input_array), input_array, ctypes.sizeof(INPUT))


def _make_keyboard_input(vk_code: int, flags: int = 0) -> INPUT:
    scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)
    ki = KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags | KEYEVENTF_SCANCODE, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_KEYBOARD, ki=ki)


def _make_mouse_input(flags: int, *, dx: int = 0, dy: int = 0) -> INPUT:
    mi = MOUSEINPUT(
        dx=int(dx),
        dy=int(dy),
        mouseData=0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return INPUT(type=INPUT_MOUSE, mi=mi)


def _post_key_message(target_hwnd: int | None, vk_code: int, *, key_up: bool) -> int:
    if target_hwnd is None:
        return 0
    scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)
    lparam = 1 | (scan_code << 16)
    if key_up:
        lparam |= 1 << 30
        lparam |= 1 << 31
    message = WM_KEYUP if key_up else WM_KEYDOWN
    return ctypes.windll.user32.PostMessageW(int(target_hwnd), message, vk_code, lparam)


class InputController:
    def __init__(self, target_hwnd: int | None = None, backend: str = "sendinput") -> None:
        self.target_hwnd = target_hwnd
        self.backend = backend.lower()

    def set_target_window(self, target_hwnd: int | None) -> None:
        self.target_hwnd = target_hwnd

    def press_key(self, key: str) -> None:
        vk = KEY_CODES.get(key.upper())
        if vk is None:
            return
        if self._uses_window_messages():
            _post_key_message(self.target_hwnd, vk, key_up=False)
            return
        _send_input([_make_keyboard_input(vk, 0)])

    def release_key(self, key: str) -> None:
        vk = KEY_CODES.get(key.upper())
        if vk is None:
            return
        if self._uses_window_messages():
            _post_key_message(self.target_hwnd, vk, key_up=True)
            return
        _send_input([_make_keyboard_input(vk, KEYEVENTF_KEYUP)])

    def tap_key(self, key: str, duration: float = 0.15) -> None:
        self.press_key(key)
        time.sleep(duration)
        self.release_key(key)

    def release_movement_keys(self) -> None:
        for key in MOVEMENT_KEYS:
            self.release_key(key)

    def _uses_window_messages(self) -> bool:
        return self.target_hwnd is not None and self.backend in {"post_message", "window", "hwnd"}


class MouseController:
    def __init__(self, target_hwnd: int | None = None) -> None:
        self._lock = threading.RLock()
        self.target_hwnd = target_hwnd

    def set_target_window(self, target_hwnd: int | None) -> None:
        self.target_hwnd = target_hwnd

    def move_to(self, screen_x: int, screen_y: int) -> None:
        with self._lock:
            ctypes.windll.user32.SetCursorPos(int(screen_x), int(screen_y))

    def button_down(self, button: str) -> None:
        """Non-blocking primitive used only by the v0.9 input executor."""
        normalized = str(button).strip().lower()
        flags = {
            "left": MOUSEEVENTF_LEFTDOWN,
            "right": MOUSEEVENTF_RIGHTDOWN,
        }.get(normalized)
        if flags is None:
            raise ValueError(f"unsupported mouse button: {button}")
        with self._lock:
            _mouse_event(flags)

    def button_up(self, button: str) -> None:
        """Release a mouse button without sleeping."""
        normalized = str(button).strip().lower()
        flags = {
            "left": MOUSEEVENTF_LEFTUP,
            "right": MOUSEEVENTF_RIGHTUP,
        }.get(normalized)
        if flags is None:
            raise ValueError(f"unsupported mouse button: {button}")
        with self._lock:
            _mouse_event(flags)

    def move_relative(self, delta_x: int, delta_y: int = 0) -> None:
        """Move the cursor/mouselook by one non-blocking executor step."""
        with self._lock:
            _send_input(
                [
                    _make_mouse_input(
                        MOUSEEVENTF_MOVE,
                        dx=int(delta_x),
                        dy=int(delta_y),
                    )
                ]
            )

    def left_click(self, duration: float = 0.05) -> None:
        with self._lock:
            _mouse_event(MOUSEEVENTF_LEFTDOWN)
            time.sleep(max(0.0, duration))
            _mouse_event(MOUSEEVENTF_LEFTUP)

    def right_click(self, duration: float = 0.05) -> None:
        with self._lock:
            _mouse_event(MOUSEEVENTF_RIGHTDOWN)
            time.sleep(max(0.0, duration))
            _mouse_event(MOUSEEVENTF_RIGHTUP)

    def right_drag_relative(
        self,
        delta_x: int,
        *,
        duration: float,
        step_interval: float = 0.012,
        restore_cursor: bool = True,
    ) -> None:
        """Turn mouselook smoothly while guaranteeing RMB is released."""
        total_x = int(delta_x)
        if total_x == 0:
            return
        duration = max(0.0, float(duration))
        step_interval = max(0.004, float(step_interval))
        steps = max(1, int(math.ceil(duration / step_interval)))
        with self._lock:
            cursor = POINT()
            cursor_saved = bool(
                restore_cursor
                and ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
            )
            center = self._client_center_screen()
            if center is not None:
                ctypes.windll.user32.SetCursorPos(center[0], center[1])
                time.sleep(min(0.012, step_interval))
            _send_input([_make_mouse_input(MOUSEEVENTF_RIGHTDOWN)])
            started_at = time.monotonic()
            sent_x = 0
            try:
                for index in range(steps):
                    target_x = round(total_x * (index + 1) / steps)
                    step_x = target_x - sent_x
                    if step_x:
                        current = POINT()
                        if ctypes.windll.user32.GetCursorPos(ctypes.byref(current)):
                            ctypes.windll.user32.SetCursorPos(
                                current.x + step_x,
                                current.y,
                            )
                        else:
                            _send_input(
                                [_make_mouse_input(MOUSEEVENTF_MOVE, dx=step_x)]
                            )
                        sent_x = target_x
                    wake_at = started_at + duration * (index + 1) / steps
                    sleep_seconds = wake_at - time.monotonic()
                    if sleep_seconds > 0.0:
                        time.sleep(sleep_seconds)
            finally:
                _send_input([_make_mouse_input(MOUSEEVENTF_RIGHTUP)])
                if cursor_saved:
                    time.sleep(min(0.012, step_interval))
                    ctypes.windll.user32.SetCursorPos(cursor.x, cursor.y)

    def right_drag_continuous(
        self,
        delta_x_per_step: int,
        *,
        stop_event: threading.Event,
        step_interval: float = 0.012,
        restore_cursor: bool = True,
    ) -> tuple[int, float]:
        """Hold RMB and move the camera until stop_event is set."""
        step_x = int(delta_x_per_step)
        if step_x == 0:
            return 0, 0.0
        step_interval = max(0.004, float(step_interval))
        sent_x = 0
        with self._lock:
            cursor = POINT()
            cursor_saved = bool(
                restore_cursor
                and ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
            )
            center = self._client_center_screen()
            if center is not None:
                ctypes.windll.user32.SetCursorPos(center[0], center[1])
                time.sleep(min(0.012, step_interval))
            _send_input([_make_mouse_input(MOUSEEVENTF_RIGHTDOWN)])
            started_at = time.monotonic()
            try:
                while not stop_event.is_set():
                    current = POINT()
                    if ctypes.windll.user32.GetCursorPos(ctypes.byref(current)):
                        ctypes.windll.user32.SetCursorPos(
                            current.x + step_x,
                            current.y,
                        )
                    else:
                        _send_input([_make_mouse_input(MOUSEEVENTF_MOVE, dx=step_x)])
                    sent_x += step_x
                    stop_event.wait(step_interval)
            finally:
                _send_input([_make_mouse_input(MOUSEEVENTF_RIGHTUP)])
                if cursor_saved:
                    time.sleep(min(0.012, step_interval))
                    ctypes.windll.user32.SetCursorPos(cursor.x, cursor.y)
        return sent_x, max(0.0, time.monotonic() - started_at)

    def _client_center_screen(self) -> tuple[int, int] | None:
        if self.target_hwnd is None:
            return None
        rect = RECT()
        if not ctypes.windll.user32.GetClientRect(
            int(self.target_hwnd),
            ctypes.byref(rect),
        ):
            return None
        point = POINT(
            max(0, (rect.right - rect.left) // 2),
            max(0, (rect.bottom - rect.top) // 2),
        )
        if not ctypes.windll.user32.ClientToScreen(
            int(self.target_hwnd),
            ctypes.byref(point),
        ):
            return None
        return point.x, point.y


@dataclass(frozen=True)
class MouseSteeringResult:
    sequence: int
    context: str
    turn_key: Literal["A", "D"]
    duration: float
    pixels: int
    pixels_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "context": self.context,
            "turn_key": self.turn_key,
            "duration": self.duration,
            "pixels": self.pixels,
            "pixels_per_second": self.pixels_per_second,
        }


class MouseSteeringController:
    """Translate logical A/D directions into bounded smooth RMB yaw drags."""

    def __init__(
        self,
        mouse_controller: MouseController,
        *,
        enabled: bool,
        pixels_per_second: dict[str, float] | None = None,
        step_interval_seconds: float = 0.012,
        min_duration_seconds: float = 0.04,
        max_duration_seconds: float = 0.40,
        min_pixels: int = 2,
        restore_cursor: bool = True,
        route_moving_max_duration_seconds: float = 0.10,
    ) -> None:
        self.mouse_controller = mouse_controller
        self.enabled = bool(enabled)
        self.pixels_per_second = {
            str(key): max(1.0, float(value))
            for key, value in (pixels_per_second or {"default": 120.0}).items()
        }
        self.step_interval_seconds = max(0.004, float(step_interval_seconds))
        self.min_duration_seconds = max(0.0, float(min_duration_seconds))
        self.max_duration_seconds = max(
            self.min_duration_seconds,
            float(max_duration_seconds),
        )
        self.min_pixels = max(1, int(min_pixels))
        self.restore_cursor = bool(restore_cursor)
        self.route_moving_max_duration_seconds = max(
            self.min_duration_seconds,
            float(route_moving_max_duration_seconds),
        )
        self.turn_count = 0
        self.last_result: MouseSteeringResult | None = None
        self._continuous_lock = threading.RLock()
        self._continuous_stop: threading.Event | None = None
        self._continuous_thread: threading.Thread | None = None
        self._continuous_turn_key: Literal["A", "D"] | None = None

    @classmethod
    def from_config(
        cls,
        mouse_controller: MouseController,
        config: dict[str, Any],
    ) -> "MouseSteeringController":
        movement_cfg = config.get("movement", {})
        cfg = movement_cfg.get("mouse_steering", {})
        if not isinstance(cfg, dict):
            cfg = {}
        raw_speeds = cfg.get("pixels_per_second", {})
        speeds = raw_speeds if isinstance(raw_speeds, dict) else {}
        return cls(
            mouse_controller,
            enabled=bool(cfg.get("enabled", False)),
            pixels_per_second={
                "default": float(speeds.get("default", 120.0)),
                "route": float(speeds.get("route", speeds.get("default", 120.0))),
                "combat": float(speeds.get("combat", speeds.get("default", 120.0))),
                "spirit": float(speeds.get("spirit", speeds.get("default", 120.0))),
                "mining": float(speeds.get("mining", speeds.get("default", 120.0))),
                "local": float(speeds.get("local", speeds.get("default", 120.0))),
                "recovery": float(speeds.get("recovery", speeds.get("default", 120.0))),
                "hostile": float(speeds.get("hostile", speeds.get("default", 120.0))),
            },
            step_interval_seconds=float(cfg.get("step_interval_seconds", 0.012)),
            min_duration_seconds=float(cfg.get("min_duration_seconds", 0.04)),
            max_duration_seconds=float(cfg.get("max_duration_seconds", 0.40)),
            min_pixels=int(cfg.get("min_pixels", 2)),
            restore_cursor=bool(cfg.get("restore_cursor", True)),
            route_moving_max_duration_seconds=float(
                cfg.get("route_moving_max_duration_seconds", 0.10)
            ),
        )

    def turn(
        self,
        turn_key: str | None,
        *,
        duration: float,
        context: str = "default",
        max_duration: float | None = None,
    ) -> bool:
        key = str(turn_key or "").upper()
        if not self.enabled or key not in {"A", "D"}:
            return False
        self.stop_continuous_turn()
        duration_limit = self.max_duration_seconds
        if max_duration is not None:
            duration_limit = min(duration_limit, max(0.0, float(max_duration)))
        bounded_duration = min(
            max(self.min_duration_seconds, duration_limit),
            max(self.min_duration_seconds, float(duration)),
        )
        speed = self.pixels_per_second.get(
            context,
            self.pixels_per_second.get("default", 120.0),
        )
        magnitude = max(self.min_pixels, round(speed * bounded_duration))
        pixels = -magnitude if key == "A" else magnitude
        self.mouse_controller.right_drag_relative(
            pixels,
            duration=bounded_duration,
            step_interval=self.step_interval_seconds,
            restore_cursor=self.restore_cursor,
        )
        self.turn_count += 1
        self.last_result = MouseSteeringResult(
            sequence=self.turn_count,
            context=context,
            turn_key=key,
            duration=bounded_duration,
            pixels=pixels,
            pixels_per_second=speed,
        )
        return True

    def start_continuous_turn(
        self,
        turn_key: str | None,
        *,
        context: str = "default",
    ) -> bool:
        key = str(turn_key or "").upper()
        if not self.enabled or key not in {"A", "D"}:
            return False
        with self._continuous_lock:
            running_thread = self._continuous_thread
            running_key = self._continuous_turn_key
            if running_thread is not None and running_thread.is_alive() and running_key == key:
                return True
        if running_thread is not None and running_thread.is_alive():
            self.stop_continuous_turn()
        with self._continuous_lock:
            speed = self.pixels_per_second.get(
                context,
                self.pixels_per_second.get("default", 120.0),
            )
            magnitude = max(self.min_pixels, round(speed * self.step_interval_seconds))
            step_pixels = -magnitude if key == "A" else magnitude
            stop_event = threading.Event()
            self._continuous_stop = stop_event
            self._continuous_turn_key = key  # type: ignore[assignment]

            def _worker() -> None:
                sent_pixels, elapsed = self.mouse_controller.right_drag_continuous(
                    step_pixels,
                    stop_event=stop_event,
                    step_interval=self.step_interval_seconds,
                    restore_cursor=self.restore_cursor,
                )
                with self._continuous_lock:
                    self.turn_count += 1
                    self.last_result = MouseSteeringResult(
                        sequence=self.turn_count,
                        context=context,
                        turn_key=key,  # type: ignore[arg-type]
                        duration=elapsed,
                        pixels=sent_pixels,
                        pixels_per_second=speed,
                    )

            self._continuous_thread = threading.Thread(
                target=_worker,
                name=f"MouseSteering-{context}-{key}",
                daemon=True,
            )
            self._continuous_thread.start()
            return True

    def stop_continuous_turn(self) -> None:
        with self._continuous_lock:
            stop_event = self._continuous_stop
            thread = self._continuous_thread
            self._continuous_stop = None
            self._continuous_thread = None
            self._continuous_turn_key = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    @property
    def continuous_turn_key(self) -> Literal["A", "D"] | None:
        with self._continuous_lock:
            thread = self._continuous_thread
            if thread is None or not thread.is_alive():
                return None
            return self._continuous_turn_key


def _mouse_event(flags: int) -> None:
    ctypes.windll.user32.mouse_event(flags, 0, 0, 0, 0)


class MovementNavigator:
    def __init__(
        self,
        input_controller: InputController,
        *,
        stuck_check_window: int = 4,
        stuck_min_progress: float = 0.03,
        stuck_min_coord_delta: float | None = None,
        obstacle_back_duration: float = 0.35,
        obstacle_strafe_duration: float = 0.55,
        obstacle_turn_duration: float | None = None,
        obstacle_jump_duration: float = 0.08,
        obstacle_jump_first: bool = True,
        obstacle_detour_duration: float = 0.35,
        obstacle_detour_max_duration: float = 0.65,
        obstacle_detour_settle_duration: float = 0.0,
        obstacle_recovery_attempts_per_side: int = 4,
        turn_duration: float = 0.12,
        turn_in_place_duration: float | None = None,
        turn_in_place_alignment: float = -0.2,
        turn_alignment_threshold: float = 0.92,
        invert_turn_direction: bool = False,
        mouse_steering: MouseSteeringController | None = None,
    ) -> None:
        self.input_controller = input_controller
        self.key_vectors: dict[str, tuple[float, float]] = {}
        self.forward_vector: tuple[float, float] | None = None
        self.held_key: str | None = None
        self.heading_held_keys: set[str] = set()
        self._heading_lock = threading.RLock()
        self._heading_turn_timer: threading.Timer | None = None
        self._heading_turn_generation = 0
        self._last_observed_coord: int | None = None
        self._last_observed_key: str | None = None
        self._last_distance_to_target: float | None = None
        self._last_target_coord: int | None = None
        self._course_correction_turn_key: Literal["A", "D"] | None = None
        self._last_progress: float | None = None
        self._last_alignment: float | None = None
        self._last_coord_delta: float | None = None
        self._stuck_samples = 0
        self._target_aligned = False
        self._recovery_attempts = 0
        self._avoid_right = True
        self._recovery_lock = threading.RLock()
        self._recovery_detour_turn_key: Literal["A", "D"] | None = None
        self._recovery_committed_turn_key: Literal["A", "D"] | None = None
        self._recovery_detour_turn_duration = 0.0
        self._recovery_detour_duration = 0.0
        self._recovery_detour_deadline: float | None = None
        self._recovery_detour_settle_until = 0.0
        self._recovery_realign_completed_key: Literal["A", "D"] | None = None
        self._recovery_realign_completed_duration: float | None = None
        self.mouse_steering = mouse_steering
        self.last_diagnostics = MovementDiagnostics()
        self.stuck_check_window = max(1, stuck_check_window)
        self.stuck_min_progress = max(0.0, stuck_min_progress)
        self.stuck_min_coord_delta = max(0.0, stuck_min_progress if stuck_min_coord_delta is None else stuck_min_coord_delta)
        self.obstacle_back_duration = max(0.0, obstacle_back_duration)
        turn_recovery_duration = obstacle_strafe_duration if obstacle_turn_duration is None else obstacle_turn_duration
        self.obstacle_turn_duration = max(0.0, turn_recovery_duration)
        self.obstacle_strafe_duration = self.obstacle_turn_duration
        self.obstacle_jump_duration = max(0.0, obstacle_jump_duration)
        self.obstacle_jump_first = obstacle_jump_first
        self.obstacle_detour_duration = max(0.0, obstacle_detour_duration)
        self.obstacle_detour_max_duration = max(
            self.obstacle_detour_duration,
            obstacle_detour_max_duration,
        )
        self.obstacle_detour_settle_duration = max(0.0, obstacle_detour_settle_duration)
        self.obstacle_recovery_attempts_per_side = max(
            1, int(obstacle_recovery_attempts_per_side)
        )
        self.turn_duration = max(0.0, turn_duration)
        self.turn_in_place_duration = max(
            0.0,
            turn_duration if turn_in_place_duration is None else turn_in_place_duration,
        )
        self.turn_in_place_alignment = max(-1.0, min(1.0, turn_in_place_alignment))
        self.turn_alignment_threshold = max(-1.0, min(1.0, turn_alignment_threshold))
        self.invert_turn_direction = invert_turn_direction

    def choose_direction(self, current_coord: int, target_coord: int) -> Literal["W", "A", "S", "D"]:
        turn_key = self.choose_turn_key(current_coord, target_coord)
        return turn_key if turn_key is not None else "W"

    def move_towards(self, current_coord: int, target_coord: int, duration: float = 0.25) -> None:
        self.release_heading_control()
        target_changed = self._last_target_coord != target_coord
        if target_changed:
            self._cancel_recovery_detour(clear_settle=True)
        self.observe_position(current_coord)
        if target_changed:
            self._last_target_coord = target_coord
            self._last_distance_to_target = None
            self._course_correction_turn_key = None
            self._last_progress = None
            self._stuck_samples = 0
            self._target_aligned = False
            self._recovery_attempts = 0
            self._recovery_committed_turn_key = None

        self._finish_recovery_detour_if_due()

        distance_to_target = self.distance(current_coord, target_coord)
        candidate_turn_key = self.choose_turn_key(current_coord, target_coord)
        is_stuck = self._is_stuck(distance_to_target)
        has_fresh_progress = (
            self._last_progress is not None
            and self._last_progress > 0.0
        ) or self._is_coord_moving()
        if has_fresh_progress and self.recovery_suppresses_stuck:
            self._cancel_recovery_detour(clear_settle=True)
        elif self.recovery_detour_active:
            self._hold_key("W", current_coord)
            self._set_diagnostics(
                "recover_detour_forward",
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                turn_key=self._recovery_detour_turn_key,
                target_coord=target_coord,
            )
            return
        elif is_stuck and self.recovery_suppresses_stuck:
            self._hold_key("W", current_coord)
            self._set_diagnostics(
                "recover_detour_settle",
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                turn_key=self._recovery_realign_completed_key,
                target_coord=target_coord,
            )
            return
        turn_key = self._resolve_course_correction_turn(candidate_turn_key)
        if is_stuck:
            progress = self._last_progress
            stuck_samples = self._stuck_samples
            recovery = self.recover_from_obstacle(current_coord, preferred_turn_key=turn_key)
            if (
                self._course_correction_turn_key is None
                and self._last_progress is not None
                and self._last_progress < self.stuck_min_progress
                and recovery.turn_key in {"A", "D"}
            ):
                self._course_correction_turn_key = recovery.turn_key
            self._set_diagnostics(
                recovery.action,
                distance_to_target=distance_to_target,
                progress=progress,
                alignment=self._last_alignment,
                turn_key=recovery.turn_key,
                stuck_samples=stuck_samples,
                target_coord=target_coord,
            )
            return

        if self._last_progress is not None and self._last_progress < -self.stuck_min_progress and turn_key is not None:
            self._target_aligned = False
            self._turn_and_hold_forward(turn_key, current_coord)
            self._set_diagnostics(
                "course_correct_and_forward",
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                turn_key=turn_key,
                target_coord=target_coord,
            )
            return

        if self._last_progress is not None and self._last_progress > 0.0:
            self._target_aligned = True
            self._recovery_attempts = 0
            self._course_correction_turn_key = None
            self._hold_key("W", current_coord)
            if self._last_distance_to_target is None:
                self._last_distance_to_target = distance_to_target
            self._set_diagnostics(
                "forward_motion"
                if self._last_progress < self.stuck_min_progress and self._is_coord_moving()
                else "forward_progress",
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                target_coord=target_coord,
            )
            return

        if turn_key is not None and self._last_progress is not None and self._last_progress < self.stuck_min_progress:
            self._target_aligned = False
            action = self._turn_and_hold_forward(turn_key, current_coord)
            self._set_diagnostics(
                "course_correct_and_forward" if action == "turn_and_forward" else action,
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                turn_key=turn_key,
                target_coord=target_coord,
            )
            return

        if (
            not target_changed
            and self._is_coord_moving()
            and self._last_progress is not None
            and self._last_progress > -self.stuck_min_progress
        ):
            self._target_aligned = True
            self._recovery_attempts = 0
            self._hold_key("W", current_coord)
            if self._last_distance_to_target is None:
                self._last_distance_to_target = distance_to_target
            self._set_diagnostics(
                "forward_motion",
                distance_to_target=distance_to_target,
                progress=self._last_progress,
                alignment=self._last_alignment,
                target_coord=target_coord,
            )
            return

        if turn_key is None:
            self._target_aligned = True

        if self._last_progress is not None and self._last_progress < -self.stuck_min_progress:
            self._target_aligned = False

        if turn_key is not None and not self._target_aligned:
            action = self._turn_and_hold_forward(turn_key, current_coord)
        else:
            action = "forward"
            self._hold_key("W", current_coord)

        if self._last_distance_to_target is None:
            self._last_distance_to_target = distance_to_target
        self._set_diagnostics(
            action,
            distance_to_target=distance_to_target,
            progress=self._last_progress,
            alignment=self._last_alignment,
            turn_key=turn_key,
            target_coord=target_coord,
        )

    def continue_forward(self, current_coord: int | None = None, target_coord: int | None = None) -> None:
        self.release_heading_control()
        if self._finish_recovery_detour_if_due():
            if self.held_key is not None:
                self.input_controller.release_key(self.held_key)
                self.held_key = None
            self._set_diagnostics(
                "recover_detour_replan",
                turn_key=self._recovery_realign_completed_key,
                target_coord=target_coord,
            )
            return

        if current_coord is not None:
            self._hold_key("W", current_coord)
        elif self.held_key != "W":
            if self.held_key is not None:
                self.input_controller.release_key(self.held_key)
            self.input_controller.press_key("W")
            self.held_key = "W"
        with self._recovery_lock:
            if self._recovery_detour_turn_key is not None:
                action = "recover_detour_forward"
                turn_key = self._recovery_detour_turn_key
            elif self._recovery_realign_completed_key is not None:
                action = "recover_detour_realign"
                turn_key = self._recovery_realign_completed_key
            else:
                action = "continue_forward"
                turn_key = None
        self._set_diagnostics(action, turn_key=turn_key, target_coord=target_coord)
        if action == "recover_detour_realign":
            with self._recovery_lock:
                self._recovery_realign_completed_key = None
                self._recovery_realign_completed_duration = None

    def apply_heading_command(
        self,
        command: HeadingNavigationCommand,
        *,
        current_coord: int,
        target_coord: int,
        turn_context: str = "route",
    ) -> None:
        """Apply nonblocking W/A/D held state produced by the heading controller."""
        # Heading-control bypasses ``move_towards``/``continue_forward``. Those
        # legacy paths normally expire the short post-recovery detour and reset
        # the jump ladder after real coordinate progress. Without the same
        # maintenance here, one jump+turn can suppress every future stuck event
        # for the rest of a live run.
        self._finish_recovery_detour_if_due()
        timed_turn_key = (
            command.turn_key
            if command.turn_key is not None and command.turn_hold_seconds is not None
            else None
        )
        with self._heading_lock:
            self._cancel_heading_turn_timer_locked()
            self.observe_position(current_coord)
            if self._is_coord_moving():
                self._stuck_samples = 0
                self._recovery_attempts = 0
                self._recovery_committed_turn_key = None
                if self.recovery_suppresses_stuck:
                    self._cancel_recovery_detour(clear_settle=True)
            desired_keys: set[str] = set()
            if command.hold_forward:
                desired_keys.add("W")
            if command.turn_key is not None and timed_turn_key is None:
                desired_keys.add(command.turn_key)

            physically_held = set(self.heading_held_keys)
            if self.held_key in MOVEMENT_KEYS:
                physically_held.add(self.held_key)

            # Release stale turning first so A and D can never overlap.
            for key in ("A", "D", "W", "S"):
                if key in physically_held and key not in desired_keys:
                    self.input_controller.release_key(key)
                    physically_held.discard(key)
            for key in ("W", "A", "D"):
                if key in desired_keys and key not in physically_held:
                    self.input_controller.press_key(key)
                    physically_held.add(key)

            self.heading_held_keys = desired_keys
            self.held_key = (
                "W"
                if "W" in desired_keys
                else command.turn_key
                if command.turn_key in desired_keys
                else None
            )
        if timed_turn_key is not None:
            # Timer callbacks can be delayed while capture/CV owns the
            # interpreter. A synchronous tap guarantees the A/D release while
            # W remains independently held.
            turn_duration = max(
                0.0,
                float(command.turn_hold_seconds or 0.0),
            )
            if (
                command.hold_forward
                and self.mouse_steering is not None
                and self.mouse_steering.enabled
            ):
                turn_duration = min(
                    turn_duration,
                    self.mouse_steering.route_moving_max_duration_seconds,
                )
            self.turn_character(
                timed_turn_key,
                duration=turn_duration,
                context=turn_context,
            )
        self._last_target_coord = target_coord
        self._set_diagnostics(
            command.reason,
            distance_to_target=self.distance(current_coord, target_coord),
            turn_key=command.turn_key,
            target_coord=target_coord,
            desired_heading_degrees=command.desired_heading_degrees,
            heading_error_degrees=command.heading_error_degrees,
            heading_braking=command.braking,
        )

    def release_heading_control(self) -> None:
        with self._heading_lock:
            self._cancel_heading_turn_timer_locked()
            if not self.heading_held_keys:
                return
            for key in ("A", "D", "W", "S"):
                if key in self.heading_held_keys:
                    self.input_controller.release_key(key)
            self.heading_held_keys.clear()
            self.held_key = None

    def _schedule_heading_turn_release_locked(self, key: str, duration: float) -> None:
        self._heading_turn_generation += 1
        generation = self._heading_turn_generation
        timer = threading.Timer(
            max(0.0, float(duration)),
            self._release_heading_turn,
            args=(key, generation),
        )
        timer.daemon = True
        self._heading_turn_timer = timer
        timer.start()

    def _release_heading_turn(self, key: str, generation: int) -> None:
        with self._heading_lock:
            if generation != self._heading_turn_generation:
                return
            self._heading_turn_timer = None
            if key not in self.heading_held_keys:
                return
            self.input_controller.release_key(key)
            self.heading_held_keys.discard(key)
            self.held_key = "W" if "W" in self.heading_held_keys else None

    def _cancel_heading_turn_timer_locked(self) -> None:
        self._heading_turn_generation += 1
        timer = self._heading_turn_timer
        self._heading_turn_timer = None
        if timer is not None:
            timer.cancel()

    def observe_position(self, current_coord: int, min_delta: float = 0.01) -> None:
        if self._last_observed_coord is None:
            self._last_observed_coord = current_coord
            self._last_observed_key = self.held_key
            self._last_coord_delta = None
            return

        observed_key = self._last_observed_key
        if observed_key is None:
            self._last_observed_coord = current_coord
            self._last_observed_key = self.held_key
            self._last_coord_delta = None
            return

        previous_x, previous_y = self._decode_coord(self._last_observed_coord)
        current_x, current_y = self._decode_coord(current_coord)
        dx = current_x - previous_x
        dy = current_y - previous_y
        distance = (dx * dx + dy * dy) ** 0.5
        self._last_coord_delta = distance
        if distance >= min_delta:
            observed_vector = (dx / distance, dy / distance)
            old_vector = self.key_vectors.get(observed_key)
            old_length = math.hypot(*old_vector) if old_vector is not None else 0.0
            if old_vector is None or old_length <= 0.0:
                self.key_vectors[observed_key] = observed_vector
            else:
                normalized_old = (old_vector[0] / old_length, old_vector[1] / old_length)
                old_weight = 0.20 if observed_key == "W" else 0.70
                new_weight = 1.0 - old_weight
                blended = (
                    normalized_old[0] * old_weight + observed_vector[0] * new_weight,
                    normalized_old[1] * old_weight + observed_vector[1] * new_weight,
                )
                blended_length = math.hypot(*blended)
                self.key_vectors[observed_key] = (
                    blended[0] / blended_length,
                    blended[1] / blended_length,
                )
            if observed_key == "W":
                self.forward_vector = self.key_vectors[observed_key]
        elif observed_key not in self.key_vectors:
            self.key_vectors[observed_key] = (0.0, 0.0)

        self._last_observed_coord = current_coord
        self._last_observed_key = self.held_key

    def stop(self) -> None:
        self._cancel_recovery_detour(clear_settle=True)
        with self._heading_lock:
            self._cancel_heading_turn_timer_locked()
        self.input_controller.release_movement_keys()
        self.heading_held_keys.clear()
        self.held_key = None
        self._last_observed_coord = None
        self._last_observed_key = None
        self._last_distance_to_target = None
        self._last_target_coord = None
        self._course_correction_turn_key = None
        self._last_progress = None
        self._last_alignment = None
        self._last_coord_delta = None
        self._stuck_samples = 0
        self._target_aligned = False
        self._recovery_attempts = 0
        self._recovery_committed_turn_key = None
        self._set_diagnostics("stopped")

    def recover_from_obstacle(
        self,
        current_coord: int | None = None,
        preferred_turn_key: Literal["A", "D"] | None = None,
    ) -> ObstacleRecovery:
        self.release_heading_control()
        if self.recovery_suppresses_stuck:
            action = (
                "recover_detour_forward"
                if self.recovery_detour_active
                else "recover_detour_settle"
            )
            turn_key = (
                self._recovery_detour_turn_key
                if self.recovery_detour_active
                else self._recovery_realign_completed_key
            )
            self._set_diagnostics(
                action,
                turn_key=turn_key,
            )
            return ObstacleRecovery(action, turn_key)

        if self.obstacle_jump_first and self._recovery_attempts == 0:
            if current_coord is not None:
                self._hold_key("W", current_coord)
            elif self.held_key != "W":
                if self.held_key is not None:
                    self.input_controller.release_key(self.held_key)
                self.input_controller.press_key("W")
                self.held_key = "W"
            self.input_controller.tap_key("SPACE", duration=self.obstacle_jump_duration)
            self._recovery_attempts += 1
            self._set_diagnostics("recover_jump")
            return ObstacleRecovery("recover_jump", None)

        if self._recovery_committed_turn_key in {"A", "D"}:
            turn_key = self._recovery_committed_turn_key
        elif preferred_turn_key is None:
            turn_key: Literal["A", "D"] = "D" if self._avoid_right else "A"
            self._avoid_right = not self._avoid_right
        else:
            turn_key = preferred_turn_key
        if (
            self._recovery_committed_turn_key in {"A", "D"}
            and self._recovery_attempts > 0
            and self._recovery_attempts
            % self.obstacle_recovery_attempts_per_side
            == 0
        ):
            turn_key = "A" if self._recovery_committed_turn_key == "D" else "D"
        self._recovery_committed_turn_key = turn_key

        if self._recovery_attempts == 1:
            if current_coord is not None:
                self._hold_key("W", current_coord)
            elif self.held_key != "W":
                if self.held_key is not None:
                    self.input_controller.release_key(self.held_key)
                self.input_controller.press_key("W")
                self.held_key = "W"
            self.input_controller.tap_key("SPACE", duration=self.obstacle_jump_duration)
            self.turn_character(
                turn_key,
                duration=self.obstacle_turn_duration,
                context="recovery",
            )
            self._recovery_attempts += 1
            self._target_aligned = False
            self._arm_recovery_detour(turn_key, self.obstacle_turn_duration)
            self._set_diagnostics("recover_jump_turn", turn_key=turn_key)
            return ObstacleRecovery("recover_jump_turn", turn_key)

        if self.held_key is not None:
            self.input_controller.release_key(self.held_key)
            self.held_key = None
        self.input_controller.release_movement_keys()
        turn_duration = self.obstacle_turn_duration * (1.0 + min(2, self._recovery_attempts - 2) * 0.35)
        self.input_controller.tap_key("S", duration=self.obstacle_back_duration)
        self.input_controller.tap_key("SPACE", duration=self.obstacle_jump_duration)
        self.turn_character(turn_key, duration=turn_duration, context="recovery")
        if current_coord is not None:
            self._hold_key("W", current_coord)
        self._recovery_attempts += 1
        self._target_aligned = False
        self._arm_recovery_detour(turn_key, turn_duration)
        self._set_diagnostics("recover_back_turn", turn_key=turn_key)
        return ObstacleRecovery("recover_back_turn", turn_key)

    @property
    def recovery_detour_active(self) -> bool:
        with self._recovery_lock:
            return self._recovery_detour_turn_key is not None

    @property
    def recovery_suppresses_stuck(self) -> bool:
        with self._recovery_lock:
            return (
                self._recovery_detour_turn_key is not None
                or time.monotonic() < self._recovery_detour_settle_until
            )

    def _arm_recovery_detour(
        self,
        turn_key: Literal["A", "D"],
        turn_duration: float,
    ) -> None:
        if self.obstacle_detour_duration <= 0.0 or turn_duration <= 0.0:
            return

        with self._recovery_lock:
            self._cancel_recovery_detour(clear_settle=False)
            scale = 1.0 + min(2, max(0, self._recovery_attempts - 2)) * 0.30
            detour_duration = min(
                self.obstacle_detour_max_duration,
                self.obstacle_detour_duration * scale,
            )
            self._recovery_detour_turn_key = turn_key
            self._recovery_detour_turn_duration = turn_duration
            self._recovery_detour_duration = detour_duration
            self._recovery_detour_deadline = time.monotonic() + detour_duration
            self._recovery_realign_completed_key = None
            self._recovery_realign_completed_duration = None

    def _finish_recovery_detour_if_due(self) -> bool:
        with self._recovery_lock:
            if self._recovery_detour_turn_key is None:
                return False
            if self._recovery_detour_deadline is None or time.monotonic() < self._recovery_detour_deadline:
                return False

            turn_key = self._recovery_detour_turn_key
            self._recovery_detour_turn_key = None
            self._recovery_detour_turn_duration = 0.0
            self._recovery_detour_deadline = None
            self._recovery_detour_settle_until = (
                time.monotonic() + self.obstacle_detour_settle_duration
            )
            self._recovery_realign_completed_key = turn_key
            self._recovery_realign_completed_duration = self._recovery_detour_duration
            self._recovery_detour_duration = 0.0
            return True

    def _cancel_recovery_detour(self, *, clear_settle: bool) -> None:
        with self._recovery_lock:
            self._recovery_detour_turn_key = None
            self._recovery_detour_turn_duration = 0.0
            self._recovery_detour_duration = 0.0
            self._recovery_detour_deadline = None
            self._recovery_realign_completed_key = None
            self._recovery_realign_completed_duration = None
            if clear_settle:
                self._recovery_detour_settle_until = 0.0

    def choose_turn_key(self, current_coord: int, target_coord: int) -> Literal["A", "D"] | None:
        desired = self._vector_to_target(current_coord, target_coord)
        forward = self.forward_vector
        if forward is None or math.hypot(forward[0], forward[1]) <= 0.0:
            self._last_alignment = None
            return None

        alignment = self._alignment_score(forward, desired)
        self._last_alignment = alignment
        if alignment >= self.turn_alignment_threshold:
            return None

        cross = forward[0] * desired[1] - forward[1] * desired[0]
        turn_key: Literal["A", "D"] = "D" if cross < 0 else "A"
        if self.invert_turn_direction:
            return "A" if turn_key == "D" else "D"
        return turn_key

    def _resolve_course_correction_turn(
        self,
        candidate_turn_key: Literal["A", "D"] | None,
    ) -> Literal["A", "D"] | None:
        if self._last_progress is None:
            return candidate_turn_key
        if self._last_progress > 0.0:
            self._course_correction_turn_key = None
            return candidate_turn_key
        if self._course_correction_turn_key is None and candidate_turn_key is not None:
            self._course_correction_turn_key = candidate_turn_key
        return self._course_correction_turn_key

    def _turn_and_hold_forward(self, turn_key: Literal["A", "D"], current_coord: int) -> str:
        if self._should_turn_in_place():
            if self.held_key is not None:
                self.input_controller.release_key(self.held_key)
                self.held_key = None
            self.turn_character(
                turn_key,
                duration=self.turn_in_place_duration,
                context="route",
            )
            action = "turn_in_place_and_forward"
        else:
            self.turn_character(
                turn_key,
                duration=self.turn_duration,
                context="route",
            )
            action = "turn_and_forward"
        self._hold_key("W", current_coord)
        self._target_aligned = (
            self._last_alignment is not None and self._last_alignment >= self.turn_alignment_threshold
        )
        return action

    def turn_character(
        self,
        turn_key: str,
        *,
        duration: float,
        context: str = "local",
    ) -> None:
        if self.mouse_steering is not None and self.mouse_steering.turn(
            turn_key,
            duration=duration,
            context=context,
        ):
            return
        self.input_controller.tap_key(turn_key, duration=max(0.0, duration))

    def _hold_key(self, key: str, current_coord: int) -> None:
        if self.held_key == key:
            return

        if self.held_key is not None:
            self.input_controller.release_key(self.held_key)
        self.input_controller.press_key(key)
        self.held_key = key
        self._last_observed_coord = current_coord
        self._last_observed_key = key
        self._last_coord_delta = None

    def _is_stuck(self, distance_to_target: float) -> bool:
        if self.held_key is None:
            self._last_distance_to_target = None
            self._last_progress = None
            self._stuck_samples = 0
            return False

        if self._last_distance_to_target is None:
            self._last_distance_to_target = distance_to_target
            self._last_progress = None
            return False

        progress = self._last_distance_to_target - distance_to_target
        self._last_progress = progress
        self._last_distance_to_target = distance_to_target
        if progress > 0.0:
            self._stuck_samples = 0
            self._recovery_attempts = 0
            self._recovery_committed_turn_key = None
            return False
        if self._is_coord_moving():
            self._stuck_samples = 0
            return False

        self._stuck_samples += 1
        return self._stuck_samples >= self.stuck_check_window

    def _should_turn_in_place(self) -> bool:
        return self._last_alignment is not None and self._last_alignment < self.turn_in_place_alignment

    def _is_coord_moving(self) -> bool:
        return self._last_coord_delta is not None and self._last_coord_delta >= self.stuck_min_coord_delta

    def _set_diagnostics(
        self,
        action: str,
        *,
        distance_to_target: float | None = None,
        progress: float | None = None,
        alignment: float | None = None,
        turn_key: str | None = None,
        stuck_samples: int | None = None,
        target_coord: int | None = None,
        desired_heading_degrees: float | None = None,
        heading_error_degrees: float | None = None,
        heading_braking: bool = False,
    ) -> None:
        with self._recovery_lock:
            detour_phase = None
            detour_turn_key = None
            detour_duration = None
            if self._recovery_detour_turn_key is not None:
                detour_phase = "forward"
                detour_turn_key = self._recovery_detour_turn_key
                detour_duration = self._recovery_detour_duration
            elif action == "recover_detour_realign":
                detour_phase = "realign"
                detour_turn_key = turn_key
                detour_duration = self._recovery_realign_completed_duration
            elif action == "recover_detour_replan":
                detour_phase = "replan"
                detour_turn_key = turn_key
                detour_duration = self._recovery_realign_completed_duration
        self.last_diagnostics = MovementDiagnostics(
            action=action,
            distance_to_target=distance_to_target,
            progress=progress,
            distance_progress=progress,
            alignment=alignment,
            turn_key=turn_key,
            held_key=self.held_key,
            stuck_samples=self._stuck_samples if stuck_samples is None else stuck_samples,
            forward_vector=self.forward_vector,
            target_coord=target_coord,
            coord_delta=self._last_coord_delta,
            movement_delta=self._last_coord_delta,
            detour_phase=detour_phase,
            detour_turn_key=detour_turn_key,
            detour_duration=detour_duration,
            held_keys=tuple(key for key in MOVEMENT_KEYS if key in self.heading_held_keys),
            desired_heading_degrees=desired_heading_degrees,
            heading_error_degrees=heading_error_degrees,
            heading_braking=heading_braking,
        )

    @staticmethod
    def distance(a: int, b: int) -> float:
        ax, ay = MovementNavigator._decode_coord(a)
        bx, by = MovementNavigator._decode_coord(b)
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def _vector_to_target(current_coord: int, target_coord: int) -> tuple[float, float]:
        current_x, current_y = MovementNavigator._decode_coord(current_coord)
        target_x, target_y = MovementNavigator._decode_coord(target_coord)
        return target_x - current_x, target_y - current_y

    @staticmethod
    def _decode_coord(coord: int) -> tuple[float, float]:
        return coord_to_xy(coord)

    @staticmethod
    def _alignment_score(vector: tuple[float, float], desired: tuple[float, float]) -> float:
        vector_length = math.hypot(vector[0], vector[1])
        desired_length = math.hypot(desired[0], desired[1])
        if vector_length <= 0.0 or desired_length <= 0.0:
            return -1.0
        return (vector[0] * desired[0] + vector[1] * desired[1]) / (vector_length * desired_length)
