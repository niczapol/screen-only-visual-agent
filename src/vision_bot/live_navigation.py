from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.capture import ScreenCapture
from vision_bot.local_navigation import NavigationFrame, analyze_local_navigation, draw_local_navigation_report
from vision_bot.movement import InputController, MovementNavigator
from vision_bot.position import read_player_position
from vision_bot.windows_security import get_current_integrity_level, get_process_integrity_level, target_requires_elevation


@dataclass(frozen=True)
class LiveNavigationAction:
    name: str
    key: str | None
    duration: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "duration": self.duration,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LiveNavigationSummary:
    output_dir: Path
    steps: int
    start_coord: int | None
    final_coord: int | None
    moved_distance: float | None
    visual_motion_mean: float | None
    stuck_events: int
    return_attempted: bool
    returned_distance: float | None


def choose_live_navigation_action(
    navigation: NavigationFrame,
    *,
    stuck_samples: int,
    stuck_window: int,
    alternate_turn_key: str,
    forward_duration: float,
    turn_duration: float,
    back_duration: float,
) -> LiveNavigationAction:
    if stuck_samples >= stuck_window:
        return LiveNavigationAction("recover_back", "S", back_duration, "stuck_samples")

    if navigation.center_blocked:
        turn_key = navigation.recommended_turn or alternate_turn_key
        return LiveNavigationAction("turn_for_blocker", turn_key, turn_duration, "center_blocked")

    return LiveNavigationAction("forward", "W", forward_duration, "center_available")


def collect_live_navigation_movement(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    steps: int,
    forward_duration: float = 0.35,
    turn_duration: float = 0.22,
    back_duration: float = 0.28,
    settle_delay: float = 0.08,
    stuck_min_delta: float = 0.015,
    visual_stuck_min_delta: float = 3.0,
    stuck_window: int = 4,
    report_interval: int = 12,
    read_coordinates: bool = True,
    return_to_start: bool = False,
    return_steps: int = 40,
) -> LiveNavigationSummary:
    output_path = Path(output_dir)
    frame_dir = output_path / "frames"
    report_dir = output_path / "reports"
    frame_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    capture = ScreenCapture(config)
    hwnd = capture.find_window()
    if hwnd is None:
        raise RuntimeError("Game window not found")
    foreground_ok = capture.activate_window()
    if not foreground_ok:
        raise RuntimeError("Game window could not be activated; live navigation input aborted")
    _ensure_input_allowed(hwnd, output_path)
    input_controller = _create_input_controller(config, hwnd)

    start_frame = capture.capture_client_region()
    start_coord = _read_coord(start_frame, config, read_coordinates=read_coordinates)
    start_navigation = analyze_local_navigation(start_frame, config)
    cv2.imwrite(str(output_path / "start_frame.png"), start_frame)
    cv2.imwrite(str(output_path / "start_report.png"), draw_local_navigation_report(start_frame, start_navigation))

    last_coord = start_coord
    final_coord = start_coord
    stuck_samples = 0
    stuck_events = 0
    alternate_turn_key = "D"
    visual_motion_values: list[float] = []

    try:
        for index in range(max(1, steps)):
            capture.activate_window()
            before_frame = capture.capture_client_region()
            before_coord = _read_coord(before_frame, config, read_coordinates=read_coordinates)
            navigation = analyze_local_navigation(before_frame, config)
            action = choose_live_navigation_action(
                navigation,
                stuck_samples=stuck_samples,
                stuck_window=max(1, stuck_window),
                alternate_turn_key=alternate_turn_key,
                forward_duration=forward_duration,
                turn_duration=turn_duration,
                back_duration=back_duration,
            )
            if action.name == "recover_back":
                stuck_events += 1
                alternate_turn_key = "A" if alternate_turn_key == "D" else "D"

            _perform_action(input_controller, action)
            time.sleep(max(0.0, settle_delay))

            after_frame = capture.capture_client_region()
            after_coord = _read_coord(after_frame, config, read_coordinates=read_coordinates)
            final_coord = after_coord if after_coord is not None else final_coord
            moved_delta = _coord_distance(before_coord, after_coord)
            visual_motion_delta = _visual_motion_delta(before_frame, after_frame)
            if visual_motion_delta is not None:
                visual_motion_values.append(visual_motion_delta)

            if action.key == "W" and _movement_was_stuck(
                coord_delta=moved_delta,
                visual_motion_delta=visual_motion_delta,
                stuck_min_delta=stuck_min_delta,
                visual_stuck_min_delta=visual_stuck_min_delta,
            ):
                stuck_samples += 1
            else:
                stuck_samples = 0

            if before_coord is None and last_coord is not None:
                before_coord = last_coord
            if after_coord is not None:
                last_coord = after_coord

            frame_path = frame_dir / f"{index:04d}.png"
            cv2.imwrite(str(frame_path), before_frame)
            report_path: Path | None = None
            if report_interval > 0 and (index % report_interval == 0 or navigation.center_blocked or action.name != "forward"):
                report_path = report_dir / f"{index:04d}.png"
                cv2.imwrite(str(report_path), draw_local_navigation_report(before_frame, navigation))

            _append_jsonl(
                metadata_path,
                {
                    "index": index,
                    "timestamp": time.time(),
                    "foreground_ok": foreground_ok,
                    "frame": _relative_posix(frame_path, output_path),
                    "report": _relative_posix(report_path, output_path) if report_path is not None else None,
                    "coord_before": before_coord,
                    "coord_after": after_coord,
                    "coord_delta": moved_delta,
                    "visual_motion_delta": visual_motion_delta,
                    "distance_from_start_before": _coord_distance(start_coord, before_coord),
                    "distance_from_start_after": _coord_distance(start_coord, after_coord),
                    "stuck_samples": stuck_samples,
                    "action": action.to_dict(),
                    "navigation": navigation.to_dict(),
                },
            )
    finally:
        input_controller.release_movement_keys()

    returned_distance: float | None = None
    if return_to_start and start_coord is not None:
        returned_distance = _return_towards_start(
            capture,
            input_controller,
            config,
            start_coord,
            max_steps=max(0, return_steps),
        )
        final_frame = capture.capture_client_region()
        final_coord = _read_coord(final_frame, config, read_coordinates=read_coordinates) or final_coord
        cv2.imwrite(str(output_path / "final_frame.png"), final_frame)
        cv2.imwrite(str(output_path / "final_report.png"), draw_local_navigation_report(final_frame, analyze_local_navigation(final_frame, config)))
    else:
        final_frame = capture.capture_client_region()
        cv2.imwrite(str(output_path / "final_frame.png"), final_frame)
        cv2.imwrite(str(output_path / "final_report.png"), draw_local_navigation_report(final_frame, analyze_local_navigation(final_frame, config)))

    input_controller.release_movement_keys()
    summary = LiveNavigationSummary(
        output_dir=output_path,
        steps=max(1, steps),
        start_coord=start_coord,
        final_coord=final_coord,
        moved_distance=_coord_distance(start_coord, final_coord),
        visual_motion_mean=(
            round(float(np.mean(visual_motion_values)), 4) if visual_motion_values else None
        ),
        stuck_events=stuck_events,
        return_attempted=return_to_start and start_coord is not None,
        returned_distance=returned_distance,
    )
    (output_path / "summary.json").write_text(json.dumps(_summary_to_dict(summary), indent=2), encoding="utf-8")
    return summary


def _perform_action(input_controller: InputController, action: LiveNavigationAction) -> None:
    if action.key is None or action.duration <= 0:
        return

    key = action.key.upper()
    input_controller.press_key(key)
    time.sleep(max(0.0, action.duration))
    input_controller.release_key(key)


def _return_towards_start(
    capture: ScreenCapture,
    input_controller: InputController,
    config: dict[str, Any],
    start_coord: int,
    *,
    max_steps: int,
) -> float | None:
    movement_cfg = config.get("movement", {})
    navigator = MovementNavigator(
        input_controller,
        stuck_check_window=int(movement_cfg.get("stuck_check_window", 4)),
        stuck_min_progress=float(movement_cfg.get("stuck_min_progress", 0.03)),
        stuck_min_coord_delta=float(
            movement_cfg.get("stuck_min_coord_delta", movement_cfg.get("stuck_min_progress", 0.03))
        ),
        obstacle_back_duration=float(movement_cfg.get("obstacle_back_duration", 0.35)),
        obstacle_turn_duration=float(movement_cfg.get("obstacle_turn_duration", 0.55)),
        obstacle_jump_duration=float(movement_cfg.get("obstacle_jump_duration", 0.08)),
        obstacle_jump_first=bool(movement_cfg.get("obstacle_jump_first", True)),
        obstacle_detour_duration=float(movement_cfg.get("obstacle_detour_duration", 0.35)),
        obstacle_detour_max_duration=float(
            movement_cfg.get("obstacle_detour_max_duration", 0.65)
        ),
        obstacle_detour_settle_duration=float(
            movement_cfg.get("obstacle_detour_settle_duration", 0.0)
        ),
        turn_duration=float(movement_cfg.get("turn_duration", 0.12)),
        turn_in_place_duration=float(movement_cfg.get("turn_in_place_duration", 0.35)),
        turn_in_place_alignment=float(movement_cfg.get("turn_in_place_alignment", -0.2)),
        turn_alignment_threshold=float(movement_cfg.get("turn_alignment_threshold", 0.92)),
        invert_turn_direction=bool(movement_cfg.get("invert_turn_direction", False)),
    )
    final_distance: float | None = None
    try:
        for _ in range(max_steps):
            frame = capture.capture_client_region()
            current_coord = read_player_position(frame, config)
            if current_coord is None:
                break
            final_distance = MovementNavigator.distance(current_coord, start_coord)
            if final_distance <= 0.20:
                break
            navigator.move_towards(current_coord, start_coord)
            time.sleep(float(movement_cfg.get("poll_interval", 0.35)))
    finally:
        navigator.stop()
    return final_distance


def _create_input_controller(config: dict[str, Any], hwnd: int | None) -> InputController:
    input_cfg = config.get("input", {})
    return InputController(target_hwnd=hwnd, backend=str(input_cfg.get("backend", "sendinput")))


def _ensure_input_allowed(hwnd: int, output_path: Path) -> None:
    try:
        import win32process
    except ImportError:
        return

    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    blocked = target_requires_elevation(pid)
    preflight = {
        "hwnd": hwnd,
        "pid": pid,
        "current_integrity": _integrity_to_dict(get_current_integrity_level()),
        "target_integrity": _integrity_to_dict(get_process_integrity_level(pid)),
        "input_blocked_by_integrity": blocked,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if blocked:
        raise RuntimeError(
            "Input is blocked by Windows integrity levels. "
            "Run Codex/Python as administrator or run the game at medium integrity before live movement."
        )


def _coord_distance(a: int | None, b: int | None) -> float | None:
    if a is None or b is None:
        return None
    return MovementNavigator.distance(a, b)


def _read_coord(frame: np.ndarray, config: dict[str, Any], *, read_coordinates: bool) -> int | None:
    if not read_coordinates:
        return None
    return read_player_position(frame, config)


def _movement_was_stuck(
    *,
    coord_delta: float | None,
    visual_motion_delta: float | None,
    stuck_min_delta: float,
    visual_stuck_min_delta: float,
) -> bool:
    if coord_delta is not None:
        return coord_delta < stuck_min_delta
    if visual_motion_delta is not None:
        return visual_motion_delta < visual_stuck_min_delta
    return True


def _visual_motion_delta(before_frame: np.ndarray, after_frame: np.ndarray) -> float | None:
    if before_frame.shape != after_frame.shape or before_frame.size == 0:
        return None

    height, width = before_frame.shape[:2]
    y1 = int(height * 0.18)
    y2 = int(height * 0.72)
    x1 = int(width * 0.18)
    x2 = int(width * 0.72)
    if y2 <= y1 or x2 <= x1:
        return None

    diff = cv2.absdiff(before_frame[y1:y2, x1:x2], after_frame[y1:y2, x1:x2])
    return round(float(np.mean(diff)), 4)


def _integrity_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"name": getattr(value, "name", None), "rid": getattr(value, "rid", None)}


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _relative_posix(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _summary_to_dict(summary: LiveNavigationSummary) -> dict[str, Any]:
    return {
        "output_dir": str(summary.output_dir),
        "steps": summary.steps,
        "start_coord": summary.start_coord,
        "final_coord": summary.final_coord,
        "moved_distance": summary.moved_distance,
        "visual_motion_mean": summary.visual_motion_mean,
        "stuck_events": summary.stuck_events,
        "return_attempted": summary.return_attempted,
        "returned_distance": summary.returned_distance,
    }
