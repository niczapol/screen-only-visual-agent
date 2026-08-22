from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.capture import ScreenCapture
from vision_bot.config import load_config, resource_path, runtime_state_path
from vision_bot.local_navigation import (
    analyze_local_navigation,
    collect_local_navigation_sources,
    collect_route_probe_metadata_sources,
    draw_local_navigation_report,
    prepare_local_navigation_dataset,
    prepare_local_navigation_outcome_dataset,
    prepare_local_navigation_review_pack,
)
from vision_bot.live_navigation import collect_live_navigation_movement
from vision_bot.live_route_probe import build_route_live_preflight, collect_route_to_node_probe
from vision_bot.mining import MiningInteractor
from vision_bot.movement import (
    InputController,
    MouseController,
    MouseSteeringController,
    MovementNavigator,
)
from vision_bot.position_filter import (
    CoordinateResyncState,
    choose_best_coord_candidate_with_resync,
)
from vision_bot.permanent_exclusions import add_permanent_exclusion, load_permanent_exclusions
from vision_bot.position import read_player_position, read_player_position_candidates
from vision_bot.recognition import recognize_dark_ore_points, recognize_ore_points, save_debug_artifacts
from vision_bot.route_decision import RouteAction, RouteDecision, decide_route_action
from vision_bot.route_planner import MiningRoutePlanner, load_nodes, load_route_waypoints
from vision_bot.routing_ui import RoutingControlPanel
from vision_bot.screen_objects import detect_screen_objects, draw_detections
from vision_bot.visual_report import draw_detection_report
from vision_bot.vision_dataset import collect_vision_dataset
from vision_bot.vision_training import prepare_yolo_training_split, train_yolo_model
from vision_bot.windows_security import get_current_integrity_level, get_process_integrity_level, target_requires_elevation


def create_input_controller(config: dict[str, Any], target_hwnd: int | None = None) -> InputController:
    input_cfg = config.get("input", {})
    return InputController(
        target_hwnd=target_hwnd,
        backend=str(input_cfg.get("backend", "sendinput")),
    )


def _load_route_nodes(route_cfg: dict[str, Any]):
    route_data_path = str(route_cfg.get("database_path") or "data/MiningData.lua")
    source_path = resource_path(route_data_path)
    if bool(route_cfg.get("follow_route_loop", False)) and source_path.suffix.lower() == ".json":
        return load_route_waypoints(source_path)
    return load_nodes(source_path)


class MiningRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.capture = ScreenCapture(config)
        self.movement_enabled = False
        self.ore_detected = False
        self.dark_ore_detected = False
        self.last_known_coord: int | None = None
        self.current_target_id: int | None = None
        self.path_failures: dict[int, int] = {}
        self.absence_confirmations: dict[int, int] = {}
        self.coord_resync_state = CoordinateResyncState()
        self.input_blocked_by_integrity = False
        route_cfg = config.get("route", {})
        self.permanent_exclusions_path = runtime_state_path(
            str(route_cfg.get("permanent_exclusions_path", "data/permanent_exclusions.json"))
        )
        self.route_planner = MiningRoutePlanner(
            nodes=_load_route_nodes(route_cfg),
            allowed_locations=set(route_cfg.get("locations", [27])),
            allowed_ores=set(route_cfg.get("ores", ["Copper"])),
            permanent_exclusions=load_permanent_exclusions(self.permanent_exclusions_path),
            route_mode=str(route_cfg.get("mode", "nearest")),
        )
        self.input_controller = create_input_controller(config)
        self.mouse_controller = MouseController(
            target_hwnd=self.capture.window_handle,
        )
        self.mouse_steering = MouseSteeringController.from_config(
            self.mouse_controller,
            config,
        )
        movement_cfg = config.get("movement", {})
        self.navigator = MovementNavigator(
            self.input_controller,
            stuck_check_window=int(movement_cfg.get("stuck_check_window", 4)),
            stuck_min_progress=float(movement_cfg.get("stuck_min_progress", 0.03)),
            stuck_min_coord_delta=float(
                movement_cfg.get("stuck_min_coord_delta", movement_cfg.get("stuck_min_progress", 0.03))
            ),
            obstacle_back_duration=float(movement_cfg.get("obstacle_back_duration", 0.35)),
            obstacle_strafe_duration=float(movement_cfg.get("obstacle_strafe_duration", 0.55)),
            obstacle_turn_duration=float(
                movement_cfg.get(
                    "obstacle_turn_duration",
                    movement_cfg.get("obstacle_strafe_duration", 0.55),
                )
            ),
            obstacle_jump_duration=float(movement_cfg.get("obstacle_jump_duration", 0.08)),
            obstacle_jump_first=bool(movement_cfg.get("obstacle_jump_first", True)),
            obstacle_detour_duration=float(movement_cfg.get("obstacle_detour_duration", 0.35)),
            obstacle_detour_max_duration=float(
                movement_cfg.get("obstacle_detour_max_duration", 0.65)
            ),
            obstacle_detour_settle_duration=float(
                movement_cfg.get("obstacle_detour_settle_duration", 0.0)
            ),
            obstacle_recovery_attempts_per_side=int(
                movement_cfg.get("obstacle_recovery_attempts_per_side", 4)
            ),
            turn_duration=float(movement_cfg.get("turn_duration", 0.12)),
            turn_in_place_duration=float(
                movement_cfg.get(
                    "turn_in_place_duration",
                    movement_cfg.get("turn_duration", 0.12),
                )
            ),
            turn_in_place_alignment=float(movement_cfg.get("turn_in_place_alignment", -0.2)),
            turn_alignment_threshold=float(movement_cfg.get("turn_alignment_threshold", 0.92)),
            invert_turn_direction=bool(movement_cfg.get("invert_turn_direction", False)),
            mouse_steering=self.mouse_steering,
        )
        self.panel = RoutingControlPanel(on_toggle=self._handle_toggle)
        self.mining_interactor = MiningInteractor(self.capture, self.mouse_controller, config)

    def _handle_toggle(self, enabled: bool) -> None:
        self.movement_enabled = enabled
        if not enabled:
            self.navigator.stop()

    def run(self) -> None:
        self.capture.find_window()
        self._bind_input_target()
        self._refresh_integrity_status()
        self.panel.set_ore_detected(False)

        def update_loop() -> None:
            while True:
                if not self.panel.root.winfo_exists():
                    break
                self._step()
                time.sleep(float(self.config.get("movement", {}).get("poll_interval", 1.0)))

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self.panel.start()

    def _bind_input_target(self) -> None:
        self.input_controller.set_target_window(self.capture.window_handle)
        self.mouse_controller.set_target_window(self.capture.window_handle)

    def _step(self) -> None:
        frame = self.capture.capture_client_region()
        minimap = self.capture.crop_minimap(frame, self.config)
        route_cfg = self.config.get("route", {})
        points = recognize_ore_points(minimap, self.config)
        dark_detection_enabled = bool(route_cfg.get("dark_exclusion_enabled", False))
        dark_points = [] if points or not dark_detection_enabled else recognize_dark_ore_points(minimap, self.config)
        self.ore_detected = bool(points)
        self.dark_ore_detected = bool(dark_points)

        if self.panel.root.winfo_exists():
            self.panel.root.after(0, lambda: self.panel.set_ore_detected(self.ore_detected))

        if self.config.get("movement", {}).get("position_ocr_enabled", False):
            player_coord_candidates = read_player_position_candidates(frame, self.config)
            max_jump = float(self.config.get("movement", {}).get("max_coord_jump_per_poll", 1.5))
            player_coord, resynced = choose_best_coord_candidate_with_resync(
                self.last_known_coord,
                player_coord_candidates,
                max_jump,
                self.route_planner.available_coords(),
                self.coord_resync_state,
                int(self.config.get("movement", {}).get("coord_resync_confirm_steps", 3)),
                float(self.config.get("movement", {}).get("coord_resync_max_distance", 5.0)),
            )
            if player_coord is not None:
                if resynced:
                    self.navigator.stop()
                self.last_known_coord = player_coord
                self.route_planner.set_current_position(player_coord)
            elif self.movement_enabled and player_coord_candidates:
                self.navigator.stop()
                self._set_status(f"Rejected OCR coordinate candidates near {self.last_known_coord}: {player_coord_candidates}")
                return
            elif self.movement_enabled:
                self.navigator.stop()
                self._set_status("Waiting for readable player coordinates")
                return
        elif self.last_known_coord is None:
            self.last_known_coord = self.route_planner.current_position

        if not self.movement_enabled:
            self.navigator.stop()
            return

        if self.capture.window_handle is None:
            self.capture.find_window()
            self._bind_input_target()
            self._refresh_integrity_status()
        if self.input_blocked_by_integrity:
            self.navigator.stop()
            return
        self.capture.activate_window()

        mining_cfg = self.config.get("mining", {})
        tracking_radius = float(route_cfg.get("tracking_radius_coord", 1.4))
        absence_cooldown = int(route_cfg.get("absence_cooldown_seconds", 300))
        absence_confirm_steps = max(1, int(route_cfg.get("absence_confirm_steps", 1)))
        reached_distance = float(mining_cfg.get("reached_distance", 0.25))
        dark_exclude_distance = float(route_cfg.get("dark_exclude_distance_coord", reached_distance))
        skipped_absent = 0
        skipped_permanent = 0

        while True:
            next_node = self._choose_active_node()
            if next_node is None:
                self.current_target_id = None
                self.navigator.stop()
                skipped_text = ""
                if skipped_absent or skipped_permanent:
                    skipped_text = f" after skipping {skipped_absent} absent and {skipped_permanent} excluded targets"
                self._set_status(f"No route targets available{skipped_text}")
                return

            self.current_target_id = next_node.node_id
            distance_to_target = self.route_planner.distance_from_current(next_node.coord)
            decision = decide_route_action(
                distance_to_target=distance_to_target,
                ore_detected=self.ore_detected,
                tracking_radius=tracking_radius,
                reached_distance=reached_distance,
                mining_enabled=bool(mining_cfg.get("enabled", True)),
                dark_ore_detected=self.dark_ore_detected,
                dark_exclude_distance=dark_exclude_distance,
                dark_exclusion_enabled=dark_detection_enabled,
            )

            if decision.action is RouteAction.WAIT_FOR_POSITION:
                self.navigator.stop()
                self._set_status("Waiting for player coordinates")
                return

            if decision.action is RouteAction.MARK_ABSENT:
                confirmations = self.absence_confirmations.get(next_node.node_id, 0) + 1
                self.absence_confirmations[next_node.node_id] = confirmations
                if confirmations < absence_confirm_steps:
                    decision = RouteDecision(
                        RouteAction.MOVE_TO_NODE,
                        f"ore_absence_unconfirmed_{confirmations}/{absence_confirm_steps}",
                    )
                    break

                skipped_absent += 1
                self.route_planner.mark_absent(next_node.node_id, cooldown_seconds=absence_cooldown)
                self.absence_confirmations.pop(next_node.node_id, None)
                self.current_target_id = None
                continue

            self.absence_confirmations.pop(next_node.node_id, None)

            if decision.action is RouteAction.MARK_PERMANENT_EXCLUDED:
                skipped_permanent += 1
                self.route_planner.exclude_permanently(next_node.coord)
                add_permanent_exclusion(self.permanent_exclusions_path, next_node, "dark_minimap_icon")
                self.absence_confirmations.pop(next_node.node_id, None)
                self.current_target_id = None
                continue

            break

        if decision.action is RouteAction.HOLD_AT_NODE:
            self.navigator.stop()
            self._set_status(f"Target {next_node.node_id} reached; mining disabled")
            return

        if decision.action is RouteAction.READY_TO_MINE:
            self.navigator.stop()
            self._attempt_mining(next_node.node_id)
            return

        if self.last_known_coord is not None:
            self.navigator.move_towards(self.last_known_coord, next_node.coord)
            if distance_to_target is not None:
                diagnostics = self.navigator.last_diagnostics
                movement_cfg = self.config.get("movement", {})
                blocked_text = ""
                if diagnostics.action.startswith("recover_"):
                    failures = self.path_failures.get(next_node.node_id, 0) + 1
                    self.path_failures[next_node.node_id] = failures
                    recovery_limit = max(1, int(movement_cfg.get("target_recovery_limit", 3)))
                    if failures >= recovery_limit:
                        cooldown = int(movement_cfg.get("path_blocked_cooldown_seconds", 90))
                        self.route_planner.mark_absent(next_node.node_id, cooldown_seconds=cooldown)
                        self.path_failures[next_node.node_id] = 0
                        self.absence_confirmations.pop(next_node.node_id, None)
                        self.current_target_id = None
                        blocked_text = f"; path blocked cooldown={cooldown}s"
                    else:
                        blocked_text = f"; recover={failures}/{recovery_limit}"
                elif diagnostics.progress is not None and diagnostics.progress >= 0.0:
                    self.path_failures[next_node.node_id] = 0
                skipped_text = f"; skipped absent={skipped_absent}" if skipped_absent else ""
                turn_text = f" turn={diagnostics.turn_key}" if diagnostics.turn_key else ""
                progress_text = (
                    f" progress={diagnostics.progress:.2f}" if diagnostics.progress is not None else ""
                )
                self._set_status(
                    f"Moving to target {next_node.node_id}: {distance_to_target:.2f} coord units "
                    f"({decision.reason}; {diagnostics.action}{turn_text}{progress_text}{skipped_text}{blocked_text})"
                )

    def _choose_active_node(self):
        if self.current_target_id is not None:
            current_node = self.route_planner.get_available_node(self.current_target_id)
            if current_node is not None:
                return current_node
            self.current_target_id = None

        return self.route_planner.choose_next_node()

    def _set_status(self, text: str) -> None:
        if self.panel.root.winfo_exists():
            self.panel.root.after(0, lambda: self.panel.set_status(text))

    def save_debug(self, minimap: np.ndarray, points: list[tuple[int, int]]) -> None:
        output_dir = Path(self.config.get("debug", {}).get("output_dir", "debug_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_debug_artifacts(minimap, points, output_dir, self.config)

    def _attempt_mining(self, node_id: int) -> bool:
        mining_cfg = self.config.get("mining", {})
        if not bool(mining_cfg.get("enabled", True)):
            return False

        cooldown_seconds = int(mining_cfg.get("cooldown_seconds", 180))
        failure_cooldown = int(mining_cfg.get("failure_cooldown_seconds", 20))

        self.navigator.stop()
        hover_result = self.mining_interactor.right_click_hover_target()
        if not hover_result.found:
            self.route_planner.mark_absent(node_id, cooldown_seconds=failure_cooldown)
            self.absence_confirmations.pop(node_id, None)
            if self.current_target_id == node_id:
                self.current_target_id = None
            self._set_status(f"Mining hover not found for target {node_id}; cooldown {failure_cooldown}s")
            return False

        frame = self.capture.capture_client_region()
        minimap = self.capture.crop_minimap(frame, self.config)
        points = recognize_ore_points(minimap, self.config)
        if not points:
            self.route_planner.mark_mined(node_id, cooldown_seconds=cooldown_seconds)
            self.absence_confirmations.pop(node_id, None)
            if self.current_target_id == node_id:
                self.current_target_id = None
            self._set_status(f"Target {node_id} mined")
            return True

        self.route_planner.mark_absent(node_id, cooldown_seconds=failure_cooldown)
        self.absence_confirmations.pop(node_id, None)
        if self.current_target_id == node_id:
            self.current_target_id = None
        self._set_status(f"Mining click sent, ore still visible for target {node_id}; cooldown {failure_cooldown}s")
        return False

    def _refresh_integrity_status(self) -> None:
        self.input_blocked_by_integrity = False
        if self.capture.window_handle is None:
            return
        try:
            import win32process

            _, pid = win32process.GetWindowThreadProcessId(self.capture.window_handle)
        except Exception:
            return

        if target_requires_elevation(pid):
            self.input_blocked_by_integrity = True
            current = get_current_integrity_level()
            target = get_process_integrity_level(pid)
            current_name = current.name if current is not None else "unknown"
            target_name = target.name if target is not None else "unknown"
            message = f"Запусти от администратора: бот {current_name}, игра {target_name}"
            if self.panel.root.winfo_exists():
                self.panel.root.after(0, lambda: self.panel.set_status(message))
        elif self.panel.root.winfo_exists():
            self.panel.root.after(0, lambda: self.panel.set_status("Готов"))


def run_movement_probe(config: dict[str, Any], steps: int, allow_mining: bool, log_path: str | Path) -> int:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        log_file.write_text("\n".join(lines), encoding="utf-8")

    capture = ScreenCapture(config)
    hwnd = capture.find_window()
    log(f"window_found={hwnd is not None}")
    if hwnd is None:
        log_file.write_text("\n".join(lines), encoding="utf-8")
        return 2

    if capture.window_handle is not None:
        try:
            import win32process

            _, pid = win32process.GetWindowThreadProcessId(capture.window_handle)
            current = get_current_integrity_level()
            target = get_process_integrity_level(pid)
            log(f"integrity_current={current}")
            log(f"integrity_target={target}")
            if target_requires_elevation(pid):
                log("blocked=target_requires_elevation")
                log_file.write_text("\n".join(lines), encoding="utf-8")
                return 3
        except Exception as exc:
            log(f"integrity_check_error={type(exc).__name__}:{exc}")

    log(f"activate={capture.activate_window()}")
    time.sleep(0.8)

    input_controller = create_input_controller(config, capture.window_handle)
    mouse_controller = MouseController(target_hwnd=capture.window_handle)
    mouse_steering = MouseSteeringController.from_config(mouse_controller, config)
    movement_cfg = config.get("movement", {})
    navigator = MovementNavigator(
        input_controller,
        stuck_check_window=int(movement_cfg.get("stuck_check_window", 4)),
        stuck_min_progress=float(movement_cfg.get("stuck_min_progress", 0.03)),
        stuck_min_coord_delta=float(
            movement_cfg.get("stuck_min_coord_delta", movement_cfg.get("stuck_min_progress", 0.03))
        ),
        obstacle_back_duration=float(movement_cfg.get("obstacle_back_duration", 0.35)),
        obstacle_strafe_duration=float(movement_cfg.get("obstacle_strafe_duration", 0.55)),
        obstacle_turn_duration=float(
            movement_cfg.get(
                "obstacle_turn_duration",
                movement_cfg.get("obstacle_strafe_duration", 0.55),
            )
        ),
        obstacle_jump_duration=float(movement_cfg.get("obstacle_jump_duration", 0.08)),
        obstacle_jump_first=bool(movement_cfg.get("obstacle_jump_first", True)),
        obstacle_detour_duration=float(movement_cfg.get("obstacle_detour_duration", 0.35)),
        obstacle_detour_max_duration=float(
            movement_cfg.get("obstacle_detour_max_duration", 0.65)
        ),
        obstacle_detour_settle_duration=float(
            movement_cfg.get("obstacle_detour_settle_duration", 0.0)
        ),
        obstacle_recovery_attempts_per_side=int(
            movement_cfg.get("obstacle_recovery_attempts_per_side", 4)
        ),
        turn_duration=float(movement_cfg.get("turn_duration", 0.12)),
        turn_in_place_duration=float(
            movement_cfg.get(
                "turn_in_place_duration",
                movement_cfg.get("turn_duration", 0.12),
            )
        ),
        turn_in_place_alignment=float(movement_cfg.get("turn_in_place_alignment", -0.2)),
        turn_alignment_threshold=float(movement_cfg.get("turn_alignment_threshold", 0.92)),
        invert_turn_direction=bool(movement_cfg.get("invert_turn_direction", False)),
        mouse_steering=mouse_steering,
    )
    route_cfg = config.get("route", {})
    mining_cfg = config.get("mining", {})
    permanent_exclusions_path = runtime_state_path(
        str(route_cfg.get("permanent_exclusions_path", "data/permanent_exclusions.json"))
    )
    route_planner = MiningRoutePlanner(
        nodes=_load_route_nodes(route_cfg),
        allowed_locations=set(route_cfg.get("locations", [])),
        allowed_ores=set(route_cfg.get("ores", [])),
        permanent_exclusions=load_permanent_exclusions(permanent_exclusions_path),
        route_mode=str(route_cfg.get("mode", "nearest")),
    )
    tracking_radius = float(route_cfg.get("tracking_radius_coord", 1.4))
    absence_cooldown = int(route_cfg.get("absence_cooldown_seconds", 300))
    absence_confirm_steps = max(1, int(route_cfg.get("absence_confirm_steps", 1)))
    reached_distance = float(mining_cfg.get("reached_distance", 0.25))
    dark_exclude_distance = float(route_cfg.get("dark_exclude_distance_coord", reached_distance))
    dark_detection_enabled = bool(route_cfg.get("dark_exclusion_enabled", False))
    mining_interactor = MiningInteractor(capture, mouse_controller, config)
    path_failures: dict[int, int] = {}
    absence_confirmations: dict[int, int] = {}
    coord_resync_state = CoordinateResyncState()
    active_target_id: int | None = None

    try:
        for step in range(max(1, steps)):
            frame = capture.capture_client_region()
            minimap = capture.crop_minimap(frame, config)
            points = recognize_ore_points(minimap, config)
            dark_points = [] if points or not dark_detection_enabled else recognize_dark_ore_points(minimap, config)
            coord_candidates = read_player_position_candidates(frame, config)
            max_jump = float(config.get("movement", {}).get("max_coord_jump_per_poll", 1.5))
            coord, resynced = choose_best_coord_candidate_with_resync(
                route_planner.current_position,
                coord_candidates,
                max_jump,
                route_planner.available_coords(),
                coord_resync_state,
                int(config.get("movement", {}).get("coord_resync_confirm_steps", 3)),
                float(config.get("movement", {}).get("coord_resync_max_distance", 5.0)),
            )
            log(
                f"step={step} coord={coord} coord_candidates={coord_candidates} ore_points={points} "
                f"dark_ore_points={dark_points} "
                f"vectors={navigator.key_vectors}"
            )

            if coord is None:
                navigator.stop()
                log("movement_stopped=coord_unreadable_or_implausible")
                time.sleep(0.5)
                continue

            if resynced:
                navigator.stop()
                log(f"coord_resynced={coord}")

            route_planner.set_current_position(coord)
            skipped_absent = 0
            skipped_permanent = 0
            while True:
                if active_target_id is not None:
                    next_node = route_planner.get_available_node(active_target_id)
                    if next_node is None:
                        active_target_id = None
                else:
                    next_node = None

                if active_target_id is None:
                    next_node = route_planner.choose_next_node()
                    if next_node is not None:
                        active_target_id = next_node.node_id

                if next_node is None:
                    active_target_id = None
                    log(
                        f"no_target=true skipped_absent={skipped_absent} "
                        f"skipped_permanent={skipped_permanent}"
                    )
                    break

                distance = route_planner.distance_from_current(next_node.coord)
                log(f"target={next_node} distance={distance}")
                decision = decide_route_action(
                    distance_to_target=distance,
                    ore_detected=bool(points),
                    tracking_radius=tracking_radius,
                    reached_distance=reached_distance,
                    mining_enabled=allow_mining,
                    dark_ore_detected=bool(dark_points),
                    dark_exclude_distance=dark_exclude_distance,
                    dark_exclusion_enabled=dark_detection_enabled,
                )
                log(f"decision={decision.action.value} reason={decision.reason}")

                if decision.action is RouteAction.MARK_ABSENT:
                    confirmations = absence_confirmations.get(next_node.node_id, 0) + 1
                    absence_confirmations[next_node.node_id] = confirmations
                    if confirmations < absence_confirm_steps:
                        decision = RouteDecision(
                            RouteAction.MOVE_TO_NODE,
                            f"ore_absence_unconfirmed_{confirmations}/{absence_confirm_steps}",
                        )
                        log(
                            f"absence_unconfirmed={next_node.node_id} "
                            f"count={confirmations}/{absence_confirm_steps}"
                        )
                        break

                    skipped_absent += 1
                    route_planner.mark_absent(next_node.node_id, cooldown_seconds=absence_cooldown)
                    absence_confirmations.pop(next_node.node_id, None)
                    active_target_id = None
                    log(f"marked_absent={next_node.node_id} cooldown={absence_cooldown}")
                    continue

                absence_confirmations.pop(next_node.node_id, None)

                if decision.action is RouteAction.MARK_PERMANENT_EXCLUDED:
                    skipped_permanent += 1
                    route_planner.exclude_permanently(next_node.coord)
                    add_permanent_exclusion(permanent_exclusions_path, next_node, "dark_minimap_icon")
                    absence_confirmations.pop(next_node.node_id, None)
                    active_target_id = None
                    log(f"permanent_excluded={next_node.coord} node_id={next_node.node_id}")
                    continue

                break

            if next_node is None:
                break

            if decision.action is RouteAction.WAIT_FOR_POSITION:
                time.sleep(0.5)
                continue

            if decision.action is RouteAction.HOLD_AT_NODE:
                navigator.stop()
                log("reached_target=true mining_attempted=false probe_mine_disabled=true")
                break

            if decision.action is RouteAction.READY_TO_MINE:
                navigator.stop()
                hover_result = mining_interactor.right_click_hover_target(frame)
                log(
                    f"mining_hover_found={hover_result.found} point={hover_result.point} "
                    f"reason={hover_result.reason} scanned={hover_result.scanned_points}"
                )
                post_frame = capture.capture_client_region()
                post_minimap = capture.crop_minimap(post_frame, config)
                post_points = recognize_ore_points(post_minimap, config)
                log(f"post_mining_ore_points={post_points}")
                if hover_result.found and not post_points:
                    route_planner.mark_mined(
                        next_node.node_id,
                        cooldown_seconds=int(mining_cfg.get("cooldown_seconds", 180)),
                    )
                    absence_confirmations.pop(next_node.node_id, None)
                    active_target_id = None
                    log("mining_attempted=true mining_verified=true")
                else:
                    route_planner.mark_absent(
                        next_node.node_id,
                        cooldown_seconds=int(mining_cfg.get("failure_cooldown_seconds", 20)),
                    )
                    absence_confirmations.pop(next_node.node_id, None)
                    active_target_id = None
                    log("mining_attempted=true mining_verified=false")
                time.sleep(float(config.get("movement", {}).get("poll_interval", 0.8)))
                continue

            navigator.move_towards(coord, next_node.coord)
            diag = navigator.last_diagnostics
            path_failure_text = ""
            if diag.action.startswith("recover_"):
                failures = path_failures.get(next_node.node_id, 0) + 1
                path_failures[next_node.node_id] = failures
                recovery_limit = max(1, int(movement_cfg.get("target_recovery_limit", 3)))
                path_failure_text = f" path_failure={failures}/{recovery_limit}"
                if failures >= recovery_limit:
                    cooldown = int(movement_cfg.get("path_blocked_cooldown_seconds", 90))
                    route_planner.mark_absent(next_node.node_id, cooldown_seconds=cooldown)
                    path_failures[next_node.node_id] = 0
                    absence_confirmations.pop(next_node.node_id, None)
                    active_target_id = None
                    path_failure_text += f" path_blocked_cooldown={cooldown}"
            elif diag.progress is not None and diag.progress >= 0.0:
                path_failures[next_node.node_id] = 0
            log(
                f"movement_action={diag.action} held={diag.held_key} turn={diag.turn_key} "
                f"distance={diag.distance_to_target} progress={diag.progress} "
                f"alignment={diag.alignment} stuck_samples={diag.stuck_samples} "
                f"forward_vector={diag.forward_vector} skipped_absent={skipped_absent}{path_failure_text}"
            )
            time.sleep(float(config.get("movement", {}).get("poll_interval", 0.8)))

        final_frame = capture.capture_client_region()
        final_coord = read_player_position(final_frame, config)
        if final_coord is not None:
            navigator.observe_position(final_coord)
        log(f"final_coord={final_coord}")
        log(f"learned_vectors={navigator.key_vectors}")
    finally:
        navigator.stop()
        log_file.write_text("\n".join(lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the mining router")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--screenshot", default=None, help="Path to a screenshot file for minimap detection testing")
    parser.add_argument("--movement-probe", type=int, default=0, help="Run a bounded live movement probe and exit")
    parser.add_argument("--probe-mine", action="store_true", help="Allow the live movement probe to press the mining key")
    parser.add_argument("--probe-log", default="debug_output/live_probe.log")
    parser.add_argument("--detect-screen-once", action="store_true", help="Capture one frame and save screen-object overlay")
    parser.add_argument("--local-nav-screenshot", default=None, help="Run local-navigation sectors on a saved screenshot")
    parser.add_argument("--local-nav-once", action="store_true", help="Capture one frame and save local-navigation sector report")
    parser.add_argument("--prepare-local-nav-dataset", action="store_true", help="Build local-navigation bootstrap data from saved frames")
    parser.add_argument("--local-nav-source-dir", action="append", default=None, help="Directory with saved frames for local navigation")
    parser.add_argument("--local-nav-output-dir", default="data/local_navigation_dataset", help="Output directory for local-navigation data")
    parser.add_argument("--local-nav-max-frames", type=int, default=0, help="Limit local-navigation source frames")
    parser.add_argument("--local-nav-no-reports", action="store_true", help="Skip local-navigation report images")
    parser.add_argument("--local-nav-no-copy", action="store_true", help="Do not copy source frames into local-navigation output")
    parser.add_argument("--local-nav-review-pack", default=None, help="Build local-navigation QA reports from metadata.jsonl")
    parser.add_argument("--local-nav-review-output-dir", default="data/local_navigation_review_pack", help="Output directory for local-navigation QA reports")
    parser.add_argument("--local-nav-review-max-per-bucket", type=int, default=8, help="Maximum local-navigation QA reports per bucket")
    parser.add_argument("--prepare-local-nav-outcome-dataset", action="store_true", help="Build action/outcome local-navigation labels from saved route probe metadata")
    parser.add_argument("--local-nav-outcome-source", action="append", default=None, help="Saved route-probe metadata.jsonl file or directory")
    parser.add_argument("--local-nav-outcome-output-dir", default="data/local_navigation_outcome_dataset", help="Output directory for outcome-labeled local-navigation data")
    parser.add_argument("--local-nav-outcome-max-samples", type=int, default=0, help="Limit outcome-labeled samples after diverse selection")
    parser.add_argument("--local-nav-outcome-no-copy", action="store_true", help="Do not copy source frames into the outcome dataset")
    parser.add_argument("--local-nav-outcome-no-reports", action="store_true", help="Skip local-navigation reports for outcome samples")
    parser.add_argument("--collect-local-nav-live", type=int, default=0, help="Collect live local-navigation movement samples")
    parser.add_argument("--local-nav-live-output-dir", default="data/live_navigation_actions", help="Output directory for live local-navigation samples")
    parser.add_argument("--local-nav-live-forward-duration", type=float, default=0.35)
    parser.add_argument("--local-nav-live-turn-duration", type=float, default=0.22)
    parser.add_argument("--local-nav-live-back-duration", type=float, default=0.28)
    parser.add_argument("--local-nav-live-report-interval", type=int, default=12)
    parser.add_argument("--local-nav-live-no-coords", action="store_true", help="Skip coordinate OCR during live local-navigation collection")
    parser.add_argument("--local-nav-live-return", action="store_true", help="Try to return toward the start coordinate after collection")
    parser.add_argument("--local-nav-live-return-steps", type=int, default=40)
    parser.add_argument("--route-live-test", type=int, default=0, help="Move toward one database mining node and save route probe frames")
    parser.add_argument("--route-live-output-dir", default="data/live_route_probe", help="Output directory for live route probe")
    parser.add_argument("--route-live-target-coord", type=int, default=0, help="Optional exact target coordinate id")
    parser.add_argument("--route-live-zone", type=int, default=0, help="Optional mining database zone override")
    parser.add_argument("--route-live-auto-zone", action="store_true", help="Choose the closest mining DB zone from the current coordinate")
    parser.add_argument("--route-live-min-target-distance", type=float, default=0.0)
    parser.add_argument("--route-live-cycle-targets", type=int, default=1)
    parser.add_argument("--route-live-reached-distance", type=float, default=0.35)
    parser.add_argument("--route-live-report-interval", type=int, default=1)
    parser.add_argument(
        "--route-live-frame-interval",
        type=float,
        default=-1.0,
        help="Seconds between ordinary saved route frames; state/action changes are saved immediately",
    )
    parser.add_argument("--route-live-poll-interval", type=float, default=0.0)
    parser.add_argument("--route-live-coord-interval", type=float, default=0.0)
    parser.add_argument("--route-live-visual-stuck-min-delta", type=float, default=3.0)
    parser.add_argument("--route-live-visual-stuck-window", type=int, default=0)
    parser.add_argument("--route-live-route-db", default=None, help="Override the route/mining JSON database")
    parser.add_argument("--route-live-follow-loop", action="store_true", help="Follow route_loop waypoints from the selected JSON")
    parser.add_argument("--route-live-enable-mining", action="store_true", help="Enable minimap-confirmed mining during route-live")
    parser.add_argument("--route-live-enable-combat-loot", action="store_true", help="Enable one bounded target-interact loot attempt after confirmed combat clear")
    parser.add_argument("--route-live-preflight", action="store_true", help="Validate route/mining inputs without opening the game or sending input")
    parser.add_argument(
        "--route-live-stop-file",
        default=None,
        help="Cooperatively stop route-live after active combat, mining, loot, or death recovery drains",
    )
    parser.add_argument(
        "--route-live-disable-combat",
        action="store_true",
        help="Disable route-live combat handling for navigation/frame-collection tests",
    )
    parser.add_argument("--collect-vision-data", type=int, default=0, help="Collect live frames for the ML vision dataset")
    parser.add_argument("--vision-data-dir", default=None, help="Output directory for collected vision data")
    parser.add_argument("--vision-interval", type=float, default=None, help="Seconds between collected frames")
    parser.add_argument("--vision-no-autolabel", action="store_true", help="Save raw frames without bootstrap YOLO labels")
    parser.add_argument("--vision-no-overlays", action="store_true", help="Skip detection overlay PNGs during dataset collection")
    parser.add_argument("--vision-turn-scan", action="store_true", help="Turn in place while collecting vision frames")
    parser.add_argument("--vision-turn-segment", type=float, default=2.0, help="Seconds to hold A/D during turn-scan collection")
    parser.add_argument("--screen-model-path", default=None, help="Optional YOLO weights for screen-object detection")
    parser.add_argument("--screen-model-only", action="store_true", help="Use only YOLO screen-object detections")
    parser.add_argument("--screen-bootstrap-only", action="store_true", help="Use only CV/bootstrap screen-object detections")
    parser.add_argument("--prepare-vision-training", action="store_true", help="Create YOLO train/val split from bootstrap labels")
    parser.add_argument("--train-vision-model", action="store_true", help="Train a YOLO screen-object model")
    parser.add_argument("--vision-val-fraction", type=float, default=0.2)
    parser.add_argument("--vision-train-model", default="yolov8n.pt")
    parser.add_argument("--vision-train-epochs", type=int, default=10)
    parser.add_argument("--vision-train-imgsz", type=int, default=640)
    parser.add_argument("--vision-train-batch", type=int, default=4)
    parser.add_argument("--vision-train-device", default=None)
    parser.add_argument("--vision-train-workers", type=int, default=0)
    parser.add_argument("--vision-train-project", default="runs/vision")
    parser.add_argument("--vision-train-name", default="screen_objects_bootstrap")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.route_live_route_db:
        config.setdefault("route", {})["database_path"] = args.route_live_route_db
    if args.route_live_zone:
        config.setdefault("route", {})["locations"] = [int(args.route_live_zone)]
    if args.route_live_follow_loop:
        config.setdefault("route", {})["follow_route_loop"] = True
    if args.route_live_enable_mining:
        config.setdefault("mining", {})["route_live_enabled"] = True
    if args.route_live_enable_combat_loot:
        config.setdefault("post_combat_loot", {})["enabled"] = True
    if args.route_live_stop_file:
        config.setdefault("training_capture", {}).setdefault("route_live", {})[
            "external_stop_file"
        ] = str(Path(args.route_live_stop_file).resolve())
    if args.screen_model_path:
        model_cfg = config.setdefault("screen_objects", {}).setdefault("model", {})
        model_cfg["enabled"] = True
        model_cfg["path"] = args.screen_model_path
        if args.screen_model_only:
            model_cfg["mode"] = "model_only"
    if args.screen_bootstrap_only:
        model_cfg = config.setdefault("screen_objects", {}).setdefault("model", {})
        model_cfg["mode"] = "bootstrap_only"

    if args.route_live_preflight:
        preflight = build_route_live_preflight(config)
        print(json.dumps(preflight, indent=2))
        sys.exit(0 if preflight["ready_offline"] else 2)

    if args.route_live_test > 0 and not bool(
        config.get("portfolio", {}).get("allow_live_input", False)
    ):
        raise RuntimeError(
            "Route-live input is disabled in this portfolio snapshot. "
            "Use an ignored local config and explicitly set "
            "portfolio.allow_live_input=true only for an authorized setup."
        )

    if args.screenshot:
        screenshot_path = Path(args.screenshot)
        frame = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        capture = ScreenCapture(config)
        minimap_cfg = config.get("minimap", {})
        minimap_width = int(minimap_cfg.get("width", 300))
        minimap_height = int(minimap_cfg.get("height", 300))

        if frame.shape[1] <= minimap_width + 40 and frame.shape[0] <= minimap_height + 120:
            minimap = frame[:minimap_height, :minimap_width]
            coord_region_top = minimap_height
            coord_region_bottom = coord_region_top + int(config.get("player_position", {}).get("height", 50))
            coord_region_width = int(config.get("player_position", {}).get("width", 200))
            pos_region = frame[coord_region_top:coord_region_bottom, :coord_region_width]
            player_coord = read_player_position(pos_region, config, direct_region=True)
        else:
            minimap = capture.crop_minimap(frame, config)
            player_coord = read_player_position(frame, config)

        points = recognize_ore_points(minimap, config)
        debug_artifacts = save_debug_artifacts(
            minimap,
            points,
            Path(config.get("debug", {}).get("output_dir", "debug_output")),
            config,
        )
        print(f"Detected {len(points)} ore points: {points}")
        print(f"Player coordinate read: {player_coord}")
        print(f"Debug images saved to: {debug_artifacts}")
        sys.exit(0)

    if args.local_nav_screenshot:
        screenshot_path = Path(args.local_nav_screenshot)
        frame = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        navigation = analyze_local_navigation(frame, config)
        output_dir = Path(config.get("debug", {}).get("output_dir", "debug_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "local_navigation_report.png"
        metadata_path = output_dir / "local_navigation_report.json"
        cv2.imwrite(str(report_path), draw_local_navigation_report(frame, navigation))
        metadata_path.write_text(json.dumps(navigation.to_dict(), indent=2), encoding="utf-8")
        print(f"Best sector: {navigation.best_sector}")
        print(f"Center blocked: {navigation.center_blocked}")
        print(f"Recommended turn: {navigation.recommended_turn}")
        print(f"Report saved to: {report_path}")
        print(f"Metadata saved to: {metadata_path}")
        sys.exit(0)

    if args.prepare_local_nav_dataset:
        source_paths = collect_local_navigation_sources(args.local_nav_source_dir)
        max_frames = args.local_nav_max_frames if args.local_nav_max_frames > 0 else None
        if not source_paths:
            print("No local-navigation source frames found")
            sys.exit(2)
        summary = prepare_local_navigation_dataset(
            source_paths,
            args.local_nav_output_dir,
            config,
            max_frames=max_frames,
            save_reports=not args.local_nav_no_reports,
            copy_images=not args.local_nav_no_copy,
        )
        print(f"Local-navigation dataset: {summary.output_dir}")
        print(f"Input sources: {summary.source_count}")
        print(f"Frames: {summary.frame_count}")
        print(f"Reports: {summary.report_count}")
        print(f"Copied images: {summary.copied_image_count}")
        sys.exit(0)

    if args.local_nav_review_pack:
        summary = prepare_local_navigation_review_pack(
            args.local_nav_review_pack,
            args.local_nav_review_output_dir,
            config,
            max_per_bucket=args.local_nav_review_max_per_bucket,
        )
        print(f"Local-navigation review pack: {summary.output_dir}")
        print(f"Items: {summary.item_count}")
        print(f"Reports: {summary.report_count}")
        print(f"Buckets: {summary.bucket_counts}")
        sys.exit(0)

    if args.prepare_local_nav_outcome_dataset:
        metadata_paths = collect_route_probe_metadata_sources(args.local_nav_outcome_source)
        if not metadata_paths:
            print("No saved route-probe metadata found")
            sys.exit(2)
        movement_cfg = config.get("movement", {})
        max_samples = args.local_nav_outcome_max_samples if args.local_nav_outcome_max_samples > 0 else None
        summary = prepare_local_navigation_outcome_dataset(
            metadata_paths,
            args.local_nav_outcome_output_dir,
            config,
            max_samples=max_samples,
            save_reports=not args.local_nav_outcome_no_reports,
            copy_images=not args.local_nav_outcome_no_copy,
            min_coord_delta=float(
                movement_cfg.get("stuck_min_coord_delta", movement_cfg.get("stuck_min_progress", 0.03))
            ),
            min_distance_progress=float(movement_cfg.get("stuck_min_progress", 0.03)),
            min_visual_motion_delta=float(movement_cfg.get("visual_stuck_min_delta", 3.0)),
            min_visual_stuck_samples=max(1, int(movement_cfg.get("visual_stuck_window", 2))),
        )
        print(f"Local-navigation outcome dataset: {summary.output_dir}")
        print(f"Source metadata files: {summary.source_metadata_count}")
        print(f"Episodes: {summary.episode_count}")
        print(f"Samples: {summary.sample_count}")
        print(f"Labels: {summary.label_counts}")
        print(f"Splits: {summary.split_counts}")
        print(f"Skipped rows: {summary.skipped_count}")
        print(f"Reports: {summary.report_count}")
        print(f"Copied images: {summary.copied_image_count}")
        sys.exit(0)

    if args.collect_local_nav_live > 0:
        summary = collect_live_navigation_movement(
            config,
            output_dir=args.local_nav_live_output_dir,
            steps=args.collect_local_nav_live,
            forward_duration=args.local_nav_live_forward_duration,
            turn_duration=args.local_nav_live_turn_duration,
            back_duration=args.local_nav_live_back_duration,
            report_interval=args.local_nav_live_report_interval,
            read_coordinates=not args.local_nav_live_no_coords,
            return_to_start=args.local_nav_live_return,
            return_steps=args.local_nav_live_return_steps,
        )
        print(f"Live local-navigation dataset: {summary.output_dir}")
        print(f"Steps: {summary.steps}")
        print(f"Start coord: {summary.start_coord}")
        print(f"Final coord: {summary.final_coord}")
        print(f"Moved distance: {summary.moved_distance}")
        print(f"Visual motion mean: {summary.visual_motion_mean}")
        print(f"Stuck events: {summary.stuck_events}")
        print(f"Return attempted: {summary.return_attempted}")
        print(f"Returned distance: {summary.returned_distance}")
        sys.exit(0)

    if args.route_live_test > 0:
        preflight = build_route_live_preflight(config)
        if not preflight["ready_offline"]:
            print(json.dumps({"route_live_preflight": preflight}, indent=2))
            sys.exit(2)
        summary = collect_route_to_node_probe(
            config,
            output_dir=args.route_live_output_dir,
            steps=args.route_live_test,
            target_coord=args.route_live_target_coord or None,
            zone_id=args.route_live_zone or None,
            auto_zone=args.route_live_auto_zone,
            min_target_distance=args.route_live_min_target_distance,
            cycle_targets=args.route_live_cycle_targets,
            reached_distance=args.route_live_reached_distance,
            report_interval=args.route_live_report_interval,
            frame_interval=args.route_live_frame_interval if args.route_live_frame_interval >= 0.0 else None,
            poll_interval=args.route_live_poll_interval or None,
            coord_interval=args.route_live_coord_interval or None,
            visual_stuck_min_delta=args.route_live_visual_stuck_min_delta,
            visual_stuck_window=args.route_live_visual_stuck_window or None,
            combat_enabled=False if args.route_live_disable_combat else None,
            mining_enabled=True if args.route_live_enable_mining else None,
            post_combat_loot_enabled=(
                True if args.route_live_enable_combat_loot else None
            ),
        )
        print(f"Live route probe: {summary.output_dir}")
        print(f"Target coord: {summary.target_coord}")
        print(f"Target node: {summary.target_node_id}")
        print(f"Target zone: {summary.target_zone_id}")
        print(f"Target ore: {summary.target_ore_type}")
        print(f"Start coord: {summary.start_coord}")
        print(f"Final coord: {summary.final_coord}")
        print(f"Start distance: {summary.start_distance}")
        print(f"Final distance: {summary.final_distance}")
        print(f"Reached: {summary.reached}")
        print(f"Stuck events: {summary.stuck_events}")
        print(f"Completed targets: {summary.completed_targets}")
        print(f"Mining attempts: {summary.mining_attempts}")
        print(f"Mining successes: {summary.mining_successes}")
        print(f"Mining failures: {summary.mining_failures}")
        print(f"Post-combat loot attempts: {summary.post_combat_loot_attempts}")
        print(f"Detected zone: {summary.detected_zone_name}")
        print(f"Route zone ids: {summary.route_zone_ids}")
        print(f"Termination: {summary.termination_reason}")
        sys.exit(0)

    if args.movement_probe > 0:
        sys.exit(run_movement_probe(config, args.movement_probe, args.probe_mine, args.probe_log))

    if args.local_nav_once:
        capture = ScreenCapture(config)
        hwnd = capture.find_window()
        if hwnd is None:
            print("Game window not found")
            sys.exit(2)
        capture.activate_window()
        frame = capture.capture_client_region()
        navigation = analyze_local_navigation(frame, config)
        output_dir = Path(config.get("debug", {}).get("output_dir", "debug_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / "local_navigation_frame.png"
        report_path = output_dir / "local_navigation_report.png"
        metadata_path = output_dir / "local_navigation_report.json"
        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(report_path), draw_local_navigation_report(frame, navigation))
        metadata_path.write_text(json.dumps(navigation.to_dict(), indent=2), encoding="utf-8")
        print(f"Best sector: {navigation.best_sector}")
        print(f"Center blocked: {navigation.center_blocked}")
        print(f"Recommended turn: {navigation.recommended_turn}")
        print(f"Frame saved to: {frame_path}")
        print(f"Report saved to: {report_path}")
        print(f"Metadata saved to: {metadata_path}")
        sys.exit(0)

    if args.detect_screen_once:
        capture = ScreenCapture(config)
        hwnd = capture.find_window()
        if hwnd is None:
            print("Game window not found")
            sys.exit(2)
        capture.activate_window()
        frame = capture.capture_client_region()
        detections = detect_screen_objects(frame, config)
        output_dir = Path(config.get("debug", {}).get("output_dir", "debug_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / "screen_objects_frame.png"
        overlay_path = output_dir / "screen_objects_overlay.png"
        report_path = output_dir / "screen_objects_report.png"
        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(overlay_path), draw_detections(frame, detections))
        cv2.imwrite(
            str(report_path),
            draw_detection_report(
                frame,
                detections,
                model_path=config.get("screen_objects", {}).get("model", {}).get("path"),
            ),
        )
        print(f"Detected {len(detections)} screen objects")
        for detection in detections:
            print(detection.to_dict())
        print(f"Frame saved to: {frame_path}")
        print(f"Overlay saved to: {overlay_path}")
        print(f"Report saved to: {report_path}")
        sys.exit(0)

    if args.collect_vision_data > 0:
        dataset_cfg = config.get("vision_dataset", {})
        output_dir = args.vision_data_dir or dataset_cfg.get("output_dir", "data/vision_dataset")
        interval = (
            float(args.vision_interval)
            if args.vision_interval is not None
            else float(dataset_cfg.get("collect_interval", 0.2))
        )
        turn_scan_stop = threading.Event()
        turn_scan_thread: threading.Thread | None = None
        turn_scan_input_controller: InputController | None = None
        if args.vision_turn_scan:
            capture = ScreenCapture(config)
            hwnd = capture.find_window()
            turn_scan_input_controller = create_input_controller(config, hwnd)

            def turn_scan_loop() -> None:
                key_index = 0
                keys = ("A", "D")
                while not turn_scan_stop.is_set():
                    key = keys[key_index % len(keys)]
                    key_index += 1
                    turn_scan_input_controller.press_key(key)
                    turn_scan_stop.wait(max(0.1, args.vision_turn_segment))
                    turn_scan_input_controller.release_key(key)
                    turn_scan_stop.wait(0.15)

            turn_scan_thread = threading.Thread(target=turn_scan_loop, daemon=True)
            turn_scan_thread.start()

        try:
            summary = collect_vision_dataset(
                config,
                count=args.collect_vision_data,
                interval=interval,
                output_dir=output_dir,
                auto_label=not args.vision_no_autolabel and bool(dataset_cfg.get("auto_label", True)),
                save_overlays=not args.vision_no_overlays and bool(dataset_cfg.get("save_overlays", True)),
            )
        finally:
            if args.vision_turn_scan:
                turn_scan_stop.set()
                if turn_scan_input_controller is not None:
                    turn_scan_input_controller.release_movement_keys()
                if turn_scan_thread is not None:
                    turn_scan_thread.join(timeout=2.0)
        print(f"Vision dataset output: {summary.output_dir}")
        print(f"Frames: {summary.frame_count}")
        print(f"Labeled frames: {summary.labeled_count}")
        print(f"Overlays: {summary.overlay_count}")
        print(f"Detections: {summary.detection_count}")
        sys.exit(0)

    if args.prepare_vision_training or args.train_vision_model:
        dataset_cfg = config.get("vision_dataset", {})
        output_dir = args.vision_data_dir or dataset_cfg.get("output_dir", "data/vision_dataset")
        split_summary = prepare_yolo_training_split(
            output_dir,
            val_fraction=args.vision_val_fraction,
        )
        print(f"Dataset: {split_summary.dataset_dir}")
        print(f"Training YAML: {split_summary.yaml_path}")
        print(f"Train images: {split_summary.train_count}")
        print(f"Val images: {split_summary.val_count}")
        print(f"Skipped without labels: {split_summary.skipped_without_labels}")

        if args.train_vision_model:
            training_summary = train_yolo_model(
                split_summary.yaml_path,
                model=args.vision_train_model,
                epochs=args.vision_train_epochs,
                imgsz=args.vision_train_imgsz,
                batch=args.vision_train_batch,
                device=args.vision_train_device,
                workers=args.vision_train_workers,
                project=args.vision_train_project,
                name=args.vision_train_name,
            )
            print(f"Training run: {training_summary.run_dir}")
            print(f"Best weights: {training_summary.best_weights}")
            print(f"Last weights: {training_summary.last_weights}")
        sys.exit(0)

    router = MiningRouter(config)
    router.run()
