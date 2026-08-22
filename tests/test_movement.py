import ctypes
import types

import vision_bot.movement as movement_module
from vision_bot.heading_navigation import HeadingNavigationCommand
from vision_bot.movement import (
    KEY_CODES,
    MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP,
    InputController,
    MouseController,
    MouseSteeringController,
    MovementNavigator,
    WM_KEYDOWN,
    WM_KEYUP,
)


class DummyInput(InputController):
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.released: list[str] = []
        self.tapped: list[tuple[str, float]] = []

    def press_key(self, key: str) -> None:
        self.pressed.append(key)

    def release_key(self, key: str) -> None:
        self.released.append(key)

    def tap_key(self, key: str, duration: float = 0.15) -> None:
        self.tapped.append((key, duration))


class ManualTimer:
    def __init__(self, interval: float, callback, args=()) -> None:
        self.interval = interval
        self.callback = callback
        self.args = args
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback(*self.args)


class DummyMouse:
    def __init__(self) -> None:
        self.drags: list[tuple[int, float, float, bool]] = []
        self.continuous_drags: list[tuple[int, float, bool]] = []

    def right_drag_relative(
        self,
        delta_x: int,
        *,
        duration: float,
        step_interval: float,
        restore_cursor: bool,
    ) -> None:
        self.drags.append((delta_x, duration, step_interval, restore_cursor))

    def right_drag_continuous(
        self,
        delta_x_per_step: int,
        *,
        stop_event,
        step_interval: float,
        restore_cursor: bool,
    ) -> tuple[int, float]:
        self.continuous_drags.append((delta_x_per_step, step_interval, restore_cursor))
        stop_event.wait(step_interval)
        return delta_x_per_step, step_interval


def test_input_key_codes_cover_configurable_actions():
    assert KEY_CODES["TAB"] == 0x09
    assert KEY_CODES["ESC"] == 0x1B
    assert KEY_CODES["ESCAPE"] == 0x1B
    assert KEY_CODES["C"] == 0x43
    assert KEY_CODES["X"] == 0x58
    assert KEY_CODES["1"] == 0x31
    assert KEY_CODES["F7"] == 0x76
    assert KEY_CODES["F8"] == 0x77
    assert KEY_CODES["F9"] == 0x78


def test_mouse_steering_maps_context_speed_and_direction_to_rmb_drag():
    mouse = DummyMouse()
    steering = MouseSteeringController(
        mouse,  # type: ignore[arg-type]
        enabled=True,
        pixels_per_second={"default": 100.0, "route": 200.0},
        min_duration_seconds=0.04,
        max_duration_seconds=0.40,
        step_interval_seconds=0.01,
    )

    assert steering.turn("A", duration=0.10, context="route")
    assert steering.turn("D", duration=0.02, context="default")

    assert mouse.drags == [
        (-20, 0.10, 0.01, True),
        (4, 0.04, 0.01, True),
    ]
    assert steering.last_result is not None
    assert steering.last_result.turn_key == "D"


def test_mouse_steering_continuous_turn_starts_and_stops_background_drag():
    mouse = DummyMouse()
    steering = MouseSteeringController(
        mouse,  # type: ignore[arg-type]
        enabled=True,
        pixels_per_second={"combat": 300.0},
        step_interval_seconds=0.01,
        min_pixels=2,
    )

    assert steering.start_continuous_turn("D", context="combat")
    steering.stop_continuous_turn()

    assert mouse.continuous_drags == [(3, 0.01, True)]
    assert steering.last_result is not None
    assert steering.last_result.context == "combat"


def test_right_drag_relative_releases_rmb_and_restores_cursor(monkeypatch):
    sent: list[tuple[int, int]] = []
    cursor_positions: list[tuple[int, int]] = []
    current_cursor = [123, 456]

    def get_cursor_pos(pointer) -> int:
        pointer._obj.x = current_cursor[0]
        pointer._obj.y = current_cursor[1]
        return 1

    def set_cursor_pos(x: int, y: int) -> int:
        current_cursor[:] = [x, y]
        cursor_positions.append((x, y))
        return 1

    fake_user32 = types.SimpleNamespace(
        GetCursorPos=get_cursor_pos,
        SetCursorPos=set_cursor_pos,
    )
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=fake_user32))
    monkeypatch.setattr(
        movement_module,
        "_send_input",
        lambda inputs: sent.extend((item.mi.dwFlags, item.mi.dx) for item in inputs) or len(inputs),
    )
    monkeypatch.setattr(movement_module.time, "sleep", lambda _seconds: None)

    MouseController().right_drag_relative(
        12,
        duration=0.03,
        step_interval=0.01,
    )

    assert sent[0] == (MOUSEEVENTF_RIGHTDOWN, 0)
    assert sent[-1] == (MOUSEEVENTF_RIGHTUP, 0)
    assert cursor_positions[-1] == (123, 456)
    assert max(x for x, _y in cursor_positions) == 135


def test_heading_turn_uses_mouse_without_tapping_a_or_d():
    inputs = DummyInput()
    mouse = DummyMouse()
    steering = MouseSteeringController(
        mouse,  # type: ignore[arg-type]
        enabled=True,
        pixels_per_second={"route": 200.0},
        min_duration_seconds=0.0,
    )
    navigator = MovementNavigator(inputs, mouse_steering=steering)

    navigator.apply_heading_command(
        HeadingNavigationCommand(
            hold_forward=True,
            turn_key="D",
            desired_heading_degrees=45.0,
            heading_error_degrees=45.0,
            braking=False,
            reason="heading_turn",
            turn_hold_seconds=0.10,
        ),
        current_coord=5000500000,
        target_coord=5000600000,
    )

    assert inputs.pressed == ["W"]
    assert inputs.tapped == []
    assert mouse.drags == [(20, 0.10, 0.012, True)]


def test_heading_turn_caps_mouse_arc_while_forward_is_held():
    inputs = DummyInput()
    mouse = DummyMouse()
    steering = MouseSteeringController(
        mouse,  # type: ignore[arg-type]
        enabled=True,
        pixels_per_second={"route": 200.0},
        min_duration_seconds=0.0,
        route_moving_max_duration_seconds=0.08,
    )
    navigator = MovementNavigator(inputs, mouse_steering=steering)

    navigator.apply_heading_command(
        HeadingNavigationCommand(
            hold_forward=True,
            turn_key="D",
            desired_heading_degrees=45.0,
            heading_error_degrees=45.0,
            braking=False,
            reason="heading_turn",
            turn_hold_seconds=0.35,
        ),
        current_coord=5000500000,
        target_coord=5000600000,
    )

    assert mouse.drags == [(16, 0.08, 0.012, True)]


def test_movement_navigator_uses_calibrated_vectors():
    controller = DummyInput()
    navigator = MovementNavigator(controller, turn_duration=0.0)
    navigator.forward_vector = (0.0, 0.1)

    navigator.move_towards(5000500000, 5100500000)

    assert controller.tapped == [("D", 0.0)]
    assert controller.pressed == ["W"]
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "turn_and_forward"
    assert navigator.last_diagnostics.turn_key == "D"


def test_movement_navigator_does_not_keep_turning_while_progressing():
    controller = DummyInput()
    navigator = MovementNavigator(controller, turn_duration=0.0)
    navigator.forward_vector = (0.0, 0.1)

    navigator.move_towards(5000500000, 5100500000)
    controller.tapped.clear()
    navigator.move_towards(5010500000, 5100500000)

    assert controller.tapped == []
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "forward_progress"
    assert navigator.last_diagnostics.progress is not None
    assert navigator.last_diagnostics.progress > 0


def test_movement_navigator_keeps_forward_for_progress_below_stuck_threshold():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.02,
        turn_duration=0.0,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    controller.tapped.clear()
    navigator.move_towards(5001500000, 5100500000)

    assert controller.tapped == []
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "forward_progress"
    assert navigator.last_diagnostics.progress is not None
    assert 0 < navigator.last_diagnostics.progress < navigator.stuck_min_progress


def test_movement_navigator_course_corrects_when_distance_gets_worse():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.01,
        turn_in_place_duration=0.0,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(4990500000, 5100500000)

    assert controller.tapped == [("A", 0.0)]
    assert navigator.last_diagnostics.action == "course_correct_and_forward"
    assert navigator.last_diagnostics.progress is not None
    assert navigator.last_diagnostics.progress < 0


def test_movement_navigator_keeps_committed_turn_when_candidate_flips():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=99,
        stuck_min_progress=0.01,
        stuck_min_coord_delta=0.01,
        turn_duration=0.0,
        turn_in_place_duration=0.0,
    )
    candidates = iter(["D", "A", "D"])
    navigator.choose_turn_key = lambda _current, _target: next(candidates)

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(4900500000, 5100500000)
    navigator.move_towards(4800500000, 5100500000)

    assert [key for key, _duration in controller.tapped] == ["D", "A", "A"]
    assert navigator.last_diagnostics.action == "course_correct_and_forward"
    assert navigator.last_diagnostics.turn_key == "A"


def test_movement_navigator_positive_progress_releases_committed_turn():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=99,
        stuck_min_progress=0.01,
        stuck_min_coord_delta=0.01,
        turn_duration=0.0,
        turn_in_place_duration=0.0,
    )
    candidates = iter(["D", "A", None, "D"])
    navigator.choose_turn_key = lambda _current, _target: next(candidates)

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(4900500000, 5100500000)
    navigator.move_towards(5050500000, 5100500000)
    navigator.move_towards(4950500000, 5100500000)

    assert [key for key, _duration in controller.tapped] == ["D", "A", "D"]
    assert navigator.last_diagnostics.action == "course_correct_and_forward"
    assert navigator.last_diagnostics.turn_key == "D"


def test_movement_navigator_target_change_releases_committed_turn():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=99,
        stuck_min_progress=0.01,
        stuck_min_coord_delta=0.01,
        turn_duration=0.0,
        turn_in_place_duration=0.0,
    )
    candidates = iter(["D", "A", "D"])
    navigator.choose_turn_key = lambda _current, _target: next(candidates)

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(4900500000, 5100500000)
    navigator.move_towards(4900500000, 4000500000)

    assert [key for key, _duration in controller.tapped] == ["D", "A", "D"]
    assert navigator.last_diagnostics.turn_key == "D"


def test_movement_navigator_does_not_recover_when_coordinates_still_move():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.01,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5002500000, 5100500000)

    assert controller.tapped == []
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "forward_motion"
    assert navigator.last_diagnostics.coord_delta is not None
    assert navigator.last_diagnostics.coord_delta >= 0.01
    assert navigator.last_diagnostics.movement_delta == navigator.last_diagnostics.coord_delta
    assert navigator.last_diagnostics.distance_progress == navigator.last_diagnostics.progress


def test_movement_navigator_course_corrects_when_moving_but_not_toward_target():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.01,
        turn_duration=0.0,
        turn_in_place_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    controller.tapped.clear()
    navigator.move_towards(5000501000, 5100500000)

    assert controller.tapped == [("D", 0.0)]
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action in {"course_correct_and_forward", "turn_in_place_and_forward"}
    assert navigator.last_diagnostics.alignment is not None
    assert navigator.last_diagnostics.alignment < navigator.turn_alignment_threshold


def test_movement_navigator_resets_recovery_attempts_after_real_movement():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.01,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5002500000, 5100500000)
    navigator.move_towards(5002500000, 5100500000)

    assert [tap[0] for tap in controller.tapped] == ["SPACE", "SPACE"]
    assert navigator.last_diagnostics.action == "recover_jump"


def test_movement_navigator_records_zero_vector_for_blocked_key():
    controller = DummyInput()
    navigator = MovementNavigator(controller)
    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5000500000, 5100500000)

    assert navigator.key_vectors["W"] == (0.0, 0.0)
    assert controller.pressed == ["W"]
    assert navigator.choose_direction(5000500000, 5100500000) == "W"


def test_movement_navigator_can_continue_forward_without_new_coordinate():
    controller = DummyInput()
    navigator = MovementNavigator(controller)

    navigator.continue_forward(target_coord=5100500000)
    navigator.continue_forward(target_coord=5100500000)

    assert controller.pressed == ["W"]
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "continue_forward"
    assert navigator.last_diagnostics.target_coord == 5100500000


def test_heading_command_holds_forward_and_turn_without_timed_taps():
    controller = DummyInput()
    navigator = MovementNavigator(controller)
    command = HeadingNavigationCommand(
        hold_forward=True,
        turn_key="A",
        desired_heading_degrees=45.0,
        heading_error_degrees=30.0,
        braking=False,
        reason="steer_to_heading",
    )

    navigator.apply_heading_command(
        command,
        current_coord=5000500000,
        target_coord=5100500000,
    )
    navigator.apply_heading_command(
        command,
        current_coord=5001500000,
        target_coord=5100500000,
    )

    assert controller.pressed == ["W", "A"]
    assert controller.tapped == []
    assert controller.released == []
    assert navigator.heading_held_keys == {"W", "A"}
    assert navigator.last_diagnostics.held_keys == ("W", "A")
    assert navigator.last_diagnostics.heading_error_degrees == 30.0


def test_heading_command_releases_old_turn_before_pressing_opposite_turn():
    events: list[tuple[str, str]] = []

    class OrderedInput(DummyInput):
        def press_key(self, key: str) -> None:
            super().press_key(key)
            events.append(("press", key))

        def release_key(self, key: str) -> None:
            super().release_key(key)
            events.append(("release", key))

    controller = OrderedInput()
    navigator = MovementNavigator(controller)
    left = HeadingNavigationCommand(True, "A", 45.0, 30.0, False, "steer_to_heading")
    right = HeadingNavigationCommand(True, "D", 315.0, -30.0, False, "steer_to_heading")

    navigator.apply_heading_command(
        left,
        current_coord=5000500000,
        target_coord=5100500000,
    )
    events.clear()
    navigator.apply_heading_command(
        right,
        current_coord=5001500000,
        target_coord=5100500000,
    )

    assert events == [("release", "A"), ("press", "D")]
    assert navigator.heading_held_keys == {"W", "D"}


def test_heading_pulse_uses_bounded_tap_without_blocking_forward():
    controller = DummyInput()
    navigator = MovementNavigator(controller)
    command = HeadingNavigationCommand(
        True,
        "A",
        45.0,
        18.0,
        False,
        "pulse_to_heading",
        turn_hold_seconds=0.07,
    )

    navigator.apply_heading_command(
        command,
        current_coord=5000500000,
        target_coord=5100500000,
    )

    assert controller.pressed == ["W"]
    assert controller.tapped == [("A", 0.07)]
    assert controller.released == []
    assert navigator.heading_held_keys == {"W"}
    assert navigator.held_key == "W"


def test_heading_pivot_releases_forward_and_legacy_takeover_releases_turn():
    controller = DummyInput()
    navigator = MovementNavigator(controller)
    pivot = HeadingNavigationCommand(False, "A", 180.0, 120.0, True, "pivot_to_heading")

    navigator.apply_heading_command(
        pivot,
        current_coord=5000500000,
        target_coord=5100500000,
    )
    navigator.continue_forward(5000500000, 5100500000)

    assert controller.pressed == ["A", "W"]
    assert controller.released == ["A"]
    assert navigator.heading_held_keys == set()
    assert navigator.held_key == "W"


def test_heading_command_expires_pending_recovery_detour(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: clock["now"])
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.35,
        obstacle_detour_settle_duration=0.0,
    )
    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    assert navigator.recovery_detour_active is True

    clock["now"] = 10.36
    navigator.apply_heading_command(
        HeadingNavigationCommand(True, None, 90.0, 0.0, False, "heading_aligned"),
        current_coord=5000500000,
        target_coord=5100500000,
    )

    assert navigator.recovery_detour_active is False
    assert navigator.recovery_suppresses_stuck is False
    assert navigator.last_diagnostics.detour_phase is None


def test_heading_progress_resets_recovery_to_jump_first(monkeypatch):
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: 10.0)
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.35,
    )
    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")

    navigator.apply_heading_command(
        HeadingNavigationCommand(True, None, 90.0, 0.0, False, "heading_aligned"),
        current_coord=5005500000,
        target_coord=5100500000,
    )
    next_recovery = navigator.recover_from_obstacle(
        5005500000,
        preferred_turn_key="A",
    )

    assert next_recovery.action == "recover_jump"
    assert next_recovery.turn_key is None


def test_movement_navigator_turns_in_place_when_facing_away():
    controller = DummyInput()
    navigator = MovementNavigator(controller, turn_in_place_duration=0.0, turn_in_place_alignment=-0.2)
    navigator.forward_vector = (0.0, 0.1)

    navigator.move_towards(5000500000, 5000490000)

    assert controller.tapped == [("A", 0.0)]
    assert controller.pressed == ["W"]
    assert navigator.last_diagnostics.action == "turn_in_place_and_forward"


def test_movement_navigator_recovers_when_distance_stops_improving():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.01,
        obstacle_back_duration=0.0,
        obstacle_turn_duration=0.0,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5000500000, 5100500000)

    assert controller.tapped == [("SPACE", 0.0)]
    assert controller.pressed == ["W"]
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "recover_jump"
    assert navigator.last_diagnostics.turn_key is None


def test_movement_navigator_turns_after_jump_recovery_fails():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        stuck_check_window=1,
        stuck_min_progress=0.01,
        obstacle_back_duration=0.0,
        obstacle_turn_duration=0.0,
        obstacle_jump_duration=0.0,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5000500000, 5100500000)
    navigator.move_towards(5000500000, 5100500000)

    assert controller.tapped == [("SPACE", 0.0), ("SPACE", 0.0), ("D", 0.0)]
    assert navigator.held_key == "W"
    assert navigator.last_diagnostics.action == "recover_jump_turn"
    assert navigator.last_diagnostics.turn_key == "D"


def test_recovery_detour_stops_and_requests_replan_when_feedback_is_delayed(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: clock["now"])
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.85,
        obstacle_detour_settle_duration=0.0,
    )

    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")

    assert navigator.recovery_detour_active is True
    assert navigator.recovery_suppresses_stuck is True
    assert controller.tapped[-2:] == [("SPACE", 0.0), ("D", 0.40)]

    clock["now"] = 10.86
    navigator.continue_forward(5000500000, 5100500000)

    assert navigator.recovery_detour_active is False
    assert navigator.recovery_suppresses_stuck is False
    assert navigator.held_key is None
    assert navigator.last_diagnostics.action == "recover_detour_replan"
    assert navigator.last_diagnostics.detour_phase == "replan"
    assert navigator.last_diagnostics.detour_turn_key == "D"
    assert navigator.last_diagnostics.detour_duration == 0.85
    assert all(key != "A" for key, _duration in controller.tapped)


def test_every_recovery_turn_is_preceded_by_jump():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_back_duration=0.0,
        obstacle_turn_duration=0.4,
        obstacle_jump_duration=0.08,
        obstacle_detour_duration=0.0,
    )

    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")

    turn_indexes = [index for index, (key, _duration) in enumerate(controller.tapped) if key in {"A", "D"}]
    assert turn_indexes
    assert all(controller.tapped[index - 1][0] == "SPACE" for index in turn_indexes)


def test_fresh_coordinate_cancels_pending_recovery_detour(monkeypatch):
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: 10.0)
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.85,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    controller.tapped.clear()
    navigator.move_towards(5005500000, 5100500000)

    assert navigator.recovery_detour_active is False
    assert ("A", 0.40) not in controller.tapped


def test_unchanged_coordinate_does_not_cancel_pending_recovery_detour(monkeypatch):
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: 10.0)
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.85,
    )

    navigator.move_towards(5000500000, 5100500000)
    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    navigator.move_towards(5000500000, 5100500000)

    assert navigator.recovery_detour_active is True
    assert navigator.last_diagnostics.action == "recover_detour_forward"


def test_stop_cancels_pending_recovery_detour(monkeypatch):
    monkeypatch.setattr(movement_module.time, "monotonic", lambda: 10.0)
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.40,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.85,
    )

    navigator.recover_from_obstacle(5000500000)
    navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    controller.tapped.clear()
    navigator.stop()

    assert navigator.recovery_detour_active is False
    assert navigator.recovery_suppresses_stuck is False
    assert controller.tapped == []


def test_recovery_keeps_one_turn_direction_until_target_progress_resumes():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.0,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.0,
    )

    navigator.recover_from_obstacle(5000500000)
    first = navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
    second = navigator.recover_from_obstacle(5000500000, preferred_turn_key="A")

    assert first.turn_key == "D"
    assert second.turn_key == "D"


def test_recovery_switches_side_after_bounded_failed_attempts():
    controller = DummyInput()
    navigator = MovementNavigator(
        controller,
        obstacle_turn_duration=0.0,
        obstacle_jump_duration=0.0,
        obstacle_detour_duration=0.0,
        obstacle_recovery_attempts_per_side=4,
    )

    attempts = [
        navigator.recover_from_obstacle(5000500000, preferred_turn_key="D")
        for _ in range(6)
    ]

    assert [attempt.turn_key for attempt in attempts[1:4]] == ["D", "D", "D"]
    assert attempts[4].turn_key == "A"
    assert attempts[5].turn_key == "A"


def test_forward_vector_tracks_recent_heading_after_turn():
    controller = DummyInput()
    navigator = MovementNavigator(controller)
    navigator._last_observed_coord = 5000500000
    navigator._last_observed_key = "W"
    navigator.held_key = "W"
    navigator.key_vectors["W"] = (1.0, 0.0)

    navigator.observe_position(5000501000)

    assert navigator.forward_vector is not None
    assert navigator.forward_vector[1] > navigator.forward_vector[0]


def test_input_controller_can_post_keys_to_target_window(monkeypatch):
    posted_messages: list[tuple[int, int, int, int]] = []

    fake_user32 = types.SimpleNamespace(
        MapVirtualKeyW=lambda _vk, _mode: 0x20,
        PostMessageW=lambda hwnd, message, wparam, lparam: posted_messages.append(
            (hwnd, message, wparam, lparam)
        )
        or 1,
    )
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=fake_user32))

    controller = InputController(target_hwnd=1234, backend="post_message")
    controller.press_key("D")
    controller.release_key("D")

    assert posted_messages[0][:3] == (1234, WM_KEYDOWN, 0x44)
    assert posted_messages[1][:3] == (1234, WM_KEYUP, 0x44)
    assert posted_messages[1][3] & (1 << 31)
