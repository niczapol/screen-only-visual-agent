from __future__ import annotations

from dataclasses import dataclass, field
import time

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.engine.input_executor import InputExecutor


@dataclass
class FakeBackend:
    calls: list[tuple] = field(default_factory=list)

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def move_cursor(self, x: int, y: int) -> None:
        self.calls.append(("move_cursor", x, y))

    def mouse_down(self, button: MouseButton) -> None:
        self.calls.append(("mouse_down", button.value))

    def mouse_up(self, button: MouseButton) -> None:
        self.calls.append(("mouse_up", button.value))

    def move_mouse_relative(self, delta_x: int, delta_y: int) -> None:
        self.calls.append(("mouse_move_relative", delta_x, delta_y))


def test_tap_key_is_scheduled_without_sleep() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    executor.set_input_permitted(True, now=1.0)
    executor.submit(
        ControlIntent(
            owner=CommandGroup.COMBAT,
            reason="attack",
            commands=(
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.COMBAT,
                    key="E",
                    duration=0.20,
                    reason="attack",
                    exclusive=True,
                ),
            ),
        ),
        now=1.0,
    )

    executor.run_due(now=1.0)
    assert backend.calls == [("key_down", "E")]
    assert executor.held_keys == {"E"}

    executor.run_due(now=1.19)
    assert executor.held_keys == {"E"}
    executor.run_due(now=1.20)
    assert backend.calls[-1] == ("key_up", "E")
    assert not executor.held_keys


def test_emergency_followup_can_be_scheduled_at_released_key_gap() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    executor.set_input_permitted(True, now=5.0)
    commands = (
        Command(
            kind=CommandKind.TAP_KEY,
            group=CommandGroup.COMBAT,
            key="F",
            duration=0.05,
            reason="critical_heal",
            exclusive=True,
        ),
        Command(
            kind=CommandKind.TAP_KEY,
            group=CommandGroup.COMBAT,
            key="V",
            duration=0.05,
            not_before=5.17,
            reason="critical_heal_followup",
            exclusive=True,
        ),
    )
    executor.submit(
        ControlIntent(CommandGroup.COMBAT, commands, "critical_heal_pair"),
        now=5.0,
    )

    executor.run_due(now=5.0)
    executor.run_due(now=5.05)
    assert not executor.held_keys
    executor.run_due(now=5.17)
    assert executor.held_keys == {"V"}
    assert backend.calls == [
        ("key_down", "F"),
        ("key_up", "F"),
        ("key_down", "V"),
    ]


def test_revoking_input_permission_releases_everything() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    executor.set_input_permitted(True, now=1.0)
    executor.submit(
        ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="travel",
                ),
            ),
            "travel",
        ),
        now=1.0,
    )
    executor.run_due(now=1.0)

    executor.set_input_permitted(False, now=1.1)

    assert backend.calls == [("key_down", "W"), ("key_up", "W")]
    assert not executor.held_keys


def test_drag_is_bounded_and_releases_mouse_button() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend, drag_step_seconds=0.01)
    executor.set_input_permitted(True, now=2.0)
    executor.submit(
        ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.DRAG_RELATIVE,
                    group=CommandGroup.TRAVEL,
                    mouse_button=MouseButton.RIGHT,
                    delta_x=12,
                    duration=0.03,
                    reason="route_yaw",
                ),
            ),
            "route_yaw",
        ),
        now=2.0,
    )

    executor.run_due(now=2.0)
    executor.run_due(now=2.01)
    executor.run_due(now=2.02)
    executor.run_due(now=2.03)

    assert backend.calls[0] == ("mouse_down", "right")
    assert backend.calls[-1] == ("mouse_up", "right")
    assert sum(call[1] for call in backend.calls if call[0] == "mouse_move_relative") == 12
    assert not executor.held_buttons


def test_latest_intent_cancels_stale_drag_tail_without_releasing_forward() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend, drag_step_seconds=0.01)
    executor.set_input_permitted(True, now=2.0)
    executor.submit(
        ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="route_forward",
                ),
                Command(
                    kind=CommandKind.DRAG_RELATIVE,
                    group=CommandGroup.TRAVEL,
                    mouse_button=MouseButton.RIGHT,
                    delta_x=120,
                    duration=0.30,
                    reason="route_yaw",
                ),
            ),
            "route_yaw",
        ),
        now=2.0,
    )
    executor.run_due(now=2.0)
    executor.run_due(now=2.10)
    moved_before_cancel = sum(
        call[1] for call in backend.calls if call[0] == "mouse_move_relative"
    )

    executor.submit(
        ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="route_aligned",
                ),
            ),
            "route_aligned",
        ),
        now=2.10,
    )
    executor.run_due(now=2.10)
    executor.run_due(now=2.40)

    assert executor.held_keys == {"W"}
    assert not executor.held_buttons
    assert ("key_up", "W") not in backend.calls
    assert backend.calls.count(("mouse_up", "right")) == 1
    assert sum(
        call[1] for call in backend.calls if call[0] == "mouse_move_relative"
    ) == moved_before_cancel
    assert executor.pending_count == 0


def test_run_due_drains_diagnostic_events_instead_of_growing_forever() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    executor.set_input_permitted(True, now=1.0)
    executor.submit(
        ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="travel",
                ),
            ),
            "travel",
        ),
        now=1.0,
    )

    assert executor.run_due(now=1.0)
    assert executor.run_due(now=1.1) == ()
    assert executor.events == []


def test_background_pump_executes_drag_steps_between_observation_frames() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend, drag_step_seconds=0.004)
    now = time.monotonic()
    executor.set_input_permitted(True, now=now)
    executor.start_background()
    try:
        executor.submit(
            ControlIntent(
                CommandGroup.TRAVEL,
                (
                    Command(
                        kind=CommandKind.DRAG_RELATIVE,
                        group=CommandGroup.TRAVEL,
                        mouse_button=MouseButton.RIGHT,
                        delta_x=24,
                        duration=0.04,
                        reason="route_yaw",
                    ),
                ),
                "route_yaw",
            ),
            now=now,
        )
        deadline = time.monotonic() + 0.50
        while time.monotonic() < deadline:
            if ("mouse_up", "right") in backend.calls:
                break
            time.sleep(0.005)
    finally:
        executor.stop_background(now=time.monotonic(), reason="test_stop")

    assert backend.calls[0] == ("mouse_down", "right")
    assert ("mouse_up", "right") in backend.calls
    assert sum(
        call[1] for call in backend.calls if call[0] == "mouse_move_relative"
    ) == 24


def test_submission_rebases_relative_deadline_after_perception_latency() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    executor.set_input_permitted(True, now=10.5)
    executor.submit(
        ControlIntent(
            CommandGroup.RECOVERY,
            (
                Command(
                    kind=CommandKind.CLICK,
                    group=CommandGroup.RECOVERY,
                    mouse_button=MouseButton.LEFT,
                    duration=0.12,
                    reason="visible_dialog",
                    deadline=10.30,
                ),
            ),
            "visible_dialog",
        ),
        now=10.5,
        reference_time=10.0,
    )

    executor.run_due(now=10.5)
    executor.run_due(now=10.62)

    assert backend.calls == [
        ("mouse_down", "left"),
        ("mouse_up", "left"),
    ]


def test_click_waits_for_cursor_settle_within_same_intent() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend, cursor_settle_seconds=0.12)
    executor.set_input_permitted(True, now=20.0)
    executor.submit(
        ControlIntent(
            CommandGroup.RECOVERY,
            (
                Command(
                    kind=CommandKind.MOVE_CURSOR,
                    group=CommandGroup.RECOVERY,
                    point=(147, 375),
                    reason="visible_dialog",
                    deadline=20.30,
                ),
                Command(
                    kind=CommandKind.CLICK,
                    group=CommandGroup.RECOVERY,
                    mouse_button=MouseButton.LEFT,
                    duration=0.12,
                    reason="visible_dialog",
                    exclusive=True,
                    deadline=20.30,
                ),
            ),
            "visible_dialog",
        ),
        now=20.0,
    )

    executor.run_due(now=20.0)
    assert backend.calls == [("move_cursor", 147, 375)]
    executor.run_due(now=20.119)
    assert backend.calls == [("move_cursor", 147, 375)]
    executor.run_due(now=20.12)
    assert backend.calls[-1] == ("mouse_down", "left")
    executor.run_due(now=20.241)
    assert backend.calls[-1] == ("mouse_up", "left")
