from vision_bot.hard_stuck_recovery import (
    HardStuckEscapeSettings,
    perform_hard_stuck_escape,
)
from vision_bot.mounting import MountTravelState


class InputStub:
    def __init__(self) -> None:
        self.taps: list[tuple[str, float]] = []

    def tap_key(self, key: str, duration: float = 0.15) -> None:
        self.taps.append((key, duration))


class NavigatorStub:
    def __init__(self) -> None:
        self.stops = 0
        self.turns: list[tuple[str, float, str]] = []

    def stop(self) -> None:
        self.stops += 1

    def turn_character(self, key: str, *, duration: float, context: str) -> None:
        self.turns.append((key, duration, context))


def test_hard_stuck_escape_dismounts_backs_turns_and_leaves_on_foot() -> None:
    input_controller = InputStub()
    navigator = NavigatorStub()
    mount_state = MountTravelState(enabled=True, key="1")
    sleeps: list[float] = []
    settings = HardStuckEscapeSettings(
        dismount_settle_seconds=0.25,
        back_seconds=0.75,
        turn_seconds=1.95,
        turn_pulse_seconds=0.39,
        forward_seconds=0.90,
        remount_delay_seconds=2.0,
    )

    action = perform_hard_stuck_escape(
        navigator=navigator,
        input_controller=input_controller,
        mount_state=mount_state,
        mounted=True,
        turn_key="A",
        settings=settings,
        now=10.0,
        sleep=sleeps.append,
    )

    assert action == "hard_stuck_dismount_reverse_escape"
    assert navigator.stops == 1
    assert input_controller.taps == [("1", 0.06), ("S", 0.75), ("W", 0.90)]
    assert sleeps == [0.25]
    assert len(navigator.turns) == 5
    assert all(turn[0] == "A" and turn[2] == "recovery" for turn in navigator.turns)
    assert abs(sum(turn[1] for turn in navigator.turns) - 1.95) < 1e-9
    assert mount_state.settle_until == 15.85


def test_hard_stuck_escape_does_not_toggle_mount_when_already_on_foot() -> None:
    input_controller = InputStub()
    navigator = NavigatorStub()

    perform_hard_stuck_escape(
        navigator=navigator,
        input_controller=input_controller,
        mount_state=MountTravelState(enabled=True, key="1"),
        mounted=False,
        turn_key="invalid",
        settings=HardStuckEscapeSettings(dismount_settle_seconds=0.0),
        now=0.0,
        sleep=lambda _seconds: None,
    )

    assert all(key != "1" for key, _duration in input_controller.taps)
    assert all(key == "D" for key, _duration, _context in navigator.turns)
