import pytest

from vision_bot.heading_navigation import (
    HeadingNavigationController,
    VisibleHeadingTracker,
)
from vision_bot.movement import InputController, MovementNavigator
from vision_bot.route_motion import RouteMotionController


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


def _controller(*, enabled: bool = True, fallback: bool = True):
    inputs = DummyInput()
    navigator = MovementNavigator(inputs)
    controller = RouteMotionController(
        navigator,
        enabled=enabled,
        fallback_to_legacy=fallback,
        heading_tracker=VisibleHeadingTracker(max_age_seconds=0.5),
        heading_controller=HeadingNavigationController(),
        reset_after_seconds=0.75,
    )
    return controller, navigator, inputs


def test_route_motion_uses_visible_heading_and_held_keys():
    controller, navigator, inputs = _controller()
    controller.observe_heading(330.0, now=10.0)

    snapshot = controller.move_towards(
        5000500000,
        5000400000,
        now=10.0,
    )

    assert snapshot.mode == "visible_heading"
    assert snapshot.command is not None
    assert snapshot.command.turn_key == "A"
    assert navigator.heading_held_keys == {"W"}
    assert inputs.pressed == ["W"]
    assert len(inputs.tapped) == 1
    assert inputs.tapped[0][0] == "A"
    assert inputs.tapped[0][1] == pytest.approx(26.0 / 110.0)


def test_route_motion_falls_back_when_visible_heading_is_stale():
    controller, navigator, _inputs = _controller(fallback=True)
    controller.observe_heading(330.0, now=10.0)

    snapshot = controller.move_towards(
        5000500000,
        5000400000,
        now=10.6,
    )

    assert snapshot.mode == "legacy_heading_unavailable"
    assert navigator.held_key == "W"


def test_route_motion_can_fail_closed_instead_of_using_legacy():
    controller, navigator, inputs = _controller(fallback=False)

    snapshot = controller.move_towards(
        5000500000,
        5000400000,
        now=10.0,
    )

    assert snapshot.mode == "heading_fail_closed"
    assert navigator.held_key is None
    assert inputs.pressed == []


def test_route_motion_suspend_releases_heading_owned_keys():
    controller, navigator, inputs = _controller()
    controller.observe_heading(330.0, now=10.0)
    controller.move_towards(5000500000, 5000400000, now=10.0)

    controller.suspend()

    assert navigator.heading_held_keys == set()
    assert inputs.released == ["W"]
    assert controller.last_snapshot.mode == "suspended"


def test_route_motion_faces_coordinate_without_holding_forward():
    controller, navigator, inputs = _controller()
    controller.observe_heading(330.0, now=10.0)

    snapshot = controller.face_towards(
        5000500000,
        5000400000,
        now=10.0,
    )

    assert snapshot.mode == "face_visible_heading"
    assert not snapshot.forward_held
    assert snapshot.command is not None
    assert snapshot.command.turn_key == "A"
    assert navigator.heading_held_keys == set()
    assert inputs.pressed == []
    assert len(inputs.tapped) == 1
    assert inputs.tapped[0][0] == "A"
    assert inputs.tapped[0][1] == pytest.approx(26.0 / 110.0)


def test_route_motion_reports_face_alignment_without_movement():
    controller, navigator, inputs = _controller()
    controller.observe_heading(0.0, now=10.0)

    snapshot = controller.face_towards(
        5000500000,
        5000400000,
        now=10.0,
    )

    assert snapshot.aligned
    assert navigator.heading_held_keys == set()
    assert inputs.pressed == []
