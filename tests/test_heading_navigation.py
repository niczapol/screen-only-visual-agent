import pytest

from vision_bot.heading_navigation import (
    HeadingNavigationController,
    VisibleHeadingTracker,
    heading_to_coord_degrees,
    signed_heading_error_degrees,
)


CENTER = 5000500000
NORTH = 5000400000
SOUTH = 5000600000
EAST = 6000500000
WEST = 4000500000


def test_heading_to_coord_matches_wow_map_cardinals() -> None:
    assert heading_to_coord_degrees(CENTER, NORTH) == 0.0
    assert heading_to_coord_degrees(CENTER, WEST) == 90.0
    assert heading_to_coord_degrees(CENTER, SOUTH) == 180.0
    assert heading_to_coord_degrees(CENTER, EAST) == 270.0
    assert heading_to_coord_degrees(CENTER, CENTER) is None


def test_heading_to_coord_can_correct_non_square_zone_axes() -> None:
    southeast = 5100510000

    raw = heading_to_coord_degrees(CENTER, southeast)
    tanaris_scaled = heading_to_coord_degrees(
        CENTER,
        southeast,
        coordinate_scale_x=6900.0,
        coordinate_scale_y=4600.0,
    )

    assert raw == pytest.approx(225.0)
    assert tanaris_scaled == pytest.approx(236.3099, abs=0.001)


def test_signed_heading_error_uses_shortest_wraparound() -> None:
    assert signed_heading_error_degrees(355.0, 5.0) == 10.0
    assert signed_heading_error_degrees(5.0, 355.0) == -10.0
    assert signed_heading_error_degrees(180.0, 0.0) == 180.0


def test_controller_pivots_without_forward_for_large_error() -> None:
    controller = HeadingNavigationController(pivot_degrees=60.0)

    command = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=180.0,
    )

    assert command.turn_key == "A"
    assert command.braking
    assert not command.hold_forward
    assert command.reason == "pivot_to_heading"
    assert command.turn_hold_seconds == 0.35


def test_controller_steers_while_moving_for_moderate_error() -> None:
    controller = HeadingNavigationController(pivot_degrees=60.0)

    command = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=330.0,
    )

    assert command.turn_key == "A"
    assert not command.braking
    assert command.hold_forward


def test_controller_uses_bounded_pulse_then_deadband_for_moderate_error() -> None:
    controller = HeadingNavigationController(
        turn_engage_degrees=10.0,
        turn_release_degrees=4.0,
        turn_rate_degrees_per_second=120.0,
        min_turn_pulse_seconds=0.03,
        max_turn_pulse_seconds=0.20,
    )

    first = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=348.0,
    )
    deadband = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=354.0,
    )
    released = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=357.0,
    )

    assert first.turn_key == "A"
    assert first.reason == "pulse_to_heading"
    assert first.turn_hold_seconds == 8.0 / 120.0
    assert deadband.turn_key is None
    assert released.turn_key is None
    assert released.hold_forward


def test_controller_suppresses_fast_small_opposite_turn_but_allows_large_error() -> None:
    controller = HeadingNavigationController(
        turn_engage_degrees=9.0,
        opposite_turn_lock_seconds=0.45,
        opposite_turn_engage_degrees=28.0,
    )

    left = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=350.0,
        now=10.0,
    )
    suppressed = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=10.0,
        now=10.1,
    )
    strong_right = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=40.0,
        now=10.2,
    )

    assert left.turn_key == "A"
    assert suppressed.turn_key is None
    assert suppressed.reason == "heading_reversal_hysteresis"
    assert suppressed.hold_forward
    assert strong_right.turn_key == "D"


def test_controller_fails_closed_without_fresh_heading() -> None:
    controller = HeadingNavigationController()

    missing = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=None,
    )
    stale = controller.plan(
        current_coord=CENTER,
        target_coord=NORTH,
        heading_degrees=0.0,
        heading_fresh=False,
    )

    assert missing.reason == "heading_unavailable"
    assert stale.reason == "heading_stale"
    assert not missing.hold_forward
    assert not stale.hold_forward


def test_visible_heading_tracker_bridges_short_decode_gaps_only() -> None:
    tracker = VisibleHeadingTracker(max_age_seconds=0.5)

    tracker.observe(361.0, now=10.0)

    assert tracker.current(now=10.4) == (1.0, True)
    assert tracker.current(now=10.6) == (1.0, False)


def test_visible_heading_tracker_ignores_invalid_observations() -> None:
    tracker = VisibleHeadingTracker(max_age_seconds=0.5)
    tracker.observe(90.0, now=10.0)

    tracker.observe(None, now=10.2)

    assert tracker.current(now=10.3) == (90.0, True)
