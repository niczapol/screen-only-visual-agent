import pytest

from vision_bot.coords import xy_to_coord
from vision_bot.mining_final_approach import (
    MiningFinalApproachAction,
    MiningFinalApproachController,
    coordinate_distance_yards,
)


def _controller(**overrides):
    values = {
        "enabled": True,
        "map_width_yards": 6900.0,
        "map_height_yards": 4600.0,
        "trigger_distance_yards": 19.0,
        "arrival_tolerance_yards": 3.5,
        "post_burst_world_scan_tolerance_yards": 6.0,
        "mounted_speed_yards_per_second": 16.8,
        "foot_speed_yards_per_second": 7.0,
        "min_burst_seconds": 0.05,
        "burst_stop_short_yards": 1.0,
        "mounted_max_burst_seconds": 1.15,
        "foot_max_burst_seconds": 2.40,
        "precision_coord_interval_seconds": 0.15,
        "feedback_settle_seconds": 0.35,
        "feedback_timeout_seconds": 1.75,
    }
    values.update(overrides)
    return MiningFinalApproachController(
        {"mining": {"final_database_approach": values}}
    )


def test_tanaris_coordinate_distance_uses_different_axis_scales():
    origin = xy_to_coord(50.0, 50.0)

    horizontal = coordinate_distance_yards(
        origin,
        xy_to_coord(50.01, 50.0),
        map_width_yards=6900.0,
        map_height_yards=4600.0,
    )
    vertical = coordinate_distance_yards(
        origin,
        xy_to_coord(50.0, 50.01),
        map_width_yards=6900.0,
        map_height_yards=4600.0,
    )

    assert horizontal == pytest.approx(0.69)
    assert vertical == pytest.approx(0.46)


def test_final_zone_stops_and_requires_a_later_fresh_coordinate():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.06, 50.0)  # 4.14 physical yards.

    entered = controller.observe(current, target, now=10.0, coord_fresh=True)
    same_tick = controller.observe(current, target, now=10.0, coord_fresh=True)
    stale_tick = controller.observe(current, target, now=10.2, coord_fresh=False)
    rechecked = controller.observe(current, target, now=10.5, coord_fresh=True)

    assert entered.action == MiningFinalApproachAction.STOP_RECHECK
    assert same_tick.action == MiningFinalApproachAction.STOP_RECHECK
    assert stale_tick.action == MiningFinalApproachAction.STOP_RECHECK
    assert rechecked.action == MiningFinalApproachAction.ALIGN


def test_empirical_braking_envelope_covers_observed_mounted_sample_step():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.27, 50.0)  # 18.63 physical yards.

    entered = controller.observe(current, target, now=10.0, coord_fresh=True)
    rechecked = controller.observe(current, target, now=10.15, coord_fresh=True)
    burst = controller.commit_burst(current, target, now=10.15, mounted=True)

    assert entered.action == MiningFinalApproachAction.STOP_RECHECK
    assert rechecked.action == MiningFinalApproachAction.ALIGN
    assert burst.burst_seconds == pytest.approx((18.63 - 1.0) / 16.8)
    assert burst.burst_seconds < controller.mounted_max_burst_seconds
    assert controller.precision_coord_interval_seconds == pytest.approx(0.15)


def test_aligned_final_approach_emits_one_calculated_burst_only():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.06, 50.0)
    controller.observe(current, target, now=10.0, coord_fresh=True)
    controller.observe(current, target, now=10.5, coord_fresh=True)

    burst = controller.commit_burst(
        current,
        target,
        now=10.5,
        mounted=True,
    )
    waiting = controller.observe(
        current,
        target,
        now=10.7,
        coord_fresh=True,
    )

    assert burst.action == MiningFinalApproachAction.WAIT_FEEDBACK
    assert burst.burst_seconds == pytest.approx((4.14 - 1.0) / 16.8)
    assert waiting.action == MiningFinalApproachAction.WAIT_FEEDBACK
    with pytest.raises(RuntimeError):
        controller.commit_burst(current, target, now=10.8, mounted=True)


def test_single_burst_accepts_fresh_arrival_feedback():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.06, 50.0)
    controller.observe(current, target, now=10.0, coord_fresh=True)
    controller.observe(current, target, now=10.5, coord_fresh=True)
    controller.commit_burst(current, target, now=10.5, mounted=True)

    arrived = controller.observe(
        xy_to_coord(50.01, 50.0),
        target,
        now=11.0,
        coord_fresh=True,
    )

    assert arrived.action == MiningFinalApproachAction.ARRIVED
    assert arrived.reason == "database_anchor_reached_after_single_burst"


def test_single_burst_miss_fails_instead_of_microstepping_again():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.10, 50.0)  # 6.9 yd: outside post-burst handoff.
    controller.observe(current, target, now=10.0, coord_fresh=True)
    controller.observe(current, target, now=10.5, coord_fresh=True)
    controller.commit_burst(current, target, now=10.5, mounted=True)

    failed = controller.observe(
        xy_to_coord(50.10, 50.0),
        target,
        now=11.0,
        coord_fresh=True,
    )

    assert failed.action == MiningFinalApproachAction.FAILED
    assert failed.reason == "ore_final_database_burst_missed"


def test_single_burst_near_miss_hands_off_to_tooltip_gated_world_scan():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.08, 50.0)  # 5.52 yd: not an initial arrival.

    entered = controller.observe(current, target, now=10.0, coord_fresh=True)
    rechecked = controller.observe(current, target, now=10.5, coord_fresh=True)
    controller.commit_burst(current, target, now=10.5, mounted=True)
    arrived = controller.observe(current, target, now=11.0, coord_fresh=True)

    assert entered.action == MiningFinalApproachAction.STOP_RECHECK
    assert rechecked.action == MiningFinalApproachAction.ALIGN
    assert arrived.action == MiningFinalApproachAction.ARRIVED
    assert arrived.reason == "database_anchor_close_after_single_burst"
    assert controller.snapshot(now=11.0)["post_burst_world_scan_tolerance_yards"] == 6.0


def test_dismounted_post_combat_burst_uses_foot_specific_cap():
    controller = _controller()
    target = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(50.0, 50.31325)  # 14.4095 physical yards.
    controller.observe(current, target, now=10.0, coord_fresh=True)
    controller.observe(current, target, now=10.15, coord_fresh=True)

    burst = controller.commit_burst(current, target, now=10.15, mounted=False)

    expected_distance = controller.distance_yards(current, target)
    assert burst.burst_seconds == pytest.approx((expected_distance - 1.0) / 7.0)
    assert burst.burst_seconds > controller.mounted_max_burst_seconds
    assert burst.burst_seconds < controller.foot_max_burst_seconds


def test_v0821_run1_mounted_residual_hands_off_to_live_marker_stage():
    controller = _controller()
    target = xy_to_coord(62.00, 39.10)
    origin = xy_to_coord(61.88, 38.82)
    observed_after_burst = xy_to_coord(62.01, 39.04)
    controller.observe(origin, target, now=10.0, coord_fresh=True)
    controller.observe(origin, target, now=10.15, coord_fresh=True)
    controller.commit_burst(origin, target, now=10.15, mounted=True)

    arrived = controller.observe(
        observed_after_burst,
        target,
        now=11.2,
        coord_fresh=True,
    )

    assert controller.distance_yards(observed_after_burst, target) == pytest.approx(
        2.845,
        abs=0.01,
    )
    assert arrived.action == MiningFinalApproachAction.ARRIVED
