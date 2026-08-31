import json

import cv2
import numpy as np
import pytest

from vision_bot.movement import MouseSteeringController

from vision_bot.live_route_probe import (
    ThreatAvoidanceCommitment,
    LocalAvoidanceCommitment,
    RouteTargetProgressWatchdog,
    SpiritSearchLegProgress,
    _autonomous_route_entry_target,
    _advance_route_stuck_samples,
    _apply_combat_fallback,
    _apply_combat_low_health_heal,
    _apply_emergency_heal_followup,
    _allowed_route_locations,
    _capture_reached_ore_training_sample,
    _choose_route_target,
    _choose_spirit_healer_coordinate_anchor,
    _combined_excluded_coords,
    _block_route_waypoint_window,
    _build_directed_route_follower,
    _configured_route_zone_ids,
    _consume_reached_route_entry,
    _covered_center_tooltip_probe_target,
    _directed_route_steering_target,
    _load_route_nodes,
    _load_mining_candidate_nodes,
    _load_configured_route_entry,
    _load_configured_route_loop,
    _load_configured_route_milestones,
    _mining_access_resume_queue,
    _mining_route_resume_target,
    _mining_world_target_turn_key,
    _mounted_escape_has_mining_conflict,
    _prepare_death_recovery_input,
    _record_target_route_node_milestone,
    _record_target_recovery,
    _target_recovery_can_block,
    _resume_generated_route_entry,
    _route_metadata_action,
    _route_coord_jump_limit,
    _route_entry_strategy,
    _route_entry_start_on_rail_sort_key,
    _route_geometry_sort_key,
    _route_target_reached_distance,
    _route_probe_requires_safe_drain,
    _external_stop_file_requested,
    _route_sort_key,
    _should_local_avoid,
    _should_complete_hazard_egress,
    _should_allow_forbidden_marker_during_hazard_egress,
    _should_handle_death_recovery,
    _should_handle_dismissable_modal,
    _should_save_route_frame,
    _should_read_route_coord,
    _should_prewarm_ore_world_detector,
    _select_stable_start_coord,
    _spirit_healer_local_search_coord,
    _spirit_healer_anchor_due,
    _spirit_search_interact_due,
    _spirit_healer_search_turn_key,
    _single_allowed_route_zone,
    _tap_due_combat_keys,
    _tap_due_low_health_combat_key,
    _turn_key_towards_click_point,
    _with_post_combat_loot_enabled,
    _with_route_live_combat_enabled,
    _with_route_live_mining_enabled,
    build_route_live_preflight,
)
from vision_bot.combat import CombatFallbackState
from vision_bot.movement import InputController
from vision_bot.ore_world_detector import OreWorldDetectorResult


def test_mining_world_target_facing_does_not_reuse_route_heading_inversion():
    assert (
        _mining_world_target_turn_key(
            "D",
            mining_config={},
            route_invert_turn_direction=True,
        )
        == "D"
    )
    assert (
        _mining_world_target_turn_key(
            "D",
            mining_config={"world_target_face_invert_turn_direction": True},
            route_invert_turn_direction=True,
        )
        == "A"
    )


@pytest.mark.parametrize(
    ("active", "coord_hazard", "marker_visible", "expected"),
    [
        (True, False, False, True),
        (True, False, True, False),
        (True, True, False, False),
        (False, False, False, False),
    ],
)
def test_hazard_egress_completes_only_after_geometry_and_visible_marker_are_safe(
    active,
    coord_hazard,
    marker_visible,
    expected,
):
    assert _should_complete_hazard_egress(
        hazard_egress_active=active,
        coord_is_hazard=coord_hazard,
        forbidden_subzone_visible=marker_visible,
    ) is expected


def test_death_recovery_releases_continuous_combat_yaw_before_cursor_work():
    class RouteMotionStub:
        suspended = False

        def suspend(self):
            self.suspended = True

    class MouseSteeringStub:
        stopped = False

        def stop_continuous_turn(self):
            self.stopped = True

    route_motion = RouteMotionStub()
    input_controller = InputController(backend="post_message")
    combat = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)
    combat.start_continuous_face_search("D", keyboard_held=False)
    mouse = MouseSteeringStub()

    _prepare_death_recovery_input(
        route_motion,
        input_controller,
        combat,
        mouse,
    )

    assert route_motion.suspended
    assert mouse.stopped
    assert combat.face_search_held_key is None


@pytest.mark.parametrize(
    ("active", "pending", "planner", "expected"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (True, True, False, False),
        (False, True, True, False),
    ],
)
def test_forbidden_marker_is_allowed_only_while_validated_hazard_egress_is_pending(
    active,
    pending,
    planner,
    expected,
):
    assert _should_allow_forbidden_marker_during_hazard_egress(
        hazard_egress_active=active,
        route_entry_pending=pending,
        planner_available=planner,
    ) is expected


@pytest.mark.parametrize(
    ("mining_enabled", "death_or_blocking_modal", "combat_active", "expected"),
    [
        (True, False, False, True),
        (True, False, True, False),
        (True, True, False, False),
        (False, False, False, False),
    ],
)
def test_ore_world_prewarm_never_blocks_initial_combat_or_death(
    mining_enabled,
    death_or_blocking_modal,
    combat_active,
    expected,
):
    assert (
        _should_prewarm_ore_world_detector(
            mining_enabled=mining_enabled,
            death_or_blocking_modal=death_or_blocking_modal,
            combat_active=combat_active,
        )
        is expected
    )


def test_route_stuck_counter_preserves_fresh_zero_progress_across_skipped_ocr_tick():
    samples = _advance_route_stuck_samples(
        0,
        coord_delta=0.0,
        movement_stuck=True,
    )
    samples = _advance_route_stuck_samples(
        samples,
        coord_delta=None,
        movement_stuck=False,
    )
    samples = _advance_route_stuck_samples(
        samples,
        coord_delta=0.0,
        movement_stuck=True,
    )

    assert samples == 2


def test_route_stuck_counter_resets_only_on_fresh_coordinate_progress():
    assert _advance_route_stuck_samples(
        2,
        coord_delta=0.04,
        movement_stuck=False,
    ) == 0


@pytest.mark.parametrize(
    ("mining", "probe", "candidate", "expected"),
    [
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
    ],
)
def test_only_active_or_actionable_mining_prevents_mounted_escape(
    mining,
    probe,
    candidate,
    expected,
):
    assert (
        _mounted_escape_has_mining_conflict(
            mining_active=mining,
            tooltip_probe_active=probe,
            candidate_available=candidate,
        )
        is expected
    )


def test_route_entry_skips_precision_pivot_when_start_is_already_on_rail():
    route = [
        1000100000,
        2000100000,
    ]

    sort_key = _route_entry_start_on_rail_sort_key(
        1500104000,
        route,
        max_distance=0.65,
    )

    assert sort_key == pytest.approx(0.5)
    assert (
        _route_entry_start_on_rail_sort_key(
            1500110000,
            route,
            max_distance=0.65,
        )
        is None
    )


def test_spirit_search_interact_spams_only_while_navigating_on_interval():
    due, next_at = _spirit_search_interact_due(
        "death_recovery_navigate_local_spirit_search",
        now=10.0,
        next_at=0.0,
        interval=0.30,
    )
    assert due
    assert next_at == 10.30

    due, unchanged = _spirit_search_interact_due(
        "death_recovery_approach_spirit_target",
        now=10.31,
        next_at=next_at,
        interval=0.30,
    )
    assert not due
    assert unchanged == next_at

    due, unchanged = _spirit_search_interact_due(
        "death_recovery_navigate_local_spirit_search",
        now=10.29,
        next_at=next_at,
        interval=0.30,
    )
    assert not due
    assert unchanged == next_at

    due, next_at = _spirit_search_interact_due(
        "death_recovery_navigate_local_spirit_search",
        now=10.31,
        next_at=next_at,
        interval=0.30,
    )
    assert due
    assert next_at == pytest.approx(10.61)

    due, unchanged = _spirit_search_interact_due(
        "death_recovery_return_to_life",
        now=11.0,
        next_at=next_at,
        interval=0.30,
    )
    assert not due
    assert unchanged == next_at


def test_route_target_progress_watchdog_detects_motion_without_target_progress():
    watchdog = RouteTargetProgressWatchdog(
        timeout_seconds=6.0,
        min_improvement=0.08,
    )

    assert not watchdog.observe(target_coord=1, distance=1.32, now=0.0, active=True)
    assert not watchdog.observe(target_coord=1, distance=1.29, now=3.0, active=True)
    assert watchdog.observe(target_coord=1, distance=1.31, now=6.1, active=True)


def test_route_target_progress_watchdog_resets_for_real_progress_and_pauses():
    watchdog = RouteTargetProgressWatchdog(
        timeout_seconds=6.0,
        min_improvement=0.08,
    )

    assert not watchdog.observe(target_coord=1, distance=1.5, now=0.0, active=True)
    assert not watchdog.observe(target_coord=1, distance=1.4, now=5.0, active=True)
    assert not watchdog.observe(target_coord=1, distance=1.4, now=10.0, active=False)
    assert not watchdog.observe(target_coord=1, distance=1.4, now=20.0, active=True)
    assert not watchdog.observe(target_coord=2, distance=1.3, now=30.0, active=True)


def test_block_route_waypoint_window_skips_adjacent_points_of_same_obstacle():
    route_loop = [100, 200, 300, 400, 500]
    node = MiningNode(
        zone_id=162,
        coord=300,
        ore_type="Route Waypoint",
        node_id=-3,
        route_index=2,
        source="route_loop_waypoint",
    )

    blocked = _block_route_waypoint_window(
        set(),
        target_node=node,
        route_loop=route_loop,
        forward_padding=2,
    )

    assert blocked == {300, 400, 500}


def test_blocked_entry_waypoint_does_not_poison_the_route_loop():
    node = MiningNode(
        zone_id=162,
        coord=250,
        ore_type="route_entry",
        node_id=-4_000_001,
        route_index=2,
        source="route_entry_waypoint",
    )

    blocked = _block_route_waypoint_window(
        set(),
        target_node=node,
        route_loop=[100, 200, 300, 400, 500],
        forward_padding=2,
    )

    assert blocked == {250}
from vision_bot.combat import CombatFallbackState, CombatHealState, CombatState, PeriodicCombatKeyState
from vision_bot.config import load_config
from vision_bot.coords import xy_to_coord
from vision_bot.game_state import GameState
from vision_bot.mining import MiningCycleController, MiningOutcome
from vision_bot.movement import MovementNavigator
from vision_bot.route_planner import MiningNode, RouteEntryPlan
from vision_bot.screen_objects import BoundingBox
from vision_bot.threat_detection import ThreatDetection


def test_covered_center_tooltip_probe_uses_nearby_available_db_hint_only():
    config = {
        "mining": {
            "route_live_enabled": True,
            "minimap_tooltip_probe": {
                "center_fallback_enabled": True,
                "center_fallback_max_distance_coord": 0.08,
            },
        },
        "recognition": {
            "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        },
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    near = MiningNode(
        zone_id=162,
        coord=xy_to_coord(50.04, 50.02),
        ore_type="Mithril",
        node_id=7,
    )
    far = MiningNode(
        zone_id=162,
        coord=xy_to_coord(50.20, 50.0),
        ore_type="Gold",
        node_id=8,
    )

    assert _covered_center_tooltip_probe_target(
        current_coord=current,
        nodes=[far, near],
        mining_cycle=controller,
        now=10.0,
        minimap_shape=(200, 200, 3),
        config=config,
    ) == (-7, (100, 96))

    controller.cooldowns[near.coord] = 20.0
    assert _covered_center_tooltip_probe_target(
        current_coord=current,
        nodes=[far, near],
        mining_cycle=controller,
        now=11.0,
        minimap_shape=(200, 200, 3),
        config=config,
    ) is None


def test_spirit_healer_coordinate_anchor_selects_ascension_point_near_live_spawn():
    config = load_config("config.yaml")

    anchor = _choose_spirit_healer_coordinate_anchor(4945590600, config)

    assert anchor is not None
    name, coord, distance, arrival_distance = anchor
    assert name == "tanaris_ascension_494_590"
    assert coord == 4940590000
    assert 0.07 < distance < 0.08
    assert arrival_distance == 0.18


@pytest.mark.parametrize(
    ("name", "coord"),
    [
        ("tanaris_ascension_494_590", 4940590000),
        ("tanaris_ascension_636_494", 6360494000),
        ("tanaris_ascension_690_407", 6900407000),
        ("tanaris_ascension_539_288", 5390288000),
        ("tanaris_ascension_363_690", 3630690000),
        ("tanaris_ascension_405_407", 4050407000),
    ],
)
def test_spirit_healer_coordinate_anchor_catalog_contains_ascension_tanaris_points(
    name,
    coord,
):
    anchor = _choose_spirit_healer_coordinate_anchor(coord, load_config("config.yaml"))

    assert anchor == (name, coord, 0.0, 0.18)


def test_spirit_healer_coordinate_anchor_ignores_unrelated_location():
    assert _choose_spirit_healer_coordinate_anchor(1000100000, load_config("config.yaml")) is None


def test_spirit_healer_local_search_covers_live_verified_gadgetzan_offset():
    config = load_config("config.yaml")
    origin = 5401286100

    points = [_spirit_healer_local_search_coord(origin, index, config) for index in range(16)]

    assert points[6] == 5388287400
    assert all(points[index] == origin for index in range(1, 16, 2))


def test_spirit_healer_local_search_expands_and_caps_its_radius():
    config = load_config("config.yaml")
    origin = 5000500000

    first = _spirit_healer_local_search_coord(origin, 0, config)
    second_ring = _spirit_healer_local_search_coord(origin, 16, config)
    capped = _spirit_healer_local_search_coord(origin, 160, config)

    assert 0.17 < MovementNavigator.distance(origin, first) < 0.19
    assert 0.27 < MovementNavigator.distance(origin, second_ring) < 0.29
    assert 0.54 < MovementNavigator.distance(origin, capped) < 0.56


def test_spirit_healer_known_anchor_preempts_local_search_after_probe_threshold():
    anchor = ("graveyard", 5000500000, 0.8, 0.25)

    assert not _spirit_healer_anchor_due(
        "death_recovery_search_spirit_healer",
        anchor,
        search_attempts=7,
        after_attempts=8,
    )
    assert _spirit_healer_anchor_due(
        "death_recovery_search_spirit_healer",
        anchor,
        search_attempts=8,
        after_attempts=8,
    )
    assert not _spirit_healer_anchor_due(
        "death_recovery_target_spirit_healer",
        anchor,
        search_attempts=8,
        after_attempts=8,
    )


def test_spirit_healer_arrived_anchor_hands_off_to_bounded_local_search():
    anchor = ("graveyard_zone", 5000500000, 0.14, 0.18)

    assert not _spirit_healer_anchor_due(
        "death_recovery_search_spirit_healer",
        anchor,
        search_attempts=8,
        after_attempts=8,
    )


def test_spirit_search_leg_completes_after_passing_close_approach():
    progress = SpiritSearchLegProgress(
        reached_distance=0.045,
        pass_distance=0.18,
        pass_margin=0.08,
        max_seconds=2.5,
    )

    assert progress.observe(target_coord=1, distance=0.30, now=0.0) == (False, "tracking")
    assert progress.observe(target_coord=1, distance=0.15, now=0.5) == (False, "tracking")
    assert progress.observe(target_coord=1, distance=0.24, now=0.8) == (
        True,
        "passed_closest_approach",
    )


def test_spirit_search_leg_timeout_bounds_one_waypoint_without_ending_search():
    progress = SpiritSearchLegProgress(
        reached_distance=0.045,
        pass_distance=0.18,
        pass_margin=0.08,
        max_seconds=2.5,
    )

    assert progress.observe(target_coord=1, distance=0.60, now=10.0) == (False, "tracking")
    assert progress.observe(target_coord=1, distance=0.50, now=12.6) == (
        True,
        "leg_timeout",
    )
    progress.reset()
    assert progress.observe(target_coord=2, distance=0.40, now=12.7) == (False, "tracking")


def test_route_frame_sampling_keeps_action_changes_and_periodic_frames():
    assert _should_save_route_frame(
        now=10.1,
        last_saved_at=10.0,
        interval=1.0,
        action="combat_attack",
        previous_action="vector_forward",
    )


def test_route_live_mining_override_is_copy_on_write():
    config = {"mining": {"route_live_enabled": False}}

    updated = _with_route_live_mining_enabled(config, True)

    assert updated["mining"]["route_live_enabled"] is True
    assert config["mining"]["route_live_enabled"] is False


def test_safe_drain_keeps_an_inflight_mining_transaction_alive():
    assert _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=False,
        death_or_blocking_modal=False,
        mining_active=True,
    )

    assert _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=False,
        death_or_blocking_modal=False,
        death_recovery_active=True,
    )


def test_external_stop_file_is_fail_closed_and_handles_missing_path(tmp_path):
    stop_path = tmp_path / "STOP_REQUESTED"

    assert not _external_stop_file_requested(None)
    assert not _external_stop_file_requested(stop_path)

    stop_path.write_text("watchdog", encoding="utf-8")

    assert _external_stop_file_requested(stop_path)


def test_mining_route_resume_continues_forward_from_current_projection():
    route_loop = [5000500000, 5100500000, 5100510000]

    target = _mining_route_resume_target(
        {"mining": {"resume_projection_reached_distance": 0.45}},
        5050500000,
        route_loop,
        zone_id=14,
    )

    assert target is not None
    assert target.coord == 5100500000
    assert target.route_index == 1
    assert target.source == "route_loop_waypoint"


def test_mining_route_resume_never_reprojects_behind_route_direction():
    route_loop = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(20.0, 10.0),
        xy_to_coord(20.0, 20.0),
        xy_to_coord(10.0, 20.0),
    ]

    target = _mining_route_resume_target(
        {"mining": {"resume_projection_reached_distance": 0.45}},
        xy_to_coord(12.0, 10.0),
        route_loop,
        zone_id=14,
        route_after_sort_key=1.1,
    )

    assert target is not None
    assert target.coord == route_loop[2]
    assert target.route_index == 2
    assert target.source == "route_loop_waypoint"


def test_mining_access_resume_queue_preserves_safe_forward_leg():
    outcome = MiningOutcome(
        success=True,
        reason="tracked_icon_and_hover_cleared",
        node_id=7,
        node_coord=5100500000,
        elapsed_seconds=4.0,
        resume_coords=(5100500000, 5100500000, 5150500000),
        resume_route_index=42,
    )

    queue = _mining_access_resume_queue(outcome, zone_id=14)

    assert [node.coord for node in queue] == [5100500000, 5150500000]
    assert all(node.route_index == 42 for node in queue)
    assert all(node.zone_id == 14 for node in queue)


def test_start_coord_consensus_rejects_single_missing_leading_digit_outlier():
    selected = _select_stable_start_coord(
        [
            [104580100, 404038000],
            [4104580100, 404038000],
            [4104580100, 404038000],
        ],
        reference_coords=[3552552400],
        max_cluster_distance=0.15,
        max_jump=1.5,
    )

    assert selected == 4104580100


def test_start_coord_consensus_uses_route_reference_for_persistent_ocr_alternate():
    selected = _select_stable_start_coord(
        [
            [3242156700, 3242756700],
            [3242156700, 3242756700],
            [3242156700, 3242756700],
        ],
        reference_coords=[3190754000, 3410754000],
        max_cluster_distance=0.15,
        max_jump=1.5,
    )

    assert selected == 3242756700


def test_route_coord_jump_limit_scales_with_elapsed_mounted_travel_time():
    config = {
        "movement": {
            "max_coord_jump_per_poll": 1.5,
            "mounted_coord_units_per_second": 0.75,
            "foot_coord_units_per_second": 0.45,
            "coord_jump_timing_margin": 0.25,
        }
    }

    assert _route_coord_jump_limit(config, elapsed_seconds=0.5, mounted=True) == 1.5
    assert round(_route_coord_jump_limit(config, elapsed_seconds=4.1, mounted=True), 3) == 3.325
    assert round(_route_coord_jump_limit(config, elapsed_seconds=4.1, mounted=False), 3) == 2.095
    assert not _should_save_route_frame(
        now=10.9,
        last_saved_at=10.0,
        interval=1.0,
        action="vector_forward",
        previous_action="vector_forward",
    )
    assert not _should_save_route_frame(
        now=10.9,
        last_saved_at=10.0,
        interval=1.0,
        action="course_correct_and_forward",
        previous_action="continue_forward",
    )
    assert _should_save_route_frame(
        now=11.0,
        last_saved_at=10.0,
        interval=1.0,
        action="vector_forward",
        previous_action="vector_forward",
    )


def test_route_frame_sampling_can_be_forced_or_unthrottled():
    assert _should_save_route_frame(
        now=1.01,
        last_saved_at=1.0,
        interval=10.0,
        action="vector_forward",
        previous_action="vector_forward",
        force=True,
    )
    assert _should_save_route_frame(
        now=1.01,
        last_saved_at=1.0,
        interval=0.0,
        action="vector_forward",
        previous_action="vector_forward",
    )


def test_choose_route_target_uses_configured_route_zone(monkeypatch):
    nodes = [
        MiningNode(zone_id=1, coord=5600505000, ore_type="Copper", node_id=1),
        MiningNode(zone_id=2, coord=5710511000, ore_type="Tin", node_id=2),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [1], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5714511900,
        target_coord=None,
        zone_id=None,
    )

    assert target == nodes[0]


def test_mining_candidate_loader_applies_permanent_exclusions(monkeypatch):
    excluded = MiningNode(
        zone_id=162,
        coord=4666298900,
        ore_type="Truesilver",
        node_id=79,
    )
    retained = MiningNode(
        zone_id=162,
        coord=6200391000,
        ore_type="Iron",
        node_id=145,
    )
    monkeypatch.setattr(
        "vision_bot.live_route_probe.load_nodes",
        lambda _path: [excluded, retained],
    )
    monkeypatch.setattr(
        "vision_bot.live_route_probe.load_permanent_exclusions",
        lambda _path: {excluded.coord},
    )

    nodes = _load_mining_candidate_nodes(
        {
            "database_path": "ignored.json",
            "permanent_exclusions_path": "ignored-exclusions.json",
        }
    )

    assert nodes == [retained]


def test_choose_route_target_accepts_manual_coordinate(monkeypatch):
    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: [])
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [1], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5714511900,
        target_coord=5760504000,
        zone_id=640,
    )

    assert target is not None
    assert target.coord == 5760504000
    assert target.zone_id == 640
    assert target.ore_type == "manual"


def test_choose_route_target_can_create_route_entry_target(monkeypatch):
    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: [])
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [5], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        4600280000,
        target_coord=4667284300,
        zone_id=5,
        manual_source="route_loop_entry",
        manual_ore_type="route_entry",
        manual_route_index=0,
        manual_route_t=0.25,
    )

    assert target is not None
    assert target.coord == 4667284300
    assert target.ore_type == "route_entry"
    assert target.source == "route_loop_entry"
    assert _route_sort_key(target) == 0.25


def test_choose_route_target_can_continue_after_route_sort_key(monkeypatch):
    nodes = [
        MiningNode(zone_id=5, coord=4500300000, ore_type="Copper", node_id=1, route_index=1, route_t=0.10),
        MiningNode(zone_id=5, coord=5000300000, ore_type="Copper", node_id=2, route_index=3, route_t=0.20),
        MiningNode(zone_id=5, coord=5500300000, ore_type="Copper", node_id=3, route_index=4, route_t=0.10),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [5], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        4510300000,
        target_coord=None,
        zone_id=None,
        route_after_sort_key=3.0,
    )

    assert target == nodes[1]


def test_choose_route_target_does_not_reverse_to_nearer_node_behind_direction(monkeypatch):
    nodes = [
        MiningNode(zone_id=5, coord=5000300000, ore_type="Copper", node_id=1, route_index=2),
        MiningNode(zone_id=5, coord=8000300000, ore_type="Copper", node_id=2, route_index=3),
    ]
    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [5], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5010300000,
        target_coord=None,
        zone_id=5,
        route_after_sort_key=2.5,
    )

    assert target == nodes[1]


def test_choose_route_target_wraps_forward_after_end_of_cycle(monkeypatch):
    nodes = [
        MiningNode(zone_id=5, coord=5000300000, ore_type="Copper", node_id=1, route_index=1),
        MiningNode(zone_id=5, coord=8000300000, ore_type="Copper", node_id=2, route_index=3),
    ]
    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [5], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        7990300000,
        target_coord=None,
        zone_id=5,
        route_after_sort_key=3.001,
    )

    assert target == nodes[0]


def test_choose_route_target_auto_zone_ignores_configured_route_zone(monkeypatch):
    nodes = [
        MiningNode(zone_id=640, coord=5760504000, ore_type="Ore236", node_id=1),
        MiningNode(zone_id=464, coord=5930510000, ore_type="Copper", node_id=2),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [640], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5937512200,
        target_coord=None,
        zone_id=None,
        auto_zone=True,
    )

    assert target == nodes[1]


def test_choose_route_target_auto_zone_uses_resolved_zone_ids(monkeypatch):
    nodes = [
        MiningNode(zone_id=4, coord=5600505000, ore_type="Copper", node_id=1),
        MiningNode(zone_id=810, coord=5660503000, ore_type="Ore242", node_id=2),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [640], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5672501600,
        target_coord=None,
        zone_id=None,
        auto_zone=True,
        auto_zone_ids={4},
    )

    assert target == nodes[0]


def test_choose_route_target_respects_minimum_distance_when_possible(monkeypatch):
    nodes = [
        MiningNode(zone_id=464, coord=5930510000, ore_type="Copper", node_id=1),
        MiningNode(zone_id=464, coord=6080541000, ore_type="Copper", node_id=2),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [464], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5937512200,
        target_coord=None,
        zone_id=None,
        min_target_distance=1.0,
    )

    assert target == nodes[1]


def test_choose_route_target_skips_temporary_cycle_exclusions(monkeypatch):
    nodes = [
        MiningNode(zone_id=464, coord=5860513000, ore_type="Iron", node_id=1),
        MiningNode(zone_id=464, coord=5850506000, ore_type="Copper", node_id=2),
    ]

    monkeypatch.setattr("vision_bot.live_route_probe.load_mining_nodes", lambda _path: nodes)
    monkeypatch.setattr("vision_bot.live_route_probe.load_permanent_exclusions", lambda _path: set())

    target = _choose_route_target(
        {"route": {"locations": [464], "ores": [], "permanent_exclusions_path": "ignored.json"}},
        5857514100,
        target_coord=None,
        zone_id=None,
        excluded_coords={5860513000},
    )

    assert target == nodes[1]


def test_combined_excluded_coords_merges_completed_and_blocked_targets():
    assert _combined_excluded_coords({1001, 1002}, {1002, 1003}, None) == {1001, 1002, 1003}


def test_local_avoid_is_suppressed_while_distance_improves():
    assert not _should_local_avoid(
        True,
        distance_progress=0.08,
        coord_delta=0.08,
        visual_motion_delta=10.0,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_local_avoid_runs_when_blocked_and_progress_stalls():
    assert _should_local_avoid(
        True,
        distance_progress=0.0,
        coord_delta=0.0,
        visual_motion_delta=0.5,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_local_avoid_is_suppressed_when_coordinates_move_even_with_low_progress():
    assert not _should_local_avoid(
        True,
        distance_progress=0.005,
        coord_delta=0.04,
        visual_motion_delta=12.0,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_route_metadata_action_mirrors_movement_turn_or_recovery():
    assert _route_metadata_action("vector_forward", "turn_and_forward") == "turn_and_forward"
    assert _route_metadata_action("vector_forward", "course_correct_and_forward") == "course_correct_and_forward"
    assert _route_metadata_action("vector_forward", "recover_jump") == "recover_jump"
    assert _route_metadata_action("local_avoid_and_vector_forward", "forward_motion") == "local_avoid_and_vector_forward"


def test_route_metadata_action_preserves_threat_guard_actions():
    assert _route_metadata_action("avoid_hostile_target", "continue_forward") == "avoid_hostile_target"
    assert _route_metadata_action("avoid_hostile_threat", "turn_and_forward") == "avoid_hostile_threat"


def test_route_metadata_action_preserves_combat_guard_actions():
    assert _route_metadata_action("combat_attack", "turn_and_forward") == "combat_attack"
    assert _route_metadata_action("combat_face_attack", "recover_jump") == "combat_face_attack"


def test_route_metadata_action_preserves_combat_heal_and_death_recovery_actions():
    assert _route_metadata_action("combat_low_health_heal", "forward_motion") == "combat_low_health_heal"
    assert _route_metadata_action("death_recovery_release_spirit", "recover_jump") == "death_recovery_release_spirit"


def test_route_metadata_action_preserves_target_blocked_actions():
    assert _route_metadata_action("target_blocked_cycle_next", "turn_and_forward") == "target_blocked_cycle_next"
    assert _route_metadata_action("target_blocked_no_target", "recover_jump") == "target_blocked_no_target"


def test_record_target_recovery_marks_blocked_at_limit():
    recovery_count, blocked = _record_target_recovery(
        "recover_jump",
        target_recovery_count=2,
        target_recovery_limit=3,
    )

    assert recovery_count == 3
    assert blocked


def test_record_target_recovery_ignores_non_recovery_actions():
    recovery_count, blocked = _record_target_recovery(
        "vector_forward",
        target_recovery_count=2,
        target_recovery_limit=3,
    )

    assert recovery_count == 2
    assert not blocked


def test_record_target_recovery_resets_when_recovery_makes_progress():
    recovery_count, blocked = _record_target_recovery(
        "recover_jump",
        target_recovery_count=2,
        target_recovery_limit=3,
        distance_progress=0.05,
        coord_delta=0.05,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
    )

    assert recovery_count == 0
    assert not blocked


def test_record_target_recovery_resets_when_non_recovery_moves_after_stuck():
    recovery_count, blocked = _record_target_recovery(
        "vector_forward",
        target_recovery_count=2,
        target_recovery_limit=3,
        distance_progress=0.04,
        coord_delta=0.04,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
    )

    assert recovery_count == 0
    assert not blocked


def test_config_disables_combat_warning_threat_fallback_by_default():
    config = load_config("config.yaml")

    assert not config["safety"]["hostile_avoidance"]["combat_warning_enabled"]


def test_config_disables_top_left_target_frame_avoidance_by_default():
    config = load_config("config.yaml")

    assert not config["safety"]["hostile_avoidance"]["target_frame_enabled"]


def test_config_uses_extensible_e_only_attack_rotation():
    config = load_config("config.yaml")
    combat = config["safety"]["combat"]

    assert combat["attack_keys"] == ["E"]
    assert combat["attack_interval"] == 0.20
    assert combat["periodic_keys"] == ["X"]
    assert combat["periodic_interval"] == 15.0
    assert combat["aligned_periodic_keys"] == ["Q"]
    assert combat["aligned_periodic_interval"] == 5.0
    assert combat["low_health_priority_keys"] == ["F"]
    assert combat["low_health_priority_threshold"] == 0.20
    assert combat["low_health_priority_interval"] == 15.0
    assert combat["heal_key"] == "V"
    assert combat["emergency_heal_followup_enabled"] is True
    assert combat["emergency_heal_followup_delay_seconds"] == 0.12
    assert combat["combat_key_min_interval"] == 0.12
    assert combat["target_frame_starts_combat"] is False
    assert combat["face_search_turn_key"] == "D"
    assert combat["damage_timeout_seconds"] == 4.0
    assert combat["target_lock_no_hit_seconds"] == 8.0
    assert combat["continuous_face_search_enabled"] is True
    assert combat["face_search_turn_180_duration"] == 0.18
    assert combat["face_search_turn_90_duration"] == 0.18
    assert config["movement"]["mouse_steering"]["pixels_per_second"]["combat"] == 300.0


def test_config_uses_separate_spirit_healer_target_and_interact_keys():
    config = load_config("config.yaml")
    recovery = config["safety"]["death_recovery"]

    assert recovery["spirit_healer_target_key"] == "C"
    assert recovery["spirit_healer_target_wait_seconds"] == 0.15
    assert recovery["spirit_healer_interact_key"] == "G"
    assert recovery["spirit_healer_target_interact_enabled"] is True
    assert recovery["spirit_healer_visual_primary"] is False
    assert recovery["spirit_healer_target_interact_max_attempts"] == 0
    assert recovery["spirit_healer_search_max_attempts"] == 0
    assert recovery["spirit_healer_approach_max_attempts"] == 12
    assert recovery["spirit_healer_follow_target_enabled"] is False
    assert recovery["return_to_graveyard_enabled"] is False


def test_directed_route_recovery_does_not_block_normal_waypoint():
    assert not _target_recovery_can_block(
        directed_route_active=True,
        target_source="route_loop_waypoint",
    )
    assert _target_recovery_can_block(
        directed_route_active=False,
        target_source="route_loop_waypoint",
    )
    assert _target_recovery_can_block(
        directed_route_active=True,
        target_source="mining_node",
    )


def test_route_live_requires_precomputed_access_plan():
    config = load_config("config.yaml")

    assert config["mining"]["require_access_plan_for_route_live"] is True
    assert config["mining"]["candidate_match_distance_coord"] == 0.70


def test_config_uses_tanaris_full_rail_cycle():
    config = load_config("config.yaml")
    route = config["route"]

    assert route["locations"] == [162]
    assert route["database_path"] == (
        "data/routes/generated/tanaris_terrain_coverage_cycle_v19_rail.json"
    )
    assert route["follow_route_loop"] is True
    assert route["reached_distance"] == 0.65
    assert route["entry"]["strategy"] == "navmesh_dynamic"
    assert route["entry"]["reached_distance"] == 0.25
    assert route["entry"]["max_anchor_distance"] == 2.0
    assert route["polyline_following"]["lookahead_distance"] == 1.5
    assert route["entry"]["guide_route_segments"] is False
    assert route["permanent_exclusions_path"] == "data/permanent_exclusions_tanaris.json"

    combat = config["safety"]["combat"]
    assert combat["combat_marker_enabled"] is True
    assert config["safety"]["hostile_avoidance"]["enabled"] is False
    assert config["movement"]["hard_stuck_escape"]["enabled"] is True
    assert config["movement"]["hard_stuck_escape"]["forward_waypoint_padding"] == 6


def test_dynamic_navmesh_preflight_rejects_a_missing_zone_profile():
    config = {
        "route": {
            "locations": [999],
            "entry": {
                "strategy": "navmesh_dynamic",
                "navmesh_profiles": {},
            },
        }
    }

    preflight = build_route_live_preflight(config)

    assert not preflight["ready_offline"]
    assert "navmesh_profile_invalid:ValueError" in preflight["issues"]


def test_route_live_preflight_rejects_unloadable_enabled_ore_model(
    monkeypatch,
    tmp_path,
):
    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "status": "test",
                "route_loop": [{"coord": 10002000}],
                "nodes": [
                    {
                        "zone_id": 162,
                        "coord": 10002000,
                        "ore_type": "Iron",
                        "access_plan": {"approach_coord": 10002000},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "vision_bot.route_probe_preflight.OreWorldDetector.load",
        lambda _self: OreWorldDetectorResult(reason="model_load_error:RuntimeError"),
    )
    config = {
        "route": {"database_path": str(route_path), "follow_route_loop": True},
        "mining": {
            "route_live_enabled": True,
            "bright_confirm_frames": 1,
            "verify_clear_frames": 1,
            "scan_max_points": 1,
            "require_cursor_change": False,
            "ore_world_detector": {"enabled": True},
        },
        "recognition": {
            "bright_templates": [
                {"path": "assets/minimap_ore_bright_reference.png"}
            ]
        },
        "cursor_classifier": {"enabled": False},
    }

    preflight = build_route_live_preflight(config)

    assert not preflight["ready_offline"]
    assert preflight["ore_world_detector_status"] == "model_load_error:RuntimeError"
    assert (
        "ore_world_detector_unavailable:model_load_error:RuntimeError"
        in preflight["issues"]
    )


def test_route_waypoint_uses_corridor_reached_distance():
    waypoint = MiningNode(
        zone_id=162,
        coord=4207263600,
        ore_type="Route Waypoint",
        node_id=None,
        source="route_loop_waypoint",
    )
    ore_node = MiningNode(zone_id=162, coord=4207263600, ore_type="Gold", node_id=None)
    config = {"route": {"reached_distance": 0.65}}

    assert _route_target_reached_distance(config, 0.35, waypoint) == 0.65
    assert _route_target_reached_distance(config, 0.35, ore_node) == 0.35


def test_follow_route_loop_uses_safe_waypoints_without_direct_ore_targets(tmp_path, monkeypatch):
    route_path = tmp_path / "route.json"
    route_path.write_text(
        '{"zone":{"id":102},"route_loop":[{"index":0,"coord":1001}],'
        '"route_nodes":[{"zone_id":102,"coord":9999,"ore_type":"Iron","node_id":7}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr("vision_bot.live_route_probe.resource_path", lambda _path: route_path)

    nodes = _load_route_nodes({"database_path": "route.json", "follow_route_loop": True})

    assert [node.coord for node in nodes] == [1001]
    assert all(node.source == "route_loop_waypoint" for node in nodes)


def test_follow_route_loop_uses_json_zone_instead_of_stale_config_location(tmp_path):
    route_path = tmp_path / "tanaris_route.json"
    route_path.write_text(
        '{"zone_id":162,"route_loop":[{"index":0,"coord":4137267100}]}',
        encoding="utf-8",
    )
    route_cfg = {
        "database_path": str(route_path),
        "follow_route_loop": True,
        "locations": [102],
    }

    assert _configured_route_zone_ids(route_cfg) == {162}
    assert _allowed_route_locations(
        route_cfg,
        zone_id=None,
        auto_zone=False,
        auto_zone_ids=None,
    ) == {162}
    assert _single_allowed_route_zone(route_cfg) == 162


def test_tanaris_full_rail_cycle_exposes_runtime_loop_and_filtered_nodes():
    config = load_config("config.yaml")

    route_loop = _load_configured_route_loop(config)

    assert len(route_loop) == 304
    assert route_loop[0] != route_loop[-1]
    assert _load_configured_route_entry(config) is None

    milestones = _load_configured_route_milestones(config)
    assert len(milestones) == 74
    assert {node.coord for node in milestones}.isdisjoint(
        {4450228000, 3770210000, 3630201000, 3621201200}
    )
    assert any(node.coord == 4140267100 for node in milestones)
    assert any(node.coord == 3819271200 for node in milestones)
    assert any(node.coord == 6110483000 for node in milestones)


def test_route_node_milestone_records_each_ore_node_once():
    milestones = [
        MiningNode(
            zone_id=102,
            coord=4770766000,
            ore_type="Iron",
            node_id=28,
            route_index=0,
            source="bounded_mining_node",
        )
    ]
    target = MiningNode(
        zone_id=102,
        coord=4770766000,
        ore_type="Route Waypoint",
        node_id=-1,
        route_index=0,
        source="route_loop_waypoint",
    )
    history = []
    completed = set()

    _record_target_route_node_milestone(
        history,
        completed,
        milestones,
        target_node=target,
        reached_coord=4771766000,
        distance=0.1,
    )
    _record_target_route_node_milestone(
        history,
        completed,
        milestones,
        target_node=target,
        reached_coord=4770766000,
        distance=0.0,
    )

    assert len(history) == 1
    assert history[0]["route_index"] == 0
    assert history[0]["ore_type"] == "Iron"


def test_route_entry_consumes_only_reached_prefix():
    queue = [
        MiningNode(zone_id=102, coord=5000500000, ore_type="entry", node_id=-1),
        MiningNode(zone_id=102, coord=5100500000, ore_type="entry", node_id=-2),
    ]

    next_node = _consume_reached_route_entry(
        queue,
        5001500000,
        reached_distance=0.2,
    )

    assert next_node is not None
    assert next_node.coord == 5100500000
    assert [node.coord for node in queue] == [5100500000]


def test_local_avoidance_keeps_one_side_until_obstacle_clears():
    state = LocalAvoidanceCommitment(retry_interval=0.45, clear_commit_seconds=0.65)

    assert state.next_turn_key(
        blocked=True,
        now=1.0,
        preferred_turn_key="D",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) == "D"
    assert state.consume_jump_only()
    assert not state.consume_jump_only()
    assert state.next_turn_key(
        blocked=True,
        now=1.2,
        preferred_turn_key="A",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) is None
    assert state.next_turn_key(
        blocked=True,
        now=1.5,
        preferred_turn_key="A",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) == "D"
    assert not state.consume_jump_only()
    assert state.next_turn_key(
        blocked=False,
        now=1.6,
        preferred_turn_key=None,
        fallback_turn_key="A",
        distance_progress=None,
        progress_threshold=0.03,
    ) is None
    assert state.should_continue_forward(1.6)
    assert state.committed_turn_key == "D"
    assert state.next_turn_key(
        blocked=False,
        now=2.2,
        preferred_turn_key=None,
        fallback_turn_key="A",
        distance_progress=None,
        progress_threshold=0.03,
    ) is None
    assert state.committed_turn_key is None


def test_local_avoidance_backtracks_and_switches_after_repeated_failed_side():
    state = LocalAvoidanceCommitment(
        retry_interval=0.45,
        clear_commit_seconds=0.65,
        max_turn_pulses_per_side=2,
    )

    assert state.next_turn_key(
        blocked=True,
        now=1.0,
        preferred_turn_key="D",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) == "D"
    assert state.next_turn_key(
        blocked=True,
        now=1.5,
        preferred_turn_key="D",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) == "D"
    assert not state.consume_side_switch()
    assert state.next_turn_key(
        blocked=True,
        now=2.0,
        preferred_turn_key="D",
        fallback_turn_key="A",
        distance_progress=-0.1,
        progress_threshold=0.03,
    ) == "A"
    assert state.consume_side_switch()
    assert not state.consume_side_switch()
    assert state.committed_turn_pulses == 1


def test_local_avoidance_progress_resets_failed_side_pulse_count():
    state = LocalAvoidanceCommitment(
        retry_interval=0.45,
        max_turn_pulses_per_side=2,
    )

    for now in (1.0, 1.5):
        assert state.next_turn_key(
            blocked=True,
            now=now,
            preferred_turn_key="D",
            fallback_turn_key="A",
            distance_progress=-0.1,
            progress_threshold=0.03,
        ) == "D"
    assert state.next_turn_key(
        blocked=True,
        now=2.0,
        preferred_turn_key="D",
        fallback_turn_key="A",
        distance_progress=0.1,
        progress_threshold=0.03,
    ) == "D"
    assert not state.consume_side_switch()
    assert state.committed_turn_pulses == 1


def test_autonomous_route_entry_targets_nearest_loop_projection_without_saved_path():
    target = _autonomous_route_entry_target(
        {"route": {"locations": [102]}},
        4000400000,
        [3000500000, 5000500000, 5000700000, 3000700000],
        excluded_coords=set(),
        zone_id=None,
    )

    assert target is not None
    assert target.coord == 4000500000
    assert target.source == "autonomous_route_projection"
    assert target.route_index == 0
    assert target.route_t == 0.5


def test_directed_route_steering_uses_lookahead_and_pass_through():
    route = [1000100000, 2000100000, 2000200000, 1000200000]
    follower = _build_directed_route_follower(
        {
            "route": {
                "polyline_following": {
                    "enabled": True,
                    "lookahead_distance": 2.0,
                    "corridor_radius": 2.0,
                    "backward_tolerance": 0.25,
                    "max_forward_advance": 4.0,
                    "pass_tolerance": 0.05,
                }
            }
        },
        route,
    )
    assert follower is not None
    follower.seed(0.4)
    target = MiningNode(
        zone_id=5,
        coord=2000100000,
        ore_type="Route Waypoint",
        node_id=-2,
        route_index=1,
        route_t=-0.5,
        route_order=1,
        source="route_loop_waypoint",
    )

    steering, passed, observation = _directed_route_steering_target(
        follower,
        current_coord=2000110000,
        target_node=target,
        route_entry_pending=False,
    )

    assert _route_geometry_sort_key(target) == 1.0
    assert steering != target.coord
    assert passed
    assert observation is not None
    assert observation.progress >= 1.0


def test_directed_route_steering_is_disabled_during_route_entry():
    route = [1000100000, 2000100000, 2000200000, 1000200000]
    follower = _build_directed_route_follower(
        {"route": {"polyline_following": {"enabled": True}}},
        route,
    )
    assert follower is not None
    target = MiningNode(
        zone_id=5,
        coord=1500100000,
        ore_type="route_entry",
        node_id=None,
        route_index=0,
        route_t=0.5,
        source="autonomous_route_projection",
    )

    steering, passed, observation = _directed_route_steering_target(
        follower,
        current_coord=1200100000,
        target_node=target,
        route_entry_pending=True,
    )

    assert steering == target.coord
    assert not passed
    assert observation is None


def test_directed_route_steering_freezes_progress_while_navigation_is_suspended():
    route = [1000100000, 2000100000, 2000200000, 1000200000]
    follower = _build_directed_route_follower(
        {"route": {"polyline_following": {"enabled": True}}},
        route,
    )
    assert follower is not None
    follower.seed(0.4)
    target = MiningNode(
        zone_id=5,
        coord=2000100000,
        ore_type="Route Waypoint",
        node_id=-2,
        route_index=1,
        route_t=-0.5,
        route_order=1,
        source="route_loop_waypoint",
    )

    steering, passed, observation = _directed_route_steering_target(
        follower,
        current_coord=2000150000,
        target_node=target,
        route_entry_pending=False,
        suspended=True,
    )

    assert steering == target.coord
    assert not passed
    assert observation is None
    assert follower.progress == 0.4


def test_autonomous_route_entry_uses_next_nearest_waypoint_after_projection_is_blocked():
    target = _autonomous_route_entry_target(
        {"route": {"locations": [102]}},
        4000400000,
        [3000500000, 5000500000, 5000700000, 3000700000],
        excluded_coords={4000500000},
        zone_id=None,
    )

    assert target is not None
    assert target.source == "autonomous_route_waypoint"
    assert target.coord in {3000500000, 5000500000, 5000700000}


def test_route_entry_strategy_defaults_to_autonomous_and_preserves_fixture_mode():
    assert _route_entry_strategy({}) == "autonomous_local"
    assert _route_entry_strategy({"route": {"entry": {"strategy": "unknown"}}}) == "autonomous_local"
    assert _route_entry_strategy({"route": {"entry": {"strategy": "validated_path"}}}) == "validated_path"
    assert _route_entry_strategy({"route": {"entry": {"strategy": "navmesh_dynamic"}}}) == "navmesh_dynamic"


def test_proactive_local_avoidance_uses_confident_blocker_before_stall():
    assert _should_local_avoid(
        True,
        proactive=True,
        distance_progress=0.1,
        coord_delta=0.1,
        visual_motion_delta=10.0,
        stuck_min_progress=0.03,
        stuck_min_coord_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_generated_route_entry_resumes_from_nearest_open_corridor_segment():
    entry = RouteEntryPlan(
        waypoints=(
            MiningNode(zone_id=102, coord=3000750000, ore_type="entry", node_id=-1),
            MiningNode(zone_id=102, coord=3200750000, ore_type="entry", node_id=-2),
            MiningNode(zone_id=102, coord=3400750000, ore_type="entry", node_id=-3),
        ),
        target_route_sort_key=35.0,
    )

    queue, projection_distance = _resume_generated_route_entry(
        entry,
        3240750300,
        reached_distance=0.65,
    )

    assert round(projection_distance, 2) == 0.03
    assert [node.coord for node in queue] == [3400750000]


def test_generated_route_entry_adds_projection_target_when_not_on_corridor():
    entry = RouteEntryPlan(
        waypoints=(
            MiningNode(zone_id=102, coord=3000750000, ore_type="entry", node_id=-1),
            MiningNode(zone_id=102, coord=3200750000, ore_type="entry", node_id=-2),
            MiningNode(zone_id=102, coord=3400750000, ore_type="entry", node_id=-3),
        ),
        target_route_sort_key=35.0,
    )

    queue, projection_distance = _resume_generated_route_entry(
        entry,
        3240760000,
        reached_distance=0.65,
    )

    assert round(projection_distance, 2) == 1.0
    assert queue[0].coord == 3240750000
    assert queue[0].source == "route_entry_projection"
    assert queue[1].coord == 3400750000


def test_with_route_live_combat_enabled_overrides_without_mutating_source():
    config = {"safety": {"combat": {"enabled": True, "attack_keys": ["E"]}}}

    updated = _with_route_live_combat_enabled(config, False)

    assert updated["safety"]["combat"]["enabled"] is True
    assert updated["safety"]["combat"]["route_live_handling_enabled"] is False
    assert config["safety"]["combat"]["enabled"] is True
    assert "route_live_handling_enabled" not in config["safety"]["combat"]
    assert updated["safety"]["combat"]["attack_keys"] == ["E"]


def test_with_post_combat_loot_enabled_overrides_without_mutating_source():
    config = {"post_combat_loot": {"enabled": False, "interact_key": "G"}}

    updated = _with_post_combat_loot_enabled(config, True)

    assert updated["post_combat_loot"]["enabled"] is True
    assert updated["post_combat_loot"]["interact_key"] == "G"
    assert config["post_combat_loot"]["enabled"] is False


def test_route_probe_safe_drain_keeps_control_during_active_combat_or_death():
    assert _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=True,
        death_or_blocking_modal=False,
    )
    assert _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=False,
        death_or_blocking_modal=True,
    )
    assert _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=False,
        death_or_blocking_modal=False,
        post_combat_loot_active=True,
    )


def test_route_probe_safe_drain_stops_when_clear_or_combat_handling_is_disabled():
    assert not _route_probe_requires_safe_drain(
        combat_handling_enabled=True,
        combat_active=False,
        death_or_blocking_modal=False,
    )
    assert not _route_probe_requires_safe_drain(
        combat_handling_enabled=False,
        combat_active=True,
        death_or_blocking_modal=False,
    )


def test_should_handle_dismissable_modal_ignores_combat_false_positive():
    state = GameState(
        death_or_blocking_modal=False,
        red_button_count=0,
        ghost_visual=False,
        dismissable_modal_click=(1200, 1180),
    )

    assert not _should_handle_dismissable_modal(state, CombatState(active=True))
    assert not _should_handle_dismissable_modal(state, CombatState(nameplate_visible=True))
    assert not _should_handle_dismissable_modal(state, CombatState(outgoing_damage_visible=True))
    assert _should_handle_dismissable_modal(state, CombatState())


def test_should_handle_death_recovery_prioritizes_ghost_action_buttons_over_combat_latch():
    state = GameState(
        death_or_blocking_modal=True,
        red_button_count=0,
        ghost_visual=False,
        ghost_button_count=2,
    )

    assert _should_handle_death_recovery(state, CombatState(active=True))
    assert _should_handle_death_recovery(state, CombatState(target_present=True))
    assert _should_handle_death_recovery(state, CombatState(facing_error_visible=True))
    assert _should_handle_death_recovery(state, CombatState())


def test_should_handle_death_recovery_prioritizes_death_dialog_over_combat_false_positive():
    state = GameState(
        death_or_blocking_modal=True,
        red_button_count=2,
        ghost_visual=False,
    )

    assert _should_handle_death_recovery(
        state,
        CombatState(active=True, nameplate_visible=True, outgoing_damage_visible=True),
    )


def test_should_handle_death_recovery_keeps_ambiguous_modal_combat_guard():
    state = GameState(
        death_or_blocking_modal=True,
        red_button_count=0,
        ghost_visual=False,
        ghost_button_count=0,
    )

    assert not _should_handle_death_recovery(state, CombatState(active=True))
    assert _should_handle_death_recovery(state, CombatState())


def test_should_handle_death_recovery_requires_blocking_state():
    state = GameState(
        death_or_blocking_modal=False,
        red_button_count=2,
        ghost_visual=True,
        ghost_button_count=2,
    )

    assert not _should_handle_death_recovery(state, CombatState())


def test_should_read_route_coord_skips_during_combat_even_when_due():
    assert not _should_read_route_coord(
        now=20.0,
        last_coord_read_at=10.0,
        coord_read_interval=0.5,
        index=3,
        held_key="W",
        reached=False,
        combat_engaged=True,
        combat_heal_casting=False,
    )


def test_should_read_route_coord_skips_during_combat_heal_cast_even_when_due():
    assert not _should_read_route_coord(
        now=20.0,
        last_coord_read_at=10.0,
        coord_read_interval=0.5,
        index=3,
        held_key=None,
        reached=False,
        combat_engaged=False,
        combat_heal_casting=True,
    )


def test_should_read_route_coord_forces_initial_noncombat_sync_before_moving():
    assert _should_read_route_coord(
        now=20.0,
        last_coord_read_at=20.0,
        coord_read_interval=2.0,
        index=0,
        held_key=None,
        reached=False,
        combat_engaged=False,
        combat_heal_casting=False,
    )


def test_should_read_route_coord_uses_interval_in_route_mode():
    assert _should_read_route_coord(
        now=12.1,
        last_coord_read_at=10.0,
        coord_read_interval=2.0,
        index=8,
        held_key="W",
        reached=False,
        combat_engaged=False,
        combat_heal_casting=False,
    )
    assert not _should_read_route_coord(
        now=11.9,
        last_coord_read_at=10.0,
        coord_read_interval=2.0,
        index=8,
        held_key="W",
        reached=False,
        combat_engaged=False,
        combat_heal_casting=False,
    )


def test_apply_combat_low_health_heal_taps_configured_key_and_starts_cast():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    heal = CombatHealState(threshold=0.50, cooldown_seconds=8.0, cast_seconds=2.0)

    action = _apply_combat_low_health_heal(
        controller,  # type: ignore[arg-type]
        heal,
        CombatState(player_health_fraction=0.49),
        now=10.0,
        heal_key="V",
        heal_tap_duration=0.06,
    )

    assert action == "combat_low_health_heal"
    assert controller.tapped == [("V", 0.06)]
    assert heal.is_casting(11.9)


def test_low_health_f_has_priority_and_rearms_only_after_health_recovers():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )
    emergency = PeriodicCombatKeyState(keys=("F",), interval=15.0)

    assert _tap_due_low_health_combat_key(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(player_health_fraction=0.20),
        emergency,
        now=10.0,
        health_threshold=0.20,
        attack_tap_duration=0.06,
    ) == ["F"]
    assert fallback.peek_next_attack_key(10.0) == "E"
    assert _tap_due_low_health_combat_key(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(player_health_fraction=0.10),
        emergency,
        now=10.13,
        health_threshold=0.20,
        attack_tap_duration=0.06,
    ) == []
    assert _tap_due_low_health_combat_key(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(player_health_fraction=0.25),
        emergency,
        now=10.20,
        health_threshold=0.20,
        attack_tap_duration=0.06,
    ) == []
    assert _tap_due_low_health_combat_key(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(player_health_fraction=0.19),
        emergency,
        now=10.26,
        health_threshold=0.20,
        attack_tap_duration=0.06,
    ) == ["F"]
    assert controller.tapped == [("F", 0.06), ("F", 0.06)]


def test_apply_combat_low_health_heal_skips_above_threshold_or_cooldown():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    heal = CombatHealState(threshold=0.50, cooldown_seconds=8.0, cast_seconds=2.0)

    assert (
        _apply_combat_low_health_heal(
            controller,  # type: ignore[arg-type]
            heal,
            CombatState(player_health_fraction=0.55),
            now=10.0,
            heal_key="V",
            heal_tap_duration=0.06,
        )
        is None
    )
    assert (
        _apply_combat_low_health_heal(
            controller,  # type: ignore[arg-type]
            heal,
            CombatState(player_health_fraction=0.40),
            now=10.0,
            heal_key="V",
            heal_tap_duration=0.06,
        )
        == "combat_low_health_heal"
    )
    assert (
        _apply_combat_low_health_heal(
            controller,  # type: ignore[arg-type]
            heal,
            CombatState(player_health_fraction=0.30),
            now=12.1,
            heal_key="V",
            heal_tap_duration=0.06,
        )
        is None
    )
    assert controller.tapped == [("V", 0.06)]


def test_emergency_f_followup_serializes_one_v_and_arms_heal_cooldown():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )
    heal = CombatHealState(threshold=0.50, cooldown_seconds=8.0, cast_seconds=2.0)
    fallback.record_tapped_keys(["F"], now=10.0)

    tapped = _apply_emergency_heal_followup(
        controller,  # type: ignore[arg-type]
        fallback,
        heal,
        ["F"],
        now=10.0,
        heal_key="V",
        heal_tap_duration=0.06,
        delay_seconds=0.0,
    )

    assert tapped == ["V"]
    assert controller.tapped == [("V", 0.06)]
    assert fallback.last_tapped_keys == ["F", "V"]
    assert heal.is_casting(10.1)
    assert not heal.should_start(CombatState(player_health_fraction=0.10), 10.1)


def test_turn_key_towards_click_point_uses_screen_side_with_deadzone():
    frame_shape = (1440, 2560, 3)

    assert _turn_key_towards_click_point((400, 700), frame_shape, deadzone_fraction=0.12) == "A"
    assert _turn_key_towards_click_point((2150, 700), frame_shape, deadzone_fraction=0.12) == "D"
    assert _turn_key_towards_click_point((1280, 700), frame_shape, deadzone_fraction=0.12) is None
    assert (
        _turn_key_towards_click_point(
            (400, 700),
            frame_shape,
            deadzone_fraction=0.12,
            invert_turn_direction=True,
        )
        == "D"
    )


def test_spirit_healer_search_turn_key_keeps_one_sweep_direction():
    assert _spirit_healer_search_turn_key("D") == "D"
    assert _spirit_healer_search_turn_key("a") == "A"
    assert _spirit_healer_search_turn_key("invalid") == "D"


def test_apply_combat_fallback_face_searches_after_e_without_damage():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    combat = CombatState(active=True, target_present=True)

    assert fallback.next_attack_key(10.0) == "E"
    assert fallback.next_attack_key(11.0) == "E"

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=13.1,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_face_search_sweep"
    assert turn_key == "D"
    assert controller.tapped[0] == ("D", 0.45)


def test_tap_due_combat_keys_serializes_periodic_x_and_attack_sequence():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2)
    periodic = PeriodicCombatKeyState(keys=("X",), interval=15.0)

    assert (
        _tap_due_combat_keys(
            controller,  # type: ignore[arg-type]
            fallback,
            now=10.0,
            attack_tap_duration=0.06,
            combat_key_min_interval=0.12,
            periodic_combat_keys=periodic,
        )
        == ["X"]
    )
    assert fallback.unconfirmed_e_started_at is None
    assert fallback.last_tapped_keys == ["X"]
    assert (
        _tap_due_combat_keys(
            controller,  # type: ignore[arg-type]
            fallback,
            now=10.05,
            attack_tap_duration=0.06,
            combat_key_min_interval=0.12,
            periodic_combat_keys=periodic,
        )
        == []
    )
    assert (
            _tap_due_combat_keys(
                controller,  # type: ignore[arg-type]
                fallback,
                now=10.13,
                attack_tap_duration=0.06,
                combat_key_min_interval=0.12,
                periodic_combat_keys=periodic,
        )
        == ["E"]
    )
    assert fallback.unconfirmed_e_started_at == 10.13
    assert (
        _tap_due_combat_keys(
            controller,  # type: ignore[arg-type]
            fallback,
            now=11.0,
            attack_tap_duration=0.06,
            combat_key_min_interval=0.12,
            periodic_combat_keys=periodic,
        )
        == []
    )
    assert fallback.unconfirmed_e_started_at == 10.13
    assert (
        _tap_due_combat_keys(
            controller,  # type: ignore[arg-type]
            fallback,
            now=12.0,
            attack_tap_duration=0.06,
            combat_key_min_interval=0.12,
            periodic_combat_keys=periodic,
        )
        == ["E"]
    )
    assert (
        _tap_due_combat_keys(
            controller,  # type: ignore[arg-type]
            fallback,
            now=25.12,
            attack_tap_duration=0.06,
            combat_key_min_interval=0.12,
            periodic_combat_keys=periodic,
        )
        == ["X"]
    )
    assert (
        _tap_due_combat_keys(
                controller,  # type: ignore[arg-type]
                fallback,
                now=25.25,
                attack_tap_duration=0.06,
                combat_key_min_interval=0.12,
                periodic_combat_keys=periodic,
        )
        == ["TAB"]
    )
    assert controller.tapped == [
        ("X", 0.06),
        ("E", 0.06),
        ("E", 0.06),
        ("X", 0.06),
        ("TAB", 0.06),
    ]


def test_periodic_x_fires_before_attacker_target_selection_on_combat_entry():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )
    periodic = PeriodicCombatKeyState(keys=("X",), interval=15.0)
    combat = CombatState(active=True, target_is_attacker=False)

    action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.18,
        face_search_turn_90_duration=0.18,
        attack_tap_duration=0.06,
        periodic_combat_keys=periodic,
        attacker_target_selection=True,
        attacker_target_cycle_key="F8",
    )
    assert action == "combat_priority_periodic"
    assert controller.tapped == [("X", 0.06)]

    action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.13,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.18,
        face_search_turn_90_duration=0.18,
        attack_tap_duration=0.06,
        periodic_combat_keys=periodic,
        attacker_target_selection=True,
        attacker_target_cycle_key="F8",
    )
    assert action == "combat_select_attacker"
    assert controller.tapped == [("X", 0.06), ("F8", 0.06)]


def test_aligned_q_waits_for_confirmed_e_hit_and_keeps_five_second_cadence():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )
    aligned = PeriodicCombatKeyState(keys=("Q",), interval=5.0)

    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=10.0,
        attack_tap_duration=0.06,
        aligned_periodic_combat_keys=aligned,
    ) == ["E"]
    assert not fallback.target_facing_locked

    fallback.observe_outgoing_damage(10.05, True)
    assert fallback.target_facing_locked
    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=10.13,
        attack_tap_duration=0.06,
        aligned_periodic_combat_keys=aligned,
    ) == ["Q"]
    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=15.0,
        attack_tap_duration=0.06,
        aligned_periodic_combat_keys=aligned,
    ) == ["E"]
    fallback.observe_outgoing_damage(15.01, True)
    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=15.14,
        attack_tap_duration=0.06,
        aligned_periodic_combat_keys=aligned,
    ) == ["Q"]


def test_periodic_x_stays_higher_priority_than_aligned_q():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )
    fallback.consume_next_attack_key(1.0)
    fallback.observe_outgoing_damage(1.05, True)
    periodic = PeriodicCombatKeyState(keys=("X",), interval=15.0)
    aligned = PeriodicCombatKeyState(keys=("Q",), interval=10.0)

    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=2.0,
        attack_tap_duration=0.06,
        periodic_combat_keys=periodic,
        aligned_periodic_combat_keys=aligned,
    ) == ["X"]
    assert _tap_due_combat_keys(
        controller,  # type: ignore[arg-type]
        fallback,
        now=2.13,
        attack_tap_duration=0.06,
        periodic_combat_keys=periodic,
        aligned_periodic_combat_keys=aligned,
    ) == ["Q"]


def test_confirmed_e_target_lock_suppresses_face_search_through_three_second_stun():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
        damage_timeout=4.0,
        target_lock_no_hit_seconds=8.0,
        face_search_cooldown=0.5,
    )
    fallback.consume_next_attack_key(10.0)
    fallback.observe_outgoing_damage(10.1, True)
    fallback.consume_next_attack_key(10.3)

    assert fallback.next_face_search_turn(
        14.5,
        turn_180_duration=0.18,
        turn_90_duration=0.18,
    ) is None
    assert fallback.has_target_facing_lock(17.9)
    assert not fallback.has_target_facing_lock(18.2)
    assert fallback.next_face_search_turn(
        18.2,
        turn_180_duration=0.18,
        turn_90_duration=0.18,
    ) == ("D", 0.18, "combat_face_search_sweep")


def test_apply_combat_fallback_latches_recent_hit_before_facing_error_search():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    assert fallback.next_attack_key(10.0) == "E"
    assert fallback.next_attack_key(11.0) == "E"
    fallback.observe_outgoing_damage(11.5, True)
    combat = CombatState(active=True, target_present=True, outgoing_damage_visible=True, facing_error_visible=True)

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=11.6,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_hold"
    assert turn_key is None
    assert controller.tapped == []

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, facing_error_visible=True),
        now=13.61,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_face_search_sweep"
    assert turn_key == "D"
    assert controller.tapped == [("D", 0.45)]


def test_apply_combat_fallback_uses_one_direction_for_facing_error_search():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    combat = CombatState(
        active=False,
        target_present=False,
        nameplate_visible=True,
        facing_error_visible=True,
        face_turn_key="A",
    )

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_face_search_sweep"
    assert turn_key == "D"
    assert controller.tapped == [("D", 0.45)]
    assert fallback.last_face_search_at == 10.0


def test_apply_combat_fallback_stops_visible_turn_after_recent_damage():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=0.5,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    assert fallback.next_attack_keys(10.0) == ["E"]
    assert fallback.next_attack_keys(10.5) == ["E"]
    fallback.observe_outgoing_damage(10.8, True)
    combat = CombatState(active=True, target_present=True, face_turn_key="A")

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=11.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_attack"
    assert turn_key is None
    assert controller.tapped == [("TAB", 0.06)]


def test_continuous_combat_facing_holds_one_turn_and_releases_on_hit():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []
            self.pressed: list[str] = []
            self.released: list[str] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def press_key(self, key: str) -> None:
            self.pressed.append(key)

        def release_key(self, key: str) -> None:
            self.released.append(key)

    controller = DummyInput()
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    combat = CombatState(active=True, target_present=True)

    first_action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
    )
    search_action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=12.01,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
    )
    hit_action, hit_turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, outgoing_damage_visible=True),
        now=12.2,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
    )

    assert first_action == "combat_attack"
    assert search_action == "combat_face_search_attack"
    assert turn_key == "D"
    assert controller.pressed == ["D"]
    assert hit_action == "combat_hold"
    assert hit_turn_key is None
    assert controller.released == ["D"]
    assert fallback.face_search_held_key is None


def test_continuous_combat_facing_uses_rmb_sweep_without_keyboard_turn():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []
            self.pressed: list[str] = []
            self.released: list[str] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def press_key(self, key: str) -> None:
            self.pressed.append(key)

        def release_key(self, key: str) -> None:
            self.released.append(key)

    class DummyMouse:
        def __init__(self) -> None:
            self.drags: list[tuple[int, float]] = []
            self.continuous_drags: list[tuple[int, float, bool]] = []

        def right_drag_relative(
            self,
            delta_x: int,
            *,
            duration: float,
            step_interval: float,
            restore_cursor: bool,
        ) -> None:
            self.drags.append((delta_x, duration))

        def right_drag_continuous(
            self,
            delta_x_per_step: int,
            *,
            stop_event,
            step_interval: float,
            restore_cursor: bool,
        ) -> tuple[int, float]:
            self.continuous_drags.append(
                (delta_x_per_step, step_interval, restore_cursor)
            )
            stop_event.wait(step_interval)
            return delta_x_per_step, step_interval

    controller = DummyInput()
    mouse = DummyMouse()
    steering = MouseSteeringController(
        mouse,  # type: ignore[arg-type]
        enabled=True,
        pixels_per_second={"combat": 300.0},
        min_duration_seconds=0.0,
    )
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )
    combat = CombatState(active=True, target_present=True)

    _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
        mouse_steering=steering,
    )
    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=12.01,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
        mouse_steering=steering,
    )
    _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, outgoing_damage_visible=True),
        now=12.2,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        continuous_face_search=True,
        mouse_steering=steering,
    )

    assert action == "combat_face_search_attack"
    assert turn_key == "D"
    assert mouse.drags == []
    assert mouse.continuous_drags == [(4, 0.012, True)]
    assert controller.pressed == []
    assert controller.released == []
    assert fallback.face_search_held_key is None


def test_combat_selects_confirmed_attacker_before_attacking_or_turning():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, target_is_attacker=False),
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        attacker_target_selection=True,
        attacker_target_cycle_key="F8",
    )

    assert action == "combat_select_attacker"
    assert turn_key is None
    assert controller.tapped == [("F8", 0.06)]
    assert fallback.last_tapped_keys == ["F8"]


def test_combat_fallback_cannot_attack_unconfirmed_target_during_clear_latch():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)

    action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=False, target_present=True, nameplate_visible=True),
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        attacker_target_selection=True,
        attacker_target_cycle_key="F8",
    )

    assert action == "combat_select_attacker"
    assert controller.tapped == [("F8", 0.06)]


def test_combat_uses_interact_to_face_confirmed_attacker_before_turning():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(
            active=True,
            target_present=True,
            target_is_attacker=True,
            facing_error_marker_visible=True,
        ),
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        target_interact_facing=True,
        target_interact_key="G",
    )

    assert action == "combat_interact_face_target"
    assert turn_key is None
    assert controller.tapped == [("G", 0.06)]


def test_combat_uses_interact_to_approach_only_confirmed_attacker():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)

    action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(
            active=True,
            target_present=True,
            target_is_attacker=True,
            out_of_range_marker_visible=True,
        ),
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        target_interact_facing=True,
        target_interact_key="G",
    )

    assert action == "combat_interact_approach_target"
    assert controller.tapped == [("G", 0.06)]


def test_combat_interact_probe_is_bounded_and_never_used_for_unconfirmed_target():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)
    fallback.unconfirmed_e_started_at = 10.0

    action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, target_is_attacker=True),
        now=10.4,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        target_interact_facing=True,
        target_interact_key="G",
        target_interact_cooldown=0.45,
        target_interact_probe_delay=0.35,
    )
    waiting_action, _ = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(active=True, target_present=True, target_is_attacker=True),
        now=10.5,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        target_interact_facing=True,
        target_interact_key="G",
        target_interact_cooldown=0.45,
        target_interact_probe_delay=0.35,
    )

    assert action == "combat_interact_probe_target"
    assert waiting_action in {"combat_attack", "combat_hold"}
    assert controller.tapped.count(("G", 0.06)) == 1


def test_combat_uses_nameplate_servo_without_damage_timeout_rotation():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

        def release_key(self, key: str) -> None:
            raise AssertionError(f"unexpected release: {key}")

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=0.2, clear_frames=2)
    fallback.unconfirmed_e_started_at = 1.0

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        CombatState(
            active=True,
            target_present=True,
            target_is_attacker=True,
            nameplate_visible=True,
            face_turn_key="A",
        ),
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        nameplate_facing=True,
        nameplate_face_turn_duration=0.08,
    )

    assert action == "combat_face_target_nameplate"
    assert turn_key == "A"
    assert controller.tapped == [("A", 0.08)]


def test_apply_combat_fallback_attacks_latched_second_enemy_evidence_without_active_target():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2)
    assert fallback.observe(CombatState(active=True, target_present=True), now=9.0)
    combat = CombatState(active=False, nameplate_visible=True, outgoing_damage_visible=True)

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
    )

    assert action == "combat_attack"
    assert turn_key is None
    assert controller.tapped == [("E", 0.06)]


def test_apply_combat_fallback_does_not_turn_from_unconfirmed_nameplate_direction():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2)
    combat = CombatState(active=True, target_present=True, face_turn_key="A")

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.45,
        attack_tap_duration=0.06,
        invert_turn_direction=True,
    )

    assert action == "combat_attack"
    assert turn_key is None
    assert controller.tapped == [("E", 0.06)]


def test_apply_combat_fallback_closes_range_when_target_nameplate_is_centered():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=1.0, clear_frames=2)
    combat = CombatState(
        active=True,
        target_present=True,
        nameplate_visible=True,
        out_of_range_marker_visible=True,
    )

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        range_approach_duration=0.35,
        range_approach_cooldown=0.55,
    )

    assert action == "combat_close_range"
    assert turn_key is None
    assert controller.tapped == [("W", 0.35)]


def test_apply_combat_fallback_approaches_out_of_range_attacker_without_nameplate_turning():
    class DummyInput:
        def __init__(self) -> None:
            self.tapped: list[tuple[str, float]] = []

        def tap_key(self, key: str, duration: float = 0.15) -> None:
            self.tapped.append((key, duration))

    controller = DummyInput()
    fallback = CombatFallbackState(attack_keys=("E",), attack_interval=1.0, clear_frames=2)
    combat = CombatState(
        active=True,
        target_present=True,
        target_is_attacker=True,
        out_of_range_marker_visible=True,
        face_turn_key="D",
    )

    action, turn_key = _apply_combat_fallback(
        controller,  # type: ignore[arg-type]
        fallback,
        combat,
        now=10.0,
        face_turn_duration=0.15,
        face_search_turn_180_duration=0.9,
        face_search_turn_90_duration=0.225,
        attack_tap_duration=0.06,
        invert_turn_direction=True,
    )

    assert action == "combat_close_range"
    assert turn_key is None
    assert controller.tapped == [("W", 0.35)]


def test_capture_reached_ore_training_sample_saves_only_when_ore_visible(tmp_path):
    class DummyCapture:
        def crop_minimap(self, frame, config):  # noqa: ANN001
            return frame

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    ore_bgr = cv2.cvtColor(np.array([[[30, 210, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.rectangle(frame, (30, 30), (35, 35), ore_bgr, -1)
    config = {
        "training_capture": {
            "reached_ore": {
                "enabled": True,
                "output_dir": str(tmp_path / "samples"),
                "save_debug_artifacts": True,
            }
        },
        "recognition": {
            "hsv_lower": [22, 120, 130],
            "hsv_upper": [38, 255, 255],
            "min_area": 4,
            "max_area": 80,
            "min_width": 2,
            "max_width": 12,
            "min_height": 2,
            "max_height": 12,
            "close_kernel": 1,
            "open_kernel": 1,
        },
    }
    node = MiningNode(zone_id=11, coord=6190403000, ore_type="Copper", node_id=1858)

    result = _capture_reached_ore_training_sample(
        DummyCapture(),  # type: ignore[arg-type]
        frame,
        config,
        target_node=node,
        target_coord=node.coord,
        reached_coord=6198394400,
        distance=0.2,
        index=7,
        route_output_dir=tmp_path / "route",
    )

    assert result is not None
    assert result["saved"]
    assert result["ore_points"]
    assert (tmp_path / "samples").exists()
    assert (next((tmp_path / "samples").iterdir()) / "metadata.json").exists()


def test_capture_reached_ore_training_sample_skips_empty_minimap(tmp_path):
    class DummyCapture:
        def crop_minimap(self, frame, config):  # noqa: ANN001
            return frame

    config = {
        "training_capture": {"reached_ore": {"enabled": True, "output_dir": str(tmp_path / "samples")}},
        "recognition": {"min_area": 4, "max_area": 80},
    }
    node = MiningNode(zone_id=11, coord=6190403000, ore_type="Copper", node_id=1858)

    result = _capture_reached_ore_training_sample(
        DummyCapture(),  # type: ignore[arg-type]
        np.zeros((80, 80, 3), dtype=np.uint8),
        config,
        target_node=node,
        target_coord=node.coord,
        reached_coord=6198394400,
        distance=0.2,
        index=7,
        route_output_dir=tmp_path / "route",
    )

    assert result is None
    assert not (tmp_path / "samples").exists()


def _threat(reason: str, turn_key: str) -> ThreatDetection:
    return ThreatDetection(
        reason=reason,
        bbox=BoundingBox(100, 100, 40, 8),
        score=0.9,
        turn_key=turn_key,
        source="test",
    )


def test_threat_avoidance_commits_turn_key_within_window():
    commitment = ThreatAvoidanceCommitment(commit_duration=1.75)

    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "A"), now=10.0, fallback_turn_key="D") == "A"
    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "D"), now=10.5, fallback_turn_key="D") == "A"
    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "D"), now=11.8, fallback_turn_key="D") == "D"


def test_threat_avoidance_clears_when_threat_disappears():
    commitment = ThreatAvoidanceCommitment(commit_duration=1.75)

    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "A"), now=10.0, fallback_turn_key="D") == "A"
    assert commitment.choose_turn_key(None, now=10.2, fallback_turn_key="D") is None
    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "D"), now=10.4, fallback_turn_key="D") == "D"


def test_hostile_target_frame_reuses_last_committed_key():
    commitment = ThreatAvoidanceCommitment(commit_duration=1.0)

    assert commitment.choose_turn_key(_threat("aggressive_nameplate", "A"), now=10.0, fallback_turn_key="D") == "A"
    assert commitment.choose_turn_key(_threat("hostile_target_frame", "D"), now=11.2, fallback_turn_key="D") == "A"
