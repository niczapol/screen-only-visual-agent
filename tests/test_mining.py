import cv2
import numpy as np
import pytest
from pathlib import Path

from vision_bot.cursor_classifier import CursorSnapshot, CursorTemplateClassifier
from types import SimpleNamespace

from vision_bot.mining import (
    BrightOreTracker,
    HoverScanPhase,
    MiningCycleController,
    MiningCandidate,
    MiningHoverScanner,
    MiningPhase,
    MinimapTooltipProbeController,
    MinimapTooltipProbePhase,
    _start_access_leg_near_current,
    build_front_target_probe_points,
    build_salient_ore_probe_points,
    build_scan_points,
    detect_mining_tooltip,
    detect_pickaxe_cursor,
    mining_target_alignment,
    minimap_marker_client_point,
)
from vision_bot.ore_world_detector import OreWorldDetection, OreWorldDetectorResult
from vision_bot.coords import coord_to_xy, xy_to_coord


def test_build_scan_points_starts_near_scan_region_center():
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "mining": {"scan_region": {"x": 760, "y": 260, "width": 940, "height": 640}, "scan_step": 32},
    }

    points = build_scan_points((1440, 2560, 3), config)

    assert points
    assert abs(points[0][0] - 1230) <= 32
    assert abs(points[0][1] - 580) <= 32


def test_salient_ore_probes_cover_colorful_world_object_outside_center_scan():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.rectangle(frame, (330, 920), (430, 1040), (255, 255, 0), -1)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "mining": {
            "salient_probe_region": {"x": 220, "y": 180, "width": 1760, "height": 950},
            "salient_probe_max_points": 12,
        },
    }

    points = build_salient_ore_probe_points(frame, config)

    assert points
    assert min(abs(x - 380) + abs(y - 980) for x, y in points) <= 4


def test_front_target_probes_cover_faced_node_before_general_scan():
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "mining": {},
    }

    points = build_front_target_probe_points((1440, 2560, 3), config)

    assert points[:7] == [
        (1280, 893),
        (1248, 893),
        (1312, 893),
        (1216, 893),
        (1344, 893),
        (1184, 893),
        (1376, 893),
    ]
    assert len(points) == 35


def test_front_target_probes_scale_horizontal_offsets_with_client_width():
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "mining": {"front_probe_y_fractions": [0.5], "front_probe_max_points": 3},
    }

    points = build_front_target_probe_points((720, 1280, 3), config)

    assert points == [(640, 360), (624, 360), (656, 360)]


def test_hover_scanner_prioritizes_salient_world_object_before_front_grid():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.rectangle(frame, (1510, 690), (1570, 770), (255, 255, 0), -1)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_max_points": 16,
            "hover_delay": 0.08,
            "front_probe_enabled": True,
            "front_probe_y_fractions": [0.50],
            "front_probe_max_points": 3,
            "salient_probe_enabled": True,
            "salient_probe_region": {"x": 220, "y": 180, "width": 1760, "height": 950},
            "salient_probe_max_points": 9,
        },
    }
    scanner = MiningHoverScanner(
        _FakeCapture(),
        _FakeMouse(),
        config,
        cursor_signature_reader=lambda: 100,
        ore_world_detector=SimpleNamespace(
            detect=lambda _frame: OreWorldDetectorResult(reason="no_detection")
        ),
    )

    scanner.start(frame.shape, now=0.0)
    probe = scanner.step(frame, now=0.1)

    assert probe.phase == HoverScanPhase.PROBE_SETTLE
    assert probe.point is not None
    assert abs(probe.point[0] - 1540) <= 4
    assert abs(probe.point[1] - 730) <= 4


def test_mining_target_alignment_turns_until_world_target_is_centered():
    config = {
        "mining": {
            "world_target_center_x_fraction": 0.5,
            "world_target_face_deadzone_fraction": 0.05,
            "world_target_face_min_seconds": 0.05,
            "world_target_face_max_seconds": 0.18,
        }
    }

    left = mining_target_alignment((400, 700), (1440, 2560, 3), config)
    center = mining_target_alignment((1285, 700), (1440, 2560, 3), config)

    assert not left.aligned
    assert left.turn_key == "A"
    assert 0.05 <= left.turn_seconds <= 0.18
    assert center.aligned
    assert center.turn_key is None


def test_detect_mining_tooltip_from_colored_tooltip_region():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "mining": {
            "tooltip_region": {"x": 2380, "y": 1010, "width": 180, "height": 130},
            "tooltip_min_yellow_pixels": 18,
            "tooltip_min_green_pixels": 18,
            "tooltip_min_dark_ratio": 0.20,
        },
    }

    cv2.rectangle(frame, (2380, 1010), (2559, 1139), (5, 5, 5), -1)
    cv2.rectangle(frame, (2410, 1040), (2490, 1052), (0, 210, 255), -1)
    cv2.rectangle(frame, (2410, 1070), (2505, 1082), (0, 220, 0), -1)

    assert detect_mining_tooltip(frame, config)


def test_detect_mining_tooltip_from_current_ui_fallback_region():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "mining": {
            "tooltip_regions": [
                {"x": 2380, "y": 1010, "width": 180, "height": 130},
                {"x": 2100, "y": 1120, "width": 350, "height": 160},
            ],
            "tooltip_min_yellow_pixels": 18,
            "tooltip_min_green_pixels": 18,
            "tooltip_min_dark_ratio": 0.20,
        },
    }
    cv2.rectangle(frame, (2140, 1140), (2419, 1259), (5, 5, 5), -1)
    cv2.rectangle(frame, (2180, 1160), (2320, 1174), (0, 210, 255), -1)
    cv2.rectangle(frame, (2180, 1190), (2340, 1204), (0, 220, 0), -1)

    assert detect_mining_tooltip(frame, config)


def test_detect_mining_tooltip_accepts_neutral_requirement_only_in_safe_fallback_region():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "mining": {
            "tooltip_regions": [
                {"x": 2380, "y": 1010, "width": 180, "height": 130},
                {"x": 2100, "y": 1120, "width": 350, "height": 160},
            ],
            "tooltip_min_yellow_pixels": 18,
            "tooltip_min_green_pixels": 18,
            "tooltip_neutral_fallback_enabled": True,
            "tooltip_neutral_fallback_region_indices": [1],
            "tooltip_min_neutral_pixels": 180,
            "tooltip_min_dark_ratio": 0.20,
        },
    }
    cv2.rectangle(frame, (2100, 1120), (2449, 1279), (5, 5, 5), -1)
    cv2.rectangle(frame, (2180, 1160), (2320, 1174), (0, 210, 255), -1)
    cv2.rectangle(frame, (2180, 1190), (2340, 1204), (150, 150, 150), -1)

    assert detect_mining_tooltip(frame, config)

    frame[:] = 0
    cv2.rectangle(frame, (2380, 1010), (2559, 1139), (5, 5, 5), -1)
    cv2.rectangle(frame, (2410, 1040), (2490, 1052), (0, 210, 255), -1)
    cv2.rectangle(frame, (2410, 1070), (2505, 1082), (150, 150, 150), -1)

    assert not detect_mining_tooltip(frame, config)


def test_detect_mining_tooltip_rejects_hostile_unit_tooltip():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "mining": {
            "tooltip_regions": [{"x": 2100, "y": 1120, "width": 350, "height": 160}],
            "tooltip_min_yellow_pixels": 18,
            "tooltip_min_green_pixels": 18,
            "tooltip_min_dark_ratio": 0.20,
            "reject_hostile_tooltip": True,
            "hostile_tooltip_min_red_pixels": 12,
            "hostile_tooltip_min_bottom_green_pixels": 55,
        },
    }
    cv2.rectangle(frame, (2100, 1120), (2449, 1279), (5, 5, 5), -1)
    cv2.rectangle(frame, (2140, 1160), (2280, 1174), (0, 50, 230), -1)
    cv2.rectangle(frame, (2140, 1248), (2380, 1260), (0, 220, 0), -1)

    assert not detect_mining_tooltip(frame, config)


def _mining_config(**overrides):
    mining = {
        "route_live_enabled": True,
        "bright_confirm_frames": 2,
        "verify_clear_frames": 2,
        "gather_wait_seconds": 3.0,
        "verify_timeout_seconds": 6.0,
        "candidate_match_distance_coord": 1.35,
        "marker_database_match_distance_coord": 0.55,
        "world_search_distance_coord": 0.22,
        "cooldown_seconds": 180.0,
        "failure_cooldown_seconds": 20.0,
    }
    mining.update(overrides)
    return {"mining": mining}


def test_mining_cycle_requires_stable_bright_icon_and_ignores_dark_only():
    controller = MiningCycleController(_mining_config())

    dark = controller.observe_minimap([], [(100, 100)])
    first = controller.observe_minimap([(120, 110)], [])
    second = controller.observe_minimap([(121, 110)], [])

    assert dark.dark_only
    assert not dark.confirmed_bright
    assert not first.confirmed_bright
    assert second.confirmed_bright


def test_mining_cycle_selects_nearest_available_node_and_applies_cooldown():
    controller = MiningCycleController(_mining_config())
    nodes = [
        SimpleNamespace(node_id=1, coord=5010501000, ore_type="Iron"),
        SimpleNamespace(node_id=2, coord=5090501000, ore_type="Gold"),
    ]

    candidate = controller.choose_candidate(5000500000, nodes, now=10.0)
    assert candidate is not None
    assert candidate.node_id == 1

    controller.begin(candidate, now=10.0)
    controller.mark_intercept_reached()
    controller.fail("hover_not_found", now=11.0)
    assert controller.phase == MiningPhase.RESUME
    assert controller.consume_outcome() is not None
    assert controller.choose_candidate(5000500000, nodes[:1], now=20.0) is None
    assert controller.choose_candidate(5000500000, nodes[:1], now=31.1) is not None


def test_failed_mining_attempt_cools_down_neighboring_spawn_records():
    controller = MiningCycleController(
        _mining_config(
            failure_cooldown_seconds=300.0,
            failure_spatial_cooldown_radius_coord=0.35,
        )
    )
    first = SimpleNamespace(node_id=1, coord=xy_to_coord(50.0, 50.0), ore_type="Iron")
    neighbor = SimpleNamespace(node_id=2, coord=xy_to_coord(50.2, 50.1), ore_type="Iron")
    distant = SimpleNamespace(node_id=3, coord=xy_to_coord(50.6, 50.0), ore_type="Iron")

    candidate = controller.choose_candidate(xy_to_coord(50.0, 50.0), [first], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.fail("ore_intercept_timeout", now=5.0)
    controller.consume_outcome()

    assert controller.choose_candidate(xy_to_coord(50.0, 50.0), [neighbor], now=10.0) is None
    assert controller.choose_candidate(xy_to_coord(50.0, 50.0), [distant], now=10.0) is not None


def test_mining_intercept_has_hard_timeout():
    controller = MiningCycleController(
        _mining_config(max_intercept_seconds=2.0, intercept_marker_missing_frames=99)
    )
    node = SimpleNamespace(node_id=1, coord=5000500000, ore_type="Iron")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)

    outcome = controller.observe_intercept_marker(now=2.1)

    assert outcome is not None
    assert outcome.reason == "ore_intercept_timeout"


def test_face_timeout_falls_back_to_authoritative_world_scan():
    controller = MiningCycleController(
        _mining_config(
            max_face_node_seconds=2.0,
            face_node_timeout_starts_world_scan=True,
        )
    )
    node = SimpleNamespace(node_id=1, coord=5000500000, ore_type="Iron")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.mark_intercept_reached(now=0.1)

    outcome = controller.observe_face_node_timeout(now=2.2)

    assert outcome is None
    assert controller.phase == MiningPhase.WORLD_SCAN


def test_face_target_uses_latest_centered_tooltip_marker_projection():
    controller = MiningCycleController(_mining_config())
    stale_database_coord = xy_to_coord(50.8, 50.8)
    live_marker_coord = xy_to_coord(50.1, 50.2)
    candidate = MiningCandidate(
        node_id=1,
        coord=stale_database_coord,
        ore_type="Mithril",
        tooltip_confirmed=True,
    )

    controller.begin(candidate, now=0.0)
    controller.last_marker_intercept_coord = live_marker_coord
    controller.center_tooltip_confirmed = True
    controller.mark_intercept_reached(now=1.0)

    assert controller.face_target_coord == live_marker_coord


def test_face_target_uses_database_coord_without_center_tooltip_confirmation():
    controller = MiningCycleController(_mining_config())
    database_coord = xy_to_coord(50.8, 50.8)
    candidate = MiningCandidate(
        node_id=1,
        coord=database_coord,
        ore_type="Mithril",
        tooltip_confirmed=True,
    )

    controller.begin(candidate, now=0.0)
    controller.last_marker_intercept_coord = xy_to_coord(50.1, 50.2)
    controller.mark_intercept_reached(now=1.0)

    assert controller.face_target_coord == database_coord


def test_face_target_holds_database_anchor_after_arrival_despite_marker_residual():
    controller = MiningCycleController(_mining_config())
    database_coord = xy_to_coord(50.8, 50.8)
    candidate = MiningCandidate(
        node_id=1,
        coord=database_coord,
        ore_type="Iron",
        tooltip_confirmed=True,
    )

    controller.begin(candidate, now=0.0)
    controller.last_marker_intercept_coord = xy_to_coord(50.1, 50.2)
    controller.center_tooltip_confirmed = True
    controller.database_anchor_arrived = True
    controller.mark_intercept_reached(now=1.0)

    assert controller.face_target_coord == database_coord


def test_tracked_bright_target_that_dims_near_player_remains_active():
    controller = MiningCycleController(
        _mining_config(
            dark_target_confirm_frames=2,
            direct_marker_intercept_enabled=False,
        )
    )
    controller.observe_minimap([(40, 50)], [])
    controller.observe_minimap([(40, 50)], [])
    node = SimpleNamespace(node_id=1, coord=5000500000, ore_type="Iron")
    candidate = controller.choose_candidate(
        5000500000,
        [node],
        now=0.0,
        marker_point=(40, 50),
        minimap_shape=(174, 183, 3),
    )
    assert candidate is not None
    controller.begin(candidate, now=0.0)

    controller.observe_minimap([], [(40, 50)])
    assert controller.observe_dark_target(now=1.0) is None
    controller.observe_minimap([], [(40, 50)])
    presence = controller.last_presence
    outcome = controller.observe_dark_target(now=2.0)

    assert presence.target_visible
    assert presence.target_dark_visible
    assert outcome is None
    assert controller.phase == MiningPhase.INTERCEPT


def test_unrelated_dark_marker_still_abandons_lost_bright_target():
    controller = MiningCycleController(_mining_config(dark_target_confirm_frames=2))
    controller.observe_minimap([(40, 50)], [])
    controller.observe_minimap([(40, 50)], [])
    node = SimpleNamespace(node_id=1, coord=5000500000, ore_type="Iron")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)

    controller.observe_minimap([], [(100, 100)])
    assert controller.observe_dark_target(now=1.0) is None
    controller.observe_minimap([], [(100, 100)])
    outcome = controller.observe_dark_target(now=2.0)

    assert outcome is not None
    assert outcome.reason == "ore_marker_dark_below"


def test_mining_candidate_uses_minimap_bearing_instead_of_nearest_database_node():
    config = _mining_config(
        marker_bearing_enabled=True,
        marker_bearing_min_alignment=0.20,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    config["route"] = {"tracking_radius_coord": 1.15}
    controller = MiningCycleController(config)
    current = xy_to_coord(58.19, 27.20)
    nodes = [
        SimpleNamespace(
            node_id=1,
            coord=xy_to_coord(57.76, 27.48),
            ore_type="Small Thorium",
        ),
        SimpleNamespace(
            node_id=2,
            coord=xy_to_coord(57.40, 27.20),
            ore_type="Small Thorium",
        ),
    ]

    candidate = controller.choose_candidate(
        current,
        nodes,
        now=0.0,
        marker_point=(45, 84),
        minimap_shape=(174, 183, 3),
    )

    assert candidate is not None
    assert candidate.node_id == 2


def test_mining_candidate_uses_live_marker_as_dynamic_intercept() -> None:
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        marker_tracking_radius_x_coord=0.72,
        marker_tracking_radius_y_coord=1.09,
        direct_marker_intercept_max_node_delta_coord=1.35,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(137, 148)], [])
    controller.observe_minimap([(137, 148)], [])
    current = xy_to_coord(71.51, 42.04)
    nodes = [
        SimpleNamespace(node_id=1, coord=xy_to_coord(71.20, 42.30), ore_type="Thorium"),
        SimpleNamespace(node_id=2, coord=xy_to_coord(71.70, 43.10), ore_type="Thorium"),
    ]

    candidate = controller.choose_candidate(
        current,
        nodes,
        now=0.0,
        marker_point=(137, 148),
        minimap_shape=(278, 293, 3),
    )

    assert candidate is not None
    assert candidate.node_id == 1
    assert candidate.marker_intercept_coord is not None
    marker_x, marker_y = coord_to_xy(candidate.marker_intercept_coord)
    assert 71.43 <= marker_x <= 71.46
    assert 42.18 <= marker_y <= 42.21
    controller.begin(candidate, now=0.0, current_coord=current)
    assert controller.direct_marker_intercept_active
    assert controller.current_intercept_coord == candidate.marker_intercept_coord
    assert controller.direct_marker_world_search_distance == 0.04


def test_zoom12_live_calibration_projects_marker_at_observed_pixel_scale() -> None:
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        marker_tracking_radius_x_coord=2.34,
        marker_tracking_radius_y_coord=3.38,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.42,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)

    projected = controller._marker_intercept_coord(
        current,
        (157, 143),
        (278, 293, 3),
    )

    assert projected is not None
    projected_x, projected_y = coord_to_xy(projected)
    assert projected_x == pytest.approx(50.21, abs=0.01)
    assert projected_y == pytest.approx(50.28, abs=0.01)


def test_unmatched_live_marker_is_ignored() -> None:
    config = _mining_config(
        direct_marker_intercept_enabled=True,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(130, 120)], [])
    presence = controller.observe_minimap([(130, 120)], [])
    current = xy_to_coord(50.0, 50.0)

    candidate = controller.choose_candidate(
        current,
        [],
        now=0.0,
        marker_point=presence.confirmed_point,
        minimap_shape=(174, 183, 3),
    )

    assert candidate is None
    snapshot = controller.snapshot(now=0.0)
    assert snapshot["candidate_rejection_reason"] == "marker_not_in_database"
    assert snapshot["marker_intercept_coord"] is not None
    assert snapshot["marker_database_distance"] is None


def test_live_marker_does_not_select_nearby_database_node_outside_match_gate() -> None:
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        marker_database_match_distance_coord=0.10,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(130, 120)], [])
    presence = controller.observe_minimap([(130, 120)], [])
    current = xy_to_coord(50.0, 50.0)
    nearby_but_unrelated = SimpleNamespace(
        node_id=1,
        coord=xy_to_coord(50.05, 50.05),
        ore_type="Iron",
    )

    candidate = controller.choose_candidate(
        current,
        [nearby_but_unrelated],
        now=0.0,
        marker_point=presence.confirmed_point,
        minimap_shape=(174, 183, 3),
    )

    assert candidate is None
    snapshot = controller.snapshot(now=0.0)
    assert snapshot["candidate_rejection_reason"] == "marker_not_in_database"
    assert snapshot["marker_database_distance"] > 0.10


def test_minimap_tooltip_probe_is_bounded_and_confirms_the_hovered_track() -> None:
    controller = MinimapTooltipProbeController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            minimap_tooltip_probe={
                "enabled": True,
                "settle_seconds": 0.10,
                "timeout_seconds": 0.70,
                "retry_cooldown_seconds": 2.0,
            },
        )
    )

    started = controller.observe(
        track_id=7,
        marker_point=(130, 120),
        now=0.0,
    )
    settling = controller.observe(
        track_id=None,
        marker_point=None,
        now=0.05,
        tooltip_ore_id=8,
        tooltip_ore_type="Small Thorium",
    )
    confirmed = controller.observe(
        track_id=None,
        marker_point=None,
        now=0.11,
        tooltip_ore_id=8,
        tooltip_ore_type="Small Thorium",
    )

    assert started.phase == MinimapTooltipProbePhase.HOVER
    assert started.move_point == (130, 120)
    assert settling.phase == MinimapTooltipProbePhase.HOVER
    assert confirmed.phase == MinimapTooltipProbePhase.CONFIRMED
    assert confirmed.release_cursor
    assert controller.confirmed_track_id == 7
    assert controller.confirmed_point == (130, 120)
    assert controller.confirmed_ore_type == "Small Thorium"

    controller.observe(track_id=7, marker_point=(126, 118), now=0.2)
    assert controller.confirmed_point == (126, 118)


def test_minimap_tooltip_probe_times_out_and_cools_down_without_tooltip() -> None:
    controller = MinimapTooltipProbeController(
        _mining_config(
            minimap_tooltip_probe={
                "enabled": True,
                "settle_seconds": 0.10,
                "timeout_seconds": 0.70,
                "retry_cooldown_seconds": 2.0,
            }
        )
    )
    controller.observe(track_id=3, marker_point=(120, 100), now=0.0)

    timed_out = controller.observe(
        track_id=None,
        marker_point=None,
        now=0.71,
    )
    cooldown = controller.observe(
        track_id=3,
        marker_point=(120, 100),
        now=1.0,
    )

    assert timed_out.action == "minimap_tooltip_probe_timeout"
    assert timed_out.release_cursor
    assert cooldown.phase == MinimapTooltipProbePhase.COOLDOWN


def test_tooltip_confirmation_relaxes_projection_and_uses_nearest_same_type_node() -> None:
    config = _mining_config(
        require_minimap_tooltip_confirmation=True,
        tooltip_confirmed_relaxed_match_enabled=True,
        marker_database_match_distance_coord=0.10,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    nodes = [
        SimpleNamespace(
            node_id=1,
            coord=xy_to_coord(50.45, 50.05),
            ore_type="Iron",
        ),
        SimpleNamespace(
            node_id=2,
            coord=xy_to_coord(50.80, 50.05),
            ore_type="Small Thorium",
        ),
        SimpleNamespace(
            node_id=3,
            coord=xy_to_coord(49.65, 50.00),
            ore_type="Small Thorium",
        ),
    ]

    assert controller.choose_candidate(
        current,
        nodes,
        now=0.0,
        marker_point=(91, 20),
        minimap_shape=(174, 183, 3),
    ) is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "minimap_tooltip_unconfirmed"
    )

    candidate = controller.choose_candidate(
        current,
        nodes,
        now=0.1,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Small Thorium",
    )

    assert candidate is not None
    assert candidate.node_id == 3
    snapshot = controller.snapshot(now=0.1)
    assert snapshot["tooltip_confirmed_relaxed_match"]
    assert snapshot["confirmed_ore_type"] == "Small Thorium"


def test_tooltip_confirmed_db_search_is_not_clipped_by_ordinary_player_radius() -> None:
    current = xy_to_coord(50.0, 50.0)
    access_plan = SimpleNamespace(
        options=(SimpleNamespace(inbound_coords=(current,)),),
    )
    node = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.1, 50.0),
        ore_type="Iron",
        access_plan=access_plan,
    )
    controller = MiningCycleController(
        _mining_config(
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
            marker_database_match_distance_coord=0.10,
            tooltip_confirmed_relaxed_match_enabled=True,
            require_access_plan_for_route_live=True,
            access_attachment_max_distance_coord=0.18,
        )
    )

    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Iron",
    )

    assert candidate is not None
    assert candidate.node_id == 17
    snapshot = controller.snapshot(now=0.0)
    assert snapshot["candidate_search_distance"] == 2.0
    assert snapshot["candidate_nodes_scanned"] == 1
    assert snapshot["candidate_same_type_nodes"] == 1
    assert snapshot["candidate_match_elapsed_ms"] >= 0.0


def test_unconfirmed_db_search_retains_ordinary_player_radius() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=False,
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
        )
    )
    node = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.1, 50.0),
        ore_type="Iron",
    )

    candidate = controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [node],
        now=0.0,
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_search_distance"] == 0.70


def test_tooltip_confirmed_db_identity_waits_for_route_attachment() -> None:
    current = xy_to_coord(50.0, 50.0)
    access_plan = SimpleNamespace(
        options=(
            SimpleNamespace(
                inbound_coords=(xy_to_coord(50.8, 50.0),),
            ),
        ),
    )
    node = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.1, 50.0),
        ore_type="Iron",
        access_plan=access_plan,
    )
    controller = MiningCycleController(
        _mining_config(
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
            marker_database_match_distance_coord=0.10,
            tooltip_confirmed_relaxed_match_enabled=True,
            require_access_plan_for_route_live=True,
            access_attachment_max_distance_coord=0.18,
        )
    )

    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Iron",
    )

    assert candidate is None
    snapshot = controller.snapshot(now=0.0)
    assert snapshot["candidate_rejection_reason"] == "access_attachment_too_far"
    assert snapshot["access_attachment_distance"] > 0.18
    assert not snapshot["tooltip_confirmed_unmatched_candidate"]


def test_tooltip_confirmed_db_identity_accepts_live_calibrated_route_attachment() -> None:
    current = xy_to_coord(50.0, 50.0)
    access_plan = SimpleNamespace(
        options=(
            SimpleNamespace(
                inbound_coords=(xy_to_coord(50.284, 50.0),),
            ),
        ),
    )
    node = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.1, 50.0),
        ore_type="Iron",
        access_plan=access_plan,
    )
    controller = MiningCycleController(
        _mining_config(
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
            marker_database_match_distance_coord=0.10,
            tooltip_confirmed_relaxed_match_enabled=True,
            require_access_plan_for_route_live=True,
            access_attachment_max_distance_coord=0.35,
        )
    )

    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Iron",
    )

    assert candidate is not None
    assert candidate.node_id == 17
    assert controller.snapshot(now=0.0)["access_attachment_distance"] < 0.35


def test_tooltip_confirmed_db_identity_reports_missing_access_plan() -> None:
    controller = MiningCycleController(
        _mining_config(
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
            tooltip_confirmed_relaxed_match_enabled=True,
            require_access_plan_for_route_live=True,
        )
    )
    node = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.1, 50.0),
        ore_type="Iron",
        access_plan=None,
    )

    candidate = controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Iron",
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "database_node_missing_access_plan"
    )


def test_tooltip_identity_does_not_fall_through_to_farther_planned_node() -> None:
    current = xy_to_coord(50.0, 50.0)
    planned = SimpleNamespace(
        node_id=18,
        coord=xy_to_coord(51.5, 50.0),
        ore_type="Iron",
        access_plan=SimpleNamespace(
            options=(SimpleNamespace(inbound_coords=(current,)),),
        ),
    )
    nearest_unplanned = SimpleNamespace(
        node_id=17,
        coord=xy_to_coord(51.0, 50.0),
        ore_type="Iron",
        access_plan=None,
    )
    controller = MiningCycleController(
        _mining_config(
            candidate_match_distance_coord=0.70,
            tooltip_confirmed_database_search_distance_coord=2.0,
            marker_database_match_distance_coord=0.10,
            tooltip_confirmed_relaxed_match_enabled=True,
            require_access_plan_for_route_live=True,
            access_attachment_max_distance_coord=0.18,
        )
    )

    candidate = controller.choose_candidate(
        current,
        [planned, nearest_unplanned],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Iron",
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "database_node_missing_access_plan"
    )


def test_tooltip_confirmed_ore_type_must_exist_in_nearby_database_nodes() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            tooltip_confirmed_relaxed_match_enabled=True,
        )
    )
    node = SimpleNamespace(
        node_id=1,
        coord=xy_to_coord(50.2, 50.0),
        ore_type="Iron",
    )

    candidate = controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Small Thorium",
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "tooltip_ore_not_in_runtime_database"
    )


def test_tooltip_confirmed_unmatched_marker_becomes_bounded_live_candidate() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            tooltip_confirmed_unmatched_enabled=True,
            tooltip_confirmed_unmatched_max_distance_coord=0.75,
        )
    )
    current = xy_to_coord(50.0, 50.0)

    candidate = controller.choose_candidate(
        current,
        [],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Small Thorium",
    )

    assert candidate is not None
    assert candidate.node_id is not None and candidate.node_id < 0
    assert candidate.coord == candidate.marker_intercept_coord
    assert candidate.ore_type == "Small Thorium"
    assert candidate.tooltip_confirmed
    assert candidate.source == "live_tooltip_marker"
    snapshot = controller.snapshot(now=0.0)
    assert snapshot["tooltip_confirmed_unmatched_candidate"]
    assert snapshot["tooltip_confirmed_unmatched_distance"] <= 0.75


def test_unmatched_marker_requires_native_tooltip_confirmation() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            tooltip_confirmed_unmatched_enabled=True,
        )
    )

    candidate = controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "minimap_tooltip_unconfirmed"
    )


def test_unmatched_fallback_does_not_bypass_unavailable_same_type_db_node() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            tooltip_confirmed_unmatched_enabled=True,
            require_access_plan_for_route_live=True,
        )
    )
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(
        node_id=1,
        coord=xy_to_coord(50.2, 50.0),
        ore_type="Small Thorium",
        access_plan=None,
    )

    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(130, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Small Thorium",
    )

    assert candidate is None
    assert not controller.snapshot(now=0.0)["tooltip_confirmed_unmatched_candidate"]


def test_unmatched_tooltip_projection_is_distance_bounded() -> None:
    controller = MiningCycleController(
        _mining_config(
            require_minimap_tooltip_confirmation=True,
            tooltip_confirmed_unmatched_enabled=True,
            tooltip_confirmed_unmatched_max_distance_coord=0.05,
        )
    )

    candidate = controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [],
        now=0.0,
        marker_point=(160, 87),
        minimap_shape=(174, 183, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Small Thorium",
    )

    assert candidate is None
    assert controller.snapshot(now=0.0)["candidate_rejection_reason"] == (
        "live_tooltip_projection_too_far"
    )


def test_minimap_marker_point_is_translated_to_client_coordinates() -> None:
    point = minimap_marker_client_point(
        (130, 120),
        (1440, 2560, 3),
        {
            "screen": {
                "reference_width": 2560,
                "reference_height": 1440,
                "scale_regions": True,
            },
            "minimap": {"x": 2267, "y": 0, "width": 293, "height": 278},
        },
    )

    assert point == (2397, 120)


def test_mining_direct_intercept_tracks_moving_live_marker() -> None:
    config = _mining_config(direct_marker_intercept_enabled=True)
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.48},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(130, 120)], [])
    controller.observe_minimap([(130, 120)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(node_id=1, coord=xy_to_coord(50.5, 50.6), ore_type="Iron")
    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(130, 120),
        minimap_shape=(174, 183, 3),
    )
    assert candidate is not None
    controller.begin(candidate, now=0.0, current_coord=current)
    first_intercept = controller.current_intercept_coord

    controller.observe_minimap([(115, 105)], [])
    refreshed = controller.refresh_direct_marker_intercept(
        xy_to_coord(50.3, 50.3),
        (174, 183, 3),
    )

    assert refreshed is not None
    assert refreshed != first_intercept


def test_tooltip_direct_intercept_stays_dynamic_final_stage() -> None:
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        final_world_scan_distance_coord=0.04,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=-1,
        coord=xy_to_coord(50.1, 50.0),
        ore_type="Small Thorium",
        marker_intercept_coord=xy_to_coord(50.1, 50.0),
        tooltip_confirmed=True,
        source="live_tooltip_marker",
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((125, 120),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(125, 120),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(125, 120),
    )

    first = controller.refresh_direct_marker_intercept(current, (200, 200, 3))
    assert first is not None
    assert len(controller.intercept_coords) == 1
    assert controller.direct_marker_stage_index == 0

    moved = xy_to_coord(50.2, 50.2)
    controller.target_marker_point = (120, 115)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 115),),
        dark_points=(),
        bright_streak=3,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 115),
        confirmed_track_id=1,
        target_visible=True,
    )
    refreshed = controller.refresh_direct_marker_intercept(
        moved,
        (200, 200, 3),
    )

    assert refreshed is not None
    assert refreshed != first
    assert len(controller.intercept_coords) == 1
    assert controller.intercept_cursor == controller.direct_marker_stage_index == 0


def test_direct_node_intercept_first_runs_to_database_coord_before_access_plan() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            direct_node_intercept_first_enabled=True,
            intercept_no_progress_seconds=1.0,
        )
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.1, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.4, 50.4),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.2, 50.2),
    )

    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.current_intercept_coord == candidate.coord
    assert controller.direct_node_intercept_active
    assert controller.access_option_rank is None
    assert controller.tried_access_option_ranks == set()
    assert controller.observe_intercept_marker(
        now=2.0,
        current_coord=xy_to_coord(50.0, 50.0),
    ) is None
    assert controller.access_option_rank == 0
    assert controller.current_intercept_coord == option.inbound_coords[0]


def test_direct_marker_uses_terrain_access_leg_before_dynamic_intercept() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            final_world_scan_distance_coord=0.005,
        )
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.1, 50.0), xy_to_coord(50.2, 50.2)),
        resume_coords=(xy_to_coord(50.1, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.32, 50.31),
    )

    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.current_intercept_coord == option.inbound_coords[0]
    assert controller.direct_marker_stage_index == 2
    assert controller.intercept_coords[-1] == candidate.coord
    controller.mark_intercept_target_reached(now=0.5)
    assert controller.current_intercept_coord == option.inbound_coords[1]
    controller.mark_intercept_target_reached(now=1.0)
    assert controller.current_intercept_coord == candidate.marker_intercept_coord
    controller.mark_intercept_target_reached(now=1.5)
    assert controller.current_intercept_coord == candidate.coord


def test_tooltip_confirmed_database_access_always_keeps_exact_anchor_last() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            final_world_scan_distance_coord=0.06,
        )
    )
    near_anchor = xy_to_coord(50.27, 50.30)
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(near_anchor,),
        resume_coords=(),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.30, 50.30),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.31, 50.30),
        tooltip_confirmed=True,
    )

    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.intercept_coords == (near_anchor, candidate.coord)
    controller.mark_intercept_target_reached(now=0.5)
    assert controller.final_intercept_uses_database_anchor

    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.target_marker_point = (120, 100)
    controller.last_minimap_shape = (200, 200, 3)
    assert controller.refresh_direct_marker_intercept(
        xy_to_coord(50.27, 50.30),
        (200, 200, 3),
    ) == candidate.coord


def test_candidate_proximity_advances_marker_stage_before_final_world_scan() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            direct_marker_world_scan_radius_pixels=12.0,
            direct_marker_world_scan_candidate_distance_coord=0.16,
            world_search_distance_coord=0.22,
        )
    )
    current = xy_to_coord(50.06, 50.04)
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.0, 50.1),
        ore_type="Iron",
        marker_intercept_coord=xy_to_coord(50.23, 50.11),
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((125, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(125, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.begin(candidate, now=0.0, current_coord=current)
    controller.target_marker_id = 1
    controller.target_marker_point = (125, 100)

    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.current_intercept_coord == candidate.coord
    assert controller.mark_intercept_target_reached(now=0.2)
    assert controller.phase == MiningPhase.FACE_NODE


def test_tooltip_confirmed_candidate_proximity_skips_unfinished_access_leg() -> None:
    controller = MiningCycleController(
        _mining_config(candidate_handoff_distance_coord=0.22)
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.7, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.0),
        ore_type="Iron",
        access_plan=plan,
        tooltip_confirmed=True,
    )
    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.observe_intercept_marker(
        now=0.5,
        current_coord=xy_to_coord(50.12, 50.0),
    ) is None

    assert controller.phase == MiningPhase.FACE_NODE
    snapshot = controller.snapshot(now=0.5)
    assert snapshot["candidate_handoff_reason"] == "candidate_proximity"
    assert snapshot["intercept_remaining"] == 0


def test_live_tooltip_candidate_does_not_handoff_on_coarse_projection_proximity() -> None:
    controller = MiningCycleController(
        _mining_config(candidate_handoff_distance_coord=0.22)
    )
    candidate = MiningCandidate(
        node_id=-1,
        coord=xy_to_coord(50.1, 50.0),
        ore_type="Small Thorium",
        marker_intercept_coord=xy_to_coord(50.1, 50.0),
        tooltip_confirmed=True,
        source="live_tooltip_marker",
    )
    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert not controller._candidate_proximity_handoff_ready(  # noqa: SLF001
        xy_to_coord(50.1, 50.0)
    )
    assert controller.snapshot(now=0.1)["candidate_handoff_reason"] is None


def test_tooltip_confirmed_candidate_segment_crossing_catches_sampled_overshoot() -> None:
    controller = MiningCycleController(
        _mining_config(candidate_handoff_distance_coord=0.12)
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.8, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.0),
        ore_type="Iron",
        access_plan=plan,
        tooltip_confirmed=True,
    )
    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.observe_intercept_marker(
        now=0.5,
        current_coord=xy_to_coord(50.55, 50.0),
    ) is None

    assert controller.phase == MiningPhase.FACE_NODE
    snapshot = controller.snapshot(now=0.5)
    assert snapshot["candidate_handoff_reason"] == "candidate_segment_crossed"
    assert snapshot["candidate_segment_distance"] <= 0.001


def test_unconfirmed_candidate_does_not_skip_terrain_access_on_proximity() -> None:
    controller = MiningCycleController(
        _mining_config(candidate_handoff_distance_coord=0.22)
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.7, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.0),
        ore_type="Iron",
        access_plan=plan,
        tooltip_confirmed=False,
    )
    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))

    assert controller.observe_intercept_marker(
        now=0.5,
        current_coord=xy_to_coord(50.12, 50.0),
    ) is None

    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.snapshot(now=0.5)["candidate_handoff_reason"] is None


def test_tooltip_confirmed_centered_marker_requires_center_tooltip_before_world_scan() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            direct_node_intercept_first_enabled=False,
            direct_marker_world_scan_radius_pixels=12.0,
            direct_marker_world_scan_candidate_distance_coord=0.16,
            world_search_distance_coord=0.22,
        )
    )
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.8, 50.8),
        ore_type="Small Thorium",
        marker_intercept_coord=xy_to_coord(50.1, 50.1),
        tooltip_confirmed=True,
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((103, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(103, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(103, 100),
    )

    assert controller.current_intercept_coord == candidate.coord
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.mark_database_anchor_arrived(now=0.2)
    assert controller.phase == MiningPhase.CENTER_TOOLTIP
    assert controller.observe_center_tooltip(
        now=0.35,
        tooltip_ore_type="Small Thorium",
    ) is None
    assert controller.phase == MiningPhase.FACE_NODE


def test_no_progress_retries_one_alternate_access_option_before_failure() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            intercept_no_progress_seconds=1.0,
            max_access_option_retries=1,
        )
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((100, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(100, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    first = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.1, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    second = SimpleNamespace(
        rank=1,
        inbound_coords=(xy_to_coord(49.9, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(
        primary_option_rank=0,
        resume_route_index=12,
        options=(first, second),
    )
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.3, 50.3),
    )
    current = xy_to_coord(50.0, 50.0)
    controller.begin(candidate, now=0.0, current_coord=current)
    controller.target_marker_id = 1
    controller.target_marker_point = (170, 100)
    controller.last_minimap_shape = (200, 200, 3)

    assert controller.observe_intercept_marker(now=0.0, current_coord=current) is None
    assert controller.observe_intercept_marker(now=1.1, current_coord=current) is None
    assert controller.access_option_rank == 1
    assert controller.access_option_retries == 1
    assert controller.current_intercept_coord == second.inbound_coords[0]

    outcome = controller.observe_intercept_marker(now=2.2, current_coord=current)
    assert outcome is not None
    assert outcome.reason == "ore_intercept_no_progress"


def test_current_target_progress_refreshes_stall_after_optimistic_transition_sample() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=False,
            intercept_no_progress_seconds=1.0,
            intercept_progress_coord=0.02,
        )
    )
    target = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(node_id=1, coord=target, ore_type="Iron")
    controller.begin(
        candidate,
        now=0.0,
        current_coord=xy_to_coord(50.138, 50.0),
    )

    # A transition-frame sample records 0.138 as the global best, then mounted
    # momentum carries the character away before it genuinely approaches again.
    assert controller.observe_intercept_marker(
        now=0.1,
        current_coord=xy_to_coord(50.260, 50.0),
    ) is None
    assert controller.observe_intercept_marker(
        now=0.9,
        current_coord=xy_to_coord(50.149, 50.0),
    ) is None

    snapshot = controller.snapshot(now=1.2)
    assert controller.phase == MiningPhase.INTERCEPT
    assert snapshot["best_intercept_coord_distance"] <= 0.141
    assert snapshot["last_intercept_target_distance"] <= 0.151
    assert snapshot["last_intercept_progress_age"] < 0.4


def test_precision_heading_alignment_does_not_consume_translation_stall_window() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=False,
            intercept_no_progress_seconds=1.0,
        )
    )
    current = xy_to_coord(50.0, 50.0)
    target = xy_to_coord(51.0, 50.0)
    controller.begin(
        MiningCandidate(node_id=1, coord=target, ore_type="Iron"),
        now=0.0,
        current_coord=current,
    )

    controller.note_intercept_heading_alignment(now=0.8)
    assert controller.observe_intercept_marker(now=1.2, current_coord=current) is None

    outcome = controller.observe_intercept_marker(now=1.9, current_coord=current)
    assert outcome is not None
    assert outcome.reason == "ore_intercept_no_progress"


def test_lost_marker_stall_near_node_starts_world_scan_instead_of_waiting() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            intercept_no_progress_seconds=1.0,
            intercept_stall_world_scan_distance_coord=0.60,
            intercept_stall_world_scan_radius_pixels=52.0,
        )
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        marker_intercept_coord=xy_to_coord(50.3, 50.3),
    )
    controller.begin(candidate, now=0.0, current_coord=current)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=(),
        dark_points=(),
        bright_streak=0,
        confirmed_bright=False,
        dark_only=False,
        confirmed_point=None,
        confirmed_track_id=None,
        target_visible=False,
    )

    outcome = controller.observe_intercept_marker(now=1.1, current_coord=current)

    assert outcome is None
    assert controller.phase == MiningPhase.FACE_NODE


def test_close_live_marker_stall_does_not_override_unfinished_access_leg() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            intercept_no_progress_seconds=1.0,
            intercept_stall_world_scan_distance_coord=0.60,
            intercept_stall_world_scan_radius_pixels=52.0,
        )
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    first = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.2, 50.0), xy_to_coord(50.4, 50.0)),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(first,))
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(51.0, 51.0),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.2, 50.0),
    )
    controller.begin(candidate, now=0.0, current_coord=current)

    assert controller.intercept_cursor < len(controller.intercept_coords) - 1
    outcome = controller.observe_intercept_marker(now=1.1, current_coord=current)

    assert outcome is not None
    assert outcome.reason == "ore_intercept_no_progress"
    assert controller.phase == MiningPhase.RESUME


def test_world_scan_timeout_retries_alternate_terrain_access_option() -> None:
    controller = MiningCycleController(
        _mining_config(
            max_world_scan_seconds=1.0,
            max_total_seconds=20.0,
            max_access_option_retries=1,
        )
    )
    first = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.1, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    second = SimpleNamespace(
        rank=1,
        inbound_coords=(xy_to_coord(49.9, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(
        primary_option_rank=0,
        resume_route_index=12,
        options=(first, second),
    )
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        access_plan=plan,
    )
    current = xy_to_coord(50.0, 50.0)
    controller.begin(candidate, now=0.0, current_coord=current)
    controller.mark_intercept_reached(now=0.1)
    controller.mark_node_faced(now=0.1)

    assert controller.observe_world_scan_timeout(now=1.2, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.access_option_rank == 1
    assert controller.current_intercept_coord == second.inbound_coords[0]


def test_final_burst_failure_retries_alternate_terrain_access_option() -> None:
    controller = MiningCycleController(
        _mining_config(max_access_option_retries=1)
    )
    first = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.1, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    second = SimpleNamespace(
        rank=1,
        inbound_coords=(xy_to_coord(49.9, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(
        primary_option_rank=0,
        resume_route_index=12,
        options=(first, second),
    )
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        access_plan=plan,
    )
    current = xy_to_coord(50.0, 50.0)
    controller.begin(candidate, now=0.0, current_coord=current)

    assert controller.observe_final_approach_failure(
        "ore_final_database_burst_missed",
        now=1.0,
        current_coord=current,
    ) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.access_option_rank == 1
    assert controller.current_intercept_coord == second.inbound_coords[0]

    outcome = controller.observe_final_approach_failure(
        "ore_final_database_burst_missed",
        now=2.0,
        current_coord=current,
    )
    assert outcome is not None
    assert outcome.reason == "ore_final_database_burst_missed"


def test_access_waypoint_coordinate_progress_prevents_false_marker_stall() -> None:
    controller = MiningCycleController(
        _mining_config(
            direct_marker_intercept_enabled=True,
            intercept_no_progress_seconds=1.0,
            intercept_progress_coord=0.02,
        )
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((100, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(100, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.2, 50.0),),
        resume_coords=(xy_to_coord(50.0, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    candidate = MiningCandidate(
        node_id=1,
        coord=xy_to_coord(50.3, 50.3),
        ore_type="Iron",
        access_plan=plan,
        marker_intercept_coord=xy_to_coord(50.3, 50.3),
    )

    controller.begin(candidate, now=0.0, current_coord=xy_to_coord(50.0, 50.0))
    outcome = controller.observe_intercept_marker(
        now=0.8,
        current_coord=xy_to_coord(50.05, 50.0),
    )
    assert outcome is None
    outcome = controller.observe_intercept_marker(
        now=1.5,
        current_coord=xy_to_coord(50.10, 50.0),
    )

    assert outcome is None
    assert controller.phase == MiningPhase.INTERCEPT


def test_mining_cycle_verifies_two_clear_frames_only_after_gather_wait():
    controller = MiningCycleController(_mining_config())
    node = SimpleNamespace(node_id=7, coord=5000500000, ore_type="Mithril")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.mark_intercept_reached()
    controller.mark_node_faced(now=0.5)
    controller.mark_clicked((1200, 700), now=1.0)

    assert controller.observe_after_click(
        target_visible=False,
        hover_mining_ready=False,
        now=3.9,
    ) is None
    assert controller.phase == MiningPhase.GATHER_WAIT
    assert controller.observe_after_click(
        target_visible=False,
        hover_mining_ready=False,
        now=4.0,
    ) is None
    outcome = controller.observe_after_click(
        target_visible=False,
        hover_mining_ready=False,
        now=4.2,
    )

    assert outcome is not None
    assert outcome.success
    assert outcome.reason == "tracked_icon_and_hover_cleared"
    assert controller.phase == MiningPhase.RESUME


def test_mining_cycle_restarts_intercept_after_combat_if_ore_remains():
    controller = MiningCycleController(_mining_config())
    node = SimpleNamespace(node_id=7, coord=5000500000, ore_type="Mithril")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.mark_intercept_reached()
    controller.suspend(now=1.0)

    assert controller.phase == MiningPhase.SUSPENDED
    assert controller.resume_after_interrupt(
        target_visible=True,
        confirmed_bright=True,
        now=2.0,
    ) is None
    assert controller.phase == MiningPhase.INTERCEPT


def test_mining_total_timeout_excludes_time_suspended_for_combat():
    controller = MiningCycleController(
        _mining_config(max_total_seconds=5.0, max_intercept_seconds=20.0)
    )
    node = SimpleNamespace(node_id=7, coord=5000500000, ore_type="Mithril")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.suspend(now=2.0)

    assert controller.snapshot(now=102.0)["elapsed_seconds"] == 2.0
    assert controller.resume_after_interrupt(
        target_visible=True,
        confirmed_bright=True,
        now=102.0,
    ) is None
    assert controller.observe_intercept_marker(now=104.9) is None
    outcome = controller.observe_intercept_marker(now=105.1)

    assert outcome is not None
    assert outcome.reason == "ore_attempt_timeout"


def test_centered_direct_marker_keeps_approaching_until_candidate_is_close():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        direct_marker_world_scan_radius_pixels=12,
        direct_marker_world_scan_candidate_distance_coord=0.12,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(108, 100)], [])
    controller.observe_minimap([(108, 100)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(node_id=1, coord=xy_to_coord(50.4, 50.4), ore_type="Iron")
    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(108, 100),
        minimap_shape=(200, 200, 3),
    )
    assert candidate is not None
    controller.begin(candidate, now=0.0, current_coord=current)
    controller.observe_minimap([(108, 100)], [])

    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.refresh_direct_marker_intercept(current, (200, 200, 3)) == node.coord

    close = xy_to_coord(50.34, 50.34)
    controller.observe_minimap([(108, 100)], [])
    assert controller.observe_intercept_marker(now=0.2, current_coord=close) is None
    assert controller.phase == MiningPhase.FACE_NODE
    controller.mark_node_faced(now=0.2)
    assert controller.phase == MiningPhase.WORLD_SCAN


def test_live_tooltip_marker_does_not_scan_from_coarse_projected_coord_proximity():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        direct_marker_world_scan_radius_pixels=12,
        world_search_distance_coord=0.22,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=-1,
        coord=xy_to_coord(50.1, 50.0),
        ore_type="Mithril",
        marker_intercept_coord=xy_to_coord(50.1, 50.0),
        tooltip_confirmed=True,
        source="live_tooltip_marker",
    )
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((140, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(140, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(140, 100),
    )

    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT

    controller.target_marker_point = (105, 100)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((105, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(105, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    assert controller.observe_intercept_marker(now=0.2, current_coord=current) is None
    assert controller.phase == MiningPhase.CENTER_TOOLTIP
    assert controller.observe_center_tooltip(
        now=0.35,
        tooltip_ore_type="Mithril",
    ) is None
    assert controller.phase == MiningPhase.FACE_NODE


def test_tooltip_candidate_at_database_coord_still_waits_for_minimap_center():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        candidate_handoff_distance_coord=0.20,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=7,
        coord=current,
        ore_type="Mithril",
        marker_intercept_coord=xy_to_coord(50.2, 50.0),
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(120, 100),
    )

    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.requires_centered_marker_confirmation


def test_database_anchor_arrival_hands_final_target_to_live_marker_projection():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(50.0, 50.0)
    current = xy_to_coord(49.9, 50.0)
    candidate = MiningCandidate(
        node_id=7,
        coord=anchor,
        ore_type="Mithril",
        marker_intercept_coord=xy_to_coord(50.2, 50.0),
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(120, 100),
    )

    assert controller.current_intercept_coord == anchor
    assert not controller.mark_database_anchor_arrived(now=0.1)
    refreshed = controller.refresh_direct_marker_intercept(
        anchor,
        (200, 200, 3),
    )

    assert controller.database_anchor_arrived
    assert refreshed != anchor
    assert controller.current_intercept_coord == refreshed
    assert not controller.final_intercept_uses_database_anchor


def test_database_anchor_tries_stopped_local_world_validation_before_modest_residual():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        anchor_local_world_validation_enabled=True,
        anchor_local_world_validation_radius_pixels=24,
        anchor_local_world_validation_seconds=3.0,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(62.0, 39.1)
    marker_coord = xy_to_coord(62.18, 39.44)
    candidate = MiningCandidate(
        node_id=145,
        coord=anchor,
        ore_type="Iron",
        marker_intercept_coord=marker_coord,
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_marker_intercept_coord = marker_coord
    controller.begin(
        candidate,
        now=0.0,
        current_coord=xy_to_coord(61.99, 39.1),
        marker_track_id=1,
        marker_point=(120, 100),
    )

    assert controller.mark_database_anchor_arrived(now=0.1)
    assert controller.phase == MiningPhase.WORLD_SCAN
    assert controller.anchor_local_world_validation_active
    assert controller.snapshot(now=0.2)["anchor_local_world_validation_active"]


def test_modest_residual_marker_loss_finishes_database_anchor_before_world_validation():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        anchor_local_world_validation_enabled=True,
        anchor_local_world_validation_radius_pixels=24,
        anchor_local_world_validation_seconds=3.0,
        intercept_marker_missing_frames=2,
        lost_marker_world_scan_distance_coord=0.10,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(62.0, 39.1)
    marker_coord = xy_to_coord(62.18, 39.44)
    candidate = MiningCandidate(
        node_id=138,
        coord=anchor,
        ore_type="Small Thorium",
        marker_intercept_coord=marker_coord,
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    current = xy_to_coord(61.88, 39.1)
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(120, 100),
    )

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.2, current_coord=current) is None
    assert controller.phase == MiningPhase.INTERCEPT

    assert controller.mark_database_anchor_arrived(now=0.3)
    assert controller.phase == MiningPhase.WORLD_SCAN
    assert controller.anchor_local_world_validation_active


def test_anchor_local_world_validation_timeout_resumes_original_frozen_residual():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        anchor_local_world_validation_enabled=True,
        anchor_local_world_validation_radius_pixels=24,
        anchor_local_world_validation_seconds=3.0,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(62.0, 39.1)
    marker_coord = xy_to_coord(62.18, 39.44)
    candidate = MiningCandidate(
        node_id=145,
        coord=anchor,
        ore_type="Iron",
        marker_intercept_coord=marker_coord,
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((120, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(120, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.last_marker_intercept_coord = marker_coord
    controller.begin(
        candidate,
        now=0.0,
        current_coord=anchor,
        marker_track_id=1,
        marker_point=(120, 100),
    )
    assert controller.mark_database_anchor_arrived(now=0.1)

    assert controller.observe_world_scan_timeout(now=3.2, current_coord=anchor) is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert not controller.anchor_local_world_validation_active
    assert controller.refresh_direct_marker_intercept(anchor, (200, 200, 3)) == marker_coord


def test_player_icon_occlusion_requires_center_tooltip_and_failure_cools_live_anchor():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        intercept_marker_missing_frames=2,
        failure_spatial_cooldown_radius_coord=0.65,
        failure_cooldown_seconds=300.0,
        center_tooltip_probe={"settle_seconds": 0.1, "timeout_seconds": 0.5},
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    marker_coord = xy_to_coord(50.4, 50.0)
    candidate = MiningCandidate(
        node_id=7,
        coord=current,
        ore_type="Mithril",
        marker_intercept_coord=marker_coord,
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((112, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(112, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(112, 100),
    )
    assert not controller.mark_database_anchor_arrived(now=0.05)
    assert controller.database_anchor_arrived

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.2, current_coord=current) is None
    assert controller.phase == MiningPhase.CENTER_TOOLTIP

    outcome = controller.observe_center_tooltip(now=0.8, tooltip_ore_type=None)
    assert outcome is not None
    assert outcome.reason == "ore_center_tooltip_unconfirmed"
    assert controller.is_spatially_cooled(current, "Mithril", now=1.0)
    assert controller.is_spatially_cooled(marker_coord, "Mithril", now=1.0)
    assert not controller.is_spatially_cooled(marker_coord, "Gold", now=1.0)


def test_database_center_tooltip_timeout_falls_back_to_bounded_world_scan_when_enabled():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        intercept_marker_missing_frames=2,
        center_tooltip_world_scan_fallback_enabled=True,
        center_tooltip_probe={"settle_seconds": 0.1, "timeout_seconds": 0.5},
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=136,
        coord=anchor,
        ore_type="Small Thorium",
        marker_intercept_coord=anchor,
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((100, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(100, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=anchor,
        marker_track_id=1,
        marker_point=(100, 100),
    )
    assert controller.mark_database_anchor_arrived(now=0.05)
    assert controller.phase == MiningPhase.CENTER_TOOLTIP

    assert controller.observe_center_tooltip(now=0.6, tooltip_ore_type=None) is None
    assert controller.phase == MiningPhase.WORLD_SCAN


def test_database_marker_loss_far_from_center_preserves_frozen_marker_stage():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        intercept_marker_missing_frames=2,
        lost_marker_world_scan_distance_coord=0.10,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    anchor = xy_to_coord(57.26, 38.59)
    current = xy_to_coord(57.23, 38.62)
    candidate = MiningCandidate(
        node_id=7,
        coord=anchor,
        ore_type="Iron",
        marker_intercept_coord=xy_to_coord(57.40, 38.70),
        tooltip_confirmed=True,
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((140, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(140, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=xy_to_coord(57.41, 38.66),
        marker_track_id=1,
        marker_point=(140, 100),
    )
    controller.refresh_direct_marker_intercept(
        xy_to_coord(57.41, 38.66),
        (200, 200, 3),
    )

    controller.observe_minimap([], [])
    controller.observe_intercept_marker(now=0.1, current_coord=current)
    controller.observe_minimap([], [])
    outcome = controller.observe_intercept_marker(now=0.2, current_coord=current)

    assert outcome is None
    assert controller.phase == MiningPhase.INTERCEPT
    assert not controller.database_anchor_arrived
    assert not controller.mark_database_anchor_arrived(now=0.3)

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.4, current_coord=anchor) is None
    assert controller.phase == MiningPhase.INTERCEPT
    refreshed = controller.refresh_direct_marker_intercept(
        anchor,
        (200, 200, 3),
    )
    assert refreshed == controller.live_marker_centering_coord
    assert refreshed != anchor


def test_tooltip_marker_loss_near_coarse_coord_does_not_bypass_center_gate():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        centered_marker_radius_pixels=6,
        centered_marker_occlusion_radius_pixels=18,
        intercept_marker_missing_frames=2,
        lost_marker_world_scan_distance_coord=0.20,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    current = xy_to_coord(50.0, 50.0)
    candidate = MiningCandidate(
        node_id=-1,
        coord=xy_to_coord(50.05, 50.0),
        ore_type="Small Thorium",
        marker_intercept_coord=xy_to_coord(50.1, 50.0),
        tooltip_confirmed=True,
        source="live_tooltip_marker",
    )
    controller.last_minimap_shape = (200, 200, 3)
    controller.last_presence = controller.last_presence.__class__(
        bright_points=((130, 100),),
        dark_points=(),
        bright_streak=2,
        confirmed_bright=True,
        dark_only=False,
        confirmed_point=(130, 100),
        confirmed_track_id=1,
        target_visible=True,
    )
    controller.begin(
        candidate,
        now=0.0,
        current_coord=current,
        marker_track_id=1,
        marker_point=(130, 100),
    )

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    controller.observe_minimap([], [])
    outcome = controller.observe_intercept_marker(now=0.2, current_coord=current)

    assert outcome is not None
    assert outcome.reason == "ore_taken_by_other_player"
    assert controller.phase == MiningPhase.RESUME


def test_mining_cycle_abandons_intercept_after_tracked_marker_disappears():
    controller = MiningCycleController(
        _mining_config(
            intercept_marker_missing_frames=3,
            continue_coordinate_approach_after_marker_loss=False,
        )
    )
    controller.observe_minimap([(40, 50)], [])
    controller.observe_minimap([(41, 50)], [])
    node = SimpleNamespace(node_id=7, coord=5000500000, ore_type="Mithril")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=1.0) is None
    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=2.0) is None
    controller.observe_minimap([], [])
    outcome = controller.observe_intercept_marker(now=3.0)

    assert outcome is not None
    assert not outcome.success
    assert outcome.reason == "ore_lost_during_intercept"
    assert controller.phase == MiningPhase.RESUME


def test_mining_cycle_keeps_bounded_coordinate_approach_after_marker_loss():
    controller = MiningCycleController(
        _mining_config(
            intercept_marker_missing_frames=2,
            continue_coordinate_approach_after_marker_loss=True,
        )
    )
    controller.observe_minimap([(40, 50)], [])
    controller.observe_minimap([(41, 50)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(
        node_id=7,
        coord=xy_to_coord(50.6, 50.0),
        ore_type="Mithril",
    )
    candidate = controller.choose_candidate(current, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0, current_coord=current)

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.2, current_coord=current) is None

    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.current_intercept_coord == node.coord


def test_tooltip_confirmed_marker_disappearance_is_treated_as_taken_and_cooled_down():
    config = _mining_config(
        require_minimap_tooltip_confirmation=True,
        intercept_marker_missing_frames=3,
        continue_coordinate_approach_after_marker_loss=True,
        failure_spatial_cooldown_radius_coord=0.35,
        failure_cooldown_seconds=20.0,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.42,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(120, 100)], [])
    presence = controller.observe_minimap([(120, 100)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(
        node_id=7,
        coord=xy_to_coord(50.6, 50.0),
        ore_type="Mithril",
    )
    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=presence.confirmed_point,
        minimap_shape=(200, 200, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Mithril",
    )
    assert candidate is not None
    assert candidate.tooltip_confirmed
    controller.begin(candidate, now=0.0, current_coord=current)

    for now in (1.0, 2.0):
        controller.observe_minimap([], [])
        assert controller.observe_intercept_marker(now=now, current_coord=current) is None
    controller.observe_minimap([], [])
    outcome = controller.observe_intercept_marker(now=3.0, current_coord=current)

    assert outcome is not None
    assert outcome.reason == "ore_taken_by_other_player"
    assert controller.phase == MiningPhase.RESUME
    controller.consume_outcome()
    assert controller.choose_candidate(
        current,
        [node],
        now=4.0,
        marker_point=(120, 100),
        minimap_shape=(200, 200, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Mithril",
    ) is None


def test_tooltip_confirmed_marker_transient_loss_does_not_cancel_intercept():
    config = _mining_config(
        require_minimap_tooltip_confirmation=True,
        intercept_marker_missing_frames=3,
        continue_coordinate_approach_after_marker_loss=True,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.42,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(120, 100)], [])
    presence = controller.observe_minimap([(120, 100)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(
        node_id=7,
        coord=xy_to_coord(50.6, 50.0),
        ore_type="Mithril",
    )
    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=presence.confirmed_point,
        minimap_shape=(200, 200, 3),
        tooltip_confirmed=True,
        confirmed_ore_type="Mithril",
    )
    assert candidate is not None
    controller.begin(candidate, now=0.0, current_coord=current)

    for now in (1.0, 2.0):
        controller.observe_minimap([], [])
        assert controller.observe_intercept_marker(now=now, current_coord=current) is None
    controller.observe_minimap([(121, 100)], [])
    assert controller.observe_intercept_marker(now=3.0, current_coord=current) is None

    assert controller.phase == MiningPhase.INTERCEPT
    assert controller.intercept_marker_missing_frames == 0


def test_close_direct_marker_loss_enters_face_node_instead_of_abandoning_ore():
    config = _mining_config(
        direct_marker_intercept_enabled=True,
        intercept_marker_missing_frames=2,
        lost_marker_world_scan_distance_coord=0.45,
    )
    config["recognition"] = {
        "sensor_circle_center_fraction": {"x": 0.50, "y": 0.50},
        "sensor_circle_radius_fraction": 0.37,
    }
    controller = MiningCycleController(config)
    controller.observe_minimap([(120, 100)], [])
    controller.observe_minimap([(120, 100)], [])
    current = xy_to_coord(50.0, 50.0)
    node = SimpleNamespace(node_id=1, coord=xy_to_coord(50.3, 50.1), ore_type="Iron")
    candidate = controller.choose_candidate(
        current,
        [node],
        now=0.0,
        marker_point=(120, 100),
        minimap_shape=(200, 200, 3),
    )
    assert candidate is not None
    controller.begin(candidate, now=0.0, current_coord=current)

    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.1, current_coord=current) is None
    controller.observe_minimap([], [])
    assert controller.observe_intercept_marker(now=0.2, current_coord=current) is None

    assert controller.phase == MiningPhase.FACE_NODE


def test_bright_ore_tracker_keeps_target_identity_when_another_marker_remains():
    tracker = BrightOreTracker(max_jump_pixels=12, max_missing_frames=1)

    first = tracker.update(((40, 50), (150, 100)))
    second = tracker.update(((44, 52), (151, 101)))
    target_id = min(second, key=lambda track: abs(track.point[0] - 44)).track_id
    third = tracker.update(((152, 101),))

    target = next(track for track in third if track.track_id == target_id)
    assert target.missing_frames == 1
    assert target.point == (44, 52)


def test_mining_cycle_does_not_verify_when_hover_still_confirms_ore():
    controller = MiningCycleController(_mining_config(verify_clear_frames=1))
    controller.observe_minimap([(40, 50)], [])
    controller.observe_minimap([(42, 51)], [])
    node = SimpleNamespace(node_id=7, coord=5000500000, ore_type="Mithril")
    candidate = controller.choose_candidate(5000500000, [node], now=0.0)
    assert candidate is not None
    controller.begin(candidate, now=0.0)
    controller.mark_intercept_reached()
    controller.mark_node_faced(now=0.5)
    controller.mark_clicked((1200, 700), now=1.0)
    controller.observe_minimap([], [])

    assert controller.observe_after_click(
        target_visible=False,
        hover_mining_ready=True,
        now=4.0,
    ) is None
    outcome = controller.observe_after_click(
        target_visible=False,
        hover_mining_ready=True,
        now=7.1,
    )
    assert outcome is not None
    assert not outcome.success
    assert outcome.reason == "mining_hover_evidence_persisted"


def test_mining_cycle_uses_precomputed_access_leg_and_preserves_resume_path():
    controller = MiningCycleController(_mining_config())
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(5000500000, 5050500000, 5100500000),
        resume_coords=(5100500000, 5150500000),
    )
    plan = SimpleNamespace(
        primary_option_rank=0,
        resume_route_index=42,
        options=(option,),
    )
    node = SimpleNamespace(
        node_id=7,
        coord=5100500000,
        ore_type="Mithril",
        access_plan=plan,
    )
    candidate = controller.choose_candidate(5050500000, [node], now=0.0)
    assert candidate is not None

    controller.begin(candidate, now=0.0, current_coord=5050500000)

    assert controller.current_intercept_coord == 5050500000
    assert not controller.mark_intercept_target_reached()
    assert controller.current_intercept_coord == 5100500000
    assert controller.mark_intercept_target_reached()
    controller.mark_node_faced(now=0.5)
    controller.mark_clicked((1200, 700), now=1.0)
    controller.fail("test_failure", now=2.0)
    outcome = controller.consume_outcome()
    assert outcome is not None
    assert outcome.resume_coords == (5100500000, 5150500000)
    assert outcome.resume_route_index == 42


def test_access_leg_only_skips_waypoints_when_already_on_the_leg():
    coords = (5000500000, 5050500000, 5100500000)

    assert _start_access_leg_near_current(coords, 5050500000) == coords[1:]
    assert _start_access_leg_near_current(coords, 5057000000) == coords


def test_route_live_mining_can_require_topographic_access_plan():
    controller = MiningCycleController(
        _mining_config(require_access_plan_for_route_live=True)
    )
    nodes = [
        SimpleNamespace(
            node_id=1,
            coord=5000500000,
            ore_type="Iron",
            access_plan=None,
        ),
        SimpleNamespace(
            node_id=2,
            coord=5010500000,
            ore_type="Iron",
            access_plan=SimpleNamespace(options=(SimpleNamespace(rank=0),)),
        ),
    ]

    candidate = controller.choose_candidate(5000500000, nodes, now=0.0)

    assert candidate is not None
    assert candidate.node_id == 2


def test_route_live_mining_waits_until_safe_access_attachment_is_near():
    controller = MiningCycleController(
        _mining_config(
            require_access_plan_for_route_live=True,
            access_attachment_max_distance_coord=0.18,
        )
    )
    option = SimpleNamespace(
        rank=0,
        inbound_coords=(xy_to_coord(50.5, 50.0), xy_to_coord(50.7, 50.1)),
        resume_coords=(xy_to_coord(50.5, 50.0),),
    )
    plan = SimpleNamespace(primary_option_rank=0, resume_route_index=12, options=(option,))
    node = SimpleNamespace(
        node_id=1,
        coord=xy_to_coord(50.7, 50.1),
        ore_type="Iron",
        access_plan=plan,
    )

    assert controller.choose_candidate(
        xy_to_coord(50.0, 50.0),
        [node],
        now=0.0,
    ) is None
    far_snapshot = controller.snapshot(now=0.0)
    assert far_snapshot["candidate_rejection_reason"] == "access_attachment_too_far"
    assert far_snapshot["access_attachment_distance"] > 0.18

    candidate = controller.choose_candidate(
        xy_to_coord(50.38, 50.0),
        [node],
        now=0.1,
    )

    assert candidate is not None
    assert controller.snapshot(now=0.1)["access_attachment_distance"] <= 0.18


class _FakeCapture:
    def __init__(self):
        self.points = []

    def capture_client_region(self):
        raise AssertionError("nonblocking scanner must not capture internally")

    def client_to_screen_point(self, x, y):
        return x + 10, y + 20


class _FakeMouse:
    def __init__(self):
        self.moves = []

    def move_to(self, x, y):
        self.moves.append((x, y))

    def right_click(self, duration=0.05):
        raise AssertionError("scanner must not click")


def test_hover_scanner_requires_tooltip_and_changed_cursor_without_blocking():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.rectangle(frame, (2380, 1010), (2559, 1139), (5, 5, 5), -1)
    cv2.rectangle(frame, (2410, 1040), (2490, 1052), (0, 210, 255), -1)
    cv2.rectangle(frame, (2410, 1070), (2505, 1082), (0, 220, 0), -1)
    signatures = iter([100, 200])
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 1000, "y": 600, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 4,
            "hover_delay": 0.08,
            "tooltip_region": {"x": 2380, "y": 1010, "width": 180, "height": 130},
            "cursor_pixel_fallback_enabled": False,
            "require_mining_tooltip": True,
            "require_cursor_change": True,
        },
    }
    mouse = _FakeMouse()
    scanner = MiningHoverScanner(
        _FakeCapture(),
        mouse,
        config,
        cursor_signature_reader=lambda: next(signatures),
    )

    start = scanner.start(frame.shape, now=0.0)
    probe = scanner.step(frame, now=0.1)
    found = scanner.step(frame, now=0.2)

    assert start.phase == HoverScanPhase.BASELINE_SETTLE
    assert probe.phase == HoverScanPhase.PROBE_SETTLE
    assert found.phase == HoverScanPhase.FOUND
    assert found.reason == "tooltip_and_mining_cursor_confirmed"
    assert len(mouse.moves) == 2


def test_hover_scanner_prioritizes_model_candidate_without_granting_click_authority():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 1600, "reference_height": 900},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 700, "y": 300, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 8,
            "hover_delay": 0.08,
            "front_probe_enabled": False,
            "salient_probe_enabled": False,
            "ore_world_detector": {"probe_offset_fractions": [0.0]},
        },
    }
    detector = SimpleNamespace(
        detect=lambda _frame: OreWorldDetectorResult(
            (OreWorldDetection(1200, 600, 1400, 800, 0.88),),
            12.0,
            "ok",
        )
    )
    mouse = _FakeMouse()
    scanner = MiningHoverScanner(
        _FakeCapture(),
        mouse,
        config,
        cursor_signature_reader=lambda: 100,
        ore_world_detector=detector,
    )

    scanner.start(frame.shape, now=0.0)
    probe = scanner.step(frame, now=0.1)

    assert probe.phase == HoverScanPhase.PROBE_SETTLE
    assert probe.point == (1300, 700)
    assert mouse.moves[-1] == (1310, 720)
    assert scanner.snapshot()["ore_world_detector"]["detections"][0]["confidence"] == 0.88


def test_hover_scanner_refreshes_model_and_prioritizes_later_sparkle_detection():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 1600, "reference_height": 900},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 700, "y": 300, "width": 128, "height": 64},
            "scan_step": 32,
            "scan_max_points": 8,
            "hover_delay": 0.08,
            "front_probe_enabled": False,
            "salient_probe_enabled": False,
            "require_mining_tooltip": True,
            "require_cursor_change": True,
            "ore_world_detector": {
                "probe_offset_fractions": [0.0],
                "refresh_interval_seconds": 0.75,
                "refresh_point_dedupe_pixels": 20,
            },
        },
    }
    results = iter(
        [
            OreWorldDetectorResult(reason="no_detection"),
            OreWorldDetectorResult(
                (OreWorldDetection(1200, 600, 1400, 800, 0.55),),
                12.0,
                "ok",
            ),
        ]
    )
    detector = SimpleNamespace(detect=lambda _frame: next(results))
    mouse = _FakeMouse()
    scanner = MiningHoverScanner(
        _FakeCapture(),
        mouse,
        config,
        cursor_signature_reader=lambda: 100,
        ore_world_detector=detector,
    )

    scanner.start(frame.shape, now=0.0)
    first_probe = scanner.step(frame, now=0.1)
    ordinary_probe = scanner.step(frame, now=0.2)
    refreshed_probe = scanner.step(frame, now=0.9)

    assert first_probe.point != (1300, 700)
    assert ordinary_probe.point != (1300, 700)
    assert refreshed_probe.point == (1300, 700)
    detector_snapshot = scanner.snapshot()["ore_world_detector"]
    assert detector_snapshot["runs"] == 2
    assert detector_snapshot["detection_runs"] == 1
    assert detector_snapshot["probe_points"] == [[1300, 700]]


def test_hover_scanner_accepts_tooltip_without_cursor_change_when_allowed():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.rectangle(frame, (2380, 1010), (2559, 1139), (5, 5, 5), -1)
    cv2.rectangle(frame, (2410, 1040), (2490, 1052), (0, 210, 255), -1)
    cv2.rectangle(frame, (2410, 1070), (2505, 1082), (0, 220, 0), -1)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 1000, "y": 600, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 4,
            "hover_delay": 0.08,
            "tooltip_region": {"x": 2380, "y": 1010, "width": 180, "height": 130},
            "cursor_pixel_fallback_enabled": False,
            "require_mining_tooltip": True,
            "require_cursor_change": True,
            "tooltip_only_click_enabled": True,
        },
    }
    scanner = MiningHoverScanner(
        _FakeCapture(),
        _FakeMouse(),
        config,
        cursor_signature_reader=lambda: 100,
    )

    scanner.start(frame.shape, now=0.0)
    scanner.step(frame, now=0.1)
    found = scanner.step(frame, now=0.2)

    assert found.phase == HoverScanPhase.FOUND
    assert found.reason == "mining_tooltip_confirmed"
    assert scanner.last_probe_evidence["tooltip_only_ready"]


def test_mining_interactor_prewarm_does_not_move_or_click_mouse():
    from vision_bot.mining import MiningInteractor

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = OreWorldDetectorResult(reason="no_detection", elapsed_ms=12.0)
    detector = SimpleNamespace(
        warm_up=lambda received: result if received is frame else None,
        detect=lambda _frame: result,
    )
    mouse = _FakeMouse()
    interactor = MiningInteractor(
        _FakeCapture(),
        mouse,
        {"cursor_classifier": {"enabled": False}},
        ore_world_detector=detector,
    )

    assert interactor.warm_up_ore_world_detector(frame) is result
    assert mouse.moves == []


def test_hover_scanner_requires_mine_class_when_templates_are_calibrated():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.rectangle(frame, (2380, 1010), (2559, 1139), (5, 5, 5), -1)
    cv2.rectangle(frame, (2410, 1040), (2490, 1052), (0, 210, 255), -1)
    cv2.rectangle(frame, (2410, 1070), (2505, 1082), (0, 220, 0), -1)
    classifier = CursorTemplateClassifier({"mine": (123,), "loot": (456,)})
    snapshots = iter(
        [
            CursorSnapshot(1, None, 456),
            CursorSnapshot(2, None, 123),
        ]
    )
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "mining": {
            "scan_region": {"x": 1000, "y": 600, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 4,
            "hover_delay": 0.08,
            "tooltip_region": {"x": 2380, "y": 1010, "width": 180, "height": 130},
            "cursor_pixel_fallback_enabled": False,
            "require_mining_tooltip": True,
            "require_cursor_change": False,
            "require_classified_cursor_when_calibrated": True,
        },
    }
    scanner = MiningHoverScanner(
        _FakeCapture(),
        _FakeMouse(),
        config,
        cursor_signature_reader=lambda: 1,
        cursor_snapshot_reader=lambda: next(snapshots),
        cursor_classifier=classifier,
    )

    scanner.start(frame.shape, now=0.0)
    scanner.step(frame, now=0.1)
    rejected = scanner.step(frame, now=0.2)
    found = scanner.step(frame, now=0.3)

    assert rejected.phase == HoverScanPhase.PROBE_SETTLE
    assert found.phase == HoverScanPhase.FOUND
    assert found.reason == "tooltip_and_mine_cursor_classified"
    assert found.cursor_label == "mine"


def test_hover_scanner_settles_briefly_on_cursor_only_mine_candidate(monkeypatch):
    import vision_bot.mining as mining_module

    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 1000, "y": 600, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 2,
            "hover_delay": 0.08,
            "cursor_pixel_fallback_enabled": True,
            "cursor_color_fallback_enabled": False,
            "cursor_candidate_extra_settle_enabled": True,
            "cursor_candidate_extra_settle_frames": 1,
            "cursor_candidate_extra_settle_seconds": 0.12,
            "require_mining_tooltip": True,
            "require_cursor_change": True,
        },
    }
    monkeypatch.setattr(mining_module, "detect_pickaxe_cursor", lambda *_args, **_kwargs: True)
    scanner = MiningHoverScanner(
        _FakeCapture(),
        _FakeMouse(),
        config,
        cursor_signature_reader=lambda: 100,
    )

    scanner.start(frame.shape, now=0.0)
    scanner.step(frame, now=0.1)
    settle = scanner.step(frame, now=0.2)
    moved_on = scanner.step(frame, now=0.4)

    assert settle.phase == HoverScanPhase.PROBE_SETTLE
    assert settle.reason == "cursor_candidate_settle"
    assert moved_on.reason == "hover_probe"


def test_hover_scanner_accepts_persistent_cursor_only_mine_candidate(monkeypatch):
    import vision_bot.mining as mining_module

    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    config = {
        "screen": {"reference_width": 2560, "reference_height": 1440},
        "cursor_classifier": {"enabled": False},
        "mining": {
            "scan_region": {"x": 1000, "y": 600, "width": 64, "height": 64},
            "scan_step": 32,
            "scan_max_points": 2,
            "hover_delay": 0.08,
            "cursor_pixel_fallback_enabled": True,
            "cursor_color_fallback_enabled": False,
            "cursor_only_click_enabled": True,
            "cursor_only_confirm_frames": 2,
            "cursor_candidate_extra_settle_enabled": True,
            "cursor_candidate_extra_settle_frames": 2,
            "cursor_candidate_extra_settle_seconds": 0.10,
            "require_mining_tooltip": True,
            "require_cursor_change": True,
        },
    }
    monkeypatch.setattr(mining_module, "detect_pickaxe_cursor", lambda *_args, **_kwargs: True)
    scanner = MiningHoverScanner(
        _FakeCapture(),
        _FakeMouse(),
        config,
        cursor_signature_reader=lambda: 100,
    )

    scanner.start(frame.shape, now=0.0)
    scanner.step(frame, now=0.1)
    settle = scanner.step(frame, now=0.2)
    found = scanner.step(frame, now=0.31)

    assert settle.reason == "cursor_candidate_settle"
    assert found.phase == HoverScanPhase.FOUND
    assert found.reason == "mine_cursor_confirmed_without_tooltip"
    assert scanner.last_probe_evidence["cursor_only_confirm_count"] == 2


def test_pickaxe_cursor_template_fallback_accepts_mine_asset_and_rejects_unable():
    root = Path(__file__).resolve().parents[1] / "data" / "cursor_templates" / "client_assets"
    if not (root / "Mine.png").exists() or not (root / "UnableMine.png").exists():
        pytest.skip("client-derived cursor fixtures are not distributed")
    mine = cv2.imread(str(root / "Mine.png"), cv2.IMREAD_UNCHANGED)
    unable = cv2.imread(str(root / "UnableMine.png"), cv2.IMREAD_UNCHANGED)
    assert mine is not None
    assert unable is not None
    config = {
        "cursor_classifier": {"asset_template_dir": str(root)},
        "mining": {
            "cursor_detection_radius": 48,
            "cursor_template_fallback_enabled": True,
            "cursor_template_min_score": 0.62,
            "cursor_template_margin": 0.04,
            "cursor_color_fallback_enabled": False,
        },
    }

    def frame_with(image):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        frame[60 : 60 + h, 60 : 60 + w] = image[:, :, :3]
        return frame

    assert detect_pickaxe_cursor(frame_with(mine), (70, 70), config)
    assert not detect_pickaxe_cursor(frame_with(unable), (70, 70), config)
