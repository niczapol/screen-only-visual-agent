from pathlib import Path

import cv2
import numpy as np

import pytest

from vision_bot.combat import (
    CombatFallbackState,
    CombatHealState,
    CombatState,
    CombatThreatEstimator,
    PeriodicCombatKeyState,
    detect_combat_facing_error_text,
    detect_combat_marker,
    detect_combat_state,
    detect_attacker_count_marker,
    detect_facing_error_marker,
    detect_out_of_range_marker,
    detect_outgoing_hit_marker,
    detect_outgoing_damage_numbers,
    detect_player_health_fraction,
    normalize_attack_keys,
    normalize_periodic_combat_keys,
)


def test_visible_attacker_count_marker_decodes_one_two_and_three_plus():
    config = _combat_test_config()
    config["safety"]["combat"].update(
        {
            "attacker_count_marker_enabled": True,
            "attacker_count_marker_region": {"x": 0, "y": 0, "width": 100, "height": 50},
            "attacker_count_marker_min_pixels": 100,
        }
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 10:30] = (0, 0, 255)
    assert detect_attacker_count_marker(frame, config) == 1

    frame[:] = 0
    orange = cv2.cvtColor(np.uint8([[[16, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[10:30, 10:30] = orange
    assert detect_attacker_count_marker(frame, config) == 2

    frame[:] = 0
    frame[10:30, 10:30] = (255, 255, 255)
    assert detect_attacker_count_marker(frame, config) == 3


def test_combat_threat_estimator_combines_attacker_count_and_hp_loss_rate():
    estimator = CombatThreatEstimator(window_seconds=5.0)
    normal = estimator.observe(
        CombatState(active=True, attacker_count=1, player_health_fraction=0.90),
        now=10.0,
        engaged=True,
    )
    assert normal.level == "normal"

    elevated = estimator.observe(
        CombatState(active=True, attacker_count=2, player_health_fraction=0.82),
        now=12.0,
        engaged=True,
    )
    assert elevated.level == "elevated"
    assert elevated.hp_loss_per_second > 0.0

    critical = estimator.observe(
        CombatState(active=True, attacker_count=3, player_health_fraction=0.18),
        now=13.0,
        engaged=True,
    )
    assert critical.level == "critical"
    assert critical.time_to_death_seconds is not None

    cleared = estimator.observe(CombatState(), now=20.0, engaged=False)
    assert cleared.level == "none"
    assert not estimator.samples


def test_fixed_addon_marker_is_authoritative_combat_entry():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (619, 4), (660, 16), (255, 0, 255), -1)
    config = _combat_test_config()
    config["safety"]["combat"].update(
        {
            "combat_marker_enabled": True,
            "combat_marker_region": {"x": 600, "y": 0, "width": 80, "height": 30},
            "combat_marker_min_pixels": 400,
        }
    )

    assert detect_combat_marker(frame, config)
    combat = detect_combat_state(frame, config)
    assert combat.active
    assert combat.combat_marker_visible
    assert combat.source == "addon_combat_marker"


def test_combat_marker_roi_ignores_magenta_outside_fixed_region():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 150), (255, 0, 255), -1)
    config = _combat_test_config()
    config["safety"]["combat"].update(
        {
            "combat_marker_enabled": True,
            "combat_marker_region": {"x": 600, "y": 0, "width": 80, "height": 30},
            "combat_marker_min_pixels": 400,
        }
    )

    assert not detect_combat_marker(frame, config)
    assert not detect_combat_state(frame, config).active


def test_fixed_outcome_markers_report_hit_facing_and_range_independently():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 4), (541, 16), (0, 255, 0), -1)
    cv2.rectangle(frame, (600, 4), (641, 16), (255, 255, 0), -1)
    cv2.rectangle(frame, (700, 4), (741, 16), (0, 115, 255), -1)
    config = _combat_test_config()
    config["safety"]["combat"].update(
        {
            "outcome_marker_enabled": True,
            "outcome_marker_min_pixels": 400,
            "outgoing_hit_marker_region": {"x": 490, "y": 0, "width": 62, "height": 30},
            "facing_error_marker_region": {"x": 590, "y": 0, "width": 62, "height": 30},
            "out_of_range_marker_region": {"x": 690, "y": 0, "width": 62, "height": 30},
            "legacy_outgoing_damage_confirmation_enabled": False,
            "legacy_facing_error_confirmation_enabled": False,
        }
    )

    assert detect_outgoing_hit_marker(frame, config)
    assert detect_facing_error_marker(frame, config)
    assert detect_out_of_range_marker(frame, config)
    combat = detect_combat_state(frame, config)
    assert combat.outgoing_hit_marker_visible
    assert combat.facing_error_marker_visible
    assert combat.out_of_range_marker_visible
    assert combat.outgoing_damage_visible
    assert combat.facing_error_visible


def test_detect_combat_state_keeps_loose_hostile_target_frame_as_inactive_hint():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (155, 34), (330, 46), (0, 0, 255), -1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert not combat.target_present
    assert combat.hostile_target_hint
    assert not combat.nameplate_visible
    assert combat.face_turn_key is None
    assert combat.bbox is not None


def test_detect_combat_state_ignores_live_ratchet_player_ui_false_positive_if_available():
    frame_path = Path("data/live_route_probe_20260729_ratchet_after_coord_spirit_v1/frames/0001.png")
    if not frame_path.exists():
        pytest.skip("live Ratchet false-combat frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    combat = detect_combat_state(frame, _live_2560_combat_config())

    assert not combat.active
    assert not combat.target_present
    assert combat.hostile_target_hint
    assert not combat.outgoing_damage_visible
    assert combat.player_health_fraction is not None
    assert combat.player_health_fraction > 0.85


def test_detect_combat_state_keeps_hostile_nameplate_visible_but_inactive():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 0, 255), -1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert not combat.target_present
    assert combat.nameplate_visible
    assert combat.face_turn_key == "A"


def test_detect_combat_state_keeps_confirmed_target_frame_with_nameplate_inactive_until_aggro():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (760, 190), (900, 202), (0, 0, 255), -1)
    cv2.rectangle(frame, (410, 47), (595, 65), (0, 0, 255), -1)
    cv2.rectangle(frame, (410, 75), (550, 87), (0, 200, 0), -1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert combat.target_present
    assert combat.nameplate_visible
    assert combat.face_turn_key == "D"


def test_detect_combat_state_keeps_confirmed_target_frame_inactive_by_default():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (410, 47), (595, 65), (0, 0, 255), -1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert combat.target_present


def test_detect_combat_state_accepts_partial_red_target_health_bar_below_green_fill_threshold():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (410, 47), (595, 65), (0, 0, 255), -1)
    for x in range(412, 595, 3):
        cv2.line(frame, (x, 49), (x, 63), (0, 0, 0), 1)
    config = _combat_test_config()
    config["safety"]["combat"]["target_health_min_fill_ratio"] = 0.75
    config["safety"]["combat"]["target_red_health_min_fill_ratio"] = 0.60

    combat = detect_combat_state(frame, config)

    assert not combat.active
    assert combat.target_present


def test_detect_combat_state_reports_real_multi_enemy_red_target_frame_as_target_hint_when_nameplate_drops():
    frame_path = Path("data/live_route_probe_after_modal_refine/frames/0036.png")
    if not frame_path.exists():
        return
    frame = cv2.imread(str(frame_path))
    config = _live_2560_combat_config()
    config["safety"]["combat"]["target_red_health_min_fill_ratio"] = 0.60

    combat = detect_combat_state(frame, config)

    assert not combat.active
    assert combat.target_present


def test_detect_combat_state_reports_outgoing_damage_without_starting_combat():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (760, 190), (900, 202), (0, 0, 255), -1)
    cv2.rectangle(frame, (410, 47), (595, 65), (0, 0, 255), -1)
    cv2.rectangle(frame, (410, 75), (550, 87), (0, 200, 0), -1)
    cv2.putText(frame, "237", (790, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 255), 4)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert combat.outgoing_damage_visible
    assert combat.outgoing_damage_bbox is not None


def test_real_distant_nameplate_and_false_damage_do_not_start_combat_when_available():
    frame_path = Path(
        "data/live_route_probe_long_three_targets_20260731_171004/frames/0007.png"
    )
    if not frame_path.exists():
        pytest.skip("live distant-nameplate regression frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    combat = detect_combat_state(frame, _live_2560_combat_config())

    assert not combat.active
    assert not combat.target_present
    assert combat.nameplate_visible
    assert combat.outgoing_damage_visible

    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=1.0,
        clear_frames=3,
        clear_seconds=3.0,
    )
    assert not fallback.observe(combat, now=10.0)
    assert not fallback.engaged


def test_detect_combat_state_can_engage_on_nameplate_when_explicitly_enabled():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (760, 190), (900, 202), (0, 0, 255), -1)
    config = _combat_test_config()
    config["safety"]["combat"]["engage_on_nameplate"] = True

    combat = detect_combat_state(frame, config)

    assert combat.active
    assert not combat.target_present
    assert combat.nameplate_visible
    assert combat.face_turn_key == "D"


def test_detect_combat_state_ignores_sparse_name_overlay():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 204), (0, 0, 255), 2)
    cv2.line(frame, (490, 197), (605, 197), (120, 0, 255), 1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert not combat.target_present
    assert not combat.nameplate_visible


def test_detect_combat_state_ignores_neutral_nameplate_red_level_fragment():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 220, 220), -1)
    cv2.rectangle(frame, (640, 192), (664, 200), (0, 0, 255), -1)

    combat = detect_combat_state(frame, _combat_test_config())

    assert not combat.active
    assert not combat.target_present
    assert not combat.nameplate_visible


def test_detect_player_health_fraction_reads_partial_green_bar():
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 50), (249, 63), (0, 200, 0), -1)
    config = {
        "screen": {"reference_width": 500, "reference_height": 200, "scale_regions": False},
        "safety": {
            "combat": {
                "player_health_region": {"x": 100, "y": 45, "width": 300, "height": 24},
                "player_health_column_min_pixels": 4,
            }
        },
    }

    health = detect_player_health_fraction(frame, config)

    assert health is not None
    assert 0.48 <= health <= 0.52


def test_detect_outgoing_damage_numbers_finds_yellow_or_white_text():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "237", (610, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 255), 4)

    bbox = detect_outgoing_damage_numbers(frame, _combat_test_config())

    assert bbox is not None
    assert 590 <= bbox.x <= 630
    assert 300 <= bbox.y <= 350


def test_detect_outgoing_damage_numbers_ignores_red_incoming_text():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "-191", (610, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

    assert detect_outgoing_damage_numbers(frame, _combat_test_config()) is None


def test_detect_combat_facing_error_text_reads_top_center_red_error():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "Target needs to be in front of you.", (370, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    config = _combat_test_config()
    config["safety"]["combat"]["facing_error_region"] = {"x": 300, "y": 70, "width": 680, "height": 90}
    config["safety"]["combat"]["facing_error_min_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_row_band_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_span_width"] = 80

    assert detect_combat_facing_error_text(frame, config)


def test_live_desolace_facing_error_calibration_rejects_idle_and_reads_error():
    run_dir = Path("data/live_desolace_adt_entry_20260801_0107/frames")
    idle_path = run_dir / "0000.png"
    error_path = run_dir / "0214.png"
    if not idle_path.exists() or not error_path.exists():
        pytest.skip("Desolace combat calibration frames are not present in this checkout")

    from vision_bot.config import load_config

    config = load_config("config.yaml")
    idle_frame = cv2.imread(str(idle_path))
    error_frame = cv2.imread(str(error_path))

    assert idle_frame is not None
    assert error_frame is not None
    assert not detect_combat_facing_error_text(idle_frame, config)
    assert detect_combat_facing_error_text(error_frame, config)


def test_detect_combat_state_reports_facing_error_with_confirmed_target():
    frame = _target_frame()
    cv2.putText(frame, "You are facing the wrong way!", (390, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    config = _combat_test_config()
    config["safety"]["combat"]["facing_error_region"] = {"x": 300, "y": 70, "width": 680, "height": 90}
    config["safety"]["combat"]["facing_error_min_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_row_band_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_span_width"] = 80

    combat = detect_combat_state(frame, config)

    assert not combat.active
    assert combat.facing_error_visible


def test_detect_combat_state_disabled_does_not_latch_on_facing_error_text():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "You are facing the wrong way!", (390, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    config = _combat_test_config()
    config["safety"]["combat"]["enabled"] = False
    config["safety"]["combat"]["facing_error_region"] = {"x": 300, "y": 70, "width": 680, "height": 90}
    config["safety"]["combat"]["facing_error_min_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_row_band_pixels"] = 20
    config["safety"]["combat"]["facing_error_min_span_width"] = 80

    combat = detect_combat_state(frame, config)

    assert not combat.facing_error_visible
    assert not combat.has_latching_evidence()


def test_combat_fallback_face_searches_after_e_without_damage():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )

    assert fallback.next_attack_key(10.0) == "E"
    assert fallback.next_attack_key(11.0) == "E"

    assert fallback.next_face_search_turn(11.9, turn_180_duration=0.9, turn_90_duration=0.45) is None
    assert fallback.next_face_search_turn(12.0, turn_180_duration=0.9, turn_90_duration=0.45) == (
        "D",
        0.45,
        "combat_face_search_sweep",
    )


def test_combat_fallback_skips_face_search_when_damage_seen_after_e():
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

    assert fallback.next_face_search_turn(13.5, turn_180_duration=0.9, turn_90_duration=0.45) is None


def test_combat_fallback_repeated_e_does_not_postpone_face_search_forever():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=0.5,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )

    assert fallback.next_attack_keys(10.0) == ["E"]
    assert fallback.next_attack_keys(10.5) == ["E"]
    assert fallback.next_attack_keys(11.0) == ["TAB"]
    assert fallback.next_attack_keys(11.5) == ["E"]

    assert fallback.next_face_search_turn(11.9, turn_180_duration=0.9, turn_90_duration=0.45) is None
    assert fallback.next_face_search_turn(12.0, turn_180_duration=0.9, turn_90_duration=0.45) == (
        "D",
        0.45,
        "combat_face_search_sweep",
    )
    assert fallback.next_face_search_turn(13.2, turn_180_duration=0.9, turn_90_duration=0.45) == (
        "D",
        0.45,
        "combat_face_search_sweep",
    )


def test_combat_fallback_facing_error_does_not_confirm_e_damage():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=0.5,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )

    assert fallback.next_attack_keys(10.0) == ["E"]
    assert fallback.next_attack_keys(10.5) == ["E"]
    fallback.observe_outgoing_damage(10.7, True, facing_error_visible=True)

    assert fallback.unconfirmed_e_started_at == 10.0
    assert fallback.next_face_search_turn(12.6, turn_180_duration=0.9, turn_90_duration=0.45) == (
        "D",
        0.45,
        "combat_face_search_sweep",
    )


def test_combat_fallback_forced_face_search_ignores_damage_after_e():
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

    assert fallback.next_face_search_turn(
        11.6,
        turn_180_duration=0.9,
        turn_90_duration=0.45,
        force=True,
    ) == ("D", 0.45, "combat_face_search_sweep")


def test_combat_fallback_clear_counter_exits_after_configured_frames():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=0.4, clear_frames=2)
    active = CombatState(active=True, target_present=True)
    inactive = detect_combat_state(np.zeros((720, 1280, 3), dtype=np.uint8), _combat_test_config())

    assert fallback.observe(active)
    assert fallback.observe(inactive)
    assert not fallback.observe(inactive)
    assert fallback.consume_clear_event()
    assert not fallback.consume_clear_event()


def test_combat_fallback_keeps_engaged_on_latching_evidence_after_target_drops():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2, clear_seconds=3.0)
    active = CombatState(active=True, target_present=True)
    second_enemy_evidence = CombatState(
        active=False,
        target_present=False,
        nameplate_visible=True,
        outgoing_damage_visible=True,
        facing_error_visible=True,
    )

    assert fallback.observe(active, now=10.0)
    assert fallback.observe(second_enemy_evidence, now=11.0)

    assert fallback.engaged
    assert not fallback.consume_clear_event()


def test_combat_fallback_keeps_engaged_on_health_drop_after_first_target_drops():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        clear_seconds=3.0,
        health_drop_latch_threshold=0.01,
    )
    active = CombatState(active=True, target_present=True, player_health_fraction=0.80)
    second_enemy_hitting = CombatState(player_health_fraction=0.76)

    assert fallback.observe(active, now=10.0)
    assert fallback.observe(second_enemy_hitting, now=10.5)

    assert fallback.engaged
    assert fallback.last_observation_latching_evidence
    assert not fallback.consume_clear_event()


def test_combat_fallback_can_start_from_player_health_drop():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        clear_seconds=3.0,
        health_drop_latch_threshold=0.01,
    )

    assert not fallback.observe(CombatState(player_health_fraction=0.90), now=10.0)
    assert fallback.observe(CombatState(player_health_fraction=0.80), now=10.5)
    assert fallback.engaged
    assert fallback.last_health_drop_at == 10.5


def test_combat_fallback_default_threshold_ignores_measured_hp_bar_quantization():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
    )

    assert not fallback.observe(CombatState(player_health_fraction=0.9631578947368421), now=10.0)
    assert not fallback.observe(CombatState(player_health_fraction=0.9473684210526315), now=10.5)
    assert not fallback.engaged


def test_combat_fallback_target_frame_alone_does_not_start_combat():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        clear_seconds=3.0,
    )

    assert not fallback.observe(CombatState(target_present=True, player_health_fraction=0.90), now=10.0)
    assert not fallback.engaged


def test_combat_fallback_target_frame_latches_after_health_drop_starts_combat():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        clear_seconds=3.0,
        health_drop_latch_threshold=0.01,
    )

    assert not fallback.observe(CombatState(target_present=True, player_health_fraction=0.90), now=10.0)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=10.5)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=11.0)
    assert fallback.engaged


def test_combat_fallback_clears_stale_soft_target_after_health_drop_grace():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
        clear_seconds=3.0,
        health_drop_latch_threshold=0.01,
        stale_soft_target_seconds=6.0,
    )

    assert not fallback.observe(CombatState(target_present=True, player_health_fraction=0.90), now=10.0)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=10.5)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=16.49)
    assert not fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=16.5)
    assert fallback.consume_clear_event()
    assert fallback.consume_stale_soft_target_clear_event()
    assert not fallback.consume_confirmed_outgoing_clear_event()


def test_combat_fallback_repeated_health_drop_extends_soft_target_grace():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
        health_drop_latch_threshold=0.01,
        stale_soft_target_seconds=6.0,
    )

    fallback.observe(CombatState(player_health_fraction=0.90), now=10.0)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.80), now=10.5)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.70), now=16.0)
    assert fallback.observe(CombatState(target_present=True, player_health_fraction=0.70), now=21.9)
    assert not fallback.consume_stale_soft_target_clear_event()


def test_combat_fallback_clear_reports_confirmed_outgoing_damage():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=0.2,
        clear_frames=2,
        clear_seconds=1.0,
    )

    assert fallback.observe(CombatState(active=True), now=10.0)
    fallback.observe_outgoing_damage(10.2, True)
    assert not fallback.observe(CombatState(), now=11.2)
    assert fallback.consume_clear_event()
    assert fallback.consume_confirmed_outgoing_clear_event()
    assert not fallback.consume_stale_soft_target_clear_event()


def test_combat_fallback_clears_only_after_time_without_latching_evidence():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2, clear_seconds=3.0)
    active = CombatState(active=True, target_present=True)
    inactive = CombatState()

    assert fallback.observe(active, now=10.0)
    assert fallback.observe(inactive, now=12.9)
    assert not fallback.consume_clear_event()
    assert not fallback.observe(inactive, now=13.1)
    assert fallback.consume_clear_event()


def test_combat_fallback_marker_clear_ignores_stale_target_hints():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=1.0,
        clear_frames=2,
        marker_clear_seconds=0.75,
    )

    assert fallback.observe(CombatState(active=True, combat_marker_visible=True), now=10.0)
    assert fallback.observe(CombatState(target_present=True, nameplate_visible=True), now=10.7)
    assert not fallback.observe(CombatState(target_present=True, nameplate_visible=True), now=10.8)
    assert fallback.consume_clear_event()


def test_combat_fallback_marker_clear_handles_zero_timestamp():
    fallback = CombatFallbackState(
        attack_keys=("E",),
        attack_interval=1.0,
        clear_frames=2,
        marker_clear_seconds=0.75,
    )

    assert fallback.observe(CombatState(active=True, combat_marker_visible=True), now=0.0)
    assert fallback.observe(CombatState(), now=0.5)
    assert not fallback.observe(CombatState(), now=0.8)


def test_combat_fallback_clear_event_requires_real_engaged_combat():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=0.4, clear_frames=2)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 0, 255), -1)
    nameplate_only = detect_combat_state(frame, _combat_test_config())
    inactive = detect_combat_state(np.zeros((720, 1280, 3), dtype=np.uint8), _combat_test_config())

    assert not nameplate_only.active
    assert not fallback.observe(nameplate_only)
    assert not fallback.observe(inactive)
    assert not fallback.consume_clear_event()


def test_combat_fallback_does_not_engage_on_inactive_nameplate_only_state():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=0.4, clear_frames=2)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 0, 255), -1)
    nameplate_only = detect_combat_state(frame, _combat_test_config())

    assert nameplate_only.nameplate_visible
    assert not nameplate_only.active
    assert not fallback.observe(nameplate_only)
    assert not fallback.engaged


def test_combat_fallback_does_not_start_from_latching_evidence_without_confirmed_combat():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=0.4, clear_frames=2, clear_seconds=3.0)
    nameplate_only = CombatState(active=False, nameplate_visible=True, outgoing_damage_visible=True)

    assert not fallback.observe(nameplate_only, now=10.0)
    assert not fallback.engaged


def test_combat_fallback_schedules_attack_keys_as_ordered_sequence():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=0.4, clear_frames=2)

    assert fallback.next_attack_key(10.0) == "E"
    assert fallback.next_attack_key(10.39) is None
    assert fallback.next_attack_key(10.4) == "E"
    assert fallback.next_attack_key(10.79) is None
    assert fallback.next_attack_key(10.8) == "TAB"
    assert fallback.next_attack_key(11.21) == "E"


def test_combat_fallback_returns_at_most_one_due_attack_key_per_tick():
    fallback = CombatFallbackState(attack_keys=("E", "E", "TAB"), attack_interval=1.0, clear_frames=2)

    assert fallback.next_attack_keys(10.0) == ["E"]
    assert fallback.next_attack_keys(10.5) == []
    assert fallback.next_attack_keys(11.0) == ["E"]
    assert fallback.next_attack_keys(12.5) == ["TAB"]


def test_combat_fallback_damage_after_tab_can_confirm_prior_e():
    fallback = CombatFallbackState(
        attack_keys=("E", "E", "TAB"),
        attack_interval=1.0,
        clear_frames=2,
        damage_timeout=2.0,
        face_search_cooldown=0.5,
    )

    assert fallback.next_attack_key(10.0) == "E"
    assert fallback.next_attack_key(11.0) == "E"
    assert fallback.next_attack_key(12.0) == "TAB"
    fallback.observe_outgoing_damage(12.2, True)

    assert fallback.unconfirmed_e_started_at is None
    assert fallback.next_face_search_turn(14.3, turn_180_duration=0.9, turn_90_duration=0.45) is None


def test_attack_key_normalization_defaults_to_extensible_e_only_rotation():
    assert normalize_attack_keys(None) == ("E",)
    assert normalize_attack_keys("E") == ("E",)
    assert normalize_attack_keys([]) == ("E",)
    assert normalize_attack_keys([" e ", "", "q"]) == ("E", "Q")


def test_periodic_key_normalization_defaults_to_disabled_but_accepts_future_keys():
    assert normalize_periodic_combat_keys(None) == ()
    assert normalize_periodic_combat_keys("X") == ()
    assert normalize_periodic_combat_keys([]) == ()
    assert normalize_periodic_combat_keys([" x ", "", "v"]) == ("X", "V")


def test_periodic_combat_keys_fire_immediately_then_on_interval():
    periodic = PeriodicCombatKeyState(keys=("X",), interval=15.0)

    assert periodic.next_keys(10.0) == ["X"]
    assert periodic.next_keys(24.9) == []
    assert periodic.next_keys(25.0) == ["X"]


def test_periodic_combat_keys_reset_for_new_combat():
    periodic = PeriodicCombatKeyState(keys=("X",), interval=15.0)

    assert periodic.next_keys(10.0) == ["X"]
    assert periodic.next_keys(11.0) == []
    periodic.reset()

    assert periodic.next_keys(12.0) == ["X"]


def test_combat_heal_state_starts_only_below_threshold():
    heal = CombatHealState(threshold=0.50, cooldown_seconds=8.0, cast_seconds=2.0)

    assert not heal.should_start(CombatState(player_health_fraction=0.51), 10.0)
    assert heal.should_start(CombatState(player_health_fraction=0.50), 10.0)


def test_combat_heal_state_tracks_cast_window_and_cooldown():
    heal = CombatHealState(threshold=0.50, cooldown_seconds=8.0, cast_seconds=2.0)
    combat = CombatState(player_health_fraction=0.40)

    assert heal.should_start(combat, 10.0)
    heal.start(10.0)

    assert heal.is_casting(11.9)
    assert not heal.is_casting(12.0)
    assert not heal.should_start(combat, 12.1)
    assert heal.should_start(combat, 18.0)


def test_combat_heal_state_ignores_unknown_health_or_disabled_config():
    assert not CombatHealState().should_start(CombatState(player_health_fraction=None), 10.0)
    assert not CombatHealState(enabled=False).should_start(CombatState(player_health_fraction=0.10), 10.0)


def _target_frame() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (410, 47), (595, 65), (0, 0, 255), -1)
    cv2.rectangle(frame, (410, 75), (550, 87), (0, 200, 0), -1)
    return frame


def _live_2560_combat_config():
    return {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "safety": {
            "combat": {
                "enabled": True,
                "engage_on_nameplate": False,
                "confirmed_target_frame_region": {"x": 300, "y": 15, "width": 390, "height": 120},
                "player_health_region": {"x": 155, "y": 70, "width": 190, "height": 25},
                "player_health_column_min_pixels": 4,
            },
            "hostile_avoidance": {
                "enabled": True,
                "target_frame_region": {"x": 230, "y": 0, "width": 420, "height": 150},
                "central_threat_region": {"x": 520, "y": 180, "width": 1320, "height": 650},
                "combat_warning_enabled": False,
                "target_frame_min_area": 90,
                "target_frame_min_width": 28,
                "target_frame_min_height": 4,
                "target_frame_min_fill_ratio": 0.55,
                "nameplate_min_area": 80,
                "nameplate_min_width": 24,
                "nameplate_min_fill_ratio": 0.60,
                "ignore_bottom_ui_fraction": 0.72,
                "ignore_right_ui_fraction": 0.78,
            },
        },
    }


def _combat_test_config():
    return {
        "screen": {"reference_width": 1280, "reference_height": 720, "scale_regions": False},
        "safety": {
            "combat": {
                "enabled": True,
                "engage_on_nameplate": False,
                "confirmed_target_frame_region": {"x": 300, "y": 15, "width": 390, "height": 120},
                "attack_keys": ["E", "E", "TAB"],
                "attack_interval": 0.40,
                "face_turn_duration": 0.15,
                "face_center_deadzone_px": 24,
                "clear_frames": 2,
            },
            "hostile_avoidance": {
                "enabled": True,
                "target_frame_region": {"x": 110, "y": 0, "width": 300, "height": 110},
                "central_threat_region": {"x": 260, "y": 90, "width": 660, "height": 340},
                "combat_warning_enabled": False,
                "target_frame_min_area": 50,
                "target_frame_min_width": 28,
                "target_frame_min_fill_ratio": 0.55,
                "nameplate_min_area": 50,
                "nameplate_min_width": 40,
                "nameplate_min_fill_ratio": 0.60,
                "ignore_bottom_ui_fraction": 0.72,
                "ignore_right_ui_fraction": 0.78,
            },
        },
    }
