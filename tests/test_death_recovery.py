from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_bot.death_recovery import (
    BoundingBox,
    DeathRecoveryController,
    _ocr_token_matches,
    clear_resurrection_sickness_state,
    detect_release_spirit_button,
    detect_death_recap_panel,
    detect_return_to_life_button,
    detect_resurrection_accept_button,
    detect_return_to_graveyard_button,
    detect_return_to_graveyard_accept_button,
    detect_safe_zone_destination_button,
    detect_safe_zone_resurrect_button,
    detect_spirit_healer_interaction_error,
    detect_spirit_healer_graveyard_candidate,
    detect_spirit_healer_tooltip,
    detect_spirit_healer_target,
    detect_spirit_healer_target_frame,
    read_resurrection_sickness_remaining,
    write_resurrection_sickness_state,
)
from vision_bot.config import load_config
from vision_bot.game_state import GameState, detect_game_state


def test_detect_release_spirit_button_chooses_left_top_center_button():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (320, 95), (480, 125), (0, 0, 150), -1)
    cv2.rectangle(frame, (520, 95), (680, 125), (0, 0, 150), -1)

    result = detect_release_spirit_button(frame, _death_test_config())

    assert result is not None
    click_point, bbox = result
    assert bbox.x == 320
    assert bbox.width == 161
    assert 390 <= click_point[0] <= 410
    assert 105 <= click_point[1] <= 115


def test_detect_return_to_graveyard_button_uses_fixed_left_ghost_action_region():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.putText(frame, "Return to", (410, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, "Graveyard", (410, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, "Resurrect in a Safe Zone", (650, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    config = _spirit_healer_test_config(
        return_to_graveyard_region={"x": 380, "y": 45, "width": 220, "height": 90},
        return_to_graveyard_min_pixels=40,
        return_to_graveyard_min_span_width=50,
        return_to_graveyard_min_span_height=20,
    )
    config["position_ocr"] = load_config("config.yaml").get("position_ocr", {})

    result = detect_return_to_graveyard_button(frame, config)

    assert result is not None
    click_point, bbox = result
    assert 430 <= click_point[0] <= 520
    assert 70 <= click_point[1] <= 105
    assert bbox.x < 600


def test_return_to_graveyard_detector_ignores_live_tanaris_sand_if_available():
    frame_path = Path(
        "data/live_overnight_20260804_035134/01_core_route_combat_mining/frames/0449.png"
    )
    if not frame_path.exists():
        pytest.skip("Overnight Tanaris regression frame is not present")
    frame = cv2.imread(str(frame_path))
    assert frame is not None

    assert detect_return_to_graveyard_button(frame, load_config("config.yaml")) is None


def test_return_to_graveyard_detector_reads_current_tanaris_button_if_available():
    frame_path = Path("data/live_tanaris_v3_directional_fix_20260803_022244/frames/1581.png")
    if not frame_path.exists():
        pytest.skip("Current Tanaris corpse frame is not present")
    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_return_to_graveyard_button(frame, load_config("config.yaml"))

    assert result is not None
    click_point, _bbox = result
    assert 1140 <= click_point[0] <= 1240
    assert 150 <= click_point[1] <= 210


def test_current_overnight_death_frame_releases_spirit_instead_of_clicking_sand_if_available():
    frame_path = Path(
        "data/live_overnight_20260804_035134/01_core_route_combat_mining/frames/0451.png"
    )
    if not frame_path.exists():
        pytest.skip("Overnight Tanaris death frame is not present")
    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    controller = DeathRecoveryController.from_config(config)
    state = detect_game_state(frame)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_release_spirit"
    assert decision.reason == "release_button"
    assert decision.click_point == (1161, 290)


def test_spirit_healer_recovery_returns_to_graveyard_before_target_macro():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.putText(frame, "Return to", (410, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, "Graveyard", (410, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    config = _spirit_healer_test_config(
        return_to_graveyard_region={"x": 380, "y": 45, "width": 220, "height": 90},
        return_to_graveyard_min_pixels=40,
        return_to_graveyard_min_span_width=50,
        return_to_graveyard_min_span_height=20,
        return_to_graveyard_wait_seconds=0.0,
    )
    controller = DeathRecoveryController.from_config(config)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_return_to_graveyard"
    assert decision.reason == "return_to_graveyard_button"
    assert decision.click_point is not None


def test_spirit_healer_recovery_prefers_yellow_graveyard_action_over_red_modal():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.putText(frame, "Return to", (410, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, "Graveyard", (410, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.rectangle(frame, (470, 160), (630, 195), (0, 0, 150), -1)
    config = _spirit_healer_test_config(
        release_button_region={"x": 280, "y": 60, "width": 440, "height": 180},
        return_to_graveyard_region={"x": 380, "y": 45, "width": 220, "height": 90},
        return_to_graveyard_min_pixels=40,
        return_to_graveyard_min_span_width=50,
        return_to_graveyard_min_span_height=20,
    )
    controller = DeathRecoveryController.from_config(config)
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=False)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_return_to_graveyard"
    assert decision.click_point is not None
    assert decision.click_point[1] < 130


def test_spirit_healer_recovery_detects_current_tanaris_corpse_frame_if_available():
    frame_path = Path("data/live_tanaris_v3_directional_fix_20260803_022244/frames/1581.png")
    if not frame_path.exists():
        pytest.skip("Current Tanaris corpse frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    controller = DeathRecoveryController.from_config(config)
    state = GameState(
        death_or_blocking_modal=True,
        red_button_count=0,
        ghost_visual=False,
        ghost_button_count=2,
    )

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_return_to_graveyard"
    assert decision.click_point is not None
    assert 1130 <= decision.click_point[0] <= 1240
    assert 155 <= decision.click_point[1] <= 210


def test_return_to_graveyard_confirmation_is_detected_on_live_frame_if_available():
    frame_path = Path("data/live_desolace_death_recovery_v2_20260801_0142/frames/0002.png")
    if not frame_path.exists():
        pytest.skip("Desolace return-to-graveyard confirmation frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    result = detect_return_to_graveyard_accept_button(frame, config)

    assert result is not None
    click_point, bbox = result
    assert 1135 <= bbox.x <= 1150
    assert 295 <= bbox.y <= 305
    assert 1145 <= click_point[0] <= 1170
    assert 300 <= click_point[1] <= 315


def test_spirit_healer_recovery_accepts_open_graveyard_confirmation_first():
    frame_path = Path("data/live_desolace_death_recovery_v2_20260801_0142/frames/0002.png")
    if not frame_path.exists():
        pytest.skip("Desolace return-to-graveyard confirmation frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    controller = DeathRecoveryController.from_config(config)
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=True, ghost_button_count=2)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_accept_return_to_graveyard"
    assert decision.reason == "return_to_graveyard_yes_button"
    assert decision.click_point is not None
    assert controller.return_to_graveyard_confirmed


def test_spirit_healer_recovery_does_not_reopen_graveyard_after_yes_if_available():
    run_dir = Path("data/live_desolace_death_recovery_v3_20260801_0152/frames")
    confirmation_path = run_dir / "0000.png"
    returned_path = run_dir / "0001.png"
    if not confirmation_path.exists() or not returned_path.exists():
        pytest.skip("Desolace graveyard transition frames are not present")

    confirmation_frame = cv2.imread(str(confirmation_path))
    returned_frame = cv2.imread(str(returned_path))
    assert confirmation_frame is not None
    assert returned_frame is not None
    config = load_config("config.yaml")
    controller = DeathRecoveryController.from_config(config)
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=True, ghost_button_count=2)

    first = controller.decide(confirmation_frame, state, config, now=10.0)
    second = controller.decide(returned_frame, state, config, now=14.0)

    assert first.action == "death_recovery_accept_return_to_graveyard"
    assert second.action == "death_recovery_target_spirit_healer"
    assert second.key == "C"


def test_graveyard_healer_candidate_accepts_two_live_healers_and_rejects_pre_graveyard_false_positive():
    positive_paths = [
        Path("data/live_desolace_death_recovery_v4_20260801_0204/frames/0073.png"),
        Path("debug_output/spirit_manual_interact_success_check_20260731.png"),
    ]
    negative_path = Path("data/live_desolace_death_recovery_20260801_0118/frames/0182.png")
    if not all(path.exists() for path in [*positive_paths, negative_path]):
        pytest.skip("Spirit Healer calibration frames are not present")

    config = load_config("config.yaml")
    positive_results = [
        detect_spirit_healer_graveyard_candidate(cv2.imread(str(path)), config)
        for path in positive_paths
    ]
    negative_result = detect_spirit_healer_graveyard_candidate(cv2.imread(str(negative_path)), config)

    assert all(result is not None for result in positive_results)
    assert negative_result is None


def test_graveyard_healer_tooltip_rejects_current_bright_mountain_false_positive():
    frame_path = Path("debug_output/screen_objects_frame.png")
    if not frame_path.exists():
        pytest.skip("Current Kodo Graveyard frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    assert detect_spirit_healer_tooltip(frame, load_config("config.yaml")) is None


def test_graveyard_healer_tooltip_accepts_live_hovered_healer():
    frame_path = Path("data/live_desolace_death_recovery_v5_20260801_0230/frames/0048.png")
    if not frame_path.exists():
        pytest.skip("Hovered Kodo Graveyard Spirit Healer frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    assert detect_spirit_healer_tooltip(frame, load_config("config.yaml")) is not None


def test_confirmed_graveyard_recovery_does_not_reopen_return_dialog():
    frame_path = Path("data/live_desolace_death_recovery_v4_20260801_0204/frames/0073.png")
    if not frame_path.exists():
        pytest.skip("Kodo Graveyard Spirit Healer frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    config["safety"]["death_recovery"]["spirit_healer_target_interact_wait_seconds"] = 0.0
    config["safety"]["death_recovery"]["spirit_healer_target_wait_seconds"] = 0.0
    config["safety"]["death_recovery"]["spirit_healer_search_wait_seconds"] = 0.0
    controller = DeathRecoveryController.from_config(config)
    controller.return_to_graveyard_confirmed = True
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True, ghost_button_count=2)

    decisions = [
        controller.decide(frame, state, config, now=float(index))
        for index in range(10)
    ]

    assert decisions[0].action == "death_recovery_target_spirit_healer"
    assert all(decision.action != "death_recovery_return_to_graveyard" for decision in decisions)
    assert any(decision.action == "death_recovery_hover_spirit_healer" for decision in decisions)
    assert controller.return_to_graveyard_confirmed


def test_confirmed_too_far_approach_uses_high_confidence_healer_not_broad_bright_cluster():
    frame_path = Path("data/live_desolace_death_recovery_v5_20260801_0230/frames/0048.png")
    if not frame_path.exists():
        pytest.skip("Kodo Graveyard too-far frame is not present")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    controller = DeathRecoveryController.from_config(config)
    controller.return_to_graveyard_confirmed = True
    controller.spirit_healer_clicked = True
    controller.spirit_healer_confirmed_visual_target = detect_spirit_healer_graveyard_candidate(
        frame,
        config,
    )
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True, ghost_button_count=2)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_approach_spirit_healer"
    assert decision.reason == "spirit_healer_interaction_too_far"
    assert decision.click_point == (817, 407)


def test_detect_release_spirit_button_on_real_death_frame_if_available():
    frame_path = Path("data/live_route_probe_20260729_combat_fallback_durotar_v1/frames/0018.png")
    if not frame_path.exists():
        pytest.skip("real live death frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_release_spirit_button(frame, _real_frame_config())

    assert result is not None
    click_point, bbox = result
    assert 1040 <= bbox.x <= 1070
    assert 270 <= bbox.y <= 290
    assert 1140 <= click_point[0] <= 1180
    assert 280 <= click_point[1] <= 305


def test_detect_safe_zone_resurrect_button_chooses_right_yellow_ghost_button():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    yellow = (0, 190, 230)
    cv2.rectangle(frame, (435, 82), (535, 118), yellow, -1)
    cv2.rectangle(frame, (610, 82), (735, 118), yellow, -1)

    result = detect_safe_zone_resurrect_button(frame, _safe_zone_test_config())

    assert result is not None
    click_point, bbox = result
    assert 600 <= bbox.x <= 615
    assert 120 <= bbox.width <= 150
    assert 660 <= click_point[0] <= 690
    assert 95 <= click_point[1] <= 105


def test_detect_safe_zone_resurrect_button_on_current_live_ghost_frame_if_available():
    frame_path = Path("debug_output/current_live_state_20260729_after_death_notes.png")
    if not frame_path.exists():
        pytest.skip("current live ghost frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_safe_zone_resurrect_button(frame, _real_frame_config())

    assert result is not None
    click_point, bbox = result
    assert 1340 <= bbox.x <= 1370
    assert 155 <= bbox.y <= 170
    assert 1390 <= click_point[0] <= 1445
    assert 175 <= click_point[1] <= 195


def test_detect_safe_zone_destination_button_chooses_left_red_choice():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    red = (0, 0, 170)
    cv2.rectangle(frame, (360, 116), (520, 145), red, -1)
    cv2.rectangle(frame, (540, 116), (700, 145), red, -1)
    cv2.rectangle(frame, (720, 116), (880, 145), red, -1)

    result = detect_safe_zone_destination_button(frame, _safe_zone_destination_test_config())

    assert result is not None
    click_point, bbox = result
    assert 355 <= bbox.x <= 365
    assert 150 <= bbox.width <= 170
    assert 430 <= click_point[0] <= 450
    assert 125 <= click_point[1] <= 140


def test_detect_safe_zone_destination_button_on_current_1080p_confirmation_if_available():
    frame_path = Path("debug_output/live_death_recovery_smoke_20260729/current_now.png")
    if not frame_path.exists():
        pytest.skip("current live safe-zone confirmation frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_safe_zone_destination_button(frame, _real_frame_config())

    assert result is not None
    click_point, bbox = result
    assert 1040 <= bbox.x <= 1075
    assert 310 <= bbox.y <= 335
    assert 1135 <= click_point[0] <= 1185
    assert 315 <= click_point[1] <= 345


def test_death_recovery_treats_safe_zone_confirmation_as_destination_if_available():
    frame_path = Path("debug_output/current_live_state_after_failed_recovery.png")
    if not frame_path.exists():
        pytest.skip("current safe-zone confirmation frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=False)
    controller = DeathRecoveryController.from_config(_real_safe_zone_frame_config())

    decision = controller.decide(frame, state, _real_safe_zone_frame_config(), now=10.0)

    assert decision.action == "death_recovery_safe_zone_destination"
    assert decision.reason == "safe_zone_destination_button"
    assert decision.click_point is not None


def test_detect_spirit_healer_target_on_current_live_ghost_frame_if_available():
    frame_path = Path("debug_output/current_live_state_20260729_after_death_notes.png")
    if not frame_path.exists():
        pytest.skip("current live ghost frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_spirit_healer_target(frame, _real_frame_config())

    assert result is not None
    click_point, bbox = result
    assert 1450 <= bbox.x <= 2100
    assert 400 <= bbox.y <= 650
    assert 1450 <= click_point[0] <= 2200
    assert 500 <= click_point[1] <= 850


def test_detect_spirit_healer_target_ignores_bottom_action_bar_live_frame_if_available():
    frame_path = Path("data/live_route_probe_barrens_run3_combat_20260731_023847/frames/0001.png")
    if not frame_path.exists():
        pytest.skip("run3 live ghost frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")

    result = detect_spirit_healer_target(frame, config)

    assert result is None


def test_detect_spirit_healer_target_frame_on_live_ghost_frame_if_available():
    frame_path = Path("data/live_route_probe_barrens_run3_combat_20260731_023847/frames/0001.png")
    if not frame_path.exists():
        pytest.skip("run3 live ghost frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_spirit_healer_target_frame(frame, load_config("config.yaml"))

    assert result is not None
    _click_point, bbox = result
    assert 440 <= bbox.x <= 470
    assert 45 <= bbox.y <= 65


def test_death_recovery_interacts_with_selected_spirit_healer_live_frame_if_available():
    frame_path = Path("data/live_route_probe_barrens_run3_combat_20260731_023847/frames/0001.png")
    if not frame_path.exists():
        pytest.skip("run3 live ghost frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    config["safety"]["death_recovery"]["return_to_graveyard_enabled"] = False
    controller = DeathRecoveryController.from_config(config)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True, ghost_button_count=2)

    target_decision = controller.decide(frame, state, config, now=10.0)
    interact_decision = controller.decide(frame, state, config, now=11.0)

    assert target_decision.action == "death_recovery_target_spirit_healer"
    assert target_decision.key == "C"
    assert interact_decision.action == "death_recovery_interact_spirit_healer_target"
    assert interact_decision.reason == "spirit_healer_target_frame_interact"
    assert interact_decision.key == "G"


def test_death_recovery_right_clicks_synthetic_spirit_healer():
    config = _spirit_healer_test_config(
        spirit_healer_wait_seconds=1.25,
        spirit_healer_self_ignore_enabled=False,
        spirit_healer_target_interact_enabled=False,
    )
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (440, 210), (540, 470), (230, 240, 245), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_right_click_spirit_healer"
    assert decision.reason == "spirit_healer_target"
    assert decision.mouse_button == "right"
    assert decision.wait_seconds == 1.25
    assert 470 <= decision.click_point[0] <= 520
    assert 300 <= decision.click_point[1] <= 380


def test_death_recovery_does_not_approach_visual_candidate_without_too_far_error():
    config = _spirit_healer_test_config(
        spirit_healer_wait_seconds=0.0,
        spirit_healer_approach_max_attempts=1,
        spirit_healer_approach_wait_seconds=0.35,
        spirit_healer_self_ignore_enabled=False,
        return_to_life_fallback_click_enabled=False,
    )
    controller = DeathRecoveryController.from_config(config)
    controller.spirit_healer_clicked = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (440, 210), (540, 470), (230, 240, 245), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, config, now=10.0)
    retarget = controller.decide(frame, state, config, now=11.0)
    retry = controller.decide(frame, state, config, now=12.0)

    assert decision.action == "death_recovery_search_spirit_healer"
    assert decision.reason == "spirit_healer_macro_no_dialog"
    assert decision.click_point is None
    assert retarget.action == "death_recovery_target_spirit_healer"
    assert retarget.reason == "spirit_healer_target_macro_primary"
    assert retarget.key == "C"
    assert retry.action == "death_recovery_interact_spirit_healer_target"
    assert retry.reason == "spirit_healer_interact_primary"
    assert retry.key == "G"


def test_death_recovery_approaches_visible_candidate_after_confirmed_too_far_error(monkeypatch):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config(
        spirit_healer_wait_seconds=0.0,
        spirit_healer_approach_max_attempts=1,
        spirit_healer_approach_wait_seconds=0.35,
        spirit_healer_self_ignore_enabled=False,
        return_to_life_fallback_click_enabled=False,
    )
    controller = DeathRecoveryController.from_config(config)
    controller.spirit_healer_clicked = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (440, 210), (540, 470), (230, 240, 245), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_interaction_error",
        lambda _frame, _config: True,
    )
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_graveyard_candidate",
        lambda _frame, _config: (
            (490, 340),
            death_recovery_module.BoundingBox(440, 210, 100, 260),
        ),
    )
    controller.spirit_healer_confirmed_visual_target = (
        (490, 340),
        death_recovery_module.BoundingBox(440, 210, 100, 260),
    )

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_approach_spirit_healer"
    assert decision.reason == "spirit_healer_interaction_too_far"
    assert decision.wait_seconds == 0.35
    assert decision.click_point == (490, 340)


def test_death_recovery_searches_cemetery_after_too_far_target_without_visual_bearing(
    monkeypatch,
):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config(
        spirit_healer_wait_seconds=0.0,
        spirit_healer_follow_target_enabled=True,
        spirit_healer_follow_target_key="F6",
        spirit_healer_follow_target_wait_seconds=1.25,
    )
    controller = DeathRecoveryController.from_config(config)
    controller.spirit_healer_clicked = True
    controller.spirit_healer_target_interact_attempts = 4
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_interaction_error",
        lambda _frame, _config: True,
    )
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_target_frame",
        lambda _frame, _config: death_recovery_module.BoundingBox(300, 10, 200, 90),
    )

    search = controller.decide(frame, state, config, now=10.0)
    retarget = controller.decide(frame, state, config, now=10.35)

    assert search.action == "death_recovery_search_spirit_healer"
    assert search.reason == "spirit_healer_target_confirmed_too_far_no_direction"
    assert search.key is None
    assert controller.spirit_healer_approach_attempts == 0
    assert controller.spirit_healer_search_attempts == 1
    assert retarget.action == "death_recovery_target_spirit_healer"
    assert retarget.key == "C"


def test_death_recovery_approaches_visible_healer_when_selected_target_is_too_far(
    monkeypatch,
):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config(
        spirit_healer_wait_seconds=0.0,
        spirit_healer_approach_max_attempts=2,
        spirit_healer_approach_wait_seconds=0.35,
        spirit_healer_self_ignore_enabled=False,
    )
    controller = DeathRecoveryController.from_config(config)
    controller.spirit_healer_clicked = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    visual_target = (
        (780, 340),
        death_recovery_module.BoundingBox(720, 200, 120, 280),
    )
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_interaction_error",
        lambda _frame, _config: True,
    )
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_target_frame",
        lambda _frame, _config: death_recovery_module.BoundingBox(300, 10, 200, 90),
    )
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_graveyard_candidate",
        lambda _frame, _config: visual_target,
    )

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_approach_spirit_healer"
    assert decision.reason == "spirit_healer_target_confirmed_visual_approach"
    assert decision.click_point == (780, 340)
    assert controller.spirit_healer_approach_attempts == 1


def test_death_recovery_can_prioritize_visible_healer_over_broken_target_macro():
    config = _spirit_healer_test_config(
        spirit_healer_visual_primary=True,
        spirit_healer_target_interact_enabled=True,
        spirit_healer_wait_seconds=0.0,
        spirit_healer_self_ignore_enabled=False,
    )
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (440, 210), (540, 470), (230, 240, 245), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_right_click_spirit_healer"
    assert decision.mouse_button == "right"
    assert decision.click_point is not None
    assert controller.spirit_healer_target_interact_attempts == 0


def test_detect_spirit_healer_prefers_left_npc_over_own_ghost():
    config = _spirit_healer_test_config(
        spirit_healer_region={"x": 0, "y": 120, "width": 900, "height": 420},
        spirit_healer_max_width=900,
        spirit_healer_center_min_x_fraction=0.0,
        spirit_healer_self_ignore_enabled=True,
    )
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (25, 170), (195, 505), (230, 240, 245), -1)
    cv2.rectangle(frame, (470, 300), (560, 470), (230, 240, 245), -1)

    result = detect_spirit_healer_target(frame, config)

    assert result is not None
    click_point, bbox = result
    assert bbox.x < 250
    assert click_point[0] < 250


def test_detect_spirit_healer_ignores_own_ghost_on_failed_recovery_frame_if_available():
    frame_path = Path("data/live_route_probe_20260729_threat_target_disabled_v1/frames/0000.png")
    if not frame_path.exists():
        pytest.skip("failed live spirit healer frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None

    result = detect_spirit_healer_target(frame, _live_spirit_healer_config())

    assert result is not None
    click_point, bbox = result
    assert bbox.x < 350
    assert click_point[0] < 350


def test_death_recovery_ignores_safe_zone_buttons_in_spirit_healer_mode():
    config = _spirit_healer_test_config(spirit_healer_wait_seconds=0.5)
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    yellow = (0, 190, 230)
    cv2.rectangle(frame, (435, 82), (535, 118), yellow, -1)
    cv2.rectangle(frame, (610, 82), (735, 118), yellow, -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_target_spirit_healer"
    assert decision.reason == "spirit_healer_target_macro_primary"
    assert decision.key == "C"


def test_death_recovery_treats_ghost_action_buttons_as_spirit_healer_state():
    config = _spirit_healer_test_config(spirit_healer_self_ignore_enabled=False)
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (440, 210), (540, 470), (230, 240, 245), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=False, ghost_button_count=2)

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_target_spirit_healer"
    assert decision.reason == "spirit_healer_target_macro_primary"
    assert decision.key == "C"


def test_death_recovery_searches_for_missing_spirit_healer_before_waiting():
    config = _spirit_healer_test_config(
        spirit_healer_region={"x": 250, "y": 120, "width": 120, "height": 120},
        spirit_healer_target_interact_enabled=False,
        spirit_healer_search_max_attempts=2,
        spirit_healer_search_wait_seconds=0.25,
    )
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    first = controller.decide(frame, state, config, now=10.0)
    second = controller.decide(frame, state, config, now=11.0)
    third = controller.decide(frame, state, config, now=12.0)

    assert first.action == "death_recovery_search_spirit_healer"
    assert first.attempts == 1
    assert first.wait_seconds == 0.25
    assert second.action == "death_recovery_search_spirit_healer"
    assert second.attempts == 2
    assert third.action == "death_recovery_failed"
    assert third.reason == "spirit_healer_missing_after_search"


def test_death_recovery_alternates_macro_and_single_search_turn_decisions():
    config = _spirit_healer_test_config(
        spirit_healer_region={"x": 250, "y": 120, "width": 120, "height": 120},
        spirit_healer_target_interact_enabled=True,
        spirit_healer_target_interact_max_attempts=2,
        spirit_healer_target_interact_wait_seconds=0.0,
        spirit_healer_search_max_attempts=2,
        spirit_healer_search_wait_seconds=0.0,
    )
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    first = controller.decide(frame, state, config, now=10.0)
    second = controller.decide(frame, state, config, now=11.0)
    third = controller.decide(frame, state, config, now=12.0)
    fourth = controller.decide(frame, state, config, now=13.0)
    fifth = controller.decide(frame, state, config, now=14.0)
    sixth = controller.decide(frame, state, config, now=15.0)
    seventh = controller.decide(frame, state, config, now=16.0)

    assert first.action == "death_recovery_target_spirit_healer"
    assert first.reason == "spirit_healer_target_macro_primary"
    assert first.key == "C"
    assert first.attempts == 1
    assert second.action == "death_recovery_interact_spirit_healer_target"
    assert second.reason == "spirit_healer_interact_primary"
    assert second.key == "G"
    assert second.attempts == 1
    assert third.action == "death_recovery_search_spirit_healer"
    assert third.reason == "spirit_healer_macro_no_dialog"
    assert third.attempts == 1
    assert fourth.action == "death_recovery_target_spirit_healer"
    assert fourth.reason == "spirit_healer_target_macro_primary"
    assert fourth.key == "C"
    assert fourth.attempts == 2
    assert fifth.action == "death_recovery_interact_spirit_healer_target"
    assert fifth.reason == "spirit_healer_interact_primary"
    assert fifth.key == "G"
    assert fifth.attempts == 2
    assert sixth.action == "death_recovery_search_spirit_healer"
    assert sixth.reason == "spirit_healer_macro_no_dialog"
    assert sixth.attempts == 2
    assert seventh.action == "death_recovery_failed"
    assert seventh.reason == "spirit_healer_missing_after_search"


def test_death_recovery_repeats_c_g_search_without_eight_heading_limit():
    config = _spirit_healer_test_config(
        spirit_healer_region={"x": 250, "y": 120, "width": 120, "height": 120},
        spirit_healer_target_interact_enabled=True,
        spirit_healer_target_interact_max_attempts=0,
        spirit_healer_target_interact_wait_seconds=0.0,
        spirit_healer_search_max_attempts=0,
        spirit_healer_search_wait_seconds=0.0,
    )
    controller = DeathRecoveryController.from_config(config)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decisions = [
        controller.decide(frame, state, config, now=float(index))
        for index in range(30)
    ]

    assert [decision.action for decision in decisions[:9]] == [
        "death_recovery_target_spirit_healer",
        "death_recovery_interact_spirit_healer_target",
        "death_recovery_search_spirit_healer",
    ] * 3
    assert all(decision.action != "death_recovery_failed" for decision in decisions)
    assert controller.spirit_healer_search_attempts == 10
    assert controller.spirit_healer_target_interact_attempts == 10


def test_detect_spirit_healer_interaction_error_reads_live_too_far_message_if_available():
    error_path = Path("debug_output/spirit_manual_interact_success_check_20260731.png")
    dialog_path = Path("debug_output/spirit_manual_g_after_approach_20260731.png")
    if not error_path.exists() or not dialog_path.exists():
        pytest.skip("manual Spirit Healer recovery frames are not present in this checkout")

    config = load_config("config.yaml")
    error_frame = cv2.imread(str(error_path))
    dialog_frame = cv2.imread(str(dialog_path))

    assert error_frame is not None
    assert dialog_frame is not None
    assert detect_spirit_healer_interaction_error(error_frame, config)
    assert not detect_spirit_healer_interaction_error(dialog_frame, config)


def test_death_recovery_normalizes_configurable_spirit_healer_macro_keys():
    controller = DeathRecoveryController.from_config(
        _spirit_healer_test_config(
            spirit_healer_target_key=" c ",
            spirit_healer_interact_key=" g ",
        )
    )

    assert controller.spirit_healer_target_key == "C"
    assert controller.spirit_healer_interact_key == "G"


def test_detect_resurrection_accept_button_chooses_left_red_button():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    red = (0, 0, 170)
    cv2.rectangle(frame, (360, 116), (520, 145), red, -1)
    cv2.rectangle(frame, (540, 116), (700, 145), red, -1)

    result = detect_resurrection_accept_button(frame, _spirit_healer_test_config())

    assert result is not None
    click_point, bbox = result
    assert 355 <= bbox.x <= 365
    assert 150 <= bbox.width <= 170
    assert 430 <= click_point[0] <= 450
    assert 125 <= click_point[1] <= 140


def test_death_recovery_clicks_accept_twice_after_return_to_life():
    config = _spirit_healer_test_config(
        accept_resurrection_max_attempts=2,
        accept_resurrection_wait_seconds=0.0,
        resurrect_sickness_wait_seconds=0.0,
    )
    controller = DeathRecoveryController.from_config(config)
    controller.death_seen = True
    controller.return_to_life_clicked = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    red = (0, 0, 170)
    cv2.rectangle(frame, (360, 116), (520, 145), red, -1)
    cv2.rectangle(frame, (540, 116), (700, 145), red, -1)
    ghost_state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=True)
    alive_state = GameState(death_or_blocking_modal=False, red_button_count=0, ghost_visual=False)

    first = controller.decide(frame, ghost_state, config, now=10.0)
    second = controller.decide(frame, ghost_state, config, now=11.0)

    assert first.action == "death_recovery_accept_resurrection"
    assert first.attempts == 1
    assert second.action == "death_recovery_accept_resurrection"
    assert second.attempts == 2
    assert controller.consume_resurrected_wait(alive_state) == 0.0
    assert controller.consume_resurrected_wait(alive_state) is None


def test_spirit_interact_attempt_alone_cannot_open_accept_stage(monkeypatch):
    controller = DeathRecoveryController.from_config(_spirit_healer_test_config())
    controller.release_clicked = True
    controller.spirit_healer_clicked = True
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "vision_bot.death_recovery.detect_resurrection_accept_button",
        lambda *_args, **_kwargs: ((1280, 300), BoundingBox(1200, 280, 160, 40)),
    )
    monkeypatch.setattr(
        "vision_bot.death_recovery.detect_return_to_life_button",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vision_bot.death_recovery.detect_spirit_healer_dialog_marker",
        lambda *_args, **_kwargs: False,
    )

    decision = controller.decide_open_dialog(frame, _spirit_healer_test_config(), now=10.0)

    assert decision is None
    assert controller.accept_resurrection_attempts == 0


def test_open_death_recap_preempts_resurrection_button_detection(monkeypatch):
    controller = DeathRecoveryController.from_config(_spirit_healer_test_config())
    controller.release_clicked = True
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "vision_bot.death_recovery.detect_death_recap_panel",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "vision_bot.death_recovery.detect_resurrection_accept_button",
        lambda *_args, **_kwargs: pytest.fail("Accept must not run behind Death Recap"),
    )

    decision = controller.decide_open_dialog(frame, _spirit_healer_test_config(), now=10.0)

    assert decision is not None
    assert decision.action == "death_recovery_dismiss_death_recap"
    assert decision.key == "ESC"


def test_detect_death_recap_panel_on_run8_final_frame_if_available():
    path = Path("data/live_v0823_run8_20260816_1824/final_frame.png")
    if not path.exists():
        pytest.skip("RUN8 live evidence is not available")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None
    assert detect_death_recap_panel(frame, load_config("config.yaml"))


def test_open_return_to_life_dialog_preempts_search_without_internal_click_flag(monkeypatch):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config()
    controller = DeathRecoveryController.from_config(config)
    controller.return_to_graveyard_confirmed = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    target = ((200, 300), death_recovery_module.BoundingBox(120, 280, 160, 40))
    monkeypatch.setattr(
        death_recovery_module,
        "detect_return_to_life_button",
        lambda _frame, _config: target,
    )

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_return_to_life"
    assert decision.click_point == (200, 300)
    assert controller.return_to_life_clicked


def test_open_spirit_dialog_marker_stops_search_while_button_is_pending(monkeypatch):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config()
    controller = DeathRecoveryController.from_config(config)
    controller.return_to_graveyard_confirmed = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    monkeypatch.setattr(
        death_recovery_module,
        "detect_spirit_healer_dialog_marker",
        lambda _frame, _config: True,
    )

    decision = controller.decide(frame, state, config, now=10.0)

    assert decision.action == "death_recovery_wait"
    assert decision.reason == "spirit_healer_dialog_visible_button_pending"


def test_open_return_to_life_dialog_does_not_repeat_during_same_stage_settle(monkeypatch):
    import vision_bot.death_recovery as death_recovery_module

    config = _spirit_healer_test_config(return_to_life_wait_seconds=1.0)
    controller = DeathRecoveryController.from_config(config)
    controller.return_to_graveyard_confirmed = True
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    target = ((200, 300), death_recovery_module.BoundingBox(120, 280, 160, 40))
    monkeypatch.setattr(
        death_recovery_module,
        "detect_return_to_life_button",
        lambda _frame, _config: target,
    )

    first = controller.decide(frame, state, config, now=10.0)
    settling = controller.decide(frame, state, config, now=10.1)

    assert first.action == "death_recovery_return_to_life"
    assert settling.action == "death_recovery_wait"
    assert settling.reason == "return_to_life_settle"
    assert controller.return_to_life_attempts == 1


def test_ocr_token_match_accepts_missing_first_letter_only_for_long_tokens():
    assert _ocr_token_matches("eturn", "return")
    assert _ocr_token_matches("sereturn", "return")
    assert _ocr_token_matches("faxeturn", "return")
    assert _ocr_token_matches("retvrn", "return")
    assert not _ocr_token_matches("e", "me")
    assert not _ocr_token_matches("xx", "to")


def test_detect_return_to_life_button_on_current_live_spirit_healer_gossip_if_available():
    frame_path = Path("data/live_route_probe_multi_enemy_20260730_195531/frames/0010.png")
    if not frame_path.exists():
        pytest.skip("current live spirit healer gossip frame is not present in this checkout")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    config = load_config("config.yaml")
    config["safety"]["death_recovery"]["return_to_life_region"] = {"x": 40, "y": 120, "width": 960, "height": 720}

    result = detect_return_to_life_button(frame, config)

    assert result is not None
    click_point, bbox = result
    assert 40 <= bbox.x <= 130
    assert 350 <= bbox.y <= 400
    assert 100 <= click_point[0] <= 260
    assert 360 <= click_point[1] <= 400


def test_death_recovery_clicks_release_once_then_waits():
    controller = DeathRecoveryController.from_config(_death_test_config(release_wait_seconds=1.5))
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (320, 95), (480, 125), (0, 0, 150), -1)
    cv2.rectangle(frame, (520, 95), (680, 125), (0, 0, 150), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=False)

    decision = controller.decide(frame, state, _death_test_config(release_wait_seconds=1.5), now=10.0)
    waiting = controller.decide(frame, state, _death_test_config(release_wait_seconds=1.5), now=10.5)

    assert decision.action == "death_recovery_release_spirit"
    assert decision.click_point is not None
    assert decision.wait_seconds == 1.5
    assert waiting.action == "death_recovery_wait"
    assert waiting.reason == "release_wait"


def test_death_recovery_clicks_safe_zone_resurrect_while_ghost():
    controller = DeathRecoveryController.from_config(_safe_zone_test_config(safe_zone_resurrect_wait_seconds=1.25))
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    yellow = (0, 190, 230)
    cv2.rectangle(frame, (435, 82), (535, 118), yellow, -1)
    cv2.rectangle(frame, (610, 82), (735, 118), yellow, -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, _safe_zone_test_config(safe_zone_resurrect_wait_seconds=1.25), now=10.0)
    waiting = controller.decide(frame, state, _safe_zone_test_config(safe_zone_resurrect_wait_seconds=1.25), now=10.5)

    assert decision.action == "death_recovery_resurrect_safe_zone"
    assert decision.reason == "safe_zone_resurrect_button"
    assert decision.attempts == 1
    assert decision.click_point is not None
    assert decision.wait_seconds == 1.25
    assert waiting.action == "death_recovery_wait"
    assert waiting.reason == "release_wait"


def test_death_recovery_clicks_safe_zone_destination_before_yellow_buttons():
    controller = DeathRecoveryController.from_config(_safe_zone_destination_test_config())
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    yellow = (0, 190, 230)
    red = (0, 0, 170)
    cv2.rectangle(frame, (435, 82), (535, 118), yellow, -1)
    cv2.rectangle(frame, (610, 82), (735, 118), yellow, -1)
    cv2.rectangle(frame, (360, 116), (520, 145), red, -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)

    decision = controller.decide(frame, state, _safe_zone_destination_test_config(), now=10.0)

    assert decision.action == "death_recovery_safe_zone_destination"
    assert decision.reason == "safe_zone_destination_button"
    assert decision.attempts == 1
    assert decision.click_point is not None


def test_death_recovery_fails_after_max_attempts():
    controller = DeathRecoveryController.from_config(_death_test_config(max_attempts=1, release_wait_seconds=0.0))
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (320, 95), (480, 125), (0, 0, 150), -1)
    cv2.rectangle(frame, (520, 95), (680, 125), (0, 0, 150), -1)
    state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=False)

    assert controller.decide(frame, state, _death_test_config(max_attempts=1, release_wait_seconds=0.0), now=10.0).action == (
        "death_recovery_release_spirit"
    )
    assert controller.decide(frame, state, _death_test_config(max_attempts=1, release_wait_seconds=0.0), now=11.0).action == (
        "death_recovery_failed"
    )


def test_death_recovery_resurrection_sickness_wait_is_consumed_once():
    controller = DeathRecoveryController.from_config(
        _death_test_config(release_wait_seconds=0.0, resurrect_sickness_wait_seconds=0.0)
    )
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (320, 95), (480, 125), (0, 0, 150), -1)
    cv2.rectangle(frame, (520, 95), (680, 125), (0, 0, 150), -1)
    death_state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=False)
    alive_state = GameState(death_or_blocking_modal=False, red_button_count=0, ghost_visual=False)

    controller.decide(frame, death_state, _death_test_config(release_wait_seconds=0.0), now=10.0)

    assert controller.consume_resurrected_wait(alive_state) == 0.0
    assert controller.consume_resurrected_wait(alive_state) is None


def test_death_recovery_resurrection_sickness_wait_after_safe_zone_click():
    controller = DeathRecoveryController.from_config(
        _safe_zone_test_config(safe_zone_resurrect_wait_seconds=0.0, resurrect_sickness_wait_seconds=0.0)
    )
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    yellow = (0, 190, 230)
    cv2.rectangle(frame, (610, 82), (735, 118), yellow, -1)
    ghost_state = GameState(death_or_blocking_modal=True, red_button_count=0, ghost_visual=True)
    alive_state = GameState(death_or_blocking_modal=False, red_button_count=0, ghost_visual=False)

    controller.decide(frame, ghost_state, _safe_zone_test_config(safe_zone_resurrect_wait_seconds=0.0), now=10.0)

    assert controller.consume_resurrected_wait(alive_state) == 0.0
    assert controller.consume_resurrected_wait(alive_state) is None


def test_death_recovery_resurrection_sickness_wait_after_safe_zone_destination_click():
    controller = DeathRecoveryController.from_config(
        _safe_zone_destination_test_config(safe_zone_destination_wait_seconds=0.0, resurrect_sickness_wait_seconds=0.0)
    )
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    red = (0, 0, 170)
    cv2.rectangle(frame, (360, 116), (520, 145), red, -1)
    ghost_state = GameState(death_or_blocking_modal=True, red_button_count=2, ghost_visual=True)
    alive_state = GameState(death_or_blocking_modal=False, red_button_count=0, ghost_visual=False)

    controller.decide(frame, ghost_state, _safe_zone_destination_test_config(), now=10.0)

    assert controller.consume_resurrected_wait(alive_state) == 0.0
    assert controller.consume_resurrected_wait(alive_state) is None


def test_resurrection_sickness_state_round_trip_and_expiry(tmp_path):
    state_path = tmp_path / "resurrection_sickness.json"

    write_resurrection_sickness_state(state_path, 60.0, now=1000.0)

    assert read_resurrection_sickness_remaining(state_path, now=1025.0) == 35.0
    assert read_resurrection_sickness_remaining(state_path, now=1061.0) is None
    assert not state_path.exists()


def test_clear_resurrection_sickness_state_ignores_missing_file(tmp_path):
    clear_resurrection_sickness_state(tmp_path / "missing.json")


def _death_test_config(**overrides):
    recovery = {
        "enabled": True,
        "max_attempts": 2,
        "release_wait_seconds": 0.0,
        "resurrect_sickness_wait_enabled": True,
        "resurrect_sickness_wait_seconds": 0.0,
        "release_button_region": {"x": 280, "y": 60, "width": 440, "height": 120},
        "release_button_min_area": 2000,
        "release_button_max_area": 7000,
        "release_button_min_width": 120,
        "release_button_max_width": 220,
        "release_button_min_height": 18,
        "release_button_max_height": 45,
        "release_button_center_min_x_fraction": 0.25,
        "release_button_center_max_x_fraction": 0.75,
        "release_button_center_min_y_fraction": 0.10,
        "release_button_center_max_y_fraction": 0.30,
        "fallback_click_enabled": False,
    }
    recovery.update(overrides)
    return {
        "screen": {"reference_width": 1000, "reference_height": 600, "scale_regions": False},
        "safety": {"death_recovery": recovery},
    }


def _safe_zone_test_config(**overrides):
    recovery = {
        "enabled": True,
        "resurrection_mode": "safe_zone",
        "max_attempts": 2,
        "release_wait_seconds": 0.0,
        "safe_zone_resurrect_enabled": True,
        "safe_zone_resurrect_max_attempts": 2,
        "safe_zone_resurrect_wait_seconds": 0.0,
        "safe_zone_destination_max_attempts": 2,
        "safe_zone_destination_wait_seconds": 0.0,
        "safe_zone_resurrect_button_region": {"x": 360, "y": 55, "width": 430, "height": 120},
        "safe_zone_resurrect_min_area": 900,
        "safe_zone_resurrect_max_area": 8000,
        "safe_zone_resurrect_min_width": 60,
        "safe_zone_resurrect_max_width": 180,
        "safe_zone_resurrect_min_height": 20,
        "safe_zone_resurrect_max_height": 60,
        "safe_zone_resurrect_center_min_x_fraction": 0.25,
        "safe_zone_resurrect_center_max_x_fraction": 0.80,
        "safe_zone_resurrect_center_min_y_fraction": 0.10,
        "safe_zone_resurrect_center_max_y_fraction": 0.30,
        "safe_zone_resurrect_fallback_click_enabled": False,
        "resurrect_sickness_wait_enabled": True,
        "resurrect_sickness_wait_seconds": 0.0,
        "fallback_click_enabled": False,
    }
    recovery.update(overrides)
    return {
        "screen": {"reference_width": 1000, "reference_height": 600, "scale_regions": False},
        "safety": {"death_recovery": recovery},
    }


def _safe_zone_destination_test_config(**overrides):
    config = _safe_zone_test_config(**overrides)
    recovery = config["safety"]["death_recovery"]
    recovery.update(
        {
            "release_button_region": {"x": 300, "y": 80, "width": 620, "height": 120},
            "release_button_min_area": 1800,
            "release_button_max_area": 7000,
            "release_button_min_width": 120,
            "release_button_max_width": 220,
            "release_button_min_height": 18,
            "release_button_max_height": 45,
            "release_button_center_min_x_fraction": 0.25,
            "release_button_center_max_x_fraction": 0.95,
            "release_button_center_min_y_fraction": 0.15,
            "release_button_center_max_y_fraction": 0.30,
        }
    )
    return config


def _spirit_healer_test_config(**overrides):
    recovery = {
        "enabled": True,
        "resurrection_mode": "spirit_healer",
        "max_attempts": 2,
        "release_wait_seconds": 0.0,
        "spirit_healer_enabled": True,
        "spirit_healer_max_attempts": 2,
        "spirit_healer_wait_seconds": 0.0,
        "spirit_healer_approach_max_attempts": 1,
        "spirit_healer_approach_wait_seconds": 0.0,
        "spirit_healer_region": {"x": 250, "y": 120, "width": 520, "height": 420},
        "spirit_healer_min_area": 1200,
        "spirit_healer_max_area": 80000,
        "spirit_healer_min_width": 35,
        "spirit_healer_max_width": 250,
        "spirit_healer_min_height": 120,
        "spirit_healer_max_height": 380,
        "spirit_healer_center_min_x_fraction": 0.15,
        "spirit_healer_center_max_x_fraction": 0.85,
        "spirit_healer_center_min_y_fraction": 0.15,
        "spirit_healer_center_max_y_fraction": 0.85,
        "spirit_healer_self_ignore_enabled": True,
        "return_to_life_max_attempts": 2,
        "return_to_life_wait_seconds": 0.0,
        "return_to_life_fallback_click_enabled": False,
        "accept_resurrection_max_attempts": 2,
        "accept_resurrection_wait_seconds": 0.0,
        "accept_button_region": {"x": 300, "y": 80, "width": 420, "height": 140},
        "accept_button_min_area": 1800,
        "accept_button_max_area": 7000,
        "accept_button_min_width": 120,
        "accept_button_max_width": 220,
        "accept_button_min_height": 18,
        "accept_button_max_height": 45,
        "accept_button_center_min_x_fraction": 0.25,
        "accept_button_center_max_x_fraction": 0.95,
        "accept_button_center_min_y_fraction": 0.15,
        "accept_button_center_max_y_fraction": 0.30,
        "resurrect_sickness_wait_enabled": True,
        "resurrect_sickness_wait_seconds": 0.0,
        "fallback_click_enabled": False,
    }
    recovery.update(overrides)
    return {
        "screen": {"reference_width": 1000, "reference_height": 600, "scale_regions": False},
        "safety": {"death_recovery": recovery},
    }


def _real_frame_config():
    return {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "safety": {
            "death_recovery": {
                "enabled": True,
                "resurrection_mode": "spirit_healer",
                "release_button_region": {"x": 640, "y": 70, "width": 1280, "height": 450},
                "spirit_healer_region": {"x": 75, "y": 260, "width": 2130, "height": 865},
                "fallback_click_enabled": False,
            },
        },
    }


def _real_safe_zone_frame_config():
    config = _real_frame_config()
    recovery = config["safety"]["death_recovery"]
    recovery.update(
        {
            "resurrection_mode": "safe_zone",
            "safe_zone_resurrect_enabled": True,
            "safe_zone_resurrect_max_attempts": 2,
            "safe_zone_resurrect_wait_seconds": 0.0,
            "safe_zone_destination_max_attempts": 4,
            "safe_zone_destination_wait_seconds": 0.0,
            "safe_zone_resurrect_button_region": {"x": 800, "y": 95, "width": 900, "height": 190},
        }
    )
    return config


def _live_spirit_healer_config():
    return {
        "screen": {"reference_width": 2560, "reference_height": 1440, "scale_regions": True},
        "safety": {
            "death_recovery": {
                "enabled": True,
                "resurrection_mode": "spirit_healer",
                "spirit_healer_region": {"x": 0, "y": 220, "width": 2205, "height": 920},
                "spirit_healer_min_area": 2000,
                "spirit_healer_max_area": 220000,
                "spirit_healer_min_width": 35,
                "spirit_healer_max_width": 900,
                "spirit_healer_min_height": 150,
                "spirit_healer_max_height": 850,
                "spirit_healer_max_saturation": 82,
                "spirit_healer_min_value": 182,
                "spirit_healer_center_min_x_fraction": 0.0,
                "spirit_healer_cluster_dx": 560,
                "spirit_healer_cluster_dy": 420,
                "spirit_healer_left_edge_click_x_fraction": 0.22,
                "spirit_healer_self_ignore_enabled": True,
                "spirit_healer_self_ignore_max_width": 260,
                "spirit_healer_self_ignore_max_height": 380,
                "spirit_healer_self_ignore_center_min_x_fraction": 0.38,
                "spirit_healer_self_ignore_center_max_x_fraction": 0.62,
                "spirit_healer_self_ignore_center_min_y_fraction": 0.42,
                "spirit_healer_self_ignore_center_max_y_fraction": 0.78,
                "fallback_click_enabled": False,
                "return_to_life_fallback_click_enabled": False,
            },
        },
    }
