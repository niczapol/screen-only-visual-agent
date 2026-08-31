from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vision_bot.game_state import detect_game_state


def test_detect_game_state_finds_top_center_red_modal_buttons():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (320, 95), (480, 125), (0, 0, 150), -1)
    cv2.rectangle(frame, (520, 95), (680, 125), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert state.death_or_blocking_modal
    assert state.red_button_count == 2


def test_detect_game_state_ignores_empty_frame():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert state.red_button_count == 0
    assert not state.ghost_visual


def test_detect_game_state_ignores_small_red_ui_or_texture_fragments():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (420, 220), (520, 235), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert state.red_button_count == 0


def test_detect_game_state_ignores_red_mob_health_bar_shape():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (380, 190), (620, 209), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert state.red_button_count == 0


def test_detect_game_state_ignores_single_red_component_without_ghost_visual():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (420, 95), (580, 125), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert state.red_button_count == 1


def test_detect_game_state_finds_desaturated_ghost_visual():
    ghost_hsv = np.full((600, 1000, 3), (105, 88, 150), dtype=np.uint8)
    frame = cv2.cvtColor(ghost_hsv, cv2.COLOR_HSV2BGR)

    state = detect_game_state(frame)

    assert state.death_or_blocking_modal
    assert state.ghost_visual


def test_detect_game_state_alive_health_bar_vetoes_sandy_ghost_palette():
    sandy_hsv = np.full((600, 1000, 3), (105, 88, 150), dtype=np.uint8)
    frame = cv2.cvtColor(sandy_hsv, cv2.COLOR_HSV2BGR)
    left = round(frame.shape[1] * (155.0 / 2560.0))
    top = round(frame.shape[0] * (70.0 / 1440.0))
    right = round(frame.shape[1] * (345.0 / 2560.0))
    bottom = round(frame.shape[0] * (95.0 / 1440.0))
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 180, 0), -1)

    state = detect_game_state(frame)

    assert not state.ghost_visual
    assert state.alive_health_bar
    assert not state.death_or_blocking_modal


def test_detect_game_state_keeps_critical_live_tanaris_frame_alive_if_available():
    frame_path = Path(
        "data/live_overnight_20260804_035134/01_core_route_combat_mining/frames/0450.png"
    )
    if not frame_path.exists():
        return
    frame = cv2.imread(str(frame_path))
    assert frame is not None

    state = detect_game_state(frame)

    assert state.ghost_visual
    assert state.alive_health_bar
    assert not state.death_or_blocking_modal


def test_detect_game_state_recognizes_following_real_death_frame_if_available():
    frame_path = Path(
        "data/live_overnight_20260804_035134/01_core_route_combat_mining/frames/0451.png"
    )
    if not frame_path.exists():
        return
    frame = cv2.imread(str(frame_path))
    assert frame is not None

    state = detect_game_state(frame)

    assert not state.alive_health_bar
    assert state.red_button_count >= 2
    assert state.death_or_blocking_modal


def test_detect_game_state_desaturated_empty_health_bar_does_not_veto_ghost():
    ghost_hsv = np.full((600, 1000, 3), (105, 88, 150), dtype=np.uint8)
    frame = cv2.cvtColor(ghost_hsv, cv2.COLOR_HSV2BGR)
    left = round(frame.shape[1] * (155.0 / 2560.0))
    top = round(frame.shape[0] * (70.0 / 1440.0))
    right = round(frame.shape[1] * (345.0 / 2560.0))
    bottom = round(frame.shape[0] * (95.0 / 1440.0))
    empty_bar_hsv = np.full((bottom - top, right - left, 3), (70, 75, 95), dtype=np.uint8)
    frame[top:bottom, left:right] = cv2.cvtColor(empty_bar_hsv, cv2.COLOR_HSV2BGR)

    state = detect_game_state(frame)

    assert state.ghost_visual
    assert not state.alive_health_bar
    assert state.death_or_blocking_modal


def test_detect_game_state_alive_health_bar_vetoes_false_ghost_buttons():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)

    with (
        patch("vision_bot.game_state._detect_alive_player_health_bar", return_value=True),
        patch("vision_bot.game_state._count_ghost_action_buttons", return_value=2) as buttons,
    ):
        state = detect_game_state(frame)

    assert state.alive_health_bar
    assert state.ghost_button_count == 0
    buttons.assert_not_called()
    assert not state.death_or_blocking_modal


def test_detect_game_state_finds_top_center_ghost_action_buttons_if_available():
    frame = cv2.imread("debug_output/live_status_20260729_after_spirit_recovery_smoke/current.png")
    if frame is None:
        return

    state = detect_game_state(frame)

    assert state.death_or_blocking_modal
    assert state.ghost_button_count >= 2


def test_detect_game_state_ignores_dark_desaturated_world_frame():
    frame = np.full((600, 1000, 3), (60, 55, 45), dtype=np.uint8)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert not state.ghost_visual


def test_detect_game_state_ignores_bottom_startup_popup_button_shape():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (610, 515), (760, 547), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert not state.death_or_blocking_modal
    assert state.dismissable_modal_click is None


def test_detect_game_state_ignores_bottom_square_action_button():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (610, 470), (650, 510), (0, 0, 150), -1)

    state = detect_game_state(frame)

    assert state.dismissable_modal_click is None


def test_detect_game_state_ignores_real_action_bar_false_positive():
    frame = cv2.imread(
        "data/live_route_probe_combat_turn_commit_20260731_1630/frames/0049.png"
    )
    if frame is None:
        return

    state = detect_game_state(frame)

    assert state.dismissable_modal_click is None


def test_detect_game_state_ignores_real_startup_popup():
    frame = cv2.imread("debug_output/after_popup_close_world.png")
    if frame is None:
        return

    state = detect_game_state(frame)

    assert state.dismissable_modal_click is None
