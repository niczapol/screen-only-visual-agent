import numpy as np

from vision_bot.live_navigation import (
    _movement_was_stuck,
    _visual_motion_delta,
    choose_live_navigation_action,
)
from vision_bot.local_navigation import NavigationFrame
from vision_bot.screen_objects import BoundingBox


def _navigation(*, center_blocked: bool, recommended_turn: str | None) -> NavigationFrame:
    return NavigationFrame(
        region=BoundingBox(0, 0, 100, 100),
        analysis_region=BoundingBox(0, 20, 100, 60),
        self_ignore_region=None,
        sectors=tuple(),
        best_sector="center",
        center_blocked=center_blocked,
        recommended_turn=recommended_turn,
        confidence=0.8,
    )


def test_choose_live_navigation_action_moves_forward_when_center_available():
    action = choose_live_navigation_action(
        _navigation(center_blocked=False, recommended_turn=None),
        stuck_samples=0,
        stuck_window=3,
        alternate_turn_key="D",
        forward_duration=0.35,
        turn_duration=0.20,
        back_duration=0.25,
    )

    assert action.name == "forward"
    assert action.key == "W"
    assert action.reason == "center_available"


def test_choose_live_navigation_action_turns_when_center_blocked():
    action = choose_live_navigation_action(
        _navigation(center_blocked=True, recommended_turn="A"),
        stuck_samples=0,
        stuck_window=3,
        alternate_turn_key="D",
        forward_duration=0.35,
        turn_duration=0.20,
        back_duration=0.25,
    )

    assert action.name == "turn_for_blocker"
    assert action.key == "A"


def test_choose_live_navigation_action_recovers_before_turning_when_stuck():
    action = choose_live_navigation_action(
        _navigation(center_blocked=True, recommended_turn="A"),
        stuck_samples=3,
        stuck_window=3,
        alternate_turn_key="D",
        forward_duration=0.35,
        turn_duration=0.20,
        back_duration=0.25,
    )

    assert action.name == "recover_back"
    assert action.key == "S"
    assert action.reason == "stuck_samples"


def test_movement_fallback_uses_visual_motion_when_coordinates_are_missing():
    assert not _movement_was_stuck(
        coord_delta=None,
        visual_motion_delta=12.0,
        stuck_min_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_movement_fallback_treats_low_visual_motion_as_stuck():
    assert _movement_was_stuck(
        coord_delta=None,
        visual_motion_delta=0.8,
        stuck_min_delta=0.015,
        visual_stuck_min_delta=3.0,
    )


def test_visual_motion_delta_compares_central_gameplay_area():
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[30:60, 30:60] = 120

    assert _visual_motion_delta(before, after) > 0.0
