from __future__ import annotations

from vision_bot.live_navigation import _movement_was_stuck
from vision_bot.movement import InputController, MovementNavigator
from vision_bot.threat_detection import ThreatDetection


def _should_local_avoid(
    center_blocked: bool,
    *,
    proactive: bool = False,
    distance_progress: float | None,
    coord_delta: float | None,
    visual_motion_delta: float | None,
    stuck_min_progress: float,
    stuck_min_coord_delta: float,
    visual_stuck_min_delta: float,
) -> bool:
    if not center_blocked:
        return False
    if proactive:
        return True
    if coord_delta is not None and coord_delta >= stuck_min_coord_delta:
        return False
    if distance_progress is not None:
        return distance_progress < stuck_min_progress
    return _movement_was_stuck(
        coord_delta=coord_delta,
        visual_motion_delta=visual_motion_delta,
        stuck_min_delta=stuck_min_coord_delta,
        visual_stuck_min_delta=visual_stuck_min_delta,
    )


def _advance_route_stuck_samples(
    samples: int,
    *,
    coord_delta: float | None,
    movement_stuck: bool,
) -> int:
    """Count consecutive fresh zero-progress samples without skipped-tick resets."""
    if coord_delta is None:
        return max(0, int(samples))
    if movement_stuck:
        return max(0, int(samples)) + 1
    return 0


def _is_threat_action(action: str) -> bool:
    return action.startswith("avoid_hostile")


def _is_combat_action(action: str) -> bool:
    return action.startswith("combat_")


def _is_recovery_action(action: str) -> bool:
    """Return whether a movement diagnostic represents a stuck recovery step."""
    return action.startswith("recover_")


def _apply_hostile_avoidance(
    navigator: MovementNavigator,
    input_controller: InputController,
    current_coord: int | None,
    target_coord: int,
    threat: ThreatDetection,
    turn_key: str,
    turn_duration: float,
) -> tuple[str, str]:
    navigator.continue_forward(current_coord, target_coord)
    navigator.turn_character(
        turn_key,
        duration=turn_duration,
        context="hostile",
    )
    if threat.reason == "hostile_target_frame":
        return "avoid_hostile_target", turn_key
    return "avoid_hostile_threat", turn_key


def _route_metadata_action(base_action: str, movement_action: str) -> str:
    if base_action in {
        "death_or_blocking_modal",
        "reached",
        "cycle_no_target",
        "no_target",
        "route_entry_no_target",
    }:
        return base_action
    if base_action.startswith("route_entry_"):
        return base_action
    if base_action.startswith("target_blocked_"):
        return base_action
    if base_action.startswith("death_recovery_"):
        return base_action
    if _is_combat_action(base_action):
        return base_action
    if _is_threat_action(base_action):
        return base_action
    if (
        movement_action.startswith("recover_")
        or movement_action.startswith("turn_")
        or movement_action.startswith("course_correct")
    ):
        return movement_action
    return base_action
