from __future__ import annotations

from typing import Any

import numpy as np

from vision_bot.movement import (
    InputController,
    MouseSteeringController,
    MovementNavigator,
)
from vision_bot.position import read_player_position_candidates
from vision_bot.position_filter import (
    CoordinateResyncState,
    choose_best_coord_candidate,
    choose_best_coord_candidate_with_resync,
)


def _read_filtered_coord(
    frame: np.ndarray,
    config: dict[str, Any],
    *,
    previous_coord: int | None,
    reference_coords: list[int],
    resync_state: CoordinateResyncState,
    max_jump: float | None = None,
    allow_distant_resync: bool = False,
) -> tuple[int | None, list[int], bool]:
    candidates = read_player_position_candidates(frame, config)
    coord, resynced = choose_best_coord_candidate_with_resync(
        previous_coord,
        candidates,
        (
            float(config.get("movement", {}).get("max_coord_jump_per_poll", 1.5))
            if max_jump is None
            else max(0.0, max_jump)
        ),
        reference_coords,
        resync_state,
        int(config.get("movement", {}).get("coord_resync_confirm_steps", 3)),
        (
            None
            if allow_distant_resync
            else float(config.get("movement", {}).get("coord_resync_max_distance", 5.0))
        ),
    )
    return coord, candidates, resynced


def _route_coord_jump_limit(
    config: dict[str, Any],
    *,
    elapsed_seconds: float,
    mounted: bool,
) -> float:
    movement_cfg = config.get("movement", {})
    base_limit = max(0.0, float(movement_cfg.get("max_coord_jump_per_poll", 1.5)))
    speed_key = "mounted_coord_units_per_second" if mounted else "foot_coord_units_per_second"
    speed_limit = max(0.0, float(movement_cfg.get(speed_key, 0.75 if mounted else 0.45)))
    timing_margin = max(0.0, float(movement_cfg.get("coord_jump_timing_margin", 0.25)))
    return max(
        base_limit,
        timing_margin + speed_limit * max(0.0, elapsed_seconds),
    )


def _select_stable_start_coord(
    candidate_sets: list[list[int]],
    *,
    reference_coords: list[int],
    max_cluster_distance: float,
    max_jump: float,
) -> int | None:
    primary_candidates = [candidates[0] for candidates in candidate_sets if candidates]
    if not primary_candidates:
        return None

    required_support = 1 if len(primary_candidates) == 1 else 2
    all_candidates = list(
        dict.fromkeys(candidate for candidates in candidate_sets for candidate in candidates)
    )
    stable_candidates: list[tuple[float, int, float, int]] = []
    for candidate in all_candidates:
        per_frame_distances = [
            min(MovementNavigator.distance(candidate, other) for other in candidates)
            for candidates in candidate_sets
            if candidates
        ]
        supporting = [
            distance
            for distance in per_frame_distances
            if distance <= max(0.0, max_cluster_distance)
        ]
        if len(supporting) < required_support:
            continue
        reference_distance = (
            min(MovementNavigator.distance(candidate, reference) for reference in reference_coords)
            if reference_coords
            else 0.0
        )
        stable_candidates.append(
            (reference_distance, -len(supporting), sum(supporting), candidate)
        )
    if stable_candidates:
        return min(stable_candidates)[3]

    scored: list[tuple[int, float, int]] = []
    for candidate in primary_candidates:
        distances = [MovementNavigator.distance(candidate, other) for other in primary_candidates]
        supporting = [
            distance
            for distance in distances
            if distance <= max(0.0, max_cluster_distance)
        ]
        scored.append((len(supporting), sum(supporting), candidate))
    support, _spread, stable_candidate = min(
        scored,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    if support >= required_support:
        return stable_candidate

    return choose_best_coord_candidate(
        None,
        all_candidates,
        max_jump,
        reference_coords,
    )


def _should_read_route_coord(
    *,
    now: float,
    last_coord_read_at: float,
    coord_read_interval: float,
    index: int,
    held_key: str | None,
    reached: bool,
    combat_engaged: bool,
    combat_heal_casting: bool,
) -> bool:
    if combat_engaged or combat_heal_casting:
        return False
    if index == 0 and held_key != "W" and not reached:
        return True
    return now - last_coord_read_at >= coord_read_interval


def _create_navigator(
    input_controller: InputController,
    config: dict[str, Any],
    *,
    mouse_steering: MouseSteeringController | None = None,
) -> MovementNavigator:
    movement_cfg = config.get("movement", {})
    return MovementNavigator(
        input_controller,
        stuck_check_window=int(movement_cfg.get("stuck_check_window", 4)),
        stuck_min_progress=float(movement_cfg.get("stuck_min_progress", 0.03)),
        stuck_min_coord_delta=float(
            movement_cfg.get("stuck_min_coord_delta", movement_cfg.get("stuck_min_progress", 0.03))
        ),
        obstacle_back_duration=float(movement_cfg.get("obstacle_back_duration", 0.35)),
        obstacle_strafe_duration=float(movement_cfg.get("obstacle_strafe_duration", 0.55)),
        obstacle_turn_duration=float(
            movement_cfg.get(
                "obstacle_turn_duration",
                movement_cfg.get("obstacle_strafe_duration", 0.55),
            )
        ),
        obstacle_jump_duration=float(movement_cfg.get("obstacle_jump_duration", 0.08)),
        obstacle_jump_first=bool(movement_cfg.get("obstacle_jump_first", True)),
        obstacle_detour_duration=float(movement_cfg.get("obstacle_detour_duration", 0.35)),
        obstacle_detour_max_duration=float(
            movement_cfg.get("obstacle_detour_max_duration", 0.65)
        ),
        obstacle_detour_settle_duration=float(
            movement_cfg.get("obstacle_detour_settle_duration", 0.0)
        ),
        obstacle_recovery_attempts_per_side=int(
            movement_cfg.get("obstacle_recovery_attempts_per_side", 4)
        ),
        turn_duration=float(movement_cfg.get("turn_duration", 0.12)),
        turn_in_place_duration=float(
            movement_cfg.get(
                "turn_in_place_duration",
                movement_cfg.get("turn_duration", 0.12),
            )
        ),
        turn_in_place_alignment=float(movement_cfg.get("turn_in_place_alignment", -0.2)),
        turn_alignment_threshold=float(movement_cfg.get("turn_alignment_threshold", 0.92)),
        invert_turn_direction=bool(movement_cfg.get("invert_turn_direction", False)),
        mouse_steering=mouse_steering,
    )
