from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vision_bot.combat import CombatState
from vision_bot.config import runtime_state_path
from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.game_state import GameState
from vision_bot.movement import MovementNavigator
from vision_bot.route_probe_combat import _maybe_invert_turn_key


def _should_handle_dismissable_modal(game_state: GameState, combat: CombatState) -> bool:
    return (
        game_state.dismissable_modal_click is not None
        and not game_state.death_or_blocking_modal
        and not combat.has_latching_evidence()
    )


def _should_handle_death_recovery(game_state: GameState, combat: CombatState) -> bool:
    if not game_state.death_or_blocking_modal:
        return False
    if (
        game_state.red_button_count >= 2
        or game_state.ghost_visual
        or game_state.ghost_button_count >= 2
    ):
        return True
    return not combat.has_latching_evidence()


def _turn_key_towards_click_point(
    click_point: tuple[int, int],
    frame_shape: tuple[int, ...],
    *,
    deadzone_fraction: float,
    invert_turn_direction: bool = False,
) -> str | None:
    frame_width = frame_shape[1]
    center_x = frame_width / 2.0
    deadzone_px = frame_width * max(0.0, float(deadzone_fraction))
    turn_key: str | None = None
    if click_point[0] < center_x - deadzone_px:
        turn_key = "A"
    elif click_point[0] > center_x + deadzone_px:
        turn_key = "D"
    return _maybe_invert_turn_key(turn_key, invert_turn_direction=invert_turn_direction)


def _spirit_healer_search_turn_key(value: Any) -> str:
    turn_key = str(value or "D").strip().upper()
    return turn_key if turn_key in {"A", "D"} else "D"


def _spirit_search_interact_due(
    action: str,
    *,
    now: float,
    next_at: float,
    interval: float,
) -> tuple[bool, float]:
    searching = action in {
        "death_recovery_navigate_to_spirit_anchor",
        "death_recovery_navigate_local_spirit_search",
    }
    if not searching or now < next_at:
        return False, next_at
    return True, now + max(0.05, interval)


def _choose_spirit_healer_coordinate_anchor(
    current_coord: int | None,
    config: dict[str, Any],
) -> tuple[str, int, float, float] | None:
    if current_coord is None:
        return None

    recovery_cfg = config.get("safety", {}).get("death_recovery", {})
    anchors = recovery_cfg.get("spirit_healer_coordinate_anchors", [])
    candidates: list[tuple[float, str, int, float]] = []
    for index, item in enumerate(anchors):
        if not isinstance(item, dict) or "coord" not in item:
            continue
        anchor_coord = int(item["coord"])
        distance = MovementNavigator.distance(current_coord, anchor_coord)
        activation_radius = max(0.0, float(item.get("activation_radius", 4.0)))
        if distance > activation_radius:
            continue
        name = str(item.get("name", f"anchor_{index}"))
        arrival_distance = max(0.0, float(item.get("arrival_distance", 0.35)))
        candidates.append((distance, name, anchor_coord, arrival_distance))

    if not candidates:
        return None
    distance, name, anchor_coord, arrival_distance = min(candidates, key=lambda item: item[0])
    return name, anchor_coord, distance, arrival_distance


def _spirit_healer_anchor_due(
    action: str,
    anchor: tuple[str, int, float, float] | None,
    *,
    search_attempts: int,
    after_attempts: int,
) -> bool:
    return bool(
        action == "death_recovery_search_spirit_healer"
        and anchor is not None
        # Public NPC coordinates are only an approximate graveyard-zone hint.
        # Once the visible character is inside that zone, hand control to the
        # bounded local search instead of trying to stand on one exact pixel.
        and anchor[2] > anchor[3]
        and search_attempts >= max(0, after_attempts)
    )


def _spirit_healer_local_search_coord(
    origin_coord: int,
    index: int,
    config: dict[str, Any],
) -> int:
    recovery_cfg = config.get("safety", {}).get("death_recovery", {})
    points_per_ring = max(4, int(recovery_cfg.get("spirit_healer_local_search_points_per_ring", 8)))
    start_radius = max(0.02, float(recovery_cfg.get("spirit_healer_local_search_start_radius", 0.18)))
    ring_step = max(0.0, float(recovery_cfg.get("spirit_healer_local_search_ring_step", 0.10)))
    max_radius = max(start_radius, float(recovery_cfg.get("spirit_healer_local_search_max_radius", 0.55)))

    normalized_index = max(0, int(index))
    return_to_origin = bool(
        recovery_cfg.get("spirit_healer_local_search_return_to_origin", True)
    )
    if return_to_origin and normalized_index % 2 == 1:
        return origin_coord
    spoke_index = normalized_index // 2 if return_to_origin else normalized_index
    ring_index = spoke_index // points_per_ring
    point_index = spoke_index % points_per_ring
    radius = min(max_radius, start_radius + ring_index * ring_step)
    phase = (ring_index % 2) * (math.pi / points_per_ring)
    angle = phase + (2.0 * math.pi * point_index / points_per_ring)
    origin_x, origin_y = coord_to_xy(origin_coord)
    return xy_to_coord(
        origin_x + radius * math.cos(angle),
        origin_y + radius * math.sin(angle),
    )


def _resurrection_sickness_state_path(config: dict[str, Any]) -> Path:
    recovery_cfg = config.get("safety", {}).get("death_recovery", {})
    return runtime_state_path(
        str(recovery_cfg.get("resurrection_sickness_state_path", "data/runtime/resurrection_sickness.json"))
    )
