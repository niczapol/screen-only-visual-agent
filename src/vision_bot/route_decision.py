from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RouteAction(str, Enum):
    WAIT_FOR_POSITION = "wait_for_position"
    MOVE_TO_LOAD_RADIUS = "move_to_load_radius"
    MARK_ABSENT = "mark_absent"
    MARK_PERMANENT_EXCLUDED = "mark_permanent_excluded"
    MOVE_TO_NODE = "move_to_node"
    HOLD_AT_NODE = "hold_at_node"
    READY_TO_MINE = "ready_to_mine"


@dataclass(frozen=True)
class RouteDecision:
    action: RouteAction
    reason: str


def decide_route_action(
    *,
    distance_to_target: float | None,
    ore_detected: bool,
    tracking_radius: float,
    reached_distance: float,
    mining_enabled: bool,
    dark_ore_detected: bool = False,
    dark_exclude_distance: float | None = None,
    dark_exclusion_enabled: bool = False,
) -> RouteDecision:
    if distance_to_target is None:
        return RouteDecision(RouteAction.WAIT_FOR_POSITION, "position_unknown")

    if distance_to_target > tracking_radius:
        return RouteDecision(RouteAction.MOVE_TO_LOAD_RADIUS, "outside_minimap_tracking_radius")

    if not dark_exclusion_enabled:
        dark_ore_detected = False

    dark_exclude_distance = reached_distance if dark_exclude_distance is None else dark_exclude_distance
    if dark_ore_detected and not ore_detected and distance_to_target <= dark_exclude_distance:
        return RouteDecision(RouteAction.MARK_PERMANENT_EXCLUDED, "dark_minimap_icon_inside_tracking_radius")

    if dark_ore_detected and not ore_detected:
        return RouteDecision(RouteAction.MOVE_TO_NODE, "dark_minimap_icon_visible_confirm_closer")

    if not ore_detected:
        return RouteDecision(RouteAction.MARK_ABSENT, "target_inside_tracking_radius_but_ore_absent")

    if distance_to_target > reached_distance:
        return RouteDecision(RouteAction.MOVE_TO_NODE, "ore_visible_on_minimap")

    if not mining_enabled:
        return RouteDecision(RouteAction.HOLD_AT_NODE, "mining_disabled")

    return RouteDecision(RouteAction.READY_TO_MINE, "ready_to_mine")
