from __future__ import annotations

import time
from typing import Any

from vision_bot.config import resource_path, runtime_state_path
from vision_bot.movement import MovementNavigator
from vision_bot.permanent_exclusions import load_permanent_exclusions
from vision_bot.route_planner import (
    MiningNode,
    ROUTE_ENTRY_WAYPOINT_SOURCE,
    ROUTE_WAYPOINT_SOURCE,
    is_route_waypoint,
    load_mining_nodes,
    load_nodes,
    load_route_waypoints,
)
from vision_bot.route_probe_routing import (
    allowed_route_locations as _allowed_route_locations,
    load_configured_route_entry as _load_configured_route_entry,
    route_entry_strategy as _route_entry_strategy,
    route_sort_key as _route_sort_key,
    single_allowed_route_zone as _single_allowed_route_zone,
)


def _load_route_nodes(
    route_cfg: dict[str, Any],
    *,
    resource_resolver: Any = resource_path,
    nodes_loader: Any = load_nodes,
    mining_loader: Any = load_mining_nodes,
    waypoints_loader: Any = load_route_waypoints,
) -> list[MiningNode]:
    if route_cfg.get("database_path"):
        source_path = resource_resolver(str(route_cfg["database_path"]))
        if bool(route_cfg.get("follow_route_loop", False)) and source_path.suffix.lower() == ".json":
            return waypoints_loader(source_path)
        return nodes_loader(source_path)
    return mining_loader(resource_resolver("data/MiningData.lua"))


def _load_mining_candidate_nodes(
    route_cfg: dict[str, Any],
    *,
    resource_resolver: Any = resource_path,
    nodes_loader: Any = load_nodes,
    mining_loader: Any = load_mining_nodes,
    state_path_resolver: Any = runtime_state_path,
    exclusions_loader: Any = load_permanent_exclusions,
) -> list[MiningNode]:
    """Load historical ore coordinates even when route-live follows dense waypoints."""
    permanent_exclusions_path = state_path_resolver(
        str(route_cfg.get("permanent_exclusions_path", "data/permanent_exclusions.json"))
    )
    permanent_exclusions = exclusions_loader(permanent_exclusions_path)
    if route_cfg.get("database_path"):
        return [
            node
            for node in nodes_loader(resource_resolver(str(route_cfg["database_path"])))
            if not is_route_waypoint(node) and node.coord not in permanent_exclusions
        ]
    return [
        node
        for node in mining_loader(resource_resolver("data/MiningData.lua"))
        if node.coord not in permanent_exclusions
    ]


def _choose_route_target(
    config: dict[str, Any],
    current_coord: int,
    *,
    target_coord: int | None,
    zone_id: int | None,
    auto_zone: bool = False,
    min_target_distance: float = 0.0,
    excluded_coords: set[int] | None = None,
    auto_zone_ids: set[int] | None = None,
    route_after_sort_key: float | None = None,
    manual_source: str = "manual",
    manual_ore_type: str = "manual",
    manual_route_index: int | None = None,
    manual_route_t: float | None = None,
    route_nodes_loader: Any = None,
    state_path_resolver: Any = runtime_state_path,
    exclusions_loader: Any = load_permanent_exclusions,
) -> MiningNode | None:
    route_cfg = config.get("route", {})
    nodes = (route_nodes_loader or _load_route_nodes)(route_cfg)
    permanent_exclusions_path = state_path_resolver(
        str(route_cfg.get("permanent_exclusions_path", "data/permanent_exclusions.json"))
    )
    permanent_exclusions = exclusions_loader(permanent_exclusions_path)

    if target_coord is not None:
        matching = [node for node in nodes if node.coord == target_coord]
        if matching:
            return matching[0]
        return MiningNode(
            zone_id=zone_id or _single_allowed_route_zone(route_cfg) or 0,
            coord=target_coord,
            ore_type=manual_ore_type,
            node_id=None,
            route_index=manual_route_index,
            route_t=manual_route_t,
            source=manual_source,
        )

    allowed_locations = _allowed_route_locations(
        route_cfg,
        zone_id=zone_id,
        auto_zone=auto_zone,
        auto_zone_ids=auto_zone_ids,
    )
    allowed_ores = set(route_cfg.get("ores", []))
    temporary_exclusions = set(excluded_coords or set())
    candidates = [
        node
        for node in nodes
        if (not allowed_locations or node.zone_id in allowed_locations)
        and (not allowed_ores or node.ore_type in allowed_ores)
        and node.coord not in permanent_exclusions
        and node.coord not in temporary_exclusions
    ]
    if not candidates:
        return None
    if min_target_distance > 0:
        distant_candidates = [
            node for node in candidates if MovementNavigator.distance(current_coord, node.coord) >= min_target_distance
        ]
        if distant_candidates:
            candidates = distant_candidates
    if route_after_sort_key is not None:
        routed_candidates = [node for node in candidates if _route_sort_key(node) is not None]
        if routed_candidates:
            ordered = sorted(
                routed_candidates,
                key=lambda node: (
                    _route_sort_key(node) or 0.0,
                    MovementNavigator.distance(current_coord, node.coord),
                ),
            )
            return next(
                (node for node in ordered if (_route_sort_key(node) or 0.0) >= route_after_sort_key),
                ordered[0],
            )
    return min(candidates, key=lambda node: MovementNavigator.distance(current_coord, node.coord))


def _route_reference_coords(
    config: dict[str, Any],
    *,
    zone_id: int | None,
    target_coord: int | None,
    auto_zone: bool = False,
    auto_zone_ids: set[int] | None = None,
    route_nodes_loader: Any = None,
) -> list[int]:
    if target_coord is not None:
        return [target_coord]

    route_cfg = config.get("route", {})
    allowed_locations = _allowed_route_locations(
        route_cfg,
        zone_id=zone_id,
        auto_zone=auto_zone,
        auto_zone_ids=auto_zone_ids,
    )
    allowed_ores = set(route_cfg.get("ores", []))
    references = [
        node.coord
        for node in (route_nodes_loader or _load_route_nodes)(route_cfg)
        if (not allowed_locations or node.zone_id in allowed_locations)
        and (not allowed_ores or node.ore_type in allowed_ores)
    ]
    entry_plan = (
        _load_configured_route_entry(config)
        if _route_entry_strategy(config) == "validated_path"
        else None
    )
    if entry_plan is not None:
        references = [node.coord for node in entry_plan.waypoints] + references
    return references


def _combined_excluded_coords(*coord_groups: set[int] | None) -> set[int]:
    combined: set[int] = set()
    for coords in coord_groups:
        if coords:
            combined.update(coords)
    return combined


def _block_route_waypoint_window(
    blocked_coords: set[int],
    *,
    target_node: MiningNode,
    route_loop: list[int],
    forward_padding: int,
) -> set[int]:
    blocked_coords.add(target_node.coord)
    if (
        target_node.source == ROUTE_ENTRY_WAYPOINT_SOURCE
        or target_node.route_index is None
        or not route_loop
    ):
        return blocked_coords
    route_index = int(target_node.route_index) % len(route_loop)
    for offset in range(max(0, forward_padding) + 1):
        blocked_coords.add(route_loop[(route_index + offset) % len(route_loop)])
    return blocked_coords


def _is_recovery_action(action: str) -> bool:
    return action.startswith("recover_")


def _record_target_recovery(
    action: str,
    *,
    target_recovery_count: int,
    target_recovery_limit: int,
    distance_progress: float | None = None,
    coord_delta: float | None = None,
    stuck_min_progress: float = 0.0,
    stuck_min_coord_delta: float = 0.0,
) -> tuple[int, bool]:
    if _target_progress_is_good(
        distance_progress=distance_progress,
        coord_delta=coord_delta,
        stuck_min_progress=stuck_min_progress,
        stuck_min_coord_delta=stuck_min_coord_delta,
    ):
        return 0, False

    if not _is_recovery_action(action):
        return target_recovery_count, False

    next_count = target_recovery_count + 1
    if target_recovery_limit <= 0:
        return next_count, False
    return next_count, next_count >= target_recovery_limit


def _target_recovery_can_block(
    *,
    directed_route_active: bool,
    target_source: str,
) -> bool:
    """Keep local recovery from consuming directed-route waypoints.

    A directed route follower owns progression along the normal route loop. Its
    local obstacle recovery may temporarily move away from the polyline, but it
    must not translate that lack of immediate progress into a blocked waypoint
    window. Explicit navmesh guidance failures are handled separately by the
    caller and remain blockable.
    """

    return not (
        directed_route_active and target_source == ROUTE_WAYPOINT_SOURCE
    )


def _target_progress_is_good(
    *,
    distance_progress: float | None,
    coord_delta: float | None,
    stuck_min_progress: float,
    stuck_min_coord_delta: float,
) -> bool:
    min_progress = max(0.0, stuck_min_progress)
    min_delta = max(0.0, stuck_min_coord_delta)
    if distance_progress is not None and distance_progress >= min_progress:
        return True
    if coord_delta is None or coord_delta < min_delta:
        return False
    if distance_progress is None:
        return True
    return distance_progress >= -min_progress


def _target_history_item(
    node: MiningNode,
    target_coord: int,
    reached_coord: int,
    distance: float,
    *,
    ore_training_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "target_coord": target_coord,
        "target_node_id": node.node_id,
        "target_zone_id": node.zone_id,
        "target_ore_type": node.ore_type,
        "route_index": node.route_index,
        "source": node.source,
        "reached_coord": reached_coord,
        "distance": distance,
        "timestamp": time.time(),
    }
    if ore_training_capture is not None:
        item["ore_training_capture"] = ore_training_capture
    return item


def _record_target_route_node_milestone(
    history: list[dict[str, Any]],
    completed_indices: set[int],
    route_milestones: list[MiningNode],
    *,
    target_node: MiningNode,
    reached_coord: int,
    distance: float,
) -> None:
    if target_node.source != ROUTE_WAYPOINT_SOURCE or target_node.route_index is None:
        return
    _record_route_node_milestone(
        history,
        completed_indices,
        route_milestones,
        route_index=target_node.route_index,
        reached_coord=reached_coord,
        distance=distance,
    )


def _record_route_node_milestone(
    history: list[dict[str, Any]],
    completed_indices: set[int],
    route_milestones: list[MiningNode],
    *,
    route_index: int,
    reached_coord: int,
    distance: float,
) -> None:
    if route_index in completed_indices:
        return
    milestone = next(
        (node for node in route_milestones if node.route_index == route_index),
        None,
    )
    if milestone is None:
        return
    completed_indices.add(route_index)
    history.append(
        {
            "route_index": route_index,
            "node_id": milestone.node_id,
            "coord": milestone.coord,
            "ore_type": milestone.ore_type,
            "reached_coord": reached_coord,
            "distance": distance,
            "timestamp": time.time(),
        }
    )


def _target_skipped_item(
    node: MiningNode,
    target_coord: int,
    current_coord: int | None,
    distance: float | None,
    *,
    recovery_count: int,
    reason: str = "target_recovery_limit",
) -> dict[str, Any]:
    return {
        "target_coord": target_coord,
        "target_node_id": node.node_id,
        "target_zone_id": node.zone_id,
        "target_ore_type": node.ore_type,
        "route_index": node.route_index,
        "source": node.source,
        "current_coord": current_coord,
        "distance": distance,
        "recovery_count": recovery_count,
        "reason": reason,
        "timestamp": time.time(),
    }
