from __future__ import annotations

import json
import math
from typing import Any

from vision_bot.config import resource_path, runtime_state_path
from vision_bot.mining import MiningOutcome
from vision_bot.movement import MovementNavigator
from vision_bot.permanent_exclusions import load_permanent_exclusions
from vision_bot.route_database import (
    load_route_loop_coords,
    project_to_loop,
    project_to_path,
)
from vision_bot.route_following import DirectedRouteFollower, DirectedRouteObservation
from vision_bot.route_planner import (
    MiningNode,
    ROUTE_WAYPOINT_SOURCE,
    RouteEntryPlan,
    is_route_waypoint,
    load_nodes,
    load_route_entry_plan,
)


def should_complete_hazard_egress(
    *,
    hazard_egress_active: bool,
    coord_is_hazard: bool,
    forbidden_subzone_visible: bool,
) -> bool:
    """Keep egress authority until both geometry and the visible addon marker are safe."""

    return bool(
        hazard_egress_active
        and not coord_is_hazard
        and not forbidden_subzone_visible
    )


def should_allow_forbidden_marker_during_hazard_egress(
    *,
    hazard_egress_active: bool,
    route_entry_pending: bool,
    planner_available: bool,
) -> bool:
    """Allow a forbidden marker only while a validated egress queue still owns movement."""

    return bool(hazard_egress_active and route_entry_pending and planner_available)


def load_configured_route_loop(config: dict[str, Any]) -> list[int]:
    route_cfg = config.get("route", {})
    database_path = route_cfg.get("database_path")
    if not database_path:
        return []
    source_path = resource_path(str(database_path))
    if source_path.suffix.lower() != ".json" or not source_path.exists():
        return []
    try:
        loop_coords = load_route_loop_coords(source_path)
        permanent_exclusions_path = runtime_state_path(
            str(
                route_cfg.get(
                    "permanent_exclusions_path",
                    "data/permanent_exclusions.json",
                )
            )
        )
        permanent_exclusions = load_permanent_exclusions(permanent_exclusions_path)
        return [coord for coord in loop_coords if coord not in permanent_exclusions]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def load_configured_route_milestones(config: dict[str, Any]) -> list[MiningNode]:
    route_cfg = config.get("route", {})
    database_path = route_cfg.get("database_path")
    if not database_path:
        return []
    source_path = resource_path(str(database_path))
    if source_path.suffix.lower() != ".json" or not source_path.exists():
        return []
    try:
        return load_nodes(source_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def load_configured_route_entry(config: dict[str, Any]) -> RouteEntryPlan | None:
    route_cfg = config.get("route", {})
    database_path = route_cfg.get("database_path")
    if not database_path:
        return None
    source_path = resource_path(str(database_path))
    if source_path.suffix.lower() != ".json" or not source_path.exists():
        return None
    try:
        return load_route_entry_plan(source_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def route_entry_strategy(config: dict[str, Any]) -> str:
    entry_cfg = config.get("route", {}).get("entry", {})
    if not isinstance(entry_cfg, dict):
        return "autonomous_local"
    strategy = str(entry_cfg.get("strategy", "autonomous_local")).strip().lower()
    return (
        strategy
        if strategy in {"autonomous_local", "validated_path", "navmesh_dynamic"}
        else "autonomous_local"
    )


def build_directed_route_follower(
    config: dict[str, Any],
    route_loop: list[int],
) -> DirectedRouteFollower | None:
    polyline_cfg = config.get("route", {}).get("polyline_following", {})
    if not isinstance(polyline_cfg, dict):
        return None
    if not bool(polyline_cfg.get("enabled", False)) or len(route_loop) < 2:
        return None
    max_forward_advance = float(polyline_cfg.get("max_forward_advance", 24.0))
    override_items = polyline_cfg.get("lookahead_overrides", [])
    lookahead_overrides: list[tuple[int, int, float]] = []
    if isinstance(override_items, list):
        for item in override_items:
            if not isinstance(item, dict):
                continue
            if not all(key in item for key in ("start_index", "end_index", "distance")):
                continue
            lookahead_overrides.append(
                (
                    int(item["start_index"]),
                    int(item["end_index"]),
                    float(item["distance"]),
                )
            )
    return DirectedRouteFollower(
        route_loop,
        lookahead_distance=float(polyline_cfg.get("lookahead_distance", 0.85)),
        corridor_radius=float(polyline_cfg.get("corridor_radius", 1.50)),
        backward_tolerance=float(polyline_cfg.get("backward_tolerance", 0.25)),
        max_forward_advance=max_forward_advance,
        pass_tolerance=float(polyline_cfg.get("pass_tolerance", 0.05)),
        projection_search_backward_segments=int(
            polyline_cfg.get("projection_search_backward_segments", 3)
        ),
        projection_search_forward_segments=int(
            polyline_cfg.get(
                "projection_search_forward_segments",
                int(math.ceil(max_forward_advance)) + 4,
            )
        ),
        lookahead_overrides=tuple(lookahead_overrides),
    )


def directed_route_steering_target(
    follower: DirectedRouteFollower | None,
    *,
    current_coord: int,
    target_node: MiningNode,
    route_entry_pending: bool,
    suspended: bool = False,
) -> tuple[int, bool, DirectedRouteObservation | None]:
    if (
        follower is None
        or route_entry_pending
        or suspended
        or not is_route_waypoint(target_node)
    ):
        observation = follower.last_observation if follower is not None else None
        return target_node.coord, False, observation
    follower.set_goal(route_geometry_sort_key(target_node))
    observation = follower.observe(current_coord)
    return observation.steering_coord, follower.goal_passed(), observation


def autonomous_route_entry_target(
    config: dict[str, Any],
    current_coord: int,
    route_loop: list[int],
    *,
    excluded_coords: set[int],
    zone_id: int | None,
) -> MiningNode | None:
    if not route_loop:
        return None
    route_cfg = config.get("route", {})
    resolved_zone_id = zone_id or single_allowed_route_zone(route_cfg) or 0
    projection = project_to_loop(current_coord, route_loop)
    candidates = [
        MiningNode(
            zone_id=resolved_zone_id,
            coord=projection.nearest_coord,
            ore_type="route_entry",
            node_id=None,
            route_index=projection.segment_index,
            route_t=projection.segment_t,
            source="autonomous_route_projection",
        )
    ]
    candidates.extend(
        MiningNode(
            zone_id=resolved_zone_id,
            coord=coord,
            ore_type="route_entry",
            node_id=None,
            route_index=index,
            route_t=0.0,
            source="autonomous_route_waypoint",
        )
        for index, coord in enumerate(route_loop)
    )
    available = [node for node in candidates if node.coord not in excluded_coords]
    if not available:
        return None
    return min(
        available,
        key=lambda node: (
            MovementNavigator.distance(current_coord, node.coord),
            route_sort_key(node) or 0.0,
        ),
    )


def mining_route_resume_target(
    config: dict[str, Any],
    current_coord: int,
    route_loop: list[int],
    *,
    zone_id: int | None,
    route_after_sort_key: float | None = None,
) -> MiningNode | None:
    """Rejoin the cyclic route at the current projection, then continue forward."""
    if len(route_loop) < 2:
        return None
    projection = project_to_loop(current_coord, route_loop)
    route_size = len(route_loop)
    floor = (
        float(route_after_sort_key) % route_size
        if route_after_sort_key is not None
        else None
    )
    projection_is_behind = False
    if floor is not None:
        forward_delta = (projection.sort_key - floor) % route_size
        projection_is_behind = forward_delta > route_size / 2.0
    reached_distance = max(
        0.0,
        float(
            config.get("mining", {}).get(
                "resume_projection_reached_distance",
                0.45,
            )
        ),
    )
    if projection_is_behind and floor is not None:
        route_index = int(math.ceil(floor)) % route_size
        coord = route_loop[route_index]
        route_t = 0.0
        source = "mining_route_forward_floor"
    elif projection.distance > reached_distance:
        coord = projection.nearest_coord
        route_index = projection.segment_index
        route_t = projection.segment_t
        source = "mining_route_rejoin_projection"
    else:
        route_index = (projection.segment_index + 1) % len(route_loop)
        coord = route_loop[route_index]
        route_t = 0.0
        source = "mining_route_forward_waypoint"
    return MiningNode(
        zone_id=zone_id or single_allowed_route_zone(config.get("route", {})) or 0,
        coord=coord,
        ore_type="Route Waypoint",
        node_id=-(2_000_000 + route_index),
        route_index=route_index,
        route_t=route_t,
        route_order=route_index,
        source=ROUTE_WAYPOINT_SOURCE,
    )


def mining_access_resume_queue(
    outcome: MiningOutcome,
    *,
    zone_id: int | None,
) -> list[MiningNode]:
    if not outcome.resume_coords or outcome.resume_route_index is None:
        return []
    coords: list[int] = []
    for coord in outcome.resume_coords:
        if not coords or coords[-1] != coord:
            coords.append(coord)
    count = max(1, len(coords))
    return [
        MiningNode(
            zone_id=zone_id or 0,
            coord=coord,
            ore_type="Route Waypoint",
            node_id=-(3_000_000 + index),
            route_index=outcome.resume_route_index,
            route_t=(index + 1) / (count + 1),
            route_order=outcome.resume_route_index,
            source=ROUTE_WAYPOINT_SOURCE,
        )
        for index, coord in enumerate(coords)
    ]


def consume_reached_route_entry(
    queue: list[MiningNode],
    current_coord: int,
    *,
    reached_distance: float,
) -> MiningNode | None:
    while (
        queue
        and MovementNavigator.distance(current_coord, queue[0].coord)
        <= reached_distance
    ):
        queue.pop(0)
    return queue[0] if queue else None


def resume_generated_route_entry(
    entry_plan: RouteEntryPlan,
    current_coord: int,
    *,
    reached_distance: float,
) -> tuple[list[MiningNode], float]:
    waypoints = list(entry_plan.waypoints)
    if not waypoints:
        return [], 0.0

    projection = project_to_path(current_coord, [node.coord for node in waypoints])
    queue = waypoints[projection.segment_index + 1 :]
    if projection.distance > reached_distance:
        projected_node = MiningNode(
            zone_id=waypoints[0].zone_id,
            coord=projection.nearest_coord,
            ore_type="route_entry",
            node_id=None,
            route_index=projection.segment_index,
            route_t=projection.segment_t,
            source="route_entry_projection",
        )
        if queue and queue[0].coord == projected_node.coord:
            queue.pop(0)
        queue.insert(0, projected_node)
    consume_reached_route_entry(
        queue,
        current_coord,
        reached_distance=reached_distance,
    )
    return queue, projection.distance


def route_entry_enabled(config: dict[str, Any], *, target_coord: int | None) -> bool:
    if target_coord is not None:
        return False
    route_cfg = config.get("route", {})
    entry_cfg = route_cfg.get("entry", {})
    if isinstance(entry_cfg, dict):
        return bool(entry_cfg.get("enabled", False))
    return bool(route_cfg.get("entry_enabled", False))


def route_entry_reached_distance(
    config: dict[str, Any],
    default_distance: float,
) -> float:
    route_cfg = config.get("route", {})
    entry_cfg = route_cfg.get("entry", {})
    if isinstance(entry_cfg, dict) and "reached_distance" in entry_cfg:
        return max(0.0, float(entry_cfg["reached_distance"]))
    if "entry_reached_distance" in route_cfg:
        return max(0.0, float(route_cfg["entry_reached_distance"]))
    return max(0.0, default_distance)


def route_entry_max_anchor_distance(config: dict[str, Any]) -> float:
    entry_cfg = config.get("route", {}).get("entry", {})
    if isinstance(entry_cfg, dict):
        return max(0.0, float(entry_cfg.get("max_anchor_distance", 2.0)))
    return 2.0


def route_sort_key(node: MiningNode) -> float | None:
    if node.route_index is not None:
        return float(node.route_index) + float(node.route_t or 0.0)
    if node.route_order is not None:
        return float(node.route_order)
    return None


def route_geometry_sort_key(node: MiningNode) -> float | None:
    if (
        node.source == ROUTE_WAYPOINT_SOURCE
        and node.route_index is not None
        and node.node_id is not None
        and -1_000_000 < node.node_id < 0
    ):
        return float(node.route_index)
    return route_sort_key(node)


def next_route_sort_key(node: MiningNode) -> float | None:
    sort_key = route_sort_key(node)
    if sort_key is None:
        return None
    return sort_key + 0.001


def allowed_route_locations(
    route_cfg: dict[str, Any],
    *,
    zone_id: int | None,
    auto_zone: bool,
    auto_zone_ids: set[int] | None,
) -> set[int]:
    if zone_id is not None:
        return {zone_id}
    if auto_zone:
        return set(auto_zone_ids or set())
    route_zone_ids = configured_route_zone_ids(route_cfg)
    if route_zone_ids:
        return route_zone_ids
    return set(route_cfg.get("locations", []))


def single_allowed_route_zone(route_cfg: dict[str, Any]) -> int | None:
    route_zone_ids = configured_route_zone_ids(route_cfg)
    if len(route_zone_ids) == 1:
        return next(iter(route_zone_ids))
    locations = route_cfg.get("locations", [])
    if not isinstance(locations, list) or len(locations) != 1:
        return None
    return int(locations[0])


def route_target_reached_distance(
    config: dict[str, Any],
    default_distance: float,
    target_node: MiningNode,
) -> float:
    if not is_route_waypoint(target_node):
        return max(0.0, float(default_distance))
    return max(
        0.0,
        float(config.get("route", {}).get("reached_distance", default_distance)),
    )


def configured_route_zone_ids(route_cfg: dict[str, Any]) -> set[int]:
    """Read authoritative zone ids from a JSON loop instead of stale config."""
    if not bool(route_cfg.get("follow_route_loop", False)):
        return set()
    database_path = route_cfg.get("database_path")
    if not database_path:
        return set()
    source_path = resource_path(str(database_path))
    if source_path.suffix.lower() != ".json" or not source_path.exists():
        return set()
    try:
        route_data = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set()

    zone_ids: set[int] = set()
    root_zone_id = route_data.get("zone_id")
    if root_zone_id is not None:
        zone_ids.add(int(root_zone_id))
    zone = route_data.get("zone")
    if isinstance(zone, dict) and zone.get("id") is not None:
        zone_ids.add(int(zone["id"]))
    for node in route_data.get("route_nodes", []):
        if isinstance(node, dict) and node.get("zone_id") is not None:
            zone_ids.add(int(node["zone_id"]))
    return zone_ids
