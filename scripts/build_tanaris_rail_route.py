from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from vision_bot.coords import coord_to_xy
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner
from vision_bot.route_database import project_to_loop


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v4.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v5_rail.json")
DEFAULT_CONFIG = Path("config.yaml")

# The two southern records are the exact DB coordinates corresponding to the
# user's spoken 36.30/20.10 and 36.21/20.12 pair. They were already uncovered,
# but keeping them explicit prevents a future regeneration from restoring them.
USER_EXCLUDED_NODES: dict[int, str] = {
    4_450_228_000: "Gold 44.50,22.80",
    3_770_210_000: "Gold 37.70,21.00",
    3_630_201_000: "Gold 36.30,20.10",
    3_621_201_200: "Gold 36.21,20.12",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full Tanaris rail route with a direct northern bridge."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--retained-start-index", type=int, default=20)
    parser.add_argument("--retained-end-index", type=int, default=444)
    parser.add_argument("--max-node-distance", type=float, default=1.25)
    return parser.parse_args(argv)


def _route_item(coord: int, *, index: int, **extra: Any) -> dict[str, Any]:
    x, y = coord_to_xy(coord)
    return {
        "index": index,
        "coord": int(coord),
        "x": round(x, 6),
        "y": round(y, 6),
        **extra,
    }


def _remap_access_plan(
    node: dict[str, Any],
    *,
    retained_start_index: int,
    retained_end_index: int,
) -> None:
    plan = node.get("terrain_access_plan")
    if not isinstance(plan, dict):
        return
    attachment = plan.get("attachment_route_index")
    resume = plan.get("resume_route_index")
    if not isinstance(attachment, int) or not isinstance(resume, int):
        node.pop("terrain_access_plan", None)
        node["terrain_access_plan_reject_reason"] = "missing_source_route_indexes"
        return
    if not (
        retained_start_index <= attachment <= retained_end_index
        and retained_start_index <= resume <= retained_end_index
    ):
        node.pop("terrain_access_plan", None)
        node["terrain_access_plan_reject_reason"] = "removed_route_attachment"
        return
    plan["source_attachment_route_index"] = attachment
    plan["source_resume_route_index"] = resume
    plan["attachment_route_index"] = attachment - retained_start_index
    plan["resume_route_index"] = resume - retained_start_index


def build_rail_route(
    source: dict[str, Any],
    *,
    bridge_coords: Sequence[int],
    retained_start_index: int = 20,
    retained_end_index: int = 444,
    max_node_distance: float = 1.25,
    excluded_nodes: dict[int, str] | None = None,
) -> dict[str, Any]:
    source_loop = [int(item["coord"]) for item in source.get("route_loop", [])]
    if not source_loop:
        raise ValueError("Source route has no route_loop")
    if not (0 <= retained_start_index <= retained_end_index < len(source_loop)):
        raise ValueError("Retained source route indexes are out of range")

    retained_source_indexes = list(range(retained_start_index, retained_end_index + 1))
    retained_coords = [source_loop[index] for index in retained_source_indexes]
    cleaned_bridge = [int(coord) for coord in bridge_coords]
    while cleaned_bridge and cleaned_bridge[0] == retained_coords[-1]:
        cleaned_bridge.pop(0)
    while cleaned_bridge and cleaned_bridge[-1] == retained_coords[0]:
        cleaned_bridge.pop()
    route_coords = retained_coords + cleaned_bridge
    if len(route_coords) < 3:
        raise ValueError("Rail route is too short")

    route_loop: list[dict[str, Any]] = []
    for source_index, coord in zip(retained_source_indexes, retained_coords):
        route_loop.append(
            _route_item(
                coord,
                index=len(route_loop),
                source="coverage_v4",
                source_index=source_index,
            )
        )
    for bridge_index, coord in enumerate(cleaned_bridge):
        route_loop.append(
            _route_item(
                coord,
                index=len(route_loop),
                source="navmesh_direct_bridge",
                bridge_index=bridge_index,
            )
        )

    exclusions = dict(excluded_nodes or USER_EXCLUDED_NODES)
    route_nodes: list[dict[str, Any]] = []
    rejected_nodes: list[dict[str, Any]] = []
    for source_node in source.get("route_nodes", []):
        node = copy.deepcopy(source_node)
        coord = int(node["coord"])
        projection = project_to_loop(coord, route_coords)
        reject_reason: str | None = None
        if coord in exclusions:
            reject_reason = "user_excluded_spawn"
            node["user_exclusion"] = exclusions[coord]
        elif node.get("coverage_status") != "covered":
            reject_reason = str(node.get("coverage_status") or "not_covered")
        elif projection.distance > max_node_distance:
            reject_reason = "outside_v5_runtime_corridor"

        if reject_reason is not None:
            node["route_reject_reason"] = reject_reason
            node["distance_to_route"] = round(projection.distance, 6)
            rejected_nodes.append(node)
            continue

        node["source_node_index"] = node.get("index")
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v5_rail_node"
        _remap_access_plan(
            node,
            retained_start_index=retained_start_index,
            retained_end_index=retained_end_index,
        )
        route_nodes.append(node)

    for node_id, node in enumerate(route_nodes, start=1):
        node["node_id"] = node_id

    scan_points: list[dict[str, Any]] = []
    for source_scan in source.get("coverage_scan_points", []):
        scan = copy.deepcopy(source_scan)
        projection = project_to_loop(int(scan["coord"]), route_coords)
        if projection.distance > max_node_distance:
            continue
        scan["source_scan_index"] = scan.get("index")
        scan["index"] = len(scan_points)
        scan["route_index"] = projection.segment_index
        scan["route_t"] = round(projection.segment_t, 6)
        scan["distance_to_route"] = round(projection.distance, 6)
        scan_points.append(scan)

    ore_counts = Counter(str(node.get("ore_type") or "Ore") for node in route_nodes)
    reject_counts = Counter(str(node["route_reject_reason"]) for node in rejected_nodes)
    bridge_start = route_coords[retained_end_index - retained_start_index]
    bridge_end = route_coords[0]
    bridge_start_xy = coord_to_xy(bridge_start)
    bridge_end_xy = coord_to_xy(bridge_end)

    return {
        "schema_version": 1,
        "name": "Tanaris terrain-aware full rail cycle v5",
        "zone_id": int(source.get("zone_id", 162)),
        "world_map_area_id": source.get("world_map_area_id"),
        "status": "offline_full_route_candidate_not_live_validated",
        "source": {
            "route": str(source.get("name") or "unknown"),
            "retained_source_indexes": [retained_start_index, retained_end_index],
            "max_node_distance_to_route": max_node_distance,
        },
        "coverage_policy": {
            **dict(source.get("coverage_policy", {})),
            "behavior": (
                "Follow the full rail route with smooth polyline lookahead; intercept only "
                "a tooltip-confirmed live ore icon. Historical nodes never create detours."
            ),
        },
        "route_override": {
            "reason": (
                "Remove the Gold detour and transition directly from the Small Thorium "
                "coverage area at 41.40,26.71 to the Mithril coverage area at 38.19,27.12."
            ),
            "bridge_start": {
                "coord": bridge_start,
                "x": bridge_start_xy[0],
                "y": bridge_start_xy[1],
            },
            "bridge_end": {
                "coord": bridge_end,
                "x": bridge_end_xy[0],
                "y": bridge_end_xy[1],
            },
            "bridge_waypoint_count": len(cleaned_bridge),
            "explicit_node_exclusions": [
                {"coord": coord, "label": label}
                for coord, label in exclusions.items()
            ],
        },
        "coverage_scan_points": scan_points,
        "route_loop": route_loop,
        "route_nodes": route_nodes,
        "rejected_nodes": rejected_nodes,
        "metrics": {
            "source_route_waypoints": len(source_loop),
            "route_waypoint_count": len(route_loop),
            "retained_source_waypoint_count": len(retained_coords),
            "bridge_waypoint_count": len(cleaned_bridge),
            "runtime_node_count": len(route_nodes),
            "rejected_node_count": len(rejected_nodes),
            "access_plan_count": sum(
                isinstance(node.get("terrain_access_plan"), dict) for node in route_nodes
            ),
            "ore_counts": dict(sorted(ore_counts.items())),
            "reject_counts": dict(sorted(reject_counts.items())),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_loop = [int(item["coord"]) for item in source["route_loop"]]
    planner = NavMeshRouteEntryPlanner.from_config(
        config,
        zone_id=int(source.get("zone_id", 162)),
    )
    bridge = planner.plan_to_target(
        source_loop[args.retained_end_index],
        source_loop[args.retained_start_index],
        target_route_index=args.retained_start_index,
    )
    bridge_coords = [node.coord for node in bridge.entry_plan.waypoints]
    route = build_rail_route(
        source,
        bridge_coords=bridge_coords,
        retained_start_index=args.retained_start_index,
        retained_end_index=args.retained_end_index,
        max_node_distance=args.max_node_distance,
    )
    route["route_override"]["bridge_world_length_yards"] = round(
        bridge.world_length_yards, 3
    )
    route["route_override"]["terrain_validation"] = bridge.terrain_validation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
