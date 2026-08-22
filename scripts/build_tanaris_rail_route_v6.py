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


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v5_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v6_rail.json")
DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_REPLACE_START = 409
DEFAULT_REPLACE_END = 414


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tanaris rail V6 with a terrain-planned camp bypass."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replace-start", type=int, default=DEFAULT_REPLACE_START)
    parser.add_argument("--replace-end", type=int, default=DEFAULT_REPLACE_END)
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


def build_camp_bypass_route(
    source: dict[str, Any],
    *,
    bypass_coords: Sequence[int],
    replace_start: int = DEFAULT_REPLACE_START,
    replace_end: int = DEFAULT_REPLACE_END,
    terrain_validation: dict[str, Any] | None = None,
    bypass_world_length_yards: float | None = None,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    source_coords = [int(item["coord"]) for item in source_items]
    if not (0 <= replace_start < replace_end < len(source_coords)):
        raise ValueError("Camp bypass replacement indexes are out of range")

    start_coord = source_coords[replace_start]
    end_coord = source_coords[replace_end]
    cleaned_bypass = [int(coord) for coord in bypass_coords]
    while cleaned_bypass and cleaned_bypass[0] == start_coord:
        cleaned_bypass.pop(0)
    while cleaned_bypass and cleaned_bypass[-1] == end_coord:
        cleaned_bypass.pop()
    if not cleaned_bypass:
        raise ValueError("Camp bypass contains no intermediate waypoints")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(replace_start + 1):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v5_index"] = old_index
        route_loop.append(item)
    for bypass_index, coord in enumerate(cleaned_bypass):
        route_loop.append(
            _route_item(
                coord,
                index=len(route_loop),
                source="navmesh_camp_bypass",
                bypass_index=bypass_index,
            )
        )
    for old_index in range(replace_end, len(source_items)):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v5_index"] = old_index
        route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    route_nodes: list[dict[str, Any]] = []
    for source_node in source.get("route_nodes", []):
        node = copy.deepcopy(source_node)
        projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v6_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"])
            resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                raise ValueError(
                    f"Node {node.get('node_id')} access plan depends on replaced V5 waypoints"
                )
            plan["v5_attachment_route_index"] = attachment
            plan["v5_resume_route_index"] = resume
            plan["attachment_route_index"] = old_to_new[attachment]
            plan["resume_route_index"] = old_to_new[resume]
        route_nodes.append(node)

    scan_points: list[dict[str, Any]] = []
    for source_scan in source.get("coverage_scan_points", []):
        scan = copy.deepcopy(source_scan)
        projection = project_to_loop(int(scan["coord"]), route_coords)
        scan["index"] = len(scan_points)
        scan["route_index"] = projection.segment_index
        scan["route_t"] = round(projection.segment_t, 6)
        scan["distance_to_route"] = round(projection.distance, 6)
        scan_points.append(scan)

    ore_counts = Counter(str(node.get("ore_type") or "Ore") for node in route_nodes)
    result = copy.deepcopy(source)
    result.update(
        {
            "name": "Tanaris terrain-aware full rail cycle v6 camp bypass",
            "status": "offline_full_route_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "replaced_v5_indexes": [replace_start + 1, replace_end - 1],
                "preserved_runtime_nodes": len(route_nodes),
            },
            "route_loop": route_loop,
            "route_nodes": route_nodes,
            "coverage_scan_points": scan_points,
            "route_override": {
                "reason": (
                    "Bypass the Bera Stonehammer tent/cart camp where the v0.8.11 "
                    "live run became stationary near 51.07,29.45."
                ),
                "preserved_v5_override": copy.deepcopy(
                    source.get("route_override", {})
                ),
                "observed_stuck_coordinate": {"x": 51.07, "y": 29.45},
                "replaced_v5_indexes": [replace_start + 1, replace_end - 1],
                "start": {"coord": start_coord, "x": coord_to_xy(start_coord)[0], "y": coord_to_xy(start_coord)[1]},
                "end": {"coord": end_coord, "x": coord_to_xy(end_coord)[0], "y": coord_to_xy(end_coord)[1]},
                "bypass_waypoint_count": len(cleaned_bypass),
                "bypass_world_length_yards": (
                    round(bypass_world_length_yards, 3)
                    if bypass_world_length_yards is not None
                    else None
                ),
                "terrain_validation": terrain_validation,
            },
            "metrics": {
                **dict(source.get("metrics", {})),
                "source_route_waypoints": len(source_coords),
                "route_waypoint_count": len(route_loop),
                "replaced_waypoint_count": replace_end - replace_start - 1,
                "bypass_waypoint_count": len(cleaned_bypass),
                "runtime_node_count": len(route_nodes),
                "access_plan_count": sum(
                    isinstance(node.get("terrain_access_plan"), dict)
                    for node in route_nodes
                ),
                "ore_counts": dict(sorted(ore_counts.items())),
            },
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_loop = [int(item["coord"]) for item in source["route_loop"]]
    planner = NavMeshRouteEntryPlanner.from_config(
        config,
        zone_id=int(source.get("zone_id", 162)),
    )
    bypass = planner.plan_to_target(
        source_loop[args.replace_start],
        source_loop[args.replace_end],
        target_route_index=args.replace_end,
    )
    route = build_camp_bypass_route(
        source,
        bypass_coords=[waypoint.coord for waypoint in bypass.entry_plan.waypoints],
        replace_start=args.replace_start,
        replace_end=args.replace_end,
        terrain_validation=bypass.terrain_validation,
        bypass_world_length_yards=bypass.world_length_yards,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
