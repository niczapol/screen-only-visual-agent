from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.route_database import project_to_loop


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v6_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v7_rail.json")
DEFAULT_REMOVE_START = 52
DEFAULT_REMOVE_END = 69
DEFAULT_EXCLUDED_NODE_IDS = frozenset({27, 30, 31, 32, 36, 37, 38})
DEFAULT_DETOUR_XY = (
    (31.70, 54.82),
    (30.55, 54.82),
    (29.45, 54.42),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tanaris rail V7 with the live-observed Noxious Lair boundary removed."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _route_item(coord: int, *, index: int, detour_index: int) -> dict[str, Any]:
    x, y = coord_to_xy(coord)
    return {
        "index": index,
        "coord": int(coord),
        "x": round(x, 6),
        "y": round(y, 6),
        "source": "live_observed_noxious_boundary_detour",
        "detour_index": detour_index,
    }


def build_noxious_boundary_detour_route(
    source: dict[str, Any],
    *,
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
    detour_coords: Sequence[int] | None = None,
    excluded_node_ids: frozenset[int] = DEFAULT_EXCLUDED_NODE_IDS,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    if not (0 < remove_start <= remove_end < len(source_items) - 1):
        raise ValueError("Noxious boundary replacement indexes are out of range")
    detour = list(detour_coords or (xy_to_coord(x, y) for x, y in DEFAULT_DETOUR_XY))
    if not detour:
        raise ValueError("Noxious boundary detour contains no waypoints")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(remove_start):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v6_index"] = old_index
        route_loop.append(item)
    for detour_index, coord in enumerate(detour):
        route_loop.append(
            _route_item(coord, index=len(route_loop), detour_index=detour_index)
        )
    for old_index in range(remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v6_index"] = old_index
        route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    route_nodes: list[dict[str, Any]] = []
    for source_node in source.get("route_nodes", []):
        node_id = int(source_node.get("node_id", -1))
        if node_id in excluded_node_ids:
            continue
        node = copy.deepcopy(source_node)
        projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v7_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"])
            resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                raise ValueError(
                    f"Retained node {node_id} depends on a removed V6 waypoint"
                )
            plan["v6_attachment_route_index"] = attachment
            plan["v6_resume_route_index"] = resume
            plan["attachment_route_index"] = old_to_new[attachment]
            plan["resume_route_index"] = old_to_new[resume]
        route_nodes.append(node)

    scan_points: list[dict[str, Any]] = []
    for source_scan in source.get("coverage_scan_points", []):
        old_route_index = int(source_scan.get("route_index", -1))
        if remove_start <= old_route_index <= remove_end:
            continue
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
            "name": "Tanaris terrain-aware full rail cycle v7 Noxious boundary detour",
            "status": "offline_full_route_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "removed_v6_indexes": [remove_start, remove_end],
                "excluded_runtime_node_ids": sorted(excluded_node_ids),
            },
            "route_loop": route_loop,
            "route_nodes": route_nodes,
            "coverage_scan_points": scan_points,
            "route_override": {
                "reason": (
                    "The 2026-08-11 live run entered the visible Noxious Lair "
                    "subzone marker at 31.35,53.32 while still on the surface."
                ),
                "observed_boundary_coordinate": {"x": 31.35, "y": 53.32},
                "removed_v6_indexes": [remove_start, remove_end],
                "detour_waypoint_count": len(detour),
                "detour_xy": [
                    {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
                    for coord in detour
                ],
                "excluded_runtime_node_ids": sorted(excluded_node_ids),
                "preserved_v6_override": copy.deepcopy(source.get("route_override", {})),
            },
            "metrics": {
                **dict(source.get("metrics", {})),
                "source_route_waypoints": len(source_items),
                "route_waypoint_count": len(route_loop),
                "removed_waypoint_count": remove_end - remove_start + 1,
                "detour_waypoint_count": len(detour),
                "runtime_node_count": len(route_nodes),
                "excluded_runtime_node_count": len(excluded_node_ids),
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
    route = build_noxious_boundary_detour_route(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
