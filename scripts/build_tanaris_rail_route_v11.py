from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.route_database import project_to_loop


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v10_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v11_rail.json")
DEFAULT_REMOVE_START = 10
DEFAULT_REMOVE_END = 30
DEFAULT_EXCLUDED_NODE_IDS = frozenset({46, 47, 49, 52, 53, 54})
DEFAULT_DETOUR_XY = (
    (34.40, 26.63), (35.35, 26.45), (36.30, 26.40), (37.25, 26.35),
    (38.20, 26.35), (39.15, 26.45), (40.10, 26.65), (41.05, 27.10),
    (41.55, 28.00), (41.65, 29.20), (41.65, 30.40), (41.65, 31.60),
    (41.65, 32.80), (41.55, 34.00), (41.50, 35.20), (41.45, 36.40),
    (41.55, 37.60), (41.80, 39.40),
)
DEFAULT_RETURN_REMOVE_START = 193
DEFAULT_RETURN_REMOVE_END = 210
DEFAULT_RETURN_DETOUR_XY = (
    (41.75, 55.20), (41.85, 54.00), (41.90, 52.80), (41.90, 51.60),
    (41.90, 50.40), (41.90, 49.20), (41.90, 48.00), (41.90, 46.80),
    (41.90, 45.60), (41.90, 44.40), (41.90, 43.20), (41.90, 42.00),
    (41.90, 40.80), (41.90, 39.60), (41.90, 38.40), (41.90, 37.20),
    (41.90, 36.00), (41.95, 34.60),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tanaris rail V11 around the live-observed north Noxious Lair boundary."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _route_item(coord: int, *, index: int, detour_index: int) -> dict[str, Any]:
    x, y = coord_to_xy(coord)
    return {"index": index, "coord": int(coord), "x": round(x, 6),
            "y": round(y, 6), "source": "live_observed_noxious_north_bypass",
            "detour_index": detour_index}


def _replace_return_segment(result: dict[str, Any]) -> dict[str, Any]:
    source_items = list(result["route_loop"])
    detour = [xy_to_coord(x, y) for x, y in DEFAULT_RETURN_DETOUR_XY]
    north_detour_count = int(result["route_override"]["detour_waypoint_count"])
    return_remove_start = 175 + north_detour_count
    return_remove_end = 192 + north_detour_count
    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(return_remove_start):
        item = copy.deepcopy(source_items[old_index]); old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop); route_loop.append(item)
    for detour_index, coord in enumerate(detour):
        item = _route_item(coord, index=len(route_loop), detour_index=detour_index)
        item["source"] = "live_observed_noxious_east_bypass"; route_loop.append(item)
    for old_index in range(return_remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index]); old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop); route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    for order, node in enumerate(result["route_nodes"]):
        projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index; node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = order; node["distance_to_route"] = round(projection.distance, 6)
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"]); resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                node.pop("terrain_access_plan", None)
            else:
                plan["v11_north_attachment_route_index"] = attachment
                plan["v11_north_resume_route_index"] = resume
                plan["attachment_route_index"] = old_to_new[attachment]
                plan["resume_route_index"] = old_to_new[resume]
    for index, scan in enumerate(result["coverage_scan_points"]):
        projection = project_to_loop(int(scan["coord"]), route_coords)
        scan["index"] = index; scan["route_index"] = projection.segment_index
        scan["route_t"] = round(projection.segment_t, 6); scan["distance_to_route"] = round(projection.distance, 6)
    result["route_loop"] = route_loop
    result["route_override"]["return_removed_indexes"] = [return_remove_start, return_remove_end]
    result["route_override"]["return_detour_xy"] = [
        {"x": coord_to_xy(c)[0], "y": coord_to_xy(c)[1]} for c in detour
    ]
    return result


def build_noxious_north_bypass_route(
    source: dict[str, Any], *, remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END, detour_coords: Sequence[int] | None = None,
    excluded_node_ids: frozenset[int] = DEFAULT_EXCLUDED_NODE_IDS,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    if not (0 < remove_start <= remove_end < len(source_items) - 1):
        raise ValueError("Noxious north replacement indexes are out of range")
    detour = list(detour_coords or (xy_to_coord(x, y) for x, y in DEFAULT_DETOUR_XY))
    if not detour:
        raise ValueError("Noxious north bypass contains no waypoints")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(remove_start):
        item = copy.deepcopy(source_items[old_index]); old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop); item["v10_index"] = old_index; route_loop.append(item)
    for detour_index, coord in enumerate(detour):
        route_loop.append(_route_item(coord, index=len(route_loop), detour_index=detour_index))
    for old_index in range(remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index]); old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop); item["v10_index"] = old_index; route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    route_nodes: list[dict[str, Any]] = []
    for source_node in source.get("route_nodes", []):
        node_id = int(source_node.get("node_id", -1))
        if node_id in excluded_node_ids: continue
        node = copy.deepcopy(source_node); projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index; node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes); node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v11_rail_node"; plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"]); resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                node.pop("terrain_access_plan", None)
            else:
                plan["v10_attachment_route_index"] = attachment; plan["v10_resume_route_index"] = resume
                plan["attachment_route_index"] = old_to_new[attachment]; plan["resume_route_index"] = old_to_new[resume]
        route_nodes.append(node)

    scan_points: list[dict[str, Any]] = []
    for source_scan in source.get("coverage_scan_points", []):
        if remove_start <= int(source_scan.get("route_index", -1)) <= remove_end: continue
        scan = copy.deepcopy(source_scan); projection = project_to_loop(int(scan["coord"]), route_coords)
        scan["index"] = len(scan_points); scan["route_index"] = projection.segment_index
        scan["route_t"] = round(projection.segment_t, 6); scan["distance_to_route"] = round(projection.distance, 6)
        scan_points.append(scan)

    result = copy.deepcopy(source)
    result.update({
        "name": "Tanaris terrain-aware full rail cycle v11 Noxious north bypass",
        "status": "offline_full_route_candidate_not_live_validated",
        "source": {"route": str(source.get("name") or "unknown"),
                   "removed_v10_indexes": [remove_start, remove_end],
                   "excluded_runtime_node_ids": sorted(excluded_node_ids)},
        "route_loop": route_loop, "route_nodes": route_nodes, "coverage_scan_points": scan_points,
        "route_override": {"reason": "The v0.8.19 live run entered the visible Noxious Lair marker on the northern surface boundary at 35.55,32.63.",
                           "observed_boundary_coordinate": {"x": 35.55, "y": 32.63},
                           "removed_v10_indexes": [remove_start, remove_end],
                           "detour_waypoint_count": len(detour),
                           "detour_xy": [{"x": coord_to_xy(c)[0], "y": coord_to_xy(c)[1]} for c in detour],
                           "excluded_runtime_node_ids": sorted(excluded_node_ids),
                           "preserved_v10_override": copy.deepcopy(source.get("route_override", {}))},
    })
    ore_counts = Counter(str(node.get("ore_type") or "Ore") for node in route_nodes)
    result = _replace_return_segment(result)
    result["metrics"].update({"source_route_waypoints": len(source_items),
                              "north_route_waypoint_count": len(route_loop),
                              "route_waypoint_count": len(result["route_loop"]),
                              "removed_waypoint_count": remove_end - remove_start + 1,
                              "detour_waypoint_count": len(detour),
                              "runtime_node_count": len(route_nodes),
                              "excluded_runtime_node_count": len(excluded_node_ids),
                              "access_plan_count": sum(isinstance(n.get("terrain_access_plan"), dict) for n in result["route_nodes"]),
                              "ore_counts": dict(sorted(ore_counts.items()))})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); source = json.loads(args.source.read_text(encoding="utf-8"))
    route = build_noxious_north_bypass_route(source); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
