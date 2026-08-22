from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner
from vision_bot.route_database import project_to_loop


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v11_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v12_rail.json")
DEFAULT_REMOVE_START = 36
DEFAULT_REMOVE_END = 63
DEFAULT_CONTROL_XY = (
    (47.70, 50.50),
    (47.70, 64.00),
    (42.00, 65.00),
    (34.00, 64.00),
    (29.27, 65.04),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tanaris rail V12 around the live-observed Dunemaul pack."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def generate_dunemaul_detour(
    config: dict[str, Any],
    *,
    control_xy: Sequence[tuple[float, float]] = DEFAULT_CONTROL_XY,
) -> tuple[int, ...]:
    planner = NavMeshRouteEntryPlanner.from_config(config, zone_id=162)
    current = xy_to_coord(41.77, 50.27)
    detour: list[int] = []
    for index, (x, y) in enumerate(control_xy):
        target = xy_to_coord(x, y)
        result = planner.plan_to_target(current, target, target_route_index=index)
        if not result.terrain_validation.get("walkable"):
            raise RuntimeError(f"Dunemaul bypass leg {index} failed terrain validation")
        for node in result.entry_plan.waypoints:
            coord = int(node.coord)
            if not detour or detour[-1] != coord:
                detour.append(coord)
        current = target
    # The last control point is the first retained V11 waypoint after the
    # replacement.  Let that original item keep its lineage and index.
    if detour and detour[-1] == xy_to_coord(*control_xy[-1]):
        detour.pop()
    return tuple(detour)


def _inside_dunemaul_exclusion(coord: int) -> bool:
    x, y = coord_to_xy(coord)
    # Conservative coordinate-space envelope around the configured polygon
    # and its 30-yard clearance.  Runtime ore records in this pack are not
    # worth an approach while this character cannot safely clear several ogres.
    return 33.5 <= x <= 47.5 and 50.8 <= y <= 63.8


def _detour_item(coord: int, *, index: int, detour_index: int) -> dict[str, Any]:
    x, y = coord_to_xy(coord)
    return {
        "index": index,
        "coord": int(coord),
        "x": round(x, 6),
        "y": round(y, 6),
        "source": "live_observed_dunemaul_pack_bypass",
        "detour_index": detour_index,
    }


def build_dunemaul_bypass_route(
    source: dict[str, Any],
    *,
    detour_coords: Sequence[int],
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    if not (0 < remove_start <= remove_end < len(source_items) - 1):
        raise ValueError("Dunemaul replacement indexes are out of range")
    detour = [int(coord) for coord in detour_coords]
    if not detour:
        raise ValueError("Dunemaul bypass contains no waypoints")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(remove_start):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v11_index"] = old_index
        route_loop.append(item)
    for detour_index, coord in enumerate(detour):
        route_loop.append(
            _detour_item(coord, index=len(route_loop), detour_index=detour_index)
        )
    for old_index in range(remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v11_index"] = old_index
        route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    route_nodes: list[dict[str, Any]] = []
    excluded_node_ids: list[int] = []
    for source_node in source.get("route_nodes", []):
        if _inside_dunemaul_exclusion(int(source_node["coord"])):
            excluded_node_ids.append(int(source_node.get("node_id", -1)))
            continue
        node = copy.deepcopy(source_node)
        projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v12_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"])
            resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                node.pop("terrain_access_plan", None)
            else:
                plan["v11_attachment_route_index"] = attachment
                plan["v11_resume_route_index"] = resume
                plan["attachment_route_index"] = old_to_new[attachment]
                plan["resume_route_index"] = old_to_new[resume]
        route_nodes.append(node)

    scan_points: list[dict[str, Any]] = []
    for source_scan in source.get("coverage_scan_points", []):
        if _inside_dunemaul_exclusion(int(source_scan["coord"])):
            continue
        scan = copy.deepcopy(source_scan)
        projection = project_to_loop(int(scan["coord"]), route_coords)
        scan["index"] = len(scan_points)
        scan["route_index"] = projection.segment_index
        scan["route_t"] = round(projection.segment_t, 6)
        scan["distance_to_route"] = round(projection.distance, 6)
        scan_points.append(scan)

    result = copy.deepcopy(source)
    result.update(
        {
            "name": "Tanaris terrain-aware full rail cycle v12 Dunemaul bypass",
            "status": "offline_full_route_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "removed_v11_indexes": [remove_start, remove_end],
                "excluded_runtime_node_ids": sorted(excluded_node_ids),
            },
            "route_loop": route_loop,
            "route_nodes": route_nodes,
            "coverage_scan_points": scan_points,
            "route_override": {
                "reason": (
                    "V11 crossed the Dunemaul surface pack and died at "
                    "37.73,54.96 with several level 46-47 ogres visible."
                ),
                "observed_death_coordinate": {"x": 37.73, "y": 54.96},
                "removed_v11_indexes": [remove_start, remove_end],
                "detour_waypoint_count": len(detour),
                "detour_xy": [
                    {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
                    for coord in detour
                ],
                "excluded_runtime_node_ids": sorted(excluded_node_ids),
                "preserved_v11_override": copy.deepcopy(source.get("route_override", {})),
            },
        }
    )
    ore_counts = Counter(str(node.get("ore_type") or "Ore") for node in route_nodes)
    result.setdefault("metrics", {}).update(
        {
            "source_route_waypoints": len(source_items),
            "route_waypoint_count": len(route_loop),
            "removed_waypoint_count": remove_end - remove_start + 1,
            "detour_waypoint_count": len(detour),
            "runtime_node_count": len(route_nodes),
            "excluded_dunemaul_runtime_node_count": len(excluded_node_ids),
            "access_plan_count": sum(
                isinstance(node.get("terrain_access_plan"), dict)
                for node in route_nodes
            ),
            "ore_counts": dict(sorted(ore_counts.items())),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    detour = generate_dunemaul_detour(load_config(args.config))
    route = build_dunemaul_bypass_route(source, detour_coords=detour)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
