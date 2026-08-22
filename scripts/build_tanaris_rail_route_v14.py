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


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v13_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v14_rail.json")
DEFAULT_REMOVE_START = 188
DEFAULT_REMOVE_END = 194
DEFAULT_EXCLUDED_NODE_IDS = frozenset({50})
DEFAULT_CONTROL_XY = (
    (48.20, 76.50),
    (48.10, 74.50),
    (48.20, 72.50),
    (48.59, 71.38),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tanaris rail V14 with a terrain-validated western Gaping "
            "Chasm bypass and the live-blocked node 50 removed."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def generate_gaping_v14_detour(
    config: dict[str, Any],
    *,
    start_xy: tuple[float, float] = (48.77, 77.45),
    control_xy: Sequence[tuple[float, float]] = DEFAULT_CONTROL_XY,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    planner = NavMeshRouteEntryPlanner.from_config(config, zone_id=162)
    current = xy_to_coord(*start_xy)
    detour: list[int] = []
    validations: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(control_xy):
        target = xy_to_coord(x, y)
        result = planner.plan_to_target(current, target, target_route_index=index)
        if not result.terrain_validation.get("walkable"):
            raise RuntimeError(f"Gaping V14 bypass leg {index} failed terrain validation")
        for waypoint in result.entry_plan.waypoints:
            coord = int(waypoint.coord)
            if not detour or detour[-1] != coord:
                detour.append(coord)
        validations.append(
            {
                "leg": index,
                "world_length_yards": round(result.world_length_yards, 4),
                **copy.deepcopy(result.terrain_validation),
            }
        )
        current = target
    # The final control point is the first retained V13 waypoint after the
    # replacement. Keep that original route item and its lineage.
    if detour and detour[-1] == xy_to_coord(*control_xy[-1]):
        detour.pop()
    return tuple(detour), validations


def _detour_item(coord: int, *, index: int, detour_index: int) -> dict[str, Any]:
    x, y = coord_to_xy(coord)
    return {
        "index": index,
        "coord": int(coord),
        "x": round(x, 6),
        "y": round(y, 6),
        "source": "live_observed_gaping_v14_west_bypass",
        "detour_index": detour_index,
    }


def build_v14(
    source: dict[str, Any],
    *,
    detour_coords: Sequence[int],
    terrain_validation: Sequence[dict[str, Any]] = (),
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
    excluded_node_ids: frozenset[int] = DEFAULT_EXCLUDED_NODE_IDS,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    if not (0 < remove_start <= remove_end < len(source_items) - 1):
        raise ValueError("Gaping V14 replacement indexes are out of range")
    detour = [int(coord) for coord in detour_coords]
    if not detour:
        raise ValueError("Gaping V14 bypass contains no waypoints")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(remove_start):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v13_index"] = old_index
        route_loop.append(item)
    for detour_index, coord in enumerate(detour):
        route_loop.append(
            _detour_item(coord, index=len(route_loop), detour_index=detour_index)
        )
    for old_index in range(remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v13_index"] = old_index
        route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    route_nodes: list[dict[str, Any]] = []
    excluded: list[int] = []
    for source_node in source.get("route_nodes", []):
        node_id = int(source_node.get("node_id", -1))
        if node_id in excluded_node_ids:
            excluded.append(node_id)
            continue
        node = copy.deepcopy(source_node)
        projection = project_to_loop(int(node["coord"]), route_coords)
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v14_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict):
            attachment = int(plan["attachment_route_index"])
            resume = int(plan["resume_route_index"])
            if attachment not in old_to_new or resume not in old_to_new:
                node.pop("terrain_access_plan", None)
                node["terrain_access_plan_reject_reason"] = "removed_v13_route_attachment"
            else:
                plan["v13_attachment_route_index"] = attachment
                plan["v13_resume_route_index"] = resume
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
            "name": "Tanaris terrain-aware full rail cycle v14 wide Gaping west bypass",
            "status": "offline_full_route_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "removed_v13_indexes": [remove_start, remove_end],
                "excluded_runtime_node_ids": sorted(excluded),
            },
            "route_loop": route_loop,
            "route_nodes": route_nodes,
            "coverage_scan_points": scan_points,
            "route_override": {
                "reason": (
                    "RUN3B entered the visibly named The Gaping Chasm near "
                    "51.19,76.07. V14 replaces the eastern bend with a wide "
                    "terrain-validated western corridor. Mithril node 50 is "
                    "removed after two live access options and exact-navmesh "
                    "replay all failed near 34.19,77.47."
                ),
                "observed_boundary_coordinate": {"x": 51.19, "y": 76.07},
                "observed_node_50_stall_coordinate": {"x": 34.19, "y": 77.47},
                "removed_v13_indexes": [remove_start, remove_end],
                "detour_waypoint_count": len(detour),
                "detour_xy": [
                    {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
                    for coord in detour
                ],
                "terrain_validation": list(terrain_validation),
                "excluded_runtime_node_ids": sorted(excluded),
                "preserved_v13_override": copy.deepcopy(source.get("route_override", {})),
            },
        }
    )
    result.setdefault("metrics", {}).update(
        {
            "source_route_waypoints": len(source_items),
            "route_waypoint_count": len(route_loop),
            "removed_waypoint_count": remove_end - remove_start + 1,
            "detour_waypoint_count": len(detour),
            "runtime_node_count": len(route_nodes),
            "excluded_v14_runtime_node_count": len(excluded),
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
    detour, validation = generate_gaping_v14_detour(load_config(args.config))
    route = build_v14(source, detour_coords=detour, terrain_validation=validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
