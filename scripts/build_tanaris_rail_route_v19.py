from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner
from vision_bot.route_database import project_to_loop
from vision_bot.terrain_routing import supercover_line


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v18_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v19_rail.json")
DEFAULT_REMOVE_START = 55
DEFAULT_REMOVE_END = 269


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build V19 by compiling V18 against every configured physical "
            "hazard instead of auditing only the latest Caverns polygon."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def generate_safe_bridge(
    planner: NavMeshRouteEntryPlanner,
    source: dict[str, Any],
    *,
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    route = source["route_loop"]
    start_coord = int(route[remove_start - 1]["coord"])
    end_coord = int(route[remove_end + 1]["coord"])
    result = planner.plan_to_target(
        start_coord,
        end_coord,
        target_route_index=remove_end + 1,
    )
    bridge = [int(item.coord) for item in result.entry_plan.waypoints]
    if bridge and bridge[-1] == end_coord:
        bridge.pop()
    return tuple(bridge), {
        "walkable": bool(result.terrain_validation.get("walkable")),
        "world_length_yards": round(result.world_length_yards, 4),
        **copy.deepcopy(result.terrain_validation),
    }


def build_v19(
    source: dict[str, Any],
    *,
    planner: NavMeshRouteEntryPlanner,
    bridge_coords: Sequence[int],
    bridge_validation: dict[str, Any],
    source_fingerprint: str,
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
) -> dict[str, Any]:
    source_items = list(source.get("route_loop", []))
    if not (0 < remove_start <= remove_end < len(source_items) - 1):
        raise ValueError("V19 replacement indexes are out of range")
    bridge = [int(coord) for coord in bridge_coords]
    if not bridge or not bridge_validation.get("walkable"):
        raise ValueError("V19 requires a non-empty terrain-validated bridge")

    old_to_new: dict[int, int] = {}
    route_loop: list[dict[str, Any]] = []
    for old_index in range(remove_start):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v18_index"] = old_index
        route_loop.append(item)
    for bridge_index, coord in enumerate(bridge):
        x, y = coord_to_xy(coord)
        route_loop.append(
            {
                "index": len(route_loop),
                "coord": coord,
                "x": round(x, 6),
                "y": round(y, 6),
                "source": "v09_all_hazard_compiler_bridge",
                "bridge_index": bridge_index,
            }
        )
    for old_index in range(remove_end + 1, len(source_items)):
        item = copy.deepcopy(source_items[old_index])
        old_to_new[old_index] = len(route_loop)
        item["index"] = len(route_loop)
        item["v18_index"] = old_index
        route_loop.append(item)

    route_coords = [int(item["coord"]) for item in route_loop]
    audit = _audit_route(planner, route_coords)
    if audit["hazard_point_indexes"] or audit["hazard_segment_indexes"]:
        raise RuntimeError(f"V19 route still crosses configured hazards: {audit}")

    route_nodes: list[dict[str, Any]] = []
    rejected_nodes = copy.deepcopy(source.get("rejected_nodes", []))
    rejection_counts: Counter[str] = Counter()
    unsafe_access_options: list[list[int]] = []
    for source_node in source.get("route_nodes", []):
        node_id = int(source_node.get("node_id", -1))
        node_coord = int(source_node["coord"])
        reject_reason: str | None = None
        if planner.coord_is_hazard(node_coord):
            reject_reason = "v19_node_inside_configured_hazard"
        plan = source_node.get("terrain_access_plan")
        if reject_reason is None and not isinstance(plan, dict):
            reject_reason = "v19_missing_access_plan"
        attachment = int(plan.get("attachment_route_index", -1)) if isinstance(plan, dict) else -1
        resume = int(plan.get("resume_route_index", -1)) if isinstance(plan, dict) else -1
        if reject_reason is None and (attachment not in old_to_new or resume not in old_to_new):
            reject_reason = "v19_access_attachment_removed"
        safe_options: list[dict[str, Any]] = []
        if reject_reason is None:
            for option in plan.get("options", []):
                coords = _option_coords(option)
                if _path_is_hazardous(planner, coords):
                    unsafe_access_options.append([node_id, int(option.get("rank", -1))])
                    continue
                safe_options.append(copy.deepcopy(option))
            if not safe_options:
                reject_reason = "v19_no_safe_access_option"
        if reject_reason is not None:
            rejection_counts[reject_reason] += 1
            rejected_nodes.append(
                {
                    "node_id": node_id,
                    "coord": node_coord,
                    "ore_type": source_node.get("ore_type"),
                    "reason": reject_reason,
                    "source": "v09_all_hazard_compiler",
                }
            )
            continue

        node = copy.deepcopy(source_node)
        projection = project_to_loop(node_coord, route_coords)
        node["route_index"] = projection.segment_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "tanaris_v19_compiled_safe_node"
        compiled_plan = copy.deepcopy(plan)
        compiled_plan["v18_attachment_route_index"] = attachment
        compiled_plan["v18_resume_route_index"] = resume
        compiled_plan["attachment_route_index"] = old_to_new[attachment]
        compiled_plan["resume_route_index"] = old_to_new[resume]
        safe_options.sort(key=lambda item: int(item.get("rank", 0)))
        for index, option in enumerate(safe_options):
            option["primary"] = index == 0
        compiled_plan["primary_option_rank"] = int(safe_options[0].get("rank", 0))
        compiled_plan["options"] = safe_options
        node["terrain_access_plan"] = compiled_plan
        route_nodes.append(node)

    ore_counts = Counter(str(node.get("ore_type") or "Ore") for node in route_nodes)
    result = copy.deepcopy(source)
    result.update(
        {
            "name": "Tanaris v0.9 compiled rail cycle v19 all-hazard contraction",
            "status": "offline_compiled_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "source_sha256": source_fingerprint,
                "compiler": "scripts/build_tanaris_rail_route_v19.py",
                "removed_v18_indexes": [remove_start, remove_end],
            },
            "route_loop": route_loop,
            "route_nodes": route_nodes,
            "rejected_nodes": rejected_nodes,
            "route_override": {
                "reason": (
                    "V18 audited only the newly narrowed Caverns gorge. The v0.9 "
                    "compiler found 25 V18 waypoints inside other configured live "
                    "hazards and contracts that disconnected southwest excursion "
                    "through one terrain/navmesh-validated bridge."
                ),
                "removed_v18_indexes": [remove_start, remove_end],
                "bridge_waypoint_count": len(bridge),
                "bridge_validation": bridge_validation,
                "all_hazard_audit": audit,
                "unsafe_access_options": unsafe_access_options,
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "preserved_v18_override": copy.deepcopy(source.get("route_override", {})),
            },
        }
    )
    result["metrics"] = {
        **copy.deepcopy(source.get("metrics", {})),
        "source_route_waypoints": len(source_items),
        "route_waypoint_count": len(route_loop),
        "removed_waypoint_count": remove_end - remove_start + 1,
        "bridge_waypoint_count": len(bridge),
        "runtime_node_count": len(route_nodes),
        "access_plan_count": len(route_nodes),
        "rejected_by_v19_count": sum(rejection_counts.values()),
        "unsafe_access_option_count": len(unsafe_access_options),
        "ore_counts": dict(sorted(ore_counts.items())),
        "all_hazard_route_point_count": len(audit["hazard_point_indexes"]),
        "all_hazard_route_segment_count": len(audit["hazard_segment_indexes"]),
    }
    return result


def _option_coords(option: dict[str, Any]) -> tuple[int, ...]:
    values: list[int] = []
    for key in ("inbound", "return", "resume"):
        leg = option.get(key, {})
        if not isinstance(leg, dict):
            continue
        for coord in leg.get("coords", []):
            value = int(coord)
            if not values or values[-1] != value:
                values.append(value)
    return tuple(values)


def _grid(planner: NavMeshRouteEntryPlanner, coord: int) -> tuple[int, int]:
    x, y = coord_to_xy(int(coord))
    return tuple(
        int(round(value))
        for value in planner.raster.ui_to_grid(x, y, planner.zone)
    )


def _segment_is_hazardous(planner: NavMeshRouteEntryPlanner, first: int, second: int) -> bool:
    return any(
        planner.hazard_mask[row, col]
        for row, col in supercover_line(_grid(planner, first), _grid(planner, second))
    )


def _path_is_hazardous(planner: NavMeshRouteEntryPlanner, coords: Sequence[int]) -> bool:
    return any(planner.coord_is_hazard(coord) for coord in coords) or any(
        _segment_is_hazardous(planner, first, second)
        for first, second in zip(coords, coords[1:])
    )


def _audit_route(planner: NavMeshRouteEntryPlanner, coords: Sequence[int]) -> dict[str, Any]:
    points = [index for index, coord in enumerate(coords) if planner.coord_is_hazard(coord)]
    segments = [
        index
        for index, (first, second) in enumerate(
            zip(coords, (*coords[1:], coords[0]))
        )
        if _segment_is_hazardous(planner, first, second)
    ]
    return {
        "hazard_point_indexes": points,
        "hazard_segment_indexes": segments,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    planner = NavMeshRouteEntryPlanner.from_config(load_config(args.config), zone_id=162)
    bridge, validation = generate_safe_bridge(planner, source)
    result = build_v19(
        source,
        planner=planner,
        bridge_coords=bridge,
        bridge_validation=validation,
        source_fingerprint=hashlib.sha256(source_bytes).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
