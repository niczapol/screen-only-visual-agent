from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.build_tanaris_rail_route_v14 import build_v14
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_tanaris_rail_route_v14 import build_v14
from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner
from vision_bot.terrain_routing import supercover_line


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v16_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v17_rail.json")
DEFAULT_REMOVE_START = 336
DEFAULT_REMOVE_END = 433
DEFAULT_START_INDEX = DEFAULT_REMOVE_START - 1
DEFAULT_END_INDEX = DEFAULT_REMOVE_END + 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tanaris rail V17 with the complete Caverns of Time surface "
            "approach removed and a terrain/hazard-validated western bridge."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def generate_v17_bridge(
    planner: NavMeshRouteEntryPlanner,
    source: dict[str, Any],
    *,
    start_index: int = DEFAULT_START_INDEX,
    end_index: int = DEFAULT_END_INDEX,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    route_loop = source.get("route_loop", [])
    start_coord = int(route_loop[start_index]["coord"])
    end_coord = int(route_loop[end_index]["coord"])
    result = planner.plan_to_target(
        start_coord,
        end_coord,
        target_route_index=end_index,
    )
    if not result.terrain_validation.get("walkable", False):
        raise RuntimeError("V17 Caverns of Time bridge is not walkable")
    coords = [int(waypoint.coord) for waypoint in result.entry_plan.waypoints]
    if coords and coords[-1] == end_coord:
        coords.pop()
    return tuple(coords), [
        {
            "leg": 0,
            "world_length_yards": round(result.world_length_yards, 4),
            **copy.deepcopy(result.terrain_validation),
        }
    ]


def _option_coords(option: dict[str, Any]) -> list[int]:
    coords: list[int] = []
    for key in ("approach_coord", "facing_stage_coord"):
        value = option.get(key)
        if value is not None:
            coords.append(int(value))
    for phase_name in ("inbound", "return", "resume"):
        phase = option.get(phase_name)
        if not isinstance(phase, dict):
            continue
        for waypoint in phase.get("waypoints", []):
            if isinstance(waypoint, dict) and waypoint.get("coord") is not None:
                coords.append(int(waypoint["coord"]))
    return coords


def _filter_hazard_access_options(
    route_nodes: list[dict[str, Any]],
    *,
    coord_is_hazard: Callable[[int], bool],
    segment_crosses_hazard: Callable[[int, int], bool] | None = None,
) -> tuple[int, int]:
    removed_options = 0
    removed_plans = 0
    for node in route_nodes:
        plan = node.get("terrain_access_plan")
        if not isinstance(plan, dict):
            continue
        options = []
        for option in plan.get("options", []):
            if not isinstance(option, dict):
                continue
            coords = _option_coords(option)
            if any(coord_is_hazard(coord) for coord in coords):
                continue
            if segment_crosses_hazard is not None and any(
                segment_crosses_hazard(first, second)
                for first, second in zip(coords, coords[1:])
            ):
                continue
            options.append(copy.deepcopy(option))
        removed_options += len(plan.get("options", [])) - len(options)
        if not options:
            node.pop("terrain_access_plan", None)
            node["terrain_access_plan_reject_reason"] = "v17_hazard_access_path"
            removed_plans += 1
            continue
        options.sort(key=lambda item: int(item.get("rank", item.get("candidate_index", 0))))
        for rank, option in enumerate(options):
            option["rank"] = rank
            option["primary"] = rank == 0
        plan["options"] = options
        plan["primary_option_rank"] = 0
    return removed_options, removed_plans


def build_v17(
    source: dict[str, Any],
    *,
    bridge_coords: Sequence[int],
    terrain_validation: Sequence[dict[str, Any]] = (),
    excluded_node_ids: frozenset[int] = frozenset(),
    coord_is_hazard: Callable[[int], bool] | None = None,
    segment_crosses_hazard: Callable[[int, int], bool] | None = None,
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
) -> dict[str, Any]:
    result = build_v14(
        source,
        detour_coords=bridge_coords,
        terrain_validation=terrain_validation,
        remove_start=remove_start,
        remove_end=remove_end,
        excluded_node_ids=excluded_node_ids,
    )
    for item in result["route_loop"]:
        if item.get("source") == "live_observed_gaping_v14_west_bypass":
            item["source"] = "caverns_of_time_v17_western_exclusion_bridge"
        if "v13_index" in item:
            item["v16_index"] = item.pop("v13_index")
    for node in result["route_nodes"]:
        node["source"] = "tanaris_v17_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict) and "v13_attachment_route_index" in plan:
            plan["v16_attachment_route_index"] = plan.pop(
                "v13_attachment_route_index"
            )
            plan["v16_resume_route_index"] = plan.pop("v13_resume_route_index")

    removed_options = 0
    removed_plans = 0
    if coord_is_hazard is not None:
        removed_options, removed_plans = _filter_hazard_access_options(
            result["route_nodes"],
            coord_is_hazard=coord_is_hazard,
            segment_crosses_hazard=segment_crosses_hazard,
        )
        result["coverage_scan_points"] = [
            scan
            for scan in result.get("coverage_scan_points", [])
            if not coord_is_hazard(int(scan["coord"]))
        ]
        for index, scan in enumerate(result["coverage_scan_points"]):
            scan["index"] = index

    result["name"] = (
        "Tanaris terrain-aware full rail cycle v17 Caverns of Time fully excluded"
    )
    result["source"] = {
        "route": str(source.get("name") or "unknown"),
        "removed_v16_indexes": [remove_start, remove_end],
        "excluded_runtime_node_ids": sorted(excluded_node_ids),
    }
    result["route_override"] = {
        "reason": (
            "V0.8.25 died at the visible coordinate 63.53,49.70 beside "
            "Anachronos and a Bronze Whelp after the V16 x=61 rail left too "
            "little clearance. V17 removes the complete eastern lobe, every "
            "Caverns-area ore/access path, and bridges around the west side "
            "of the expanded 45-yard exclusion."
        ),
        "observed_death_coordinate": {"x": 63.53, "y": 49.70},
        "stale_filtered_coordinate_at_death": {"x": 61.03, "y": 50.27},
        "removed_v16_indexes": [remove_start, remove_end],
        "bridge_waypoint_count": len(bridge_coords),
        "bridge_xy": [
            {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
            for coord in bridge_coords
        ],
        "terrain_validation": list(terrain_validation),
        "excluded_runtime_node_ids": sorted(excluded_node_ids),
        "removed_hazard_access_option_count": removed_options,
        "removed_hazard_access_plan_count": removed_plans,
        "preserved_v16_override": copy.deepcopy(source.get("route_override", {})),
    }
    result["metrics"].update(
        {
            "excluded_v17_hazard_node_count": len(excluded_node_ids),
            "removed_v17_hazard_access_option_count": removed_options,
            "removed_v17_hazard_access_plan_count": removed_plans,
            "access_plan_count": sum(
                isinstance(node.get("terrain_access_plan"), dict)
                for node in result["route_nodes"]
            ),
            "coverage_scan_point_count": len(result.get("coverage_scan_points", [])),
        }
    )
    result["metrics"].pop("excluded_v14_runtime_node_count", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    planner = NavMeshRouteEntryPlanner.from_config(load_config(args.config), zone_id=162)
    bridge, validation = generate_v17_bridge(planner, source)
    excluded = frozenset(
        int(node.get("node_id", -1))
        for node in source.get("route_nodes", [])
        if planner.coord_is_hazard(int(node["coord"]))
    )

    def segment_crosses_hazard(first: int, second: int) -> bool:
        first_xy = coord_to_xy(first)
        second_xy = coord_to_xy(second)
        first_grid = tuple(
            int(round(value))
            for value in planner.raster.ui_to_grid(*first_xy, planner.zone)
        )
        second_grid = tuple(
            int(round(value))
            for value in planner.raster.ui_to_grid(*second_xy, planner.zone)
        )
        rows, cols = planner.hazard_mask.shape
        return any(
            0 <= row < rows
            and 0 <= col < cols
            and bool(planner.hazard_mask[row, col])
            for row, col in supercover_line(first_grid, second_grid)
        )

    route = build_v17(
        source,
        bridge_coords=bridge,
        terrain_validation=validation,
        excluded_node_ids=excluded,
        coord_is_hazard=planner.coord_is_hazard,
        segment_crosses_hazard=segment_crosses_hazard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
