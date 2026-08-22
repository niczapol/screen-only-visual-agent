from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.build_tanaris_rail_route_v14 import build_v14
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_tanaris_rail_route_v14 import build_v14
from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v15_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v16_rail.json")
DEFAULT_REMOVE_START = 206
DEFAULT_REMOVE_END = 207
DEFAULT_START_XY = (31.68, 74.18)
DEFAULT_END_XY = (29.09, 72.83)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tanaris rail V16 around the live-confirmed southwest "
            "wreckage contact at 30.65,73.82."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def generate_v16_detour(
    planner: NavMeshRouteEntryPlanner,
    *,
    start_xy: tuple[float, float] = DEFAULT_START_XY,
    end_xy: tuple[float, float] = DEFAULT_END_XY,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    result = planner.plan_to_target(
        xy_to_coord(*start_xy),
        xy_to_coord(*end_xy),
        target_route_index=0,
    )
    if not result.terrain_validation.get("walkable", False):
        raise RuntimeError("V16 southwest hard-contact detour is not walkable")
    coords = [int(waypoint.coord) for waypoint in result.entry_plan.waypoints]
    if coords and coords[-1] == xy_to_coord(*end_xy):
        coords.pop()
    return tuple(coords), [
        {
            "leg": 0,
            "world_length_yards": round(result.world_length_yards, 4),
            **result.terrain_validation,
        }
    ]


def build_v16(
    source: dict[str, Any],
    *,
    detour_coords: Sequence[int],
    terrain_validation: Sequence[dict[str, Any]] = (),
    remove_start: int = DEFAULT_REMOVE_START,
    remove_end: int = DEFAULT_REMOVE_END,
) -> dict[str, Any]:
    result = build_v14(
        source,
        detour_coords=detour_coords,
        terrain_validation=terrain_validation,
        remove_start=remove_start,
        remove_end=remove_end,
        excluded_node_ids=frozenset(),
    )
    for item in result["route_loop"]:
        if item.get("source") == "live_observed_gaping_v14_west_bypass":
            item["source"] = "live_observed_southwest_v16_wreckage_bypass"
        if "v13_index" in item:
            item["v15_index"] = item.pop("v13_index")
    for node in result["route_nodes"]:
        node["source"] = "tanaris_v16_rail_node"
        plan = node.get("terrain_access_plan")
        if isinstance(plan, dict) and "v13_attachment_route_index" in plan:
            plan["v15_attachment_route_index"] = plan.pop(
                "v13_attachment_route_index"
            )
            plan["v15_resume_route_index"] = plan.pop(
                "v13_resume_route_index"
            )

    result["name"] = "Tanaris terrain-aware full rail cycle v16 southwest hard-contact bypass"
    result["source"] = {
        "route": str(source.get("name") or "unknown"),
        "removed_v15_indexes": [remove_start, remove_end],
    }
    result["route_override"] = {
        "reason": (
            "RUN2, RUN3 and RUN4 remained fixed at 30.65,73.82 through "
            "mounted jump/back/turn recovery on both sides. V16 permanently "
            "routes north of the live-confirmed wreckage contact."
        ),
        "observed_hard_stuck_coordinate": {"x": 30.65, "y": 73.82},
        "removed_v15_indexes": [remove_start, remove_end],
        "detour_waypoint_count": len(detour_coords),
        "detour_xy": [
            {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
            for coord in detour_coords
        ],
        "terrain_validation": list(terrain_validation),
        "preserved_v15_override": copy.deepcopy(source.get("route_override", {})),
    }
    result["metrics"]["excluded_v16_hazard_node_count"] = 0
    result["metrics"].pop("excluded_v14_runtime_node_count", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    planner = NavMeshRouteEntryPlanner.from_config(load_config(args.config), zone_id=162)
    detour, validation = generate_v16_detour(planner)
    route = build_v16(
        source,
        detour_coords=detour,
        terrain_validation=validation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
