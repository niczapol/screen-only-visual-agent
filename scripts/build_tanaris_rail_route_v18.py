from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.build_tanaris_rail_route_v17 import _option_coords
except ModuleNotFoundError:
    from build_tanaris_rail_route_v17 import _option_coords
from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy
from vision_bot.navmesh_route_entry import NavMeshRouteEntryPlanner, build_hazard_mask
from vision_bot.terrain_routing import supercover_line


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v16_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v18_rail.json")
DEFAULT_HAZARDS = Path("data/routes/live_hazards/tanaris.json")
GORGE_KIND = "caverns_of_time_gorge_interior"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V18 by restoring V16 coverage around a narrow Caverns gorge veto."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hazards", type=Path, default=DEFAULT_HAZARDS)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def build_v18(source: dict[str, Any], *, audit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result["name"] = "Tanaris terrain-aware full rail cycle v18 narrow Caverns gorge veto"
    result["status"] = "offline_full_route_candidate_not_live_validated"
    result["source"] = {
        "route": str(source.get("name") or "unknown"),
        "restored_v16_route_geometry": True,
        "restored_v16_route_nodes": True,
    }
    result["route_override"] = {
        "reason": (
            "V17 over-excluded the entire eastern mining sector. V18 restores "
            "all V16 route geometry and marked DB nodes because exact raster "
            "audit shows the rail, nodes and access paths remain outside the "
            "actual dark Caverns gorge. Only the gorge interior around the "
            "lethal 63.53,49.70 coordinate remains forbidden."
        ),
        "observed_death_coordinate": {"x": 63.53, "y": 49.70},
        "gorge_hazard_kind": GORGE_KIND,
        "audit": copy.deepcopy(audit),
        "preserved_v16_override": copy.deepcopy(source.get("route_override", {})),
    }
    metrics = result.setdefault("metrics", {})
    metrics.update(
        {
            "route_waypoint_count": len(result.get("route_loop", [])),
            "runtime_node_count": len(result.get("route_nodes", [])),
            "access_plan_count": sum(
                isinstance(node.get("terrain_access_plan"), dict)
                for node in result.get("route_nodes", [])
            ),
            "v18_gorge_route_point_count": int(audit["route_point_count"]),
            "v18_gorge_route_segment_count": int(audit["route_segment_count"]),
            "v18_gorge_node_count": int(audit["node_count"]),
            "v18_gorge_access_option_count": int(audit["access_option_count"]),
            "v18_restored_v16_node_count": len(result.get("route_nodes", [])),
        }
    )
    return result


def audit_gorge(
    source: dict[str, Any],
    planner: NavMeshRouteEntryPlanner,
    hazard: dict[str, Any],
) -> dict[str, Any]:
    mask = build_hazard_mask(
        planner.hazard_mask.shape,
        raster=planner.raster,
        zone=planner.zone,
        hazards=[hazard],
    )

    def grid(coord: int) -> tuple[int, int]:
        x, y = coord_to_xy(int(coord))
        return tuple(
            int(round(value))
            for value in planner.raster.ui_to_grid(x, y, planner.zone)
        )

    def inside(coord: int) -> bool:
        row, col = grid(coord)
        return bool(mask[row, col])

    def crosses(first: int, second: int) -> bool:
        return any(mask[row, col] for row, col in supercover_line(grid(first), grid(second)))

    route_coords = [int(item["coord"]) for item in source.get("route_loop", [])]
    route_points = [index for index, coord in enumerate(route_coords) if inside(coord)]
    route_segments = [
        index
        for index, (first, second) in enumerate(zip(route_coords, route_coords[1:]))
        if crosses(first, second)
    ]
    nodes = [
        int(node["node_id"])
        for node in source.get("route_nodes", [])
        if inside(int(node["coord"]))
    ]
    access_options: list[list[int]] = []
    for node in source.get("route_nodes", []):
        plan = node.get("terrain_access_plan")
        if not isinstance(plan, dict):
            continue
        for option in plan.get("options", []):
            coords = _option_coords(option)
            if any(inside(coord) for coord in coords) or any(
                crosses(first, second) for first, second in zip(coords, coords[1:])
            ):
                access_options.append([int(node["node_id"]), int(option.get("rank", -1))])
    return {
        "route_point_count": len(route_points),
        "route_point_indexes": route_points,
        "route_segment_count": len(route_segments),
        "route_segment_indexes": route_segments,
        "node_count": len(nodes),
        "node_ids": nodes,
        "access_option_count": len(access_options),
        "access_options": access_options,
        "death_coordinate_inside": inside(6353497000),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    hazards = json.loads(args.hazards.read_text(encoding="utf-8"))
    hazard = next(item for item in hazards["hazards"] if item.get("kind") == GORGE_KIND)
    planner = NavMeshRouteEntryPlanner.from_config(load_config(args.config), zone_id=162)
    audit = audit_gorge(source, planner, hazard)
    if not audit["death_coordinate_inside"]:
        raise RuntimeError("The lethal Caverns coordinate is not inside the V18 gorge veto")
    if any(
        audit[key]
        for key in (
            "route_point_count",
            "route_segment_count",
            "node_count",
            "access_option_count",
        )
    ):
        raise RuntimeError(f"V18 restored coverage intersects the gorge: {audit}")
    route = build_v18(source, audit=audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": route["metrics"], "audit": audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
