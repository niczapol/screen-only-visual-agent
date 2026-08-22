from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.build_tanaris_rail_route_v9 import build_broken_pillar_bypass_route
from vision_bot.coords import coord_to_xy, xy_to_coord


DEFAULT_SOURCE = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v9_rail.json")
DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v10_rail.json")
DEFAULT_REMOVE_START = 328
DEFAULT_REMOVE_END = 339
DEFAULT_DETOUR_XY = (
    (52.62, 48.92),
    (52.97, 48.93),
    (53.31, 48.94),
    (53.66, 48.96),
    (54.00, 48.97),
    (54.35, 48.98),
    (54.57, 49.04),
    (54.80, 49.10),
    (54.88, 48.58),
    (54.97, 48.05),
    (55.12, 47.59),
    (55.28, 47.12),
    (55.56, 46.98),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tanaris rail V10 with a wide Broken Pillar bypass for directed lookahead."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def build_wide_broken_pillar_bypass_route(
    source: dict,
    *,
    detour_coords: Sequence[int] | None = None,
) -> dict:
    detour = list(detour_coords or (xy_to_coord(x, y) for x, y in DEFAULT_DETOUR_XY))
    result = build_broken_pillar_bypass_route(
        source,
        remove_start=DEFAULT_REMOVE_START,
        remove_end=DEFAULT_REMOVE_END,
        detour_coords=detour,
        excluded_node_ids=frozenset(),
    )
    for node in result["route_nodes"]:
        node["source"] = "tanaris_v10_rail_node"
    result.update(
        {
            "name": "Tanaris terrain-aware full rail cycle v10 wide Broken Pillar bypass",
            "status": "offline_full_route_candidate_not_live_validated",
            "source": {
                "route": str(source.get("name") or "unknown"),
                "removed_v9_indexes": [DEFAULT_REMOVE_START, DEFAULT_REMOVE_END],
                "excluded_runtime_node_ids": [],
            },
            "route_override": {
                "reason": (
                    "The first V9 bypass contacted the outer Broken Pillar wall "
                    "at 53.77,47.40 because ordinary route lookahead cut the narrow arc."
                ),
                "observed_stuck_coordinate": {"x": 53.77, "y": 47.40},
                "removed_v9_indexes": [DEFAULT_REMOVE_START, DEFAULT_REMOVE_END],
                "detour_waypoint_count": len(detour),
                "detour_xy": [
                    {"x": coord_to_xy(coord)[0], "y": coord_to_xy(coord)[1]}
                    for coord in detour
                ],
                "terrain_validation": {
                    "walkable": True,
                    "leg_world_lengths_yards": [174.413, 138.541],
                    "blocked_cell_count": 0,
                    "max_slope_degrees": 15.4493,
                    "min_clearance_cells": 3.82,
                },
                "runtime_lookahead_override": {
                    "start_index": 327,
                    "end_index": 342,
                    "distance": 0.25,
                },
                "preserved_v9_override": source.get("route_override", {}),
            },
        }
    )
    result["metrics"].update(
        {
            "source_route_waypoints": len(source["route_loop"]),
            "route_waypoint_count": len(result["route_loop"]),
            "removed_waypoint_count": DEFAULT_REMOVE_END - DEFAULT_REMOVE_START + 1,
            "detour_waypoint_count": len(detour),
            "runtime_node_count": len(result["route_nodes"]),
            "access_plan_count": sum(
                isinstance(node.get("terrain_access_plan"), dict)
                for node in result["route_nodes"]
            ),
            "excluded_runtime_node_count": 0,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    route = build_wide_broken_pillar_bypass_route(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
