from __future__ import annotations

import json

from scripts.build_tanaris_rail_route_v10 import build_wide_broken_pillar_bypass_route
from vision_bot.coords import xy_to_coord


def test_v10_replaces_narrow_bypass_with_wide_dense_arc() -> None:
    source = json.loads(
        open(
            "data/routes/generated/tanaris_terrain_coverage_cycle_v9_rail.json",
            encoding="utf-8",
        ).read()
    )
    detour = (
        xy_to_coord(52.62, 48.92),
        xy_to_coord(54.80, 49.10),
        xy_to_coord(55.56, 46.98),
    )

    route = build_wide_broken_pillar_bypass_route(source, detour_coords=detour)

    assert len(route["route_loop"]) == len(source["route_loop"]) - 12 + 3
    assert route["route_loop"][328]["source"] == "live_observed_broken_pillar_bypass"
    assert route["route_override"]["runtime_lookahead_override"]["distance"] == 0.25
    assert len(route["route_nodes"]) == len(source["route_nodes"])
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
