from __future__ import annotations

import json

from scripts.build_tanaris_rail_route_v12 import build_dunemaul_bypass_route
from vision_bot.coords import xy_to_coord


def test_v12_replaces_dunemaul_pack_and_remaps_route_metadata() -> None:
    source = json.loads(
        open(
            "data/routes/generated/tanaris_terrain_coverage_cycle_v11_rail.json",
            encoding="utf-8",
        ).read()
    )
    detour = (
        xy_to_coord(47.70, 50.50),
        xy_to_coord(47.70, 64.00),
        xy_to_coord(34.00, 64.00),
    )

    route = build_dunemaul_bypass_route(source, detour_coords=detour)

    assert len(route["route_loop"]) == len(source["route_loop"]) - 28 + 3
    assert route["route_loop"][36]["source"] == "live_observed_dunemaul_pack_bypass"
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
    assert all(
        not (33.5 <= float(node["x"]) <= 47.5 and 50.8 <= float(node["y"]) <= 63.8)
        for node in route["route_nodes"]
    )
    assert route["route_override"]["observed_death_coordinate"] == {
        "x": 37.73,
        "y": 54.96,
    }
