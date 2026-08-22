from __future__ import annotations

import json

from scripts.build_tanaris_rail_route_v11 import build_noxious_north_bypass_route
from vision_bot.coords import xy_to_coord


def test_v11_replaces_noxious_north_surface_branch_and_excludes_enclosed_nodes() -> None:
    source = json.loads(open("data/routes/generated/tanaris_terrain_coverage_cycle_v10_rail.json", encoding="utf-8").read())
    detour = (xy_to_coord(34.40, 26.63), xy_to_coord(41.65, 32.80), xy_to_coord(41.25, 38.55))
    route = build_noxious_north_bypass_route(source, detour_coords=detour)
    assert len(route["route_loop"]) == len(source["route_loop"]) - 21 + 3 - 18 + 18
    assert route["route_loop"][10]["source"] == "live_observed_noxious_north_bypass"
    assert {n["node_id"] for n in route["route_nodes"]}.isdisjoint({46, 47, 49, 52, 53, 54})
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
    assert all(
        not (32.0 <= float(item["x"]) <= 40.8 and 29.0 <= float(item["y"]) <= 53.5)
        for item in route["route_loop"]
    )
    assert any(item["source"] == "live_observed_noxious_east_bypass" for item in route["route_loop"])
