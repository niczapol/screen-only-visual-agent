from __future__ import annotations

import json

from scripts.build_tanaris_rail_route_v8 import (
    DEFAULT_EXCLUDED_NODE_IDS,
    build_gaping_boundary_detour_route,
)
from vision_bot.coords import xy_to_coord


def test_v8_removes_live_observed_gaping_south_boundary_spur_and_nodes() -> None:
    source = json.loads(
        open(
            "data/routes/generated/tanaris_terrain_coverage_cycle_v7_rail.json",
            encoding="utf-8",
        ).read()
    )
    route = build_gaping_boundary_detour_route(
        source,
        detour_coords=(
            xy_to_coord(49.25, 77.25),
            xy_to_coord(49.25, 76.65),
        ),
    )

    assert len(route["route_loop"]) == len(source["route_loop"]) - 10 + 2
    assert route["route_loop"][121]["source"] == "live_observed_gaping_south_boundary_detour"
    assert {node["node_id"] for node in route["route_nodes"]}.isdisjoint(
        DEFAULT_EXCLUDED_NODE_IDS
    )
    assert route["metrics"]["excluded_runtime_node_count"] == 2
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
    assert all(
        0 <= node["route_index"] < len(route["route_loop"])
        for node in route["route_nodes"]
    )
    assert all(
        0 <= node["terrain_access_plan"]["attachment_route_index"]
        < len(route["route_loop"])
        for node in route["route_nodes"]
        if isinstance(node.get("terrain_access_plan"), dict)
    )
