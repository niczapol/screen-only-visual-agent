from __future__ import annotations

import json

from scripts.build_tanaris_rail_route_v9 import (
    DEFAULT_EXCLUDED_NODE_IDS,
    build_broken_pillar_bypass_route,
)
from vision_bot.coords import xy_to_coord


def test_v9_bypasses_broken_pillar_structure_and_enclosed_nodes() -> None:
    source = json.loads(
        open(
            "data/routes/generated/tanaris_terrain_coverage_cycle_v8_rail.json",
            encoding="utf-8",
        ).read()
    )
    route = build_broken_pillar_bypass_route(
        source,
        detour_coords=(xy_to_coord(52.56, 47.50), xy_to_coord(54.75, 46.55)),
    )

    assert len(route["route_loop"]) == len(source["route_loop"]) - 7 + 2
    assert route["route_loop"][329]["source"] == "live_observed_broken_pillar_bypass"
    assert {node["node_id"] for node in route["route_nodes"]}.isdisjoint(
        DEFAULT_EXCLUDED_NODE_IDS
    )
    assert route["metrics"]["excluded_runtime_node_count"] == 2
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
    assert all(
        0 <= node["terrain_access_plan"]["attachment_route_index"] < len(route["route_loop"])
        for node in route["route_nodes"]
        if isinstance(node.get("terrain_access_plan"), dict)
    )
