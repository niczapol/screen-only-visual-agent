from scripts.build_tanaris_rail_route_v14 import build_v14
from vision_bot.coords import xy_to_coord


def _item(index: int, x: float, y: float = 70.0) -> dict:
    return {"index": index, "coord": xy_to_coord(x, y), "x": x, "y": y}


def test_v14_replaces_boundary_remaps_access_and_excludes_blocked_node() -> None:
    source = {
        "name": "v13",
        "route_loop": [_item(index, float(index)) for index in range(10)],
        "route_nodes": [
            {
                **_item(0, 8.1),
                "node_id": 8,
                "ore_type": "Gold",
                "terrain_access_plan": {
                    "attachment_route_index": 8,
                    "resume_route_index": 9,
                    "options": [],
                },
            },
            {**_item(1, 4.1), "node_id": 50, "ore_type": "Mithril"},
        ],
        "coverage_scan_points": [{**_item(0, 8.2), "index": 0}],
        "metrics": {},
    }

    built = build_v14(
        source,
        detour_coords=[xy_to_coord(3.5, 69.0), xy_to_coord(6.5, 69.0)],
        remove_start=3,
        remove_end=6,
    )

    assert len(built["route_loop"]) == 8
    assert built["route_loop"][3]["source"] == "live_observed_gaping_v14_west_bypass"
    assert [node["node_id"] for node in built["route_nodes"]] == [8]
    plan = built["route_nodes"][0]["terrain_access_plan"]
    assert plan["v13_attachment_route_index"] == 8
    assert plan["attachment_route_index"] == 6
    assert plan["resume_route_index"] == 7
    assert built["metrics"]["excluded_v14_runtime_node_count"] == 1
