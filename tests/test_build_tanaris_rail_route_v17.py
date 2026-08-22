from scripts.build_tanaris_rail_route_v17 import build_v17
from vision_bot.coords import xy_to_coord


def _item(index: int, x: float, y: float = 50.0) -> dict:
    return {"index": index, "coord": xy_to_coord(x, y), "x": x, "y": y}


def test_v17_removes_complete_caverns_lobe_and_hazard_nodes() -> None:
    source = {
        "name": "v16",
        "route_loop": [_item(index, float(index)) for index in range(10)],
        "route_nodes": [
            {**_item(0, 2.5), "node_id": 25, "ore_type": "Gold"},
            {**_item(0, 8.5), "node_id": 85, "ore_type": "Iron"},
        ],
        "coverage_scan_points": [],
        "metrics": {},
        "route_override": {"reason": "v16"},
    }
    built = build_v17(
        source,
        bridge_coords=[xy_to_coord(2.5, 49.0), xy_to_coord(7.5, 49.0)],
        excluded_node_ids=frozenset({25}),
        remove_start=2,
        remove_end=7,
    )

    assert len(built["route_loop"]) == 6
    assert built["route_loop"][2]["source"] == (
        "caverns_of_time_v17_western_exclusion_bridge"
    )
    assert built["route_loop"][4]["v16_index"] == 8
    assert [node["node_id"] for node in built["route_nodes"]] == [85]
    assert built["source"]["removed_v16_indexes"] == [2, 7]
    assert built["route_override"]["preserved_v16_override"]["reason"] == "v16"


def test_v17_removes_hazard_access_options_and_scan_points() -> None:
    safe = xy_to_coord(10.0, 10.0)
    danger = xy_to_coord(63.5, 49.7)
    plan = {
        "attachment_route_index": 1,
        "resume_route_index": 8,
        "primary_option_rank": 0,
        "options": [
            {
                "rank": 0,
                "primary": True,
                "approach_coord": danger,
                "inbound": {"waypoints": [{"coord": danger}]},
            },
            {
                "rank": 1,
                "primary": False,
                "approach_coord": safe,
                "inbound": {"waypoints": [{"coord": safe}]},
            },
        ],
    }
    source = {
        "name": "v16",
        "route_loop": [_item(index, float(index)) for index in range(10)],
        "route_nodes": [
            {
                **_item(0, 8.5),
                "node_id": 85,
                "ore_type": "Iron",
                "terrain_access_plan": plan,
            }
        ],
        "coverage_scan_points": [
            {"coord": safe, "index": 0},
            {"coord": danger, "index": 1},
        ],
        "metrics": {},
        "route_override": {},
    }
    built = build_v17(
        source,
        bridge_coords=[xy_to_coord(2.5, 49.0), xy_to_coord(7.5, 49.0)],
        coord_is_hazard=lambda coord: coord == danger,
        remove_start=2,
        remove_end=7,
    )

    options = built["route_nodes"][0]["terrain_access_plan"]["options"]
    assert len(options) == 1
    assert options[0]["approach_coord"] == safe
    assert options[0]["rank"] == 0
    assert options[0]["primary"] is True
    assert [scan["coord"] for scan in built["coverage_scan_points"]] == [safe]
    assert built["metrics"]["removed_v17_hazard_access_option_count"] == 1
