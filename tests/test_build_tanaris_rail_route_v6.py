from scripts.build_tanaris_rail_route_v6 import build_camp_bypass_route
from vision_bot.coords import xy_to_coord


def _item(index: int, x: float, y: float = 10.0) -> dict:
    return {"index": index, "coord": xy_to_coord(x, y), "x": x, "y": y}


def test_camp_bypass_replaces_interior_and_remaps_access_plans() -> None:
    source = {
        "name": "v5",
        "zone_id": 162,
        "route_loop": [_item(index, float(index)) for index in range(8)],
        "route_nodes": [
            {
                **_item(1, 6.1),
                "node_id": 1,
                "ore_type": "Iron",
                "terrain_access_plan": {
                    "attachment_route_index": 6,
                    "resume_route_index": 7,
                    "options": [],
                },
            }
        ],
        "coverage_scan_points": [{**_item(0, 6.5), "index": 0}],
        "rejected_nodes": [],
        "metrics": {},
    }

    built = build_camp_bypass_route(
        source,
        bypass_coords=[xy_to_coord(3.4, 10.5), xy_to_coord(4.0, 10.0)],
        replace_start=3,
        replace_end=4,
    )

    assert len(built["route_loop"]) == 9
    assert built["route_loop"][4]["source"] == "navmesh_camp_bypass"
    assert built["route_loop"][5]["v5_index"] == 4
    plan = built["route_nodes"][0]["terrain_access_plan"]
    assert plan["v5_attachment_route_index"] == 6
    assert plan["attachment_route_index"] == 7
    assert plan["resume_route_index"] == 8


def test_camp_bypass_rejects_access_plan_on_removed_waypoint() -> None:
    source = {
        "route_loop": [_item(index, float(index)) for index in range(8)],
        "route_nodes": [
            {
                **_item(1, 3.5),
                "node_id": 1,
                "terrain_access_plan": {
                    "attachment_route_index": 3,
                    "resume_route_index": 4,
                    "options": [],
                },
            }
        ],
        "coverage_scan_points": [],
    }

    try:
        build_camp_bypass_route(
            source,
            bypass_coords=[xy_to_coord(2.5, 10.5), xy_to_coord(5.0, 10.0)],
            replace_start=2,
            replace_end=5,
        )
    except ValueError as exc:
        assert "depends on replaced V5 waypoints" in str(exc)
    else:
        raise AssertionError("Expected replaced access-plan dependency to fail")
