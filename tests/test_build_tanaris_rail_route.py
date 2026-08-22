from scripts.build_tanaris_rail_route import build_rail_route
from vision_bot.coords import xy_to_coord


def _node(index: int, x: float, y: float, *, status: str = "covered") -> dict:
    return {
        "index": index,
        "zone_id": 162,
        "coord": xy_to_coord(x, y),
        "x": x,
        "y": y,
        "ore_type": "Gold",
        "coverage_status": status,
    }


def test_build_rail_route_rotates_loop_adds_bridge_and_filters_nodes() -> None:
    source = {
        "name": "source",
        "zone_id": 162,
        "route_loop": [
            {"index": index, "coord": xy_to_coord(float(index), 10.0)}
            for index in range(8)
        ],
        "route_nodes": [
            _node(1, 2.1, 10.0),
            _node(2, 6.1, 10.0),
            _node(3, 0.1, 10.0),
            _node(4, 4.1, 10.0, status="not_covered"),
        ],
        "coverage_scan_points": [],
    }
    excluded = {xy_to_coord(6.1, 10.0): "explicit"}

    built = build_rail_route(
        source,
        bridge_coords=[xy_to_coord(3.5, 10.5), xy_to_coord(2.0, 10.0)],
        retained_start_index=2,
        retained_end_index=6,
        max_node_distance=0.5,
        excluded_nodes=excluded,
    )

    assert [item["source_index"] for item in built["route_loop"][:5]] == [2, 3, 4, 5, 6]
    assert built["route_loop"][-1]["source"] == "navmesh_direct_bridge"
    assert built["route_loop"][-1]["coord"] == xy_to_coord(3.5, 10.5)
    assert [item["source_node_index"] for item in built["route_nodes"]] == [1]
    reasons = {item["index"]: item["route_reject_reason"] for item in built["rejected_nodes"]}
    assert reasons[2] == "user_excluded_spawn"
    assert reasons[3] == "outside_v5_runtime_corridor"
    assert reasons[4] == "not_covered"


def test_build_rail_route_remaps_or_drops_access_plan_indexes() -> None:
    source = {
        "route_loop": [
            {"index": index, "coord": xy_to_coord(float(index), 10.0)}
            for index in range(8)
        ],
        "route_nodes": [
            {
                **_node(1, 3.1, 10.0),
                "terrain_access_plan": {
                    "attachment_route_index": 3,
                    "resume_route_index": 4,
                    "options": [],
                },
            },
            {
                **_node(2, 5.1, 10.0),
                "terrain_access_plan": {
                    "attachment_route_index": 6,
                    "resume_route_index": 7,
                    "options": [],
                },
            },
        ],
    }

    built = build_rail_route(
        source,
        bridge_coords=[],
        retained_start_index=2,
        retained_end_index=6,
        max_node_distance=0.5,
        excluded_nodes={},
    )

    first, second = built["route_nodes"]
    assert first["terrain_access_plan"]["attachment_route_index"] == 1
    assert first["terrain_access_plan"]["resume_route_index"] == 2
    assert "terrain_access_plan" not in second
    assert second["terrain_access_plan_reject_reason"] == "removed_route_attachment"
