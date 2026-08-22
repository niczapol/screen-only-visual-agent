from __future__ import annotations

from scripts.build_route_slice import build_route_slice, choose_best_window
from vision_bot.coords import xy_to_coord


def _access_plan(*, attachment: int, resume: int, coord: int) -> dict:
    return {
        "method": "test",
        "attachment_route_index": attachment,
        "resume_route_index": resume,
        "primary_option_rank": 0,
        "options": [
            {
                "rank": 0,
                "candidate_index": 0,
                "primary": True,
                "approach_coord": coord,
                "sector": 0,
                "bearing_degrees": 0.0,
                "objective": 0.0,
                "inbound": {"waypoints": [{"coord": coord}]},
                "return": {"waypoints": [{"coord": coord}]},
                "resume": {"waypoints": [{"coord": coord}]},
            }
        ],
    }


def test_build_route_slice_remaps_loop_and_access_plan_indexes():
    loop = [
        {"index": index, "coord": xy_to_coord(float(index), 10.0), "x": index, "y": 10.0}
        for index in range(6)
    ]
    node_coord = xy_to_coord(2.1, 10.0)
    data = {
        "schema_version": 1,
        "name": "source route",
        "zone_id": 162,
        "route_loop": loop,
        "route_nodes": [
            {
                "index": 10,
                "zone_id": 162,
                "coord": node_coord,
                "x": 2.1,
                "y": 10.0,
                "ore_type": "Iron",
                "ore_id": 1731,
                "coverage_status": "covered",
                "terrain_access_plan": _access_plan(
                    attachment=2,
                    resume=3,
                    coord=node_coord,
                ),
            },
            {
                "index": 11,
                "zone_id": 162,
                "coord": xy_to_coord(5.0, 10.0),
                "x": 5.0,
                "y": 10.0,
                "ore_type": "Iron",
                "ore_id": 1731,
                "coverage_status": "covered",
            },
        ],
    }

    sliced = build_route_slice(
        data,
        start_index=1,
        window_size=3,
        max_node_distance_to_route=0.5,
        output_name="slice",
    )

    assert [item["source_index"] for item in sliced["route_loop"]] == [1, 2, 3]
    assert len(sliced["route_nodes"]) == 1
    node = sliced["route_nodes"][0]
    assert node["route_index"] == 1
    assert node["source_route_index"] == 2
    plan = node["terrain_access_plan"]
    assert plan["source_attachment_route_index"] == 2
    assert plan["source_resume_route_index"] == 3
    assert plan["attachment_route_index"] == 1
    assert plan["resume_route_index"] == 2


def test_choose_best_window_prefers_covered_node_density():
    loop = [
        {"index": index, "coord": xy_to_coord(float(index), 20.0), "x": index, "y": 20.0}
        for index in range(8)
    ]
    nodes = [
        {
            "index": index,
            "zone_id": 162,
            "coord": xy_to_coord(float(index) + 0.1, 20.0),
            "x": float(index) + 0.1,
            "y": 20.0,
            "ore_type": "Iron",
            "coverage_status": "covered",
        }
        for index in (4, 5, 6)
    ]
    data = {"route_loop": loop, "route_nodes": nodes}

    assert choose_best_window(data, size=3, max_distance=0.5) == 4
