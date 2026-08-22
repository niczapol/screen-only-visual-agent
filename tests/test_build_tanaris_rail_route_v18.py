from scripts.build_tanaris_rail_route_v18 import build_v18
from vision_bot.coords import xy_to_coord


def test_v18_restores_v16_geometry_and_nodes_with_zero_gorge_audit() -> None:
    source = {
        "name": "v16",
        "route_loop": [
            {"index": 0, "coord": xy_to_coord(61.0, 48.0)},
            {"index": 1, "coord": xy_to_coord(61.0, 49.0)},
        ],
        "route_nodes": [
            {"node_id": 135, "coord": xy_to_coord(61.1, 48.3), "terrain_access_plan": {}}
        ],
        "route_override": {"reason": "v16"},
        "metrics": {},
    }
    audit = {
        "route_point_count": 0,
        "route_segment_count": 0,
        "node_count": 0,
        "access_option_count": 0,
        "death_coordinate_inside": True,
    }

    built = build_v18(source, audit=audit)

    assert built["route_loop"] == source["route_loop"]
    assert built["route_nodes"] == source["route_nodes"]
    assert built["metrics"]["route_waypoint_count"] == 2
    assert built["metrics"]["runtime_node_count"] == 1
    assert built["route_override"]["preserved_v16_override"]["reason"] == "v16"
