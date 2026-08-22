from scripts.build_tanaris_rail_route_v15 import build_v15
from vision_bot.coords import xy_to_coord


def _item(index: int, x: float, y: float = 70.0) -> dict:
    return {"index": index, "coord": xy_to_coord(x, y), "x": x, "y": y}


def test_v15_removes_complete_spur_and_renames_source_lineage() -> None:
    source = {
        "name": "v14",
        "route_loop": [_item(index, float(index)) for index in range(10)],
        "route_nodes": [{**_item(0, 8.1), "node_id": 8, "ore_type": "Gold"}],
        "coverage_scan_points": [],
        "metrics": {},
    }
    built = build_v15(
        source,
        detour_coords=[xy_to_coord(3.5, 69.0)],
        remove_start=3,
        remove_end=6,
    )

    assert len(built["route_loop"]) == 7
    assert built["route_loop"][3]["source"] == "live_observed_gaping_v15_spur_bridge"
    assert built["route_loop"][4]["v14_index"] == 7
    assert "v13_index" not in built["route_loop"][4]
    assert built["source"]["removed_v14_indexes"] == [3, 6]
