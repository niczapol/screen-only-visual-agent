from scripts.build_tanaris_rail_route_v16 import build_v16
from vision_bot.coords import xy_to_coord


def _item(index: int, x: float, y: float = 70.0) -> dict:
    return {"index": index, "coord": xy_to_coord(x, y), "x": x, "y": y}


def test_v16_replaces_hard_contact_segment_and_preserves_lineage() -> None:
    source = {
        "name": "v15",
        "route_loop": [_item(index, float(index)) for index in range(10)],
        "route_nodes": [{**_item(0, 8.1), "node_id": 8, "ore_type": "Gold"}],
        "coverage_scan_points": [],
        "metrics": {},
        "route_override": {"reason": "v15"},
    }
    built = build_v16(
        source,
        detour_coords=[xy_to_coord(3.5, 69.0), xy_to_coord(4.0, 69.0)],
        remove_start=3,
        remove_end=4,
    )

    assert len(built["route_loop"]) == 10
    assert built["route_loop"][3]["source"] == "live_observed_southwest_v16_wreckage_bypass"
    assert built["route_loop"][5]["v15_index"] == 5
    assert "v13_index" not in built["route_loop"][5]
    assert built["source"]["removed_v15_indexes"] == [3, 4]
    assert built["route_override"]["preserved_v15_override"]["reason"] == "v15"
