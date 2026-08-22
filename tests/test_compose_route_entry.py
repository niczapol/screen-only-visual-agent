import pytest

from scripts.compose_route_entry import compose_route_entry


def _base_route():
    return {
        "name": "base",
        "zone": {"id": 102, "name": "Desolace"},
        "generation": {"topographic": {"component_selection": {}}},
        "route_loop": [{"coord": 3000300000}],
        "entry_route": {
            "status": "candidate_anchor_entry",
            "target_route_index": 35,
            "path_world_length": 12.0,
            "waypoints": [
                {"coord": 2000200000, "x": 20.0, "y": 20.0},
                {"coord": 3000300000, "x": 30.0, "y": 30.0},
            ],
        },
    }


def test_compose_route_entry_prepends_staging_and_preserves_loop():
    result = compose_route_entry(
        _base_route(),
        {
            "zone": {"id": 102},
            "output_name": "staged",
            "waypoints": [
                {"x": 10.0, "y": 10.0},
                {"x": 20.0, "y": 20.0},
            ],
        },
    )

    assert result["name"] == "staged"
    assert result["route_loop"] == [{"coord": 3000300000}]
    assert [item["coord"] for item in result["entry_route"]["waypoints"]] == [
        1000100000,
        2000200000,
        3000300000,
    ]
    assert result["entry_route"]["target_route_index"] == 35
    assert result["generation"]["topographic"]["component_selection"][
        "runtime_entry_anchor_ui"
    ] == [10.0, 10.0]


def test_compose_route_entry_rejects_zone_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        compose_route_entry(
            _base_route(),
            {"zone": {"id": 1}, "waypoints": [{"x": 1, "y": 1}, {"x": 2, "y": 2}]},
        )
