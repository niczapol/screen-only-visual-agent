from scripts.render_route_briefing_map import _inside_ui_crop, rejection_counts


def test_rejection_counts_separates_spacing_and_terrain():
    counts = rejection_counts(
        {
            "rejected_nodes": [
                {"reason": "too_close_to_accepted_route_node"},
                {"reason": "terrain_no_primary_component_approach"},
                {"reason": "terrain_no_primary_component_approach"},
            ]
        }
    )

    assert counts["too_close_to_accepted_route_node"] == 1
    assert counts["terrain_no_primary_component_approach"] == 2


def test_ui_crop_edges_are_inclusive():
    crop = (24.0, 82.0, 55.0, 84.0)

    assert _inside_ui_crop(24.0, 84.0, crop)
    assert not _inside_ui_crop(83.0, 60.0, crop)
