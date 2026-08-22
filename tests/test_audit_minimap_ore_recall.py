from scripts.audit_minimap_ore_recall import (
    _nearby_runtime_detection,
    _runtime_points,
    _strong_reasons,
)


def test_runtime_points_treats_bright_and_legacy_dark_matches_as_recognized():
    row = {"mining": {"bright_points": [[10, 20]], "dark_points": [[30, 40]]}}

    assert _runtime_points(row) == [(10, 20), (30, 40)]


def test_temporal_sandwich_requires_recognition_on_both_sides():
    rows = [
        {"mining": {"bright_points": [[10, 20]]}},
        {"mining": {"bright_points": []}},
        {"mining": {"bright_points": [[11, 20]]}},
    ]

    assert _nearby_runtime_detection(rows, 1, 1)
    assert not _nearby_runtime_detection(rows[:2], 1, 1)


def test_strong_reasons_retains_visible_confirmation_without_calling_it_truth():
    row = {
        "minimap_ore_tooltip": {"ore_type": "Iron"},
        "minimap_tooltip_probe": {"confirmed_track_id": 7},
        "mining": {"confirmed_ore_type": "Iron", "target_marker_id": 7},
    }

    assert _strong_reasons(row) == [
        "visible_tooltip_marker",
        "tooltip_confirmed_track",
        "mining_tooltip_confirmed",
        "retained_marker_track",
    ]
