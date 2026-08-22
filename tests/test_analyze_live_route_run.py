from __future__ import annotations

from scripts.analyze_live_route_run import (
    analyze_rows,
    parse_index_ranges,
    project_trajectory,
    replay_directed_route_following,
)
from vision_bot.coords import xy_to_coord


def test_parse_index_ranges_is_bounded_and_deduplicated() -> None:
    assert parse_index_ranges("1:5,3,9:20", max_index=10, step=2) == [1, 3, 5, 9]


def test_route_projection_detects_backward_progress() -> None:
    route = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(20.0, 10.0),
        xy_to_coord(20.0, 20.0),
        xy_to_coord(10.0, 20.0),
    ]
    rows = [
        {"index": 0, "coord": xy_to_coord(15.0, 10.0), "coord_fresh": True},
        {"index": 1, "coord": xy_to_coord(12.0, 10.0), "coord_fresh": True},
    ]
    projected = project_trajectory(rows, route)
    assert projected[1]["signed_route_delta"] < 0.0


def test_analysis_reports_stale_position_regression_and_unconfirmed_x() -> None:
    rows = [
        {"index": 0, "coord": xy_to_coord(10.0, 10.0), "coord_fresh": True},
        {
            "index": 1,
            "coord": xy_to_coord(20.0, 20.0),
            "coord_fresh": False,
            "combat_keys_tapped": ["X"],
            "combat": {"active": True, "combat_marker_visible": True, "target_is_attacker": False},
        },
    ]
    summary = analyze_rows(rows, [], 4)
    assert summary["stale_coordinate_regressions_after_fresh_read"] == 1
    assert summary["x_without_confirmed_combat"] == 1


def test_analysis_deduplicates_mouse_turns_and_reports_reversals() -> None:
    rows = [
        {
            "index": 0,
            "timestamp": 10.0,
            "mouse_steering": {
                "sequence": 1,
                "context": "route",
                "turn_key": "D",
                "duration": 0.10,
            },
        },
        {
            "index": 1,
            "timestamp": 20.0,
            "mouse_steering": {
                "sequence": 1,
                "context": "route",
                "turn_key": "D",
                "duration": 0.10,
            },
        },
        {
            "index": 2,
            "timestamp": 40.0,
            "mouse_steering": {
                "sequence": 2,
                "context": "combat",
                "turn_key": "A",
                "duration": 0.15,
            },
        },
    ]

    steering = analyze_rows(rows, [], 4)["mouse_steering"]

    assert steering == {
        "turns": 2,
        "turns_per_minute": 4.0,
        "direction_reversals": 1,
        "contexts": {"route": 1, "combat": 1},
        "total_drag_seconds": 0.25,
    }


def test_directed_route_replay_rejects_backward_and_large_jump_without_regressing() -> None:
    route = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(20.0, 10.0),
        xy_to_coord(20.0, 20.0),
        xy_to_coord(10.0, 20.0),
    ]
    rows = [
        {"index": 0, "coord": xy_to_coord(15.0, 10.0), "coord_fresh": True},
        {"index": 1, "coord": xy_to_coord(18.0, 10.0), "coord_fresh": True},
        {"index": 2, "coord": xy_to_coord(12.0, 10.0), "coord_fresh": True},
        {"index": 3, "coord": xy_to_coord(80.0, 80.0), "coord_fresh": True},
    ]

    summary, replay = replay_directed_route_following(
        rows,
        route,
        corridor_radius=2.0,
        backward_tolerance=0.1,
        max_forward_advance=2.0,
    )

    assert summary["progress_regressions"] == 0
    assert summary["accepted_projections"] == 2
    assert summary["rejected_projections"] == 2
    assert replay[-1]["progress"] == replay[1]["progress"]


def test_directed_route_replay_reports_loop_progress_coverage() -> None:
    route = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(20.0, 10.0),
        xy_to_coord(20.0, 20.0),
        xy_to_coord(10.0, 20.0),
    ]
    rows = [
        {"index": 0, "coord": xy_to_coord(12.0, 10.0), "coord_fresh": True},
        {"index": 1, "coord": xy_to_coord(18.0, 10.0), "coord_fresh": True},
        {"index": 2, "coord": xy_to_coord(20.0, 14.0), "coord_fresh": True},
    ]

    summary, _replay = replay_directed_route_following(
        rows,
        route,
        corridor_radius=2.0,
        max_forward_advance=2.0,
    )

    assert summary["route_progress_span"] < len(route)
    assert 0.0 < summary["route_progress_coverage_fraction"] < 1.0
    assert summary["completed_laps"] == 0
