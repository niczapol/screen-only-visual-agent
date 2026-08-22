from vision_bot.coords import xy_to_coord
from vision_bot.route_hazards import RouteHazardGuard


def test_route_hazard_guard_includes_polygon_boundary_and_interior() -> None:
    guard = RouteHazardGuard.from_hazards(
        [
            {
                "runtime_margin_x_coord": 0.65,
                "runtime_margin_y_coord": 0.98,
                "points": [
                    [58.0, 34.0],
                    [76.5, 34.0],
                    [76.5, 58.5],
                    [57.5, 55.0],
                    [57.5, 41.0],
                ]
            }
        ]
    )

    assert guard.contains_coord(xy_to_coord(63.53, 49.70))
    assert guard.contains_any_coord(
        [xy_to_coord(61.03, 50.27), xy_to_coord(63.53, 49.70)]
    )
    assert guard.contains_xy(58.0, 34.0)
    assert guard.contains_xy(57.5, 40.5)
    assert not guard.contains_coord(xy_to_coord(55.30, 45.0))


def test_route_hazard_guard_ignores_yard_circle_without_runtime_radius() -> None:
    guard = RouteHazardGuard.from_hazards(
        [{"x": 30.65, "y": 73.82, "radius_yards": 12.0}]
    )

    assert not guard.contains_xy(30.65, 73.82)
