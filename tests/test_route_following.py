import pytest

from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.route_following import (
    DirectedRouteFollower,
    coord_at_distance_along_loop,
    unwrap_at_or_after,
    unwrap_nearest,
)


def test_local_lookahead_override_preserves_tight_bypass_vertices() -> None:
    route = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(11.0, 10.0),
        xy_to_coord(11.0, 11.0),
        xy_to_coord(12.0, 11.0),
    ]
    follower = DirectedRouteFollower(
        route,
        lookahead_distance=1.5,
        corridor_radius=1.0,
        backward_tolerance=0.25,
        max_forward_advance=4.0,
        pass_tolerance=0.05,
        lookahead_overrides=((0, 2, 0.25),),
    )

    observation = follower.observe(xy_to_coord(10.5, 10.0))

    assert coord_to_xy(observation.steering_coord) == pytest.approx((10.75, 10.0))


ROUTE = [
    1000100000,
    2000100000,
    2000200000,
    1000200000,
]


def test_coord_at_distance_advances_across_segment_boundary():
    coord, sort_key = coord_at_distance_along_loop(ROUTE, 0.8, 5.0)

    assert coord == 2000130000
    assert round(sort_key, 3) == 1.3


def test_unwrap_helpers_preserve_forward_wrap_direction():
    assert unwrap_nearest(0.2, 3.9, 4.0) == 4.2
    assert unwrap_at_or_after(0.2, 3.9, 4.0) == 4.2
    assert unwrap_at_or_after(3.0, 3.1, 4.0) == 7.0


def test_follower_never_regresses_on_projection_jitter():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.3,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    first = follower.observe(1500100000)
    second = follower.observe(1490100000)

    assert first.progress == 0.5
    assert second.progress == first.progress
    assert second.progress_delta == 0.0


def test_follower_rejects_far_off_corridor_projection():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=1.0,
        backward_tolerance=0.3,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    follower.seed(0.4)
    observation = follower.observe(1500500000)

    assert observation.progress == 0.4
    assert not observation.on_corridor
    assert not observation.projection_accepted


def test_follower_marks_goal_passed_without_radius_hit():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.3,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    follower.seed(0.4)
    follower.set_goal(1.0)
    follower.observe(2000110000)

    assert follower.goal_passed()


def test_follower_wraps_goal_to_next_lap():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.3,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    follower.seed(3.8)
    follower.set_goal(0.2)
    follower.observe(1200100000)

    assert follower.progress == 4.2
    assert follower.goal_progress == 4.2
    assert follower.goal_passed()


def test_follower_treats_small_sample_overshoot_as_passed_not_next_lap():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.25,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    follower.seed(1.219)

    follower.set_goal(1.0)

    assert follower.goal_progress == 1.0
    assert follower.goal_passed()


def test_follower_keeps_real_wrap_when_goal_is_materially_behind():
    follower = DirectedRouteFollower(
        ROUTE,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.25,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )
    follower.seed(1.40)

    follower.set_goal(1.0)

    assert follower.goal_progress == 5.0
    assert not follower.goal_passed()


def test_active_lap_seam_goal_blocks_overlapping_far_forward_projection():
    route = [
        xy_to_coord(0.0, 0.0),
        xy_to_coord(1.0, 0.0),
        xy_to_coord(2.0, 0.0),
        xy_to_coord(3.0, 0.0),
        xy_to_coord(4.0, 0.0),
        xy_to_coord(5.0, 0.0),
        xy_to_coord(6.0, 0.0),
        xy_to_coord(7.0, 0.0),
        xy_to_coord(8.0, 0.0),
        xy_to_coord(9.0, 0.0),
        xy_to_coord(10.0, 0.0),
        xy_to_coord(11.0, 0.0),
        xy_to_coord(12.0, 0.0),
        xy_to_coord(13.0, 0.0),
        # A later section overlaps the seam more closely than the actual
        # closing leg, reproducing V13 segment 14 near route index 493.
        xy_to_coord(0.20, 0.05),
        xy_to_coord(0.40, 0.05),
        xy_to_coord(6.0, 3.0),
        xy_to_coord(4.0, 2.0),
        xy_to_coord(2.0, 1.0),
        xy_to_coord(0.30, 0.30),
    ]
    follower = DirectedRouteFollower(
        route,
        lookahead_distance=0.5,
        corridor_radius=2.0,
        backward_tolerance=0.25,
        max_forward_advance=24.0,
        pass_tolerance=0.05,
        projection_search_backward_segments=3,
        projection_search_forward_segments=28,
    )
    follower.seed(18.8)
    follower.set_goal(0.0)

    observation = follower.observe(xy_to_coord(0.20, 0.05))

    assert observation.progress <= 20.0
    assert observation.progress < 21.0
    assert follower.goal_progress == 20.0


def test_follower_uses_local_projection_window_on_parallel_route_sections():
    route = [
        xy_to_coord(0.0, 0.0),
        xy_to_coord(10.0, 0.0),
        xy_to_coord(20.0, 5.0),
        xy_to_coord(10.0, 1.0),
        xy_to_coord(0.0, 1.0),
        xy_to_coord(0.0, 10.0),
        xy_to_coord(20.0, 10.0),
    ]
    follower = DirectedRouteFollower(
        route,
        lookahead_distance=2.0,
        corridor_radius=2.0,
        backward_tolerance=0.3,
        max_forward_advance=10.0,
        pass_tolerance=0.05,
        projection_search_backward_segments=0,
        projection_search_forward_segments=1,
    )
    follower.seed(0.5)

    observation = follower.observe(xy_to_coord(5.0, 0.9))

    assert observation.projection_accepted
    assert observation.projection.segment_index == 0
    assert observation.progress < 1.0
