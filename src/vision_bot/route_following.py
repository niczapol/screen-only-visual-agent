from __future__ import annotations

import math
from dataclasses import dataclass

from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.route_database import RouteProjection, project_to_loop


@dataclass(frozen=True)
class DirectedRouteObservation:
    projection: RouteProjection
    progress: float
    progress_delta: float
    cross_track_distance: float
    on_corridor: bool
    projection_accepted: bool
    steering_coord: int
    steering_sort_key: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "progress": self.progress,
            "progress_delta": self.progress_delta,
            "cross_track_distance": self.cross_track_distance,
            "on_corridor": self.on_corridor,
            "projection_accepted": self.projection_accepted,
            "projection_sort_key": self.projection.sort_key,
            "projection_segment_index": self.projection.segment_index,
            "projection_segment_t": self.projection.segment_t,
            "steering_coord": self.steering_coord,
            "steering_sort_key": self.steering_sort_key,
        }


class DirectedRouteFollower:
    """Track monotonic progress and steer toward a geometric loop lookahead."""

    def __init__(
        self,
        route_coords: list[int],
        *,
        lookahead_distance: float,
        corridor_radius: float,
        backward_tolerance: float,
        max_forward_advance: float,
        pass_tolerance: float,
        projection_search_backward_segments: int = 3,
        projection_search_forward_segments: int | None = None,
        lookahead_overrides: tuple[tuple[int, int, float], ...] = (),
    ) -> None:
        if len(route_coords) < 2:
            raise ValueError("directed route following requires at least two coordinates")
        self.route_coords = tuple(route_coords)
        self.lookahead_distance = max(0.01, float(lookahead_distance))
        self.corridor_radius = max(0.0, float(corridor_radius))
        self.backward_tolerance = max(0.0, float(backward_tolerance))
        self.max_forward_advance = max(0.01, float(max_forward_advance))
        self.pass_tolerance = max(0.0, float(pass_tolerance))
        self.projection_search_backward_segments = max(
            0,
            int(projection_search_backward_segments),
        )
        self.projection_search_forward_segments = max(
            0,
            (
                int(projection_search_forward_segments)
                if projection_search_forward_segments is not None
                else int(math.ceil(self.max_forward_advance)) + 4
            ),
        )
        self.lookahead_overrides = tuple(
            (
                int(start_index),
                int(end_index),
                max(0.01, float(distance)),
            )
            for start_index, end_index, distance in lookahead_overrides
        )
        self.progress: float | None = None
        self.goal_sort_key: float | None = None
        self.goal_progress: float | None = None
        self.last_observation: DirectedRouteObservation | None = None

    @property
    def route_size(self) -> float:
        return float(len(self.route_coords))

    def seed(self, sort_key: float) -> None:
        normalized = float(sort_key) % self.route_size
        self.progress = normalized
        self.goal_sort_key = None
        self.goal_progress = None

    def set_goal(self, sort_key: float | None) -> None:
        if sort_key is None:
            self.goal_sort_key = None
            self.goal_progress = None
            return
        normalized = float(sort_key) % self.route_size
        if self.goal_sort_key is not None and math.isclose(
            normalized,
            self.goal_sort_key,
            abs_tol=1e-9,
        ):
            return
        self.goal_sort_key = normalized
        if self.progress is None:
            self.goal_progress = normalized
            return
        goal_progress = unwrap_at_or_after(normalized, self.progress, self.route_size)
        previous_lap_goal = goal_progress - self.route_size
        # A fresh coordinate sample can project slightly past the diagnostic
        # waypoint before the outer route loop consumes it.  Treat that small
        # local overshoot as an already-passed goal.  Without this guard,
        # unwrap_at_or_after moves the goal one complete lap forward and the
        # controller backtracks forever toward a waypoint only 0.2 segments
        # behind it (observed live on V12 at sort key 135.219 -> goal 135.0).
        if (
            previous_lap_goal <= self.progress
            and self.progress - previous_lap_goal
            <= self.backward_tolerance + self.pass_tolerance
        ):
            goal_progress = previous_lap_goal
        self.goal_progress = goal_progress

    def goal_passed(self) -> bool:
        return bool(
            self.progress is not None
            and self.goal_progress is not None
            and self.progress + self.pass_tolerance >= self.goal_progress
        )

    def observe(self, coord: int) -> DirectedRouteObservation:
        previous = self.progress
        if previous is None:
            projection = project_to_loop(coord, list(self.route_coords))
            accepted_progress = projection.sort_key
            accepted = True
        else:
            active_goal_ceiling = (
                self.goal_progress
                if self.goal_progress is not None
                and previous + self.pass_tolerance < self.goal_progress
                else None
            )
            projection, candidate = self._project_near_progress(
                coord,
                previous,
                max_progress=active_goal_ceiling,
            )
            backward_jitter = 0.0
            if candidate < previous:
                backward_jitter = previous - candidate
                candidate = previous
            advance = candidate - previous
            accepted = bool(
                projection.distance <= self.corridor_radius
                and advance <= self.max_forward_advance
                and backward_jitter <= self.backward_tolerance
            )
            accepted_progress = candidate if accepted else previous
        self.progress = accepted_progress
        lookahead_distance = self._lookahead_distance_for_progress(accepted_progress)
        steering_coord, steering_sort_key = coord_at_distance_along_loop(
            list(self.route_coords),
            accepted_progress,
            lookahead_distance,
        )
        observation = DirectedRouteObservation(
            projection=projection,
            progress=accepted_progress,
            progress_delta=0.0 if previous is None else accepted_progress - previous,
            cross_track_distance=projection.distance,
            on_corridor=projection.distance <= self.corridor_radius,
            projection_accepted=accepted,
            steering_coord=steering_coord,
            steering_sort_key=steering_sort_key,
        )
        self.last_observation = observation
        return observation

    def _lookahead_distance_for_progress(self, progress: float) -> float:
        segment_index = int(math.floor(progress)) % len(self.route_coords)
        selected = self.lookahead_distance
        for start_index, end_index, distance in self.lookahead_overrides:
            normalized_start = start_index % len(self.route_coords)
            normalized_end = end_index % len(self.route_coords)
            if normalized_start <= normalized_end:
                matched = normalized_start <= segment_index <= normalized_end
            else:
                matched = segment_index >= normalized_start or segment_index <= normalized_end
            if matched:
                selected = min(selected, distance)
        return selected

    def _project_near_progress(
        self,
        coord: int,
        progress: float,
        *,
        max_progress: float | None = None,
    ) -> tuple[RouteProjection, float]:
        if (
            self.projection_search_backward_segments <= 0
            and self.projection_search_forward_segments <= 0
        ):
            projection = project_to_loop(coord, list(self.route_coords))
            return projection, unwrap_nearest(
                projection.sort_key,
                progress,
                self.route_size,
            )

        route_size = len(self.route_coords)
        point = coord_to_xy(coord)
        center_segment = int(math.floor(progress))
        start_segment = center_segment - self.projection_search_backward_segments
        end_segment = center_segment + self.projection_search_forward_segments
        if max_progress is not None:
            # Do not let an overlapping section many segments ahead steal the
            # projection before the currently assigned diagnostic waypoint is
            # consumed. V13's lap seam is physically close to segment 14; the
            # unrestricted 24-segment window jumped 490 -> 508 at the seam and
            # then sent the character around almost an entire extra lap toward
            # waypoint 1. The next goal will expand this ceiling immediately.
            end_segment = min(end_segment, int(math.floor(max_progress)))
        best_projection: RouteProjection | None = None
        best_progress = progress
        best_key: tuple[float, float, float] | None = None

        for raw_segment_index in range(start_segment, end_segment + 1):
            segment_index = raw_segment_index % route_size
            segment_start = coord_to_xy(self.route_coords[segment_index])
            segment_end = coord_to_xy(self.route_coords[(segment_index + 1) % route_size])
            distance, segment_t, nearest_x, nearest_y = _point_segment_projection(
                point,
                segment_start,
                segment_end,
            )
            candidate_progress = raw_segment_index + segment_t
            key = (
                distance,
                abs(candidate_progress - progress),
                candidate_progress,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_progress = candidate_progress
                best_projection = RouteProjection(
                    distance=distance,
                    segment_index=segment_index,
                    segment_t=segment_t,
                    nearest_coord=xy_to_coord(nearest_x, nearest_y),
                    nearest_x=nearest_x,
                    nearest_y=nearest_y,
                )

        if best_projection is None:
            projection = project_to_loop(coord, list(self.route_coords))
            return projection, unwrap_nearest(
                projection.sort_key,
                progress,
                self.route_size,
            )
        return best_projection, best_progress


def unwrap_nearest(sort_key: float, reference: float, route_size: float) -> float:
    if route_size <= 0.0:
        raise ValueError("route_size must be positive")
    normalized = float(sort_key) % route_size
    lap = math.floor(reference / route_size)
    candidates = (
        normalized + (lap - 1) * route_size,
        normalized + lap * route_size,
        normalized + (lap + 1) * route_size,
    )
    return min(candidates, key=lambda candidate: (abs(candidate - reference), candidate))


def unwrap_at_or_after(sort_key: float, reference: float, route_size: float) -> float:
    if route_size <= 0.0:
        raise ValueError("route_size must be positive")
    normalized = float(sort_key) % route_size
    lap = math.floor(reference / route_size)
    candidate = normalized + lap * route_size
    if candidate + 1e-9 < reference:
        candidate += route_size
    return candidate


def coord_at_distance_along_loop(
    route_coords: list[int],
    start_sort_key: float,
    distance: float,
) -> tuple[int, float]:
    if len(route_coords) < 2:
        raise ValueError("route loop requires at least two coordinates")
    route_size = len(route_coords)
    progress = float(start_sort_key)
    segment_index = int(math.floor(progress)) % route_size
    segment_t = progress - math.floor(progress)
    remaining = max(0.0, float(distance))

    for _ in range(route_size * 2 + 1):
        start = coord_to_xy(route_coords[segment_index])
        end_index = (segment_index + 1) % route_size
        end = coord_to_xy(route_coords[end_index])
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        available = segment_length * max(0.0, 1.0 - segment_t)
        if segment_length > 0.0 and remaining <= available:
            target_t = segment_t + remaining / segment_length
            x = start[0] + (end[0] - start[0]) * target_t
            y = start[1] + (end[1] - start[1]) * target_t
            target_progress = progress + remaining / segment_length
            return xy_to_coord(x, y), target_progress
        remaining -= available
        progress = math.floor(progress) + 1.0
        segment_index = end_index
        segment_t = 0.0

    return route_coords[segment_index], progress


def _point_segment_projection(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float, float]:
    px, py = point
    ax, ay = start
    bx, by = end
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denominator = vx * vx + vy * vy
    if denominator <= 0.0:
        return math.hypot(px - ax, py - ay), 0.0, ax, ay
    segment_t = max(0.0, min(1.0, (wx * vx + wy * vy) / denominator))
    nearest_x = ax + vx * segment_t
    nearest_y = ay + vy * segment_t
    return math.hypot(px - nearest_x, py - nearest_y), segment_t, nearest_x, nearest_y
