from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from vision_bot.coords import coord_to_xy, xy_to_coord


@dataclass(frozen=True)
class MapPoint:
    """A visible UI-map position in zone percentage units."""

    x_percent: float
    y_percent: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_percent) or not math.isfinite(self.y_percent):
            raise ValueError("map coordinates must be finite")

    @classmethod
    def from_coord(cls, coord: int) -> "MapPoint":
        x_percent, y_percent = coord_to_xy(coord)
        return cls(x_percent=x_percent, y_percent=y_percent)

    def to_coord(self) -> int:
        return xy_to_coord(self.x_percent, self.y_percent)


@dataclass(frozen=True)
class WorldPoint:
    """A local zone position in physical yards."""

    x_yards: float
    y_yards: float


@dataclass(frozen=True)
class ZoneGeometry:
    """Convert visible zone percentages into a consistent physical plane."""

    zone_id: int
    width_yards: float
    height_yards: float

    def __post_init__(self) -> None:
        if int(self.zone_id) <= 0:
            raise ValueError("zone_id must be positive")
        if not math.isfinite(self.width_yards) or self.width_yards <= 0.0:
            raise ValueError("zone width must be positive and finite")
        if not math.isfinite(self.height_yards) or self.height_yards <= 0.0:
            raise ValueError("zone height must be positive and finite")

    def to_world(self, point: MapPoint) -> WorldPoint:
        return WorldPoint(
            x_yards=point.x_percent * self.width_yards / 100.0,
            y_yards=point.y_percent * self.height_yards / 100.0,
        )

    def to_map(self, point: WorldPoint) -> MapPoint:
        return MapPoint(
            x_percent=point.x_yards * 100.0 / self.width_yards,
            y_percent=point.y_yards * 100.0 / self.height_yards,
        )

    def distance(self, first: MapPoint, second: MapPoint) -> float:
        first_world = self.to_world(first)
        second_world = self.to_world(second)
        return math.hypot(
            second_world.x_yards - first_world.x_yards,
            second_world.y_yards - first_world.y_yards,
        )

    def heading_degrees(self, first: MapPoint, second: MapPoint) -> float | None:
        first_world = self.to_world(first)
        second_world = self.to_world(second)
        delta_x = second_world.x_yards - first_world.x_yards
        delta_y = second_world.y_yards - first_world.y_yards
        if math.hypot(delta_x, delta_y) <= 1.0e-9:
            return None
        # Preserve the GetPlayerFacing convention used by the existing runtime.
        return math.degrees(math.pi + math.atan2(delta_x, delta_y)) % 360.0


@dataclass(frozen=True)
class RouteProjection:
    distance_along_yards: float
    cross_track_yards: float
    segment_index: int
    segment_fraction: float
    point: MapPoint


class PhysicalRoute:
    """A loop route parameterised by physical arc length, not point index."""

    def __init__(
        self,
        points: Iterable[MapPoint],
        geometry: ZoneGeometry,
        *,
        loop: bool = True,
    ) -> None:
        route_points = tuple(points)
        if len(route_points) < 2:
            raise ValueError("a physical route requires at least two points")
        self.points = route_points
        self.geometry = geometry
        self.loop = bool(loop)
        segment_count = len(route_points) if self.loop else len(route_points) - 1
        lengths: list[float] = []
        cumulative = [0.0]
        for index in range(segment_count):
            start = route_points[index]
            end = route_points[(index + 1) % len(route_points)]
            length = geometry.distance(start, end)
            if length <= 1.0e-6:
                raise ValueError(f"route segment {index} has zero physical length")
            lengths.append(length)
            cumulative.append(cumulative[-1] + length)
        self.segment_lengths = tuple(lengths)
        self.cumulative_lengths = tuple(cumulative)
        self.total_length_yards = cumulative[-1]

    def point_at(self, distance_along_yards: float) -> MapPoint:
        distance = float(distance_along_yards)
        if self.loop:
            distance %= self.total_length_yards
        else:
            distance = max(0.0, min(self.total_length_yards, distance))
        segment_index = self._segment_for_distance(distance)
        start_distance = self.cumulative_lengths[segment_index]
        fraction = (distance - start_distance) / self.segment_lengths[segment_index]
        start = self.points[segment_index]
        end = self.points[(segment_index + 1) % len(self.points)]
        return MapPoint(
            x_percent=start.x_percent + (end.x_percent - start.x_percent) * fraction,
            y_percent=start.y_percent + (end.y_percent - start.y_percent) * fraction,
        )

    def project(self, point: MapPoint) -> RouteProjection:
        target = self.geometry.to_world(point)
        best: RouteProjection | None = None
        for index, segment_length in enumerate(self.segment_lengths):
            start_map = self.points[index]
            end_map = self.points[(index + 1) % len(self.points)]
            start = self.geometry.to_world(start_map)
            end = self.geometry.to_world(end_map)
            dx = end.x_yards - start.x_yards
            dy = end.y_yards - start.y_yards
            denominator = dx * dx + dy * dy
            fraction = 0.0 if denominator <= 1.0e-12 else (
                (target.x_yards - start.x_yards) * dx
                + (target.y_yards - start.y_yards) * dy
            ) / denominator
            fraction = max(0.0, min(1.0, fraction))
            projected_world = WorldPoint(
                x_yards=start.x_yards + dx * fraction,
                y_yards=start.y_yards + dy * fraction,
            )
            cross_track = math.hypot(
                target.x_yards - projected_world.x_yards,
                target.y_yards - projected_world.y_yards,
            )
            candidate = RouteProjection(
                distance_along_yards=(
                    self.cumulative_lengths[index] + segment_length * fraction
                ),
                cross_track_yards=cross_track,
                segment_index=index,
                segment_fraction=fraction,
                point=self.geometry.to_map(projected_world),
            )
            if best is None or candidate.cross_track_yards < best.cross_track_yards:
                best = candidate
        assert best is not None
        return best

    def _segment_for_distance(self, distance: float) -> int:
        for index, end_distance in enumerate(self.cumulative_lengths[1:]):
            if distance <= end_distance:
                return index
        return len(self.segment_lengths) - 1
