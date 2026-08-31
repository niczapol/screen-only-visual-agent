from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vision_bot.config import resource_path
from vision_bot.core.geometry import MapPoint, WorldPoint, ZoneGeometry


@dataclass(frozen=True)
class PolygonHazard:
    kind: str
    points: tuple[MapPoint, ...]
    margin_yards: float


@dataclass(frozen=True)
class CircleHazard:
    kind: str
    center: MapPoint
    radius_yards: float


@dataclass(frozen=True)
class PhysicalHazardGuard:
    geometry: ZoneGeometry
    polygons: tuple[PolygonHazard, ...] = ()
    circles: tuple[CircleHazard, ...] = ()
    source_path: Path | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        geometry: ZoneGeometry,
    ) -> "PhysicalHazardGuard":
        profiles = config.get("route", {}).get("entry", {}).get("navmesh_profiles", {})
        profile = profiles.get(str(geometry.zone_id), {})
        value = profile.get("hazards_path") if isinstance(profile, dict) else None
        if not value:
            return cls(geometry)
        path = resource_path(str(value)).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        polygons: list[PolygonHazard] = []
        circles: list[CircleHazard] = []
        for item in data.get("hazards", []):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "unnamed")
            margin = max(0.0, float(item.get("margin_yards", 0.0)))
            raw_points = item.get("points")
            if isinstance(raw_points, list) and len(raw_points) >= 3:
                points = tuple(
                    MapPoint(
                        float(point["x"] if isinstance(point, dict) else point[0]),
                        float(point["y"] if isinstance(point, dict) else point[1]),
                    )
                    for point in raw_points
                )
                polygons.append(PolygonHazard(kind, points, margin))
            elif all(key in item for key in ("x", "y", "radius_yards")):
                circles.append(
                    CircleHazard(
                        kind,
                        MapPoint(float(item["x"]), float(item["y"])),
                        max(0.0, float(item["radius_yards"])) + margin,
                    )
                )
        return cls(geometry, tuple(polygons), tuple(circles), path)

    def containing_kinds(self, point: MapPoint) -> tuple[str, ...]:
        found: list[str] = []
        for hazard in self.polygons:
            if _point_in_polygon(point, hazard.points) or self._distance_to_polygon(
                point, hazard.points
            ) <= hazard.margin_yards:
                found.append(hazard.kind)
        for hazard in self.circles:
            if self.geometry.distance(point, hazard.center) <= hazard.radius_yards:
                found.append(hazard.kind)
        return tuple(found)

    def contains(self, point: MapPoint) -> bool:
        return bool(self.containing_kinds(point))

    def segment_crosses(
        self,
        start: MapPoint,
        end: MapPoint,
        *,
        sample_step_yards: float = 2.0,
    ) -> bool:
        distance = self.geometry.distance(start, end)
        steps = max(1, int(math.ceil(distance / max(0.25, sample_step_yards))))
        return any(
            self.contains(
                MapPoint(
                    start.x_percent + (end.x_percent - start.x_percent) * index / steps,
                    start.y_percent + (end.y_percent - start.y_percent) * index / steps,
                )
            )
            for index in range(steps + 1)
        )

    def audit_path(self, points: Iterable[MapPoint]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        items = tuple(points)
        point_hits = tuple(index for index, point in enumerate(items) if self.contains(point))
        segment_hits = tuple(
            index
            for index, (start, end) in enumerate(zip(items, items[1:]))
            if self.segment_crosses(start, end)
        )
        return point_hits, segment_hits

    def _distance_to_polygon(
        self,
        point: MapPoint,
        polygon: tuple[MapPoint, ...],
    ) -> float:
        target = self.geometry.to_world(point)
        world = tuple(self.geometry.to_world(item) for item in polygon)
        return min(
            _distance_to_segment(target, first, second)
            for first, second in zip(world, (*world[1:], world[0]))
        )


def _point_in_polygon(point: MapPoint, polygon: tuple[MapPoint, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        crosses = (current.y_percent > point.y_percent) != (
            previous.y_percent > point.y_percent
        )
        if crosses:
            intersection_x = (
                (previous.x_percent - current.x_percent)
                * (point.y_percent - current.y_percent)
                / (previous.y_percent - current.y_percent)
                + current.x_percent
            )
            if point.x_percent < intersection_x:
                inside = not inside
        previous = current
    return inside


def _distance_to_segment(
    point: WorldPoint,
    start: WorldPoint,
    end: WorldPoint,
) -> float:
    dx = end.x_yards - start.x_yards
    dy = end.y_yards - start.y_yards
    denominator = dx * dx + dy * dy
    fraction = 0.0 if denominator <= 1.0e-12 else (
        (point.x_yards - start.x_yards) * dx
        + (point.y_yards - start.y_yards) * dy
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    return math.hypot(
        point.x_yards - (start.x_yards + dx * fraction),
        point.y_yards - (start.y_yards + dy * fraction),
    )
