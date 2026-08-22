from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_bot.config import resource_path
from vision_bot.coords import coord_to_xy


@dataclass(frozen=True)
class RouteHazardGuard:
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    polygon_margins: tuple[tuple[float, float], ...] = ()
    circles: tuple[tuple[float, float, float], ...] = ()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        zone_id: int,
    ) -> RouteHazardGuard:
        profiles = (
            config.get("route", {})
            .get("entry", {})
            .get("navmesh_profiles", {})
        )
        profile = profiles.get(str(int(zone_id)), {})
        hazards_path = profile.get("hazards_path") if isinstance(profile, dict) else None
        if not hazards_path:
            return cls()
        try:
            source = json.loads(
                resource_path(str(hazards_path)).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls()
        return cls.from_hazards(source.get("hazards", []))

    @classmethod
    def from_hazards(cls, hazards: object) -> RouteHazardGuard:
        polygons: list[tuple[tuple[float, float], ...]] = []
        polygon_margins: list[tuple[float, float]] = []
        circles: list[tuple[float, float, float]] = []
        if not isinstance(hazards, list):
            return cls()
        for hazard in hazards:
            if not isinstance(hazard, dict):
                continue
            points = hazard.get("points")
            if isinstance(points, list):
                polygon: list[tuple[float, float]] = []
                for point in points:
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        polygon.append((float(point["x"]), float(point["y"])))
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        polygon.append((float(point[0]), float(point[1])))
                if len(polygon) >= 3:
                    polygons.append(tuple(polygon))
                    polygon_margins.append(
                        (
                            max(0.0, float(hazard.get("runtime_margin_x_coord", 0.0))),
                            max(0.0, float(hazard.get("runtime_margin_y_coord", 0.0))),
                        )
                    )
                    continue
            if all(key in hazard for key in ("x", "y", "radius_yards")):
                # Runtime guard is intentionally conservative for circular local
                # contacts. The exact yard-to-map transform remains an offline
                # planner responsibility; the visible-coordinate guard uses the
                # explicit center only when a map-percent radius is supplied.
                radius = float(hazard.get("runtime_radius_coord", 0.0))
                if radius > 0.0:
                    circles.append((float(hazard["x"]), float(hazard["y"]), radius))
        return cls(
            polygons=tuple(polygons),
            polygon_margins=tuple(polygon_margins),
            circles=tuple(circles),
        )

    def contains_coord(self, coord: int | None) -> bool:
        if coord is None:
            return False
        x, y = coord_to_xy(int(coord))
        return self.contains_xy(x, y)

    def contains_any_coord(self, coords: list[int] | tuple[int, ...]) -> bool:
        return any(self.contains_coord(coord) for coord in coords)

    def contains_xy(self, x: float, y: float) -> bool:
        return any(
            _point_in_polygon(x, y, polygon)
            or _point_within_anisotropic_margin(
                x,
                y,
                polygon,
                *(self.polygon_margins[index] if index < len(self.polygon_margins) else (0.0, 0.0)),
            )
            for index, polygon in enumerate(self.polygons)
        ) or any(
            math.hypot(x - center_x, y - center_y) <= radius
            for center_x, center_y, radius in self.circles
        )


def _point_in_polygon(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _point_on_segment(
            x,
            y,
            previous_x,
            previous_y,
            current_x,
            current_y,
        ):
            return True
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(
        y1, y2
    ) - 1e-9 <= y <= max(y1, y2) + 1e-9


def _point_within_anisotropic_margin(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...],
    margin_x: float,
    margin_y: float,
) -> bool:
    if margin_x <= 0.0 or margin_y <= 0.0:
        return False
    scaled_x = x / margin_x
    scaled_y = y / margin_y
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _distance_to_segment(
            scaled_x,
            scaled_y,
            previous_x / margin_x,
            previous_y / margin_y,
            current_x / margin_x,
            current_y / margin_y,
        ) <= 1.0:
            return True
        previous_x, previous_y = current_x, current_y
    return False


def _distance_to_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(x - x1, y - y1)
    projection = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_squared))
    return math.hypot(x - (x1 + projection * dx), y - (y1 + projection * dy))
