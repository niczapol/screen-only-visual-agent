from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_bot.config import resource_path
from vision_bot.core.geometry import MapPoint, PhysicalRoute, ZoneGeometry
from vision_bot.core.hazards import PhysicalHazardGuard
from vision_bot.route_planner import MiningNode, load_nodes


@dataclass(frozen=True)
class RouteCompileReport:
    source_path: Path
    source_sha256: str
    source_status: str
    source_point_count: int
    compiled_point_count: int
    normalized_duplicate_count: int
    mining_node_count: int
    permanently_excluded_node_count: int
    hazard_point_indexes: tuple[int, ...]
    hazard_segment_indexes: tuple[int, ...]


@dataclass(frozen=True)
class CompiledRoute:
    route: PhysicalRoute
    nodes: tuple[MiningNode, ...]
    hazards: PhysicalHazardGuard
    report: RouteCompileReport


def compile_configured_route(
    project_config: dict[str, Any],
    geometry: ZoneGeometry,
) -> CompiledRoute:
    value = project_config.get("route", {}).get("database_path")
    if not value:
        raise ValueError("v0.9 route database_path is required")
    path = resource_path(str(value)).resolve()
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported route artifact schema")
    if int(data.get("zone_id", 0)) != geometry.zone_id:
        raise ValueError("route artifact zone does not match v0.9 geometry")
    route_items = data.get("route_loop")
    if not isinstance(route_items, list) or len(route_items) < 2:
        raise ValueError("route artifact has no usable route_loop")
    source_points = tuple(MapPoint.from_coord(int(item["coord"])) for item in route_items)
    points: list[MapPoint] = []
    duplicates = 0
    for point in source_points:
        if points and point == points[-1]:
            duplicates += 1
            continue
        points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
        duplicates += 1
    route = PhysicalRoute(points, geometry, loop=True)
    hazards = PhysicalHazardGuard.from_config(project_config, geometry)
    point_hits = tuple(index for index, point in enumerate(points) if hazards.contains(point))
    segment_hits = tuple(
        index
        for index, (start, end) in enumerate(zip(points, (*points[1:], points[0])))
        if hazards.segment_crosses(start, end)
    )
    if point_hits or segment_hits:
        raise ValueError(
            "compiled route intersects configured hazards: "
            f"points={point_hits[:12]} segments={segment_hits[:12]}"
        )
    exclusion_value = project_config.get("route", {}).get("permanent_exclusions_path")
    excluded_coords = _load_permanent_exclusions(exclusion_value)
    source_nodes = tuple(
        node
        for node in load_nodes(path)
        if node.zone_id == geometry.zone_id and node.access_plan is not None
    )
    nodes = tuple(node for node in source_nodes if node.coord not in excluded_coords)
    for node in nodes:
        if hazards.contains(MapPoint.from_coord(node.coord)):
            raise ValueError(f"mining node {node.node_id} is inside a configured hazard")
        for option in node.access_plan.options:
            coords = (*option.inbound_coords, *option.return_coords, *option.resume_coords)
            path_points = tuple(MapPoint.from_coord(coord) for coord in coords)
            path_hits, path_segment_hits = hazards.audit_path(path_points)
            if path_hits or path_segment_hits:
                raise ValueError(
                    f"mining node {node.node_id} access option {option.rank} intersects a hazard"
                )
    report = RouteCompileReport(
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_status=str(data.get("status") or "unknown"),
        source_point_count=len(source_points),
        compiled_point_count=len(points),
        normalized_duplicate_count=duplicates,
        mining_node_count=len(nodes),
        permanently_excluded_node_count=len(source_nodes) - len(nodes),
        hazard_point_indexes=point_hits,
        hazard_segment_indexes=segment_hits,
    )
    return CompiledRoute(route, nodes, hazards, report)


def _load_permanent_exclusions(value: Any) -> frozenset[int]:
    if not value:
        return frozenset()
    path = resource_path(str(value)).resolve()
    if not path.is_file():
        raise ValueError(f"permanent exclusions file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("excluded_coords", [])
    if not isinstance(items, list):
        raise ValueError("permanent exclusions must contain an excluded_coords list")
    return frozenset(int(item["coord"]) for item in items)
