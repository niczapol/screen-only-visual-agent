from __future__ import annotations

import heapq
import math
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


MMAP_MAGIC = 0x4D4D4150
MMAP_VERSION = 20
DT_NAVMESH_MAGIC = 0x444E4156
DT_EXT_LINK = 0x8000
DT_POLYTYPE_GROUND = 0
NAV_GROUND = 0x01
TILE_SIZE = 533.3333333333

_MMAP_FILE_HEADER = struct.Struct("<4I?3x f4B3I4f")
_MESH_HEADER = struct.Struct("<5iI9i10f")
_POLY = struct.Struct("<I6H6HHBB")

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class NavLocation:
    polygon_id: int
    point: Point3
    horizontal_distance: float
    vertical_distance: float


@dataclass(frozen=True)
class NavPath:
    polygon_ids: tuple[int, ...]
    points: tuple[Point3, ...]
    graph_cost: float

    @property
    def world_length(self) -> float:
        return sum(
            math.dist((first[0], first[2]), (second[0], second[2]))
            for first, second in zip(self.points, self.points[1:])
        )


@dataclass
class NavPolygon:
    polygon_id: int
    tile_x: int
    tile_y: int
    local_index: int
    vertices: tuple[Point3, ...]
    flags: int
    area: int
    neighbors: dict[int, tuple[Point3, Point3]] = field(default_factory=dict)

    @property
    def center(self) -> Point3:
        count = float(len(self.vertices))
        return tuple(sum(vertex[index] for vertex in self.vertices) / count for index in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class _ParsedPolygon:
    local_index: int
    vertices: tuple[Point3, ...]
    neighbor_values: tuple[int, ...]
    flags: int
    area: int


@dataclass(frozen=True)
class _ExternalEdge:
    polygon_id: int
    tile_x: int
    tile_y: int
    first: Point3
    second: Point3


class MMapNavMesh:
    def __init__(self) -> None:
        self.polygons: dict[int, NavPolygon] = {}
        self.polygons_by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._component_ids: dict[int, int] | None = None

    @classmethod
    def load_tiles(
        cls,
        mmap_dir: str | Path,
        *,
        map_id: int,
        tiles: Iterable[tuple[int, int]],
        include_flags: int = NAV_GROUND,
    ) -> MMapNavMesh:
        mesh = cls()
        local_to_global: dict[tuple[int, int, int], int] = {}
        parsed_by_tile: dict[tuple[int, int], list[_ParsedPolygon]] = {}
        external_edges: list[_ExternalEdge] = []

        for tile_x, tile_y in sorted(set(tiles)):
            path = mmtile_path(mmap_dir, map_id=map_id, tile_x=tile_x, tile_y=tile_y)
            if not path.exists():
                continue
            parsed = parse_mmtile(path, include_flags=include_flags)
            parsed_by_tile[(tile_x, tile_y)] = parsed
            for item in parsed:
                polygon_id = len(mesh.polygons)
                local_to_global[(tile_x, tile_y, item.local_index)] = polygon_id
                polygon = NavPolygon(
                    polygon_id=polygon_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    local_index=item.local_index,
                    vertices=item.vertices,
                    flags=item.flags,
                    area=item.area,
                )
                mesh.polygons[polygon_id] = polygon
                mesh.polygons_by_tile[(tile_x, tile_y)].append(polygon_id)

        for tile, parsed in parsed_by_tile.items():
            tile_x, tile_y = tile
            for item in parsed:
                polygon_id = local_to_global[(tile_x, tile_y, item.local_index)]
                for edge_index, neighbor_value in enumerate(item.neighbor_values):
                    if edge_index >= len(item.vertices) or neighbor_value == 0:
                        continue
                    first = item.vertices[edge_index]
                    second = item.vertices[(edge_index + 1) % len(item.vertices)]
                    if neighbor_value & DT_EXT_LINK:
                        external_edges.append(
                            _ExternalEdge(polygon_id, tile_x, tile_y, first, second)
                        )
                        continue
                    neighbor_local_index = neighbor_value - 1
                    neighbor_id = local_to_global.get(
                        (tile_x, tile_y, neighbor_local_index)
                    )
                    if neighbor_id is not None:
                        _connect(mesh, polygon_id, neighbor_id, (first, second))

        _connect_external_edges(mesh, external_edges)
        return mesh

    def component_id(self, polygon_id: int) -> int:
        if self._component_ids is None:
            self._component_ids = self._build_component_ids()
        return self._component_ids[polygon_id]

    def component_size(self, polygon_id: int) -> int:
        component = self.component_id(polygon_id)
        assert self._component_ids is not None
        return sum(1 for value in self._component_ids.values() if value == component)

    def nearest_location(
        self,
        *,
        world_x: float,
        world_y: float,
        world_z: float,
        max_horizontal_distance: float = 24.0,
        max_vertical_distance: float = 80.0,
        tile_radius: int = 1,
    ) -> NavLocation | None:
        tile_x, tile_y = world_to_tile(world_x, world_y)
        candidate_ids: list[int] = []
        for offset_x in range(-tile_radius, tile_radius + 1):
            for offset_y in range(-tile_radius, tile_radius + 1):
                candidate_ids.extend(
                    self.polygons_by_tile.get((tile_x + offset_x, tile_y + offset_y), [])
                )

        recast_x = float(world_y)
        recast_z = float(world_x)
        best: tuple[tuple[float, float], NavLocation] | None = None
        for polygon_id in candidate_ids:
            polygon = self.polygons[polygon_id]
            point_2d, horizontal_distance = _closest_point_on_polygon(
                (recast_x, recast_z), polygon.vertices
            )
            if horizontal_distance > max_horizontal_distance:
                continue
            height = _polygon_height(polygon.vertices, point_2d)
            vertical_distance = abs(height - float(world_z))
            if vertical_distance > max_vertical_distance:
                continue
            score = (horizontal_distance, vertical_distance)
            location = NavLocation(
                polygon_id=polygon_id,
                point=(point_2d[0], height, point_2d[1]),
                horizontal_distance=horizontal_distance,
                vertical_distance=vertical_distance,
            )
            if best is None or score < best[0]:
                best = score, location
        return None if best is None else best[1]

    def find_path(
        self,
        start: NavLocation,
        end: NavLocation,
        *,
        max_expansions: int = 500_000,
        transition_cost: Callable[[NavPolygon, NavPolygon, tuple[Point3, Point3]], float | None]
        | None = None,
    ) -> NavPath | None:
        if self.component_id(start.polygon_id) != self.component_id(end.polygon_id):
            return None
        if start.polygon_id == end.polygon_id:
            return NavPath(
                polygon_ids=(start.polygon_id,),
                points=(start.point, end.point),
                graph_cost=math.dist(start.point, end.point),
            )

        frontier: list[tuple[float, float, int]] = []
        best_cost = {start.polygon_id: 0.0}
        came_from: dict[int, int] = {}
        goal_center = self.polygons[end.polygon_id].center
        heapq.heappush(
            frontier,
            (_horizontal_distance(self.polygons[start.polygon_id].center, goal_center), 0.0, start.polygon_id),
        )
        expansions = 0
        while frontier:
            _, cost, polygon_id = heapq.heappop(frontier)
            if cost > best_cost.get(polygon_id, math.inf) + 1.0e-9:
                continue
            if polygon_id == end.polygon_id:
                corridor = _reconstruct(came_from, polygon_id)
                portals = [
                    self.polygons[first].neighbors[second]
                    for first, second in zip(corridor, corridor[1:])
                ]
                points = _funnel_path(
                    start.point,
                    end.point,
                    portals,
                    [self.polygons[item].center for item in corridor],
                )
                return NavPath(tuple(corridor), tuple(points), cost)

            expansions += 1
            if expansions > max_expansions:
                return None
            current = self.polygons[polygon_id]
            for neighbor_id in current.neighbors:
                neighbor = self.polygons[neighbor_id]
                portal = current.neighbors[neighbor_id]
                edge_cost = (
                    transition_cost(current, neighbor, portal)
                    if transition_cost is not None
                    else _horizontal_distance(current.center, neighbor.center)
                )
                if edge_cost is None or not math.isfinite(edge_cost) or edge_cost < 0.0:
                    continue
                next_cost = cost + edge_cost
                if next_cost + 1.0e-9 >= best_cost.get(neighbor_id, math.inf):
                    continue
                best_cost[neighbor_id] = next_cost
                came_from[neighbor_id] = polygon_id
                estimate = next_cost + _horizontal_distance(neighbor.center, goal_center)
                heapq.heappush(frontier, (estimate, next_cost, neighbor_id))
        return None

    def _build_component_ids(self) -> dict[int, int]:
        result: dict[int, int] = {}
        component = 0
        for polygon_id in self.polygons:
            if polygon_id in result:
                continue
            queue = deque([polygon_id])
            result[polygon_id] = component
            while queue:
                current = queue.popleft()
                for neighbor in self.polygons[current].neighbors:
                    if neighbor not in result:
                        result[neighbor] = component
                        queue.append(neighbor)
            component += 1
        return result


def world_to_tile(world_x: float, world_y: float) -> tuple[int, int]:
    return (
        int(math.floor(32.0 - float(world_y) / TILE_SIZE)),
        int(math.floor(32.0 - float(world_x) / TILE_SIZE)),
    )


def mmtile_path(
    mmap_dir: str | Path,
    *,
    map_id: int,
    tile_x: int,
    tile_y: int,
) -> Path:
    # AzerothCore writes mmap file names as mapId + tileY + tileX.
    return Path(mmap_dir) / f"{map_id:03d}{tile_y:02d}{tile_x:02d}.mmtile"


def parse_mmtile(path: str | Path, *, include_flags: int = NAV_GROUND) -> list[_ParsedPolygon]:
    data = Path(path).read_bytes()
    if len(data) < _MMAP_FILE_HEADER.size + _MESH_HEADER.size:
        raise ValueError(f"Truncated mmtile: {path}")
    file_header = _MMAP_FILE_HEADER.unpack_from(data)
    magic, _, version, payload_size = file_header[:4]
    if magic != MMAP_MAGIC or version != MMAP_VERSION:
        raise ValueError(f"Unsupported mmtile header: {path}")
    if payload_size != len(data) - _MMAP_FILE_HEADER.size:
        raise ValueError(f"Mismatched mmtile payload size: {path}")

    payload_offset = _MMAP_FILE_HEADER.size
    mesh_header = _MESH_HEADER.unpack_from(data, payload_offset)
    if mesh_header[0] != DT_NAVMESH_MAGIC:
        raise ValueError(f"Unsupported Detour mesh payload: {path}")
    poly_count = int(mesh_header[6])
    vert_count = int(mesh_header[7])
    off_mesh_base = int(mesh_header[14])
    vertices_offset = payload_offset + _align4(_MESH_HEADER.size)
    vertices = np.frombuffer(
        data,
        dtype="<f4",
        count=vert_count * 3,
        offset=vertices_offset,
    ).reshape((-1, 3))
    polygons_offset = vertices_offset + _align4(vert_count * 3 * 4)

    polygons: list[_ParsedPolygon] = []
    for local_index in range(min(poly_count, off_mesh_base)):
        values = _POLY.unpack_from(data, polygons_offset + local_index * _POLY.size)
        vertex_count = int(values[14])
        flags = int(values[13])
        area_and_type = int(values[15])
        if vertex_count < 3 or area_and_type >> 6 != DT_POLYTYPE_GROUND:
            continue
        if not flags & include_flags:
            continue
        vertex_indexes = values[1 : 1 + vertex_count]
        polygon_vertices = tuple(
            tuple(float(value) for value in vertices[index]) for index in vertex_indexes
        )
        polygons.append(
            _ParsedPolygon(
                local_index=local_index,
                vertices=polygon_vertices,
                neighbor_values=tuple(int(value) for value in values[7 : 7 + vertex_count]),
                flags=flags,
                area=area_and_type & 0x3F,
            )
        )
    return polygons


def densify_path(points: Sequence[Point3], max_spacing: float) -> list[Point3]:
    if max_spacing <= 0.0:
        raise ValueError("max_spacing must be positive")
    if len(points) < 2:
        return list(points)
    output = [points[0]]
    for first, second in zip(points, points[1:]):
        distance = _horizontal_distance(first, second)
        steps = max(1, int(math.ceil(distance / max_spacing)))
        for index in range(1, steps + 1):
            ratio = index / steps
            output.append(
                tuple(
                    float(first[axis]) + (float(second[axis]) - float(first[axis])) * ratio
                    for axis in range(3)
                )
            )
    return output


def _connect(
    mesh: MMapNavMesh,
    first_id: int,
    second_id: int,
    portal: tuple[Point3, Point3],
) -> None:
    if first_id == second_id:
        return
    mesh.polygons[first_id].neighbors.setdefault(second_id, portal)
    mesh.polygons[second_id].neighbors.setdefault(first_id, (portal[1], portal[0]))


def _connect_external_edges(mesh: MMapNavMesh, edges: Sequence[_ExternalEdge]) -> None:
    buckets: dict[tuple[str, int], list[_ExternalEdge]] = defaultdict(list)
    tolerance = 1.0e-3
    for edge in edges:
        delta_x = abs(edge.first[0] - edge.second[0])
        delta_z = abs(edge.first[2] - edge.second[2])
        if delta_x <= tolerance:
            buckets[("x", int(round((edge.first[0] + edge.second[0]) * 0.5 / tolerance)))].append(edge)
        elif delta_z <= tolerance:
            buckets[("z", int(round((edge.first[2] + edge.second[2]) * 0.5 / tolerance)))].append(edge)

    for (axis, _), bucket in buckets.items():
        for index, first in enumerate(bucket):
            for second in bucket[index + 1 :]:
                if (first.tile_x, first.tile_y) == (second.tile_x, second.tile_y):
                    continue
                portal = _overlapping_portal(first, second, axis=axis)
                if portal is not None:
                    _connect(mesh, first.polygon_id, second.polygon_id, portal)


def _overlapping_portal(
    first: _ExternalEdge,
    second: _ExternalEdge,
    *,
    axis: str,
) -> tuple[Point3, Point3] | None:
    varying = 2 if axis == "x" else 0
    first_values = sorted((first.first[varying], first.second[varying]))
    second_values = sorted((second.first[varying], second.second[varying]))
    low = max(first_values[0], second_values[0])
    high = min(first_values[1], second_values[1])
    if high - low <= 0.02:
        return None

    first_low = _point_on_edge(first.first, first.second, varying, low)
    first_high = _point_on_edge(first.first, first.second, varying, high)
    second_low = _point_on_edge(second.first, second.second, varying, low)
    second_high = _point_on_edge(second.first, second.second, varying, high)
    if max(abs(first_low[1] - second_low[1]), abs(first_high[1] - second_high[1])) > 2.0:
        return None
    low_point = tuple((first_low[index] + second_low[index]) * 0.5 for index in range(3))
    high_point = tuple((first_high[index] + second_high[index]) * 0.5 for index in range(3))
    if first.first[varying] <= first.second[varying]:
        return low_point, high_point
    return high_point, low_point


def _point_on_edge(first: Point3, second: Point3, axis: int, value: float) -> Point3:
    span = second[axis] - first[axis]
    ratio = 0.0 if abs(span) <= 1.0e-9 else (value - first[axis]) / span
    return tuple(first[index] + (second[index] - first[index]) * ratio for index in range(3))  # type: ignore[return-value]


def _closest_point_on_polygon(point: Point2, vertices: Sequence[Point3]) -> tuple[Point2, float]:
    polygon = [(vertex[0], vertex[2]) for vertex in vertices]
    if _point_in_polygon(point, polygon):
        return point, 0.0
    candidates = [
        _closest_point_on_segment(point, first, second)
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    ]
    closest = min(candidates, key=lambda candidate: math.dist(point, candidate))
    return closest, math.dist(point, closest)


def _point_in_polygon(point: Point2, polygon: Sequence[Point2]) -> bool:
    sign = 0
    for first, second in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        cross = _triarea2(first, second, point)
        if abs(cross) <= 1.0e-7:
            continue
        current = 1 if cross > 0 else -1
        if sign and current != sign:
            return False
        sign = current
    return True


def _closest_point_on_segment(point: Point2, first: Point2, second: Point2) -> Point2:
    delta = (second[0] - first[0], second[1] - first[1])
    length_squared = delta[0] * delta[0] + delta[1] * delta[1]
    if length_squared <= 1.0e-12:
        return first
    ratio = ((point[0] - first[0]) * delta[0] + (point[1] - first[1]) * delta[1]) / length_squared
    ratio = min(1.0, max(0.0, ratio))
    return first[0] + delta[0] * ratio, first[1] + delta[1] * ratio


def _polygon_height(vertices: Sequence[Point3], point: Point2) -> float:
    matrix = np.asarray([[vertex[0], vertex[2], 1.0] for vertex in vertices], dtype=np.float64)
    heights = np.asarray([vertex[1] for vertex in vertices], dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(matrix, heights, rcond=None)
    return float(coefficients[0] * point[0] + coefficients[1] * point[1] + coefficients[2])


def _funnel_path(
    start: Point3,
    end: Point3,
    portals: Sequence[tuple[Point3, Point3]],
    centers: Sequence[Point3],
) -> list[Point3]:
    if len(centers) != len(portals) + 1:
        raise ValueError("Expected one polygon center per corridor polygon")
    # Detour stores each from-polygon edge as left -> right. Reverse graph links
    # carry the reversed edge, so no heuristic reorientation is needed here.
    oriented: list[tuple[Point3, Point3]] = [(start, start)]
    oriented.extend(portals)
    oriented.append((end, end))

    output = [start]
    apex = start
    left = start
    right = start
    apex_index = left_index = right_index = 0
    index = 1
    while index < len(oriented):
        new_left, new_right = oriented[index]
        if _triarea3(apex, right, new_right) <= 0.0:
            if _same_point(apex, right) or _triarea3(apex, left, new_right) > 0.0:
                right = new_right
                right_index = index
            else:
                output.append(left)
                apex = left
                apex_index = left_index
                left = right = apex
                left_index = right_index = apex_index
                index = apex_index + 1
                continue
        if _triarea3(apex, left, new_left) >= 0.0:
            if _same_point(apex, left) or _triarea3(apex, right, new_left) < 0.0:
                left = new_left
                left_index = index
            else:
                output.append(right)
                apex = right
                apex_index = right_index
                left = right = apex
                left_index = right_index = apex_index
                index = apex_index + 1
                continue
        index += 1
    if not _same_point(output[-1], end):
        output.append(end)
    return output


def _reconstruct(came_from: dict[int, int], goal: int) -> list[int]:
    output = [goal]
    while goal in came_from:
        goal = came_from[goal]
        output.append(goal)
    output.reverse()
    return output


def _horizontal_distance(first: Point3, second: Point3) -> float:
    return math.dist((first[0], first[2]), (second[0], second[2]))


def _triarea2(first: Point2, second: Point2, third: Point2) -> float:
    # Match Detour's dtTriArea2D sign convention exactly.
    return (third[0] - first[0]) * (second[1] - first[1]) - (second[0] - first[0]) * (third[1] - first[1])


def _triarea3(first: Point3, second: Point3, third: Point3) -> float:
    return _triarea2((first[0], first[2]), (second[0], second[2]), (third[0], third[2]))


def _same_point(first: Point3, second: Point3) -> bool:
    return _horizontal_distance(first, second) <= 1.0e-6


def _align4(value: int) -> int:
    return (int(value) + 3) & ~3
