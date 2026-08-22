from __future__ import annotations

import base64
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_bot.coords import coord_to_xy, normalize_coord, xy_to_coord
from vision_bot.route_planner import MiningNode


@dataclass(frozen=True)
class ImportedRoute:
    name: str
    zone_name: str
    coords: list[int]
    source_line: str


@dataclass(frozen=True)
class RouteBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, coord: int) -> bool:
        x, y = coord_to_xy(coord)
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }


@dataclass(frozen=True)
class RouteProjection:
    distance: float
    segment_index: int
    segment_t: float
    nearest_coord: int
    nearest_x: float
    nearest_y: float

    @property
    def sort_key(self) -> float:
        return self.segment_index + self.segment_t


def load_wotlk_routes_from_markdown(path: str | Path) -> list[ImportedRoute]:
    routes: list[ImportedRoute] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("WOTLKC:Routes:"):
            routes.append(decode_wotlk_route_line(line))
    return routes


def load_route_loop_coords(path: str | Path) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [int(item["coord"]) for item in data.get("route_loop", []) if "coord" in item]


def decode_wotlk_route_line(line: str) -> ImportedRoute:
    prefix, payload = line.rsplit(":", 1)
    parts = prefix.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Unsupported route line prefix: {prefix}")
    _, _, route_name, zone_name = parts
    raw = base64.b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8", errors="replace")
    coords: list[int] = []
    for match in re.finditer(r"\^N(\d{7,8})(?=\^|$)", raw):
        coord = normalize_coord(int(match.group(1)))
        x, y = coord_to_xy(coord)
        if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0:
            coords.append(coord)
    return ImportedRoute(name=route_name, zone_name=zone_name, coords=_dedupe_consecutive(coords), source_line=line)


def build_safe_route_database(
    *,
    route: ImportedRoute,
    nodes: list[MiningNode],
    zone_id: int,
    corridor_radius: float,
    name: str,
    source_mining_data: str,
    source_route_data: str,
    min_node_spacing: float = 0.0,
) -> dict[str, Any]:
    route_nodes: list[dict[str, Any]] = []
    rejected_nodes: list[dict[str, Any]] = []

    for node in nodes:
        if node.zone_id != zone_id:
            continue
        projection = project_to_loop(node.coord, route.coords)
        x, y = coord_to_xy(node.coord)
        base_item = {
            "zone_id": node.zone_id,
            "coord": normalize_coord(node.coord),
            "x": round(x, 2),
            "y": round(y, 2),
            "ore_id": node.ore_id,
            "ore_type": node.ore_type,
            "source_node_id": node.node_id,
            "route_index": projection.segment_index,
            "route_t": round(projection.segment_t, 4),
            "distance_to_route": round(projection.distance, 4),
            "source": "snapped_mining_node",
        }
        if projection.distance <= corridor_radius:
            route_nodes.append(base_item)
        else:
            rejected = dict(base_item)
            rejected["reason"] = "outside_route_corridor"
            rejected_nodes.append(rejected)

    route_nodes.sort(key=lambda item: (int(item["route_index"]), float(item["route_t"]), float(item["distance_to_route"])))
    if min_node_spacing > 0:
        route_nodes = _filter_spaced_nodes(route_nodes, min_node_spacing, rejected_nodes)

    for index, item in enumerate(route_nodes, start=1):
        item["node_id"] = index
        item["route_order"] = index - 1

    waypoints = []
    for index, coord in enumerate(route.coords):
        x, y = coord_to_xy(coord)
        waypoints.append({"index": index, "coord": coord, "x": round(x, 2), "y": round(y, 2)})

    ore_counts: dict[str, int] = {}
    for item in route_nodes:
        ore_type = str(item["ore_type"])
        ore_counts[ore_type] = ore_counts.get(ore_type, 0) + 1

    return {
        "schema_version": 1,
        "name": name,
        "zone": {"id": zone_id, "name": route.zone_name},
        "mode": "cyclic",
        "source": {
            "route_name": route.name,
            "route_data": source_route_data,
            "mining_data": source_mining_data,
        },
        "generation": {
            "corridor_radius": corridor_radius,
            "min_node_spacing": min_node_spacing,
            "route_waypoints": len(route.coords),
            "accepted_nodes": len(route_nodes),
            "rejected_nodes": len(rejected_nodes),
            "ore_counts": ore_counts,
        },
        "route_loop": waypoints,
        "route_nodes": route_nodes,
        "rejected_nodes": rejected_nodes,
    }


def build_constrained_route_database(
    *,
    nodes: list[MiningNode],
    zone_id: int,
    zone_name: str,
    bounds: RouteBounds,
    name: str,
    source_mining_data: str,
    allowed_ores: set[str] | None = None,
    min_node_spacing: float = 0.0,
    max_two_opt_iterations: int = 80,
) -> dict[str, Any]:
    route_nodes: list[dict[str, Any]] = []
    rejected_nodes: list[dict[str, Any]] = []
    ore_filter = set(allowed_ores or set())

    for node in nodes:
        if node.zone_id != zone_id:
            continue
        x, y = coord_to_xy(node.coord)
        base_item = {
            "zone_id": node.zone_id,
            "coord": normalize_coord(node.coord),
            "x": round(x, 2),
            "y": round(y, 2),
            "ore_id": node.ore_id,
            "ore_type": node.ore_type,
            "source_node_id": node.node_id,
            "source": "bounded_mining_node",
        }
        if ore_filter and node.ore_type not in ore_filter:
            rejected = dict(base_item)
            rejected["reason"] = "ore_not_allowed"
            rejected_nodes.append(rejected)
            continue
        if not bounds.contains(node.coord):
            rejected = dict(base_item)
            rejected["reason"] = "outside_route_bounds"
            rejected_nodes.append(rejected)
            continue
        route_nodes.append(base_item)

    route_nodes.sort(key=lambda item: (float(item["y"]), float(item["x"]), int(item["coord"])))
    if min_node_spacing > 0:
        route_nodes = _filter_spaced_nodes(route_nodes, min_node_spacing, rejected_nodes)

    route_nodes = _order_route_nodes_as_cycle(route_nodes, max_two_opt_iterations=max_two_opt_iterations)

    for index, item in enumerate(route_nodes):
        item["node_id"] = index + 1
        item["route_order"] = index
        item["route_index"] = index
        item["route_t"] = 0.0
        item["distance_to_route"] = 0.0

    route_loop = [
        {
            "index": index,
            "coord": int(item["coord"]),
            "x": float(item["x"]),
            "y": float(item["y"]),
        }
        for index, item in enumerate(route_nodes)
    ]

    ore_counts: dict[str, int] = {}
    for item in route_nodes:
        ore_type = str(item["ore_type"])
        ore_counts[ore_type] = ore_counts.get(ore_type, 0) + 1

    return {
        "schema_version": 1,
        "name": name,
        "zone": {"id": zone_id, "name": zone_name},
        "mode": "cyclic",
        "source": {
            "route_name": "bounded_mining_node_cycle",
            "route_data": "generated_from_bounds",
            "mining_data": source_mining_data,
        },
        "generation": {
            "bounds": bounds.to_dict(),
            "ordering": "nearest_neighbor_2opt",
            "min_node_spacing": min_node_spacing,
            "max_two_opt_iterations": max_two_opt_iterations,
            "route_waypoints": len(route_loop),
            "accepted_nodes": len(route_nodes),
            "rejected_nodes": len(rejected_nodes),
            "ore_counts": ore_counts,
        },
        "route_loop": route_loop,
        "route_nodes": route_nodes,
        "rejected_nodes": rejected_nodes,
    }


def project_to_loop(coord: int, route_coords: list[int]) -> RouteProjection:
    return _project_to_segments(coord, route_coords, closed=True)


def project_to_path(coord: int, route_coords: list[int]) -> RouteProjection:
    """Project a coordinate onto an open route without a synthetic last-to-first edge."""
    return _project_to_segments(coord, route_coords, closed=False)


def _project_to_segments(
    coord: int,
    route_coords: list[int],
    *,
    closed: bool,
) -> RouteProjection:
    if len(route_coords) < 2:
        x, y = coord_to_xy(coord)
        return RouteProjection(
            distance=0.0,
            segment_index=0,
            segment_t=0.0,
            nearest_coord=normalize_coord(coord),
            nearest_x=x,
            nearest_y=y,
        )

    point = coord_to_xy(coord)
    best = RouteProjection(
        distance=float("inf"),
        segment_index=0,
        segment_t=0.0,
        nearest_coord=normalize_coord(coord),
        nearest_x=point[0],
        nearest_y=point[1],
    )
    segment_ends = route_coords[1:] + (route_coords[:1] if closed else [])
    segments = zip(route_coords, segment_ends)
    for index, (start, end) in enumerate(segments):
        distance, segment_t, nearest_x, nearest_y = _point_segment_distance(point, coord_to_xy(start), coord_to_xy(end))
        if distance < best.distance:
            best = RouteProjection(
                distance=distance,
                segment_index=index,
                segment_t=segment_t,
                nearest_coord=xy_to_coord(nearest_x, nearest_y),
                nearest_x=nearest_x,
                nearest_y=nearest_y,
            )
    return best


def write_route_database(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_route_csv(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "route_order",
        "route_index",
        "route_t",
        "distance_to_route",
        "zone_id",
        "coord",
        "x",
        "y",
        "ore_id",
        "ore_type",
        "source_node_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in data.get("route_nodes", []):
            writer.writerow({field: item.get(field) for field in fieldnames})


def write_route_svg(data: dict[str, Any], output_path: str | Path, *, size: int = 900) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    padding = 35
    scale = size - padding * 2

    def point(coord: int) -> tuple[float, float]:
        x, y = coord_to_xy(coord)
        return padding + (x / 100.0) * scale, padding + (y / 100.0) * scale

    loop = data.get("route_loop", [])
    polyline = " ".join(f"{point(int(item['coord']))[0]:.1f},{point(int(item['coord']))[1]:.1f}" for item in loop)
    node_circles = []
    for item in data.get("route_nodes", []):
        x, y = point(int(item["coord"]))
        color = "#d38a26" if item.get("ore_type") == "Copper" else "#d8d8d8"
        node_circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="#202020" stroke-width="1" />'
        )

    title = _xml_escape(str(data.get("name", "route")))
    accepted = int(data.get("generation", {}).get("accepted_nodes", 0))
    rejected = int(data.get("generation", {}).get("rejected_nodes", 0))
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
                '<rect width="100%" height="100%" fill="#f3ead8" />',
                f'<text x="{padding}" y="24" font-family="Segoe UI, Arial" font-size="16" fill="#1d1d1d">{title}</text>',
                f'<text x="{padding}" y="{size - 14}" font-family="Segoe UI, Arial" font-size="13" fill="#333">accepted={accepted} rejected={rejected}</text>',
                f'<polyline points="{polyline}" fill="none" stroke="#246a92" stroke-width="3" stroke-linejoin="round" />',
                *node_circles,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def _point_segment_distance(
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
    nearest_x = ax + segment_t * vx
    nearest_y = ay + segment_t * vy
    return math.hypot(px - nearest_x, py - nearest_y), segment_t, nearest_x, nearest_y


def _dedupe_consecutive(coords: list[int]) -> list[int]:
    deduped: list[int] = []
    for coord in coords:
        if not deduped or deduped[-1] != coord:
            deduped.append(coord)
    return deduped


def _filter_spaced_nodes(
    route_nodes: list[dict[str, Any]],
    min_node_spacing: float,
    rejected_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for item in route_nodes:
        if all(_coord_distance(int(item["coord"]), int(existing["coord"])) >= min_node_spacing for existing in accepted):
            accepted.append(item)
            continue
        rejected = dict(item)
        rejected["reason"] = "too_close_to_accepted_route_node"
        rejected_nodes.append(rejected)
    return accepted


def _order_route_nodes_as_cycle(
    route_nodes: list[dict[str, Any]],
    *,
    max_two_opt_iterations: int,
) -> list[dict[str, Any]]:
    if len(route_nodes) <= 2:
        return list(route_nodes)

    order = _nearest_neighbor_order(route_nodes)
    order = _two_opt_cycle_order(route_nodes, order, max_iterations=max_two_opt_iterations)
    return [route_nodes[index] for index in order]


def _nearest_neighbor_order(route_nodes: list[dict[str, Any]]) -> list[int]:
    start_index = min(
        range(len(route_nodes)),
        key=lambda index: (float(route_nodes[index]["y"]), float(route_nodes[index]["x"]), int(route_nodes[index]["coord"])),
    )
    order = [start_index]
    unvisited = set(range(len(route_nodes)))
    unvisited.remove(start_index)
    current = start_index
    while unvisited:
        next_index = min(
            unvisited,
            key=lambda index: (
                _route_item_distance(route_nodes[current], route_nodes[index]),
                float(route_nodes[index]["y"]),
                float(route_nodes[index]["x"]),
                int(route_nodes[index]["coord"]),
            ),
        )
        order.append(next_index)
        unvisited.remove(next_index)
        current = next_index
    return order


def _two_opt_cycle_order(
    route_nodes: list[dict[str, Any]],
    order: list[int],
    *,
    max_iterations: int,
) -> list[int]:
    if len(order) <= 3 or max_iterations <= 0:
        return list(order)

    improved_order = list(order)
    for _ in range(max_iterations):
        improved = False
        for i in range(len(improved_order) - 2):
            for k in range(i + 2, len(improved_order)):
                if i == 0 and k == len(improved_order) - 1:
                    continue
                a = route_nodes[improved_order[i]]
                b = route_nodes[improved_order[(i + 1) % len(improved_order)]]
                c = route_nodes[improved_order[k]]
                d = route_nodes[improved_order[(k + 1) % len(improved_order)]]
                current = _route_item_distance(a, b) + _route_item_distance(c, d)
                candidate = _route_item_distance(a, c) + _route_item_distance(b, d)
                if candidate + 1e-9 < current:
                    improved_order[i + 1 : k + 1] = reversed(improved_order[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return improved_order


def _route_item_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _coord_distance(a: int, b: int) -> float:
    ax, ay = coord_to_xy(a)
    bx, by = coord_to_xy(b)
    return math.hypot(ax - bx, ay - by)


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
