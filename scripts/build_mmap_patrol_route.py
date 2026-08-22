from __future__ import annotations

import argparse
import heapq
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from scripts.build_mmap_route import (
    build_terrain_transition_cost,
    locate_ui,
    render_overlay,
    validate_navmesh_path,
    zone_from_manifest,
)
from vision_bot.coords import xy_to_coord
from vision_bot.mmap_navmesh import MMapNavMesh, NavLocation, NavPolygon, densify_path
from vision_bot.terrain_routing import TerrainCostConfig, build_terrain_cost_map, build_terrain_raster


TransitionCost = Callable[
    [NavPolygon, NavPolygon, tuple[tuple[float, float, float], tuple[float, float, float]]],
    float | None,
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a terrain-constrained out-and-back patrol from a live-safe anchor"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adt-dir", required=True)
    parser.add_argument("--mmap-dir", required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--anchor-x", type=float, required=True)
    parser.add_argument("--anchor-y", type=float, required=True)
    parser.add_argument("--output-route", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-overlay", required=True)
    parser.add_argument("--max-nav-snap", type=float, default=30.0)
    parser.add_argument("--max-nav-vertical", type=float, default=50.0)
    parser.add_argument("--max-waypoint-spacing", type=float, default=18.0)
    parser.add_argument("--min-clearance", type=float, default=1.0)
    parser.add_argument("--min-patrol-length", type=float, default=80.0)
    parser.add_argument("--max-patrol-cost", type=float, default=500.0)
    parser.add_argument("--soft-slope-degrees", type=float, default=12.0)
    parser.add_argument("--hard-slope-degrees", type=float, default=32.0)
    parser.add_argument("--slope-weight", type=float, default=12.0)
    parser.add_argument("--roughness-weight", type=float, default=0.12)
    parser.add_argument("--clearance-cells", type=float, default=5.0)
    parser.add_argument("--clearance-weight", type=float, default=9.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.max_waypoint_spacing <= 0.0:
        raise ValueError("--max-waypoint-spacing must be positive")
    if args.min_patrol_length < 0.0 or args.max_patrol_cost <= 0.0:
        raise ValueError("Patrol distance limits must be non-negative")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(Path(args.adt_dir).glob("*.adt"))
    terrain_config = TerrainCostConfig(
        soft_slope_degrees=float(args.soft_slope_degrees),
        hard_slope_degrees=float(args.hard_slope_degrees),
        slope_weight=float(args.slope_weight),
        roughness_weight=float(args.roughness_weight),
        clearance_cells=float(args.clearance_cells),
        clearance_weight=float(args.clearance_weight),
    )
    cost_map = build_terrain_cost_map(raster, terrain_config)
    transition_cost = build_terrain_transition_cost(
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        min_clearance=float(args.min_clearance),
    )
    tiles = [(int(item["x"]), int(item["y"])) for item in manifest["tiles"]]
    mesh = MMapNavMesh.load_tiles(args.mmap_dir, map_id=int(args.map_id), tiles=tiles)
    anchor = locate_ui(
        mesh,
        raster,
        zone,
        float(args.anchor_x),
        float(args.anchor_y),
        max_nav_snap=float(args.max_nav_snap),
        max_nav_vertical=float(args.max_nav_vertical),
    )
    if anchor is None:
        raise RuntimeError("Anchor coordinate does not resolve to a ground navmesh polygon")

    target, reachable_count = farthest_reachable_location(
        mesh,
        anchor,
        transition_cost=transition_cost,
        max_graph_cost=float(args.max_patrol_cost),
    )
    route_points: list[tuple[float, float, float]] = []
    path = None
    validation: dict[str, Any] | None = None
    status = "rejected_no_patrol_corridor"
    if target is not None:
        path = mesh.find_path(anchor, target, transition_cost=transition_cost)
    if path is not None:
        outward = densify_path(path.points, float(args.max_waypoint_spacing))
        validation = validate_navmesh_path(
            outward,
            raster=raster,
            zone=zone,
            cost_map=cost_map,
            min_clearance=float(args.min_clearance),
        )
        if validation["walkable"] and path.world_length >= float(args.min_patrol_length):
            route_points = outward + list(reversed(outward[:-1]))
            status = "candidate_local_patrol"
        elif validation["walkable"]:
            status = "rejected_patrol_too_short"
        else:
            status = "rejected_patrol_crosses_client_terrain_blocker"

    route_loop = []
    for index, point in enumerate(route_points):
        ui_x, ui_y = zone.world_to_ui(point[2], point[0])
        route_loop.append(
            {
                "index": index,
                "coord": xy_to_coord(ui_x, ui_y),
                "x": round(ui_x, 6),
                "y": round(ui_y, 6),
                "world_z": round(point[1], 4),
                "source": "azerothcore_mmap_local_patrol",
            }
        )

    metrics = {
        "map_id": int(args.map_id),
        "status": status,
        "anchor_ui": [float(args.anchor_x), float(args.anchor_y)],
        "anchor_polygon_id": int(anchor.polygon_id),
        "anchor_component_id": int(mesh.component_id(anchor.polygon_id)),
        "anchor_component_polygons": mesh.component_size(anchor.polygon_id),
        "terrain_safe_reachable_polygons": reachable_count,
        "target_polygon_id": int(target.polygon_id) if target is not None else None,
        "one_way_world_length": round(path.world_length, 4) if path is not None else 0.0,
        "route_world_length": round(_polyline_length(route_points), 4),
        "route_waypoint_count": len(route_loop),
        "corridor_polygons": len(path.polygon_ids) if path is not None else 0,
        "terrain_validation": validation,
        "terrain_cost_config": terrain_config.to_dict(),
        "min_clearance": float(args.min_clearance),
        "max_patrol_cost": float(args.max_patrol_cost),
        "min_patrol_length": float(args.min_patrol_length),
    }
    result = {
        "schema_version": 3,
        "name": "desolace_local_navigation_validation_patrol",
        "zone": {"id": 102, "name": manifest.get("zone_name", "Desolace")},
        "mode": "cyclic" if status == "candidate_local_patrol" else "blocked",
        "live_status": status,
        "source": {
            "navigation": "AzerothCore mmap v20 ground polygons constrained by external client ADT",
            "purpose": "local navigation/combat/recovery validation; not an ore farming cycle",
        },
        "generation": metrics,
        "route_loop": route_loop,
        "route_nodes": [],
        "rejected_nodes": [],
    }
    _write_json(result, args.output_route)
    _write_json(metrics, args.output_metrics)
    render_overlay(
        raster=raster,
        zone=zone,
        route_loop=route_loop,
        accepted_nodes=[],
        rejected_nodes=[],
        anchor_ui=(float(args.anchor_x), float(args.anchor_y)),
        output_path=args.output_overlay,
    )
    print(
        f"status={status} reachable={reachable_count} waypoints={len(route_loop)} "
        f"one_way={metrics['one_way_world_length']} total={metrics['route_world_length']}"
    )


def farthest_reachable_location(
    mesh: MMapNavMesh,
    start: NavLocation,
    *,
    transition_cost: TransitionCost,
    max_graph_cost: float,
) -> tuple[NavLocation | None, int]:
    frontier: list[tuple[float, int]] = [(0.0, start.polygon_id)]
    best_cost = {start.polygon_id: 0.0}
    best_polygon = start.polygon_id
    best_distance = 0.0
    while frontier:
        cost, polygon_id = heapq.heappop(frontier)
        if cost > best_cost.get(polygon_id, math.inf) + 1.0e-9:
            continue
        current = mesh.polygons[polygon_id]
        distance = _horizontal_distance(start.point, current.center)
        if (distance, cost, -polygon_id) > (
            best_distance,
            best_cost.get(best_polygon, 0.0),
            -best_polygon,
        ):
            best_polygon = polygon_id
            best_distance = distance
        for neighbor_id, portal in current.neighbors.items():
            neighbor = mesh.polygons[neighbor_id]
            edge_cost = transition_cost(current, neighbor, portal)
            if edge_cost is None or not math.isfinite(edge_cost) or edge_cost < 0.0:
                continue
            next_cost = cost + edge_cost
            if next_cost > max_graph_cost:
                continue
            if next_cost + 1.0e-9 >= best_cost.get(neighbor_id, math.inf):
                continue
            best_cost[neighbor_id] = next_cost
            heapq.heappush(frontier, (next_cost, neighbor_id))

    if best_polygon == start.polygon_id:
        return None, len(best_cost)
    center = mesh.polygons[best_polygon].center
    return NavLocation(best_polygon, center, 0.0, 0.0), len(best_cost)


def _horizontal_distance(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return math.dist((first[0], first[2]), (second[0], second[2]))


def _polyline_length(points: Sequence[tuple[float, float, float]]) -> float:
    return sum(_horizontal_distance(first, second) for first, second in zip(points, points[1:]))


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
