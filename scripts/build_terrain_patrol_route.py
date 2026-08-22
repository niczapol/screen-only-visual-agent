from __future__ import annotations

import argparse
import heapq
import json
import math
from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import Any

from scripts.build_mmap_route import render_overlay, zone_from_manifest
from vision_bot.coords import xy_to_coord
from vision_bot.terrain_routing import (
    TerrainCostConfig,
    TerrainCostMap,
    TerrainPath,
    build_terrain_cost_map,
    build_terrain_raster,
    nearest_walkable,
    supercover_line,
)


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict ADT local validation patrol")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adt-dir", required=True)
    parser.add_argument("--anchor-x", type=float, required=True)
    parser.add_argument("--anchor-y", type=float, required=True)
    parser.add_argument("--output-route", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-overlay", required=True)
    parser.add_argument("--snap-radius-cells", type=float, default=20.0)
    parser.add_argument("--max-radius-cells", type=int, default=55)
    parser.add_argument("--min-one-way-cells", type=float, default=20.0)
    parser.add_argument("--waypoint-stride", type=int, default=3)
    parser.add_argument("--min-clearance", type=float, default=1.0)
    parser.add_argument("--soft-slope-degrees", type=float, default=12.0)
    parser.add_argument("--hard-slope-degrees", type=float, default=32.0)
    parser.add_argument("--slope-weight", type=float, default=12.0)
    parser.add_argument("--roughness-weight", type=float, default=0.12)
    parser.add_argument("--clearance-cells", type=float, default=5.0)
    parser.add_argument("--clearance-weight", type=float, default=9.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.max_radius_cells <= 0 or args.waypoint_stride <= 0:
        raise ValueError("Patrol radius and waypoint stride must be positive")

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
    anchor_grid_float = raster.ui_to_grid(float(args.anchor_x), float(args.anchor_y), zone)
    anchor_grid = int(round(anchor_grid_float[0])), int(round(anchor_grid_float[1]))
    start = nearest_walkable(
        cost_map,
        anchor_grid,
        float(args.snap_radius_cells),
        min_clearance=float(args.min_clearance),
    )
    if start is None:
        raise RuntimeError("No terrain-safe cell exists near the live anchor")

    outward, reachable_count = farthest_terrain_path(
        cost_map,
        start,
        max_radius_cells=int(args.max_radius_cells),
        min_clearance=float(args.min_clearance),
    )
    one_way_cells = _grid_polyline_length(outward.points) if outward is not None else 0.0
    status = "rejected_no_terrain_patrol"
    route_points: list[tuple[int, int]] = []
    if outward is not None and one_way_cells >= float(args.min_one_way_cells):
        sampled = _sample_path(outward.points, int(args.waypoint_stride))
        route_points = sampled + list(reversed(sampled[:-1]))
        if _validate_route(route_points, cost_map, float(args.min_clearance)):
            status = "candidate_local_terrain_patrol"
        else:
            status = "rejected_sampled_patrol_crosses_blocker"
            route_points = []
    elif outward is not None:
        status = "rejected_terrain_patrol_too_short"

    route_loop = []
    for index, (row, col) in enumerate(route_points):
        ui_x, ui_y = raster.grid_to_ui(row, col, zone)
        route_loop.append(
            {
                "index": index,
                "coord": xy_to_coord(ui_x, ui_y),
                "x": round(ui_x, 6),
                "y": round(ui_y, 6),
                "world_z": round(float(raster.heights[row, col]), 4),
                "source": "ascension_adt_local_patrol",
            }
        )

    start_ui = raster.grid_to_ui(start[0], start[1], zone)
    metrics = {
        "status": status,
        "anchor_ui": [float(args.anchor_x), float(args.anchor_y)],
        "anchor_grid": list(anchor_grid),
        "snapped_start_grid": list(start),
        "snapped_start_ui": [round(start_ui[0], 6), round(start_ui[1], 6)],
        "anchor_snap_cells": round(math.dist(anchor_grid, start), 4),
        "terrain_safe_reachable_cells": reachable_count,
        "one_way_grid_cells": round(one_way_cells, 4),
        "route_waypoint_count": len(route_loop),
        "route_grid_cells": round(_grid_polyline_length(route_points), 4),
        "terrain_cost_config": terrain_config.to_dict(),
        "min_clearance": float(args.min_clearance),
        "max_radius_cells": int(args.max_radius_cells),
    }
    result = {
        "schema_version": 3,
        "name": "desolace_kodo_graveyard_adt_validation_patrol",
        "zone": {"id": 102, "name": manifest.get("zone_name", "Desolace")},
        "mode": "cyclic" if status == "candidate_local_terrain_patrol" else "blocked",
        "live_status": status,
        "source": {
            "navigation": "external client ADT V9 slope/roughness/clearance",
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
        f"status={status} reachable={reachable_count} snap={metrics['anchor_snap_cells']} "
        f"one_way_cells={metrics['one_way_grid_cells']} waypoints={len(route_loop)}"
    )


def farthest_terrain_path(
    cost_map: TerrainCostMap,
    start: tuple[int, int],
    *,
    max_radius_cells: int,
    min_clearance: float,
) -> tuple[TerrainPath | None, int]:
    if not cost_map.is_walkable(start, min_clearance=min_clearance):
        return None, 0
    sequence = count()
    frontier: list[tuple[float, int, tuple[int, int]]] = [(0.0, next(sequence), start)]
    best_cost = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_point = start
    best_distance = 0.0
    rows, cols = cost_map.cost.shape

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)
        if current_cost > best_cost.get(current, math.inf) + 1.0e-9:
            continue
        distance = math.dist(start, current)
        if distance > best_distance:
            best_point = current
            best_distance = distance
        row, col = current
        for delta_row, delta_col, step_length in _NEIGHBORS:
            next_point = row + delta_row, col + delta_col
            next_row, next_col = next_point
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if math.dist(start, next_point) > max_radius_cells:
                continue
            if not cost_map.is_walkable(next_point, min_clearance=min_clearance):
                continue
            if delta_row and delta_col:
                if not cost_map.is_walkable(
                    (row + delta_row, col), min_clearance=min_clearance
                ) or not cost_map.is_walkable(
                    (row, col + delta_col), min_clearance=min_clearance
                ):
                    continue
            edge_cost = step_length * 0.5 * (
                float(cost_map.cost[current]) + float(cost_map.cost[next_point])
            )
            next_cost = current_cost + edge_cost
            if next_cost + 1.0e-9 >= best_cost.get(next_point, math.inf):
                continue
            best_cost[next_point] = next_cost
            came_from[next_point] = current
            heapq.heappush(frontier, (next_cost, next(sequence), next_point))

    if best_point == start:
        return None, len(best_cost)
    points = [best_point]
    while points[-1] != start:
        points.append(came_from[points[-1]])
    points.reverse()
    return TerrainPath(points=points, total_cost=best_cost[best_point]), len(best_cost)


def _sample_path(points: list[tuple[int, int]], stride: int) -> list[tuple[int, int]]:
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _validate_route(
    points: list[tuple[int, int]], cost_map: TerrainCostMap, min_clearance: float
) -> bool:
    for first, second in zip(points, points[1:]):
        for point in supercover_line(first, second):
            if not cost_map.is_walkable(point, min_clearance=min_clearance):
                return False
    return bool(points)


def _grid_polyline_length(points: Sequence[tuple[int, int]]) -> float:
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
