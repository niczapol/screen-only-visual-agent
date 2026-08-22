from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.coords import xy_to_coord
from vision_bot.terrain_routing import (
    GRID_SPACING,
    TerrainCostMap,
    TerrainCostConfig,
    TerrainPath,
    TerrainRaster,
    ZoneTransform,
    astar,
    build_terrain_cost_map,
    build_terrain_raster,
    line_is_walkable,
    nearest_walkable,
    path_length,
    simplify_path,
    supercover_line,
)


@dataclass(frozen=True)
class PlannedSegment:
    index: int
    start_node_index: int
    end_node_index: int
    raw_path: TerrainPath
    simplified_points: list[tuple[int, int]]
    search_mode: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a terrain-aware cyclic mining route from external client ADT heights"
    )
    parser.add_argument(
        "--input-route",
        default="data/routes/generated/barrens_central_east_mining_safe_cycle.json",
    )
    parser.add_argument(
        "--manifest",
        default="data/extracted_client_data/barrens_v2/manifest.json",
    )
    parser.add_argument(
        "--adt-dir",
        default="data/extracted_client_data/barrens_v2/world/maps/Kalimdor",
    )
    parser.add_argument(
        "--output-route",
        default="data/routes/generated/barrens_topographic_safe_cycle.json",
    )
    parser.add_argument(
        "--output-metrics",
        default="data/routes/generated/barrens_topographic_safe_cycle_metrics.json",
    )
    parser.add_argument(
        "--output-overlay",
        default="data/routes/generated/barrens_topographic_safe_cycle.png",
    )
    parser.add_argument("--max-snap-radius", type=float, default=8.0)
    parser.add_argument("--approach-candidate-sectors", type=int, default=8)
    parser.add_argument("--approach-refinement-passes", type=int, default=2)
    parser.add_argument("--approach-distance-weight", type=float, default=4.0)
    parser.add_argument("--approach-max-extra-cells", type=float, default=2.0)
    parser.add_argument("--approach-access-options", type=int, default=3)
    parser.add_argument("--anchor-x", type=float)
    parser.add_argument("--anchor-y", type=float)
    parser.add_argument("--anchor-snap-radius", type=float, default=20.0)
    parser.add_argument("--min-clearance", type=float, default=1.0)
    parser.add_argument("--max-waypoint-spacing", type=float, default=16.0)
    parser.add_argument("--route-bound-margin", type=float, default=1.0)
    parser.add_argument("--soft-slope-degrees", type=float, default=15.0)
    parser.add_argument("--hard-slope-degrees", type=float, default=35.0)
    parser.add_argument("--slope-weight", type=float, default=8.0)
    parser.add_argument("--roughness-weight", type=float, default=0.08)
    parser.add_argument("--clearance-cells", type=float, default=3.0)
    parser.add_argument("--clearance-weight", type=float, default=5.0)
    parser.add_argument("--optimize-node-order", action="store_true")
    parser.add_argument("--max-two-opt-iterations", type=int, default=80)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def zone_from_manifest(manifest: dict[str, Any]) -> ZoneTransform:
    area = manifest["world_map_area"]
    return ZoneTransform(
        left=float(area["left"]),
        right=float(area["right"]),
        top=float(area["top"]),
        bottom=float(area["bottom"]),
    )


def restrict_to_primary_component(
    cost_map: TerrainCostMap,
    *,
    min_clearance: float = 0.0,
) -> tuple[TerrainCostMap, np.ndarray, int]:
    preliminary_blocked = np.asarray(cost_map.blocked, dtype=bool).copy()
    if min_clearance > 0.0:
        preliminary_blocked |= cost_map.clearance < min_clearance
    foreground = (~preliminary_blocked).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=4,
    )
    if label_count <= 1:
        raise ValueError("Terrain cost map contains no walkable component")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    primary_label = int(np.argmax(component_areas)) + 1
    primary_mask = labels == primary_label
    primary_area = int(stats[primary_label, cv2.CC_STAT_AREA])

    restricted_blocked = preliminary_blocked
    restricted_blocked |= ~primary_mask
    restricted_cost = np.asarray(cost_map.cost, dtype=np.float32).copy()
    restricted_cost[restricted_blocked] = np.inf
    return (
        TerrainCostMap(
            cost=restricted_cost,
            slope_degrees=cost_map.slope_degrees,
            roughness=cost_map.roughness,
            clearance=cost_map.clearance,
            blocked=restricted_blocked,
            config=cost_map.config,
        ),
        primary_mask,
        primary_area,
    )


def restrict_to_anchor_component(
    cost_map: TerrainCostMap,
    anchor: tuple[int, int],
    *,
    max_snap_radius: float,
    min_clearance: float = 0.0,
) -> tuple[TerrainCostMap, np.ndarray, int, tuple[int, int]]:
    preliminary_blocked = np.asarray(cost_map.blocked, dtype=bool).copy()
    if min_clearance > 0.0:
        preliminary_blocked |= cost_map.clearance < min_clearance
    preliminary_cost = np.asarray(cost_map.cost, dtype=np.float32).copy()
    preliminary_cost[preliminary_blocked] = np.inf
    preliminary_map = TerrainCostMap(
        cost=preliminary_cost,
        slope_degrees=cost_map.slope_degrees,
        roughness=cost_map.roughness,
        clearance=cost_map.clearance,
        blocked=preliminary_blocked,
        config=cost_map.config,
    )
    snapped_anchor = nearest_walkable(
        preliminary_map,
        anchor,
        max_snap_radius,
        min_clearance=min_clearance,
    )
    if snapped_anchor is None:
        raise ValueError("No walkable terrain component exists near the route anchor")

    foreground = (~preliminary_blocked).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=4,
    )
    if label_count <= 1:
        raise ValueError("Terrain cost map contains no walkable component")
    anchor_label = int(labels[snapped_anchor])
    if anchor_label <= 0:
        raise ValueError("Snapped route anchor is outside walkable terrain")
    component_mask = labels == anchor_label
    component_area = int(stats[anchor_label, cv2.CC_STAT_AREA])

    restricted_blocked = preliminary_blocked | ~component_mask
    restricted_cost = np.asarray(cost_map.cost, dtype=np.float32).copy()
    restricted_cost[restricted_blocked] = np.inf
    return (
        TerrainCostMap(
            cost=restricted_cost,
            slope_degrees=cost_map.slope_degrees,
            roughness=cost_map.roughness,
            clearance=cost_map.clearance,
            blocked=restricted_blocked,
            config=cost_map.config,
        ),
        component_mask,
        component_area,
        snapped_anchor,
    )


def restrict_to_ui_bounds(
    cost_map: TerrainCostMap,
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    bounds: dict[str, Any],
    margin: float,
) -> tuple[TerrainCostMap, tuple[int, int, int, int]]:
    min_x = max(0.0, float(bounds["min_x"]) - margin)
    max_x = min(100.0, float(bounds["max_x"]) + margin)
    min_y = max(0.0, float(bounds["min_y"]) - margin)
    max_y = min(100.0, float(bounds["max_y"]) + margin)
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Route UI bounds collapse after applying margin")

    grid_corners = [
        raster.ui_to_grid(ui_x, ui_y, zone)
        for ui_x in (min_x, max_x)
        for ui_y in (min_y, max_y)
    ]
    row_min = max(0, int(math.floor(min(point[0] for point in grid_corners))))
    row_max = min(
        cost_map.cost.shape[0] - 1,
        int(math.ceil(max(point[0] for point in grid_corners))),
    )
    col_min = max(0, int(math.floor(min(point[1] for point in grid_corners))))
    col_max = min(
        cost_map.cost.shape[1] - 1,
        int(math.ceil(max(point[1] for point in grid_corners))),
    )
    if row_min > row_max or col_min > col_max:
        raise ValueError("Route UI bounds do not intersect the terrain raster")
    allowed = np.zeros(cost_map.cost.shape, dtype=bool)
    allowed[row_min : row_max + 1, col_min : col_max + 1] = True
    bounded_blocked = np.asarray(cost_map.blocked, dtype=bool).copy() | ~allowed
    bounded_cost = np.asarray(cost_map.cost, dtype=np.float32).copy()
    bounded_cost[bounded_blocked] = np.inf
    return (
        TerrainCostMap(
            cost=bounded_cost,
            slope_degrees=cost_map.slope_degrees,
            roughness=cost_map.roughness,
            clearance=cost_map.clearance,
            blocked=bounded_blocked,
            config=cost_map.config,
        ),
        (row_min, row_max, col_min, col_max),
    )


def snap_route_nodes(
    *,
    nodes: list[dict[str, Any]],
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
    max_snap_radius: float,
    min_clearance: float,
    approach_candidate_sectors: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_order, node in enumerate(nodes):
        row_float, col_float = raster.ui_to_grid(float(node["x"]), float(node["y"]), zone)
        raw_point = int(round(row_float)), int(round(col_float))
        approach_candidates = find_ore_approach_candidates(
            cost_map,
            raw_point,
            max_snap_radius,
            min_clearance=min_clearance,
            sector_count=approach_candidate_sectors,
        )
        if not approach_candidates:
            rejected_node = dict(node)
            rejected_node.update(
                {
                    "reason": "terrain_no_primary_component_approach",
                    "terrain_source_route_order": source_order,
                    "terrain_raw_grid": [raw_point[0], raw_point[1]],
                    "terrain_snap_limit_cells": float(max_snap_radius),
                }
            )
            rejected.append(rejected_node)
            continue

        approach = tuple(approach_candidates[0]["grid"])
        accepted_node = dict(node)
        accepted_node.update(
            {
                "route_order": len(accepted),
                "route_index": -1,
                "route_t": 0.0,
                "distance_to_route": 0.0,
                "terrain_source_route_order": source_order,
                "terrain_raw_grid": [raw_point[0], raw_point[1]],
                "terrain_approach_grid": [approach[0], approach[1]],
                "terrain_snap_cells": round(math.dist(raw_point, approach), 4),
                "terrain_approach_candidates": approach_candidates,
                "terrain_selected_approach_index": 0,
            }
        )
        accepted.append(accepted_node)
    return accepted, rejected


def find_ore_approach_candidates(
    cost_map: TerrainCostMap,
    raw_point: tuple[int, int],
    max_radius: float,
    *,
    min_clearance: float,
    sector_count: int = 8,
) -> list[dict[str, Any]]:
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")
    if sector_count <= 0:
        raise ValueError("sector_count must be positive")

    raw_row, raw_col = raw_point
    radius = int(math.ceil(max_radius))
    best_by_sector: dict[int, tuple[tuple[float, ...], dict[str, Any]]] = {}
    for row in range(raw_row - radius, raw_row + radius + 1):
        for col in range(raw_col - radius, raw_col + radius + 1):
            distance = math.hypot(row - raw_row, col - raw_col)
            if distance > max_radius or not cost_map.is_walkable(
                (row, col),
                min_clearance=min_clearance,
            ):
                continue
            if distance <= 1.0e-9:
                sector = -1
                bearing = 0.0
            else:
                bearing = (math.degrees(math.atan2(row - raw_row, col - raw_col)) + 360.0) % 360.0
                sector = int((bearing / 360.0) * sector_count) % sector_count
            slope = float(cost_map.slope_degrees[row, col])
            clearance = float(cost_map.clearance[row, col])
            terrain_cost = float(cost_map.cost[row, col])
            local_score = (
                distance
                + slope * 0.04
                + terrain_cost * 0.03
                - min(clearance, 8.0) * 0.08
            )
            item = {
                "grid": [row, col],
                "sector": sector,
                "bearing_degrees": round(bearing, 3),
                "snap_cells": round(distance, 4),
                "slope_degrees": round(slope, 4),
                "clearance_cells": round(clearance, 4),
                "terrain_cost": round(terrain_cost, 4),
                "local_score": round(local_score, 4),
            }
            rank = (local_score, distance, slope, -clearance, row, col)
            existing = best_by_sector.get(sector)
            if existing is None or rank < existing[0]:
                best_by_sector[sector] = (rank, item)

    return [
        item
        for _rank, item in sorted(
            best_by_sector.values(),
            key=lambda candidate: candidate[0],
        )
    ]


def optimize_ore_node_approaches(
    cost_map: TerrainCostMap,
    accepted_nodes: list[dict[str, Any]],
    *,
    refinement_passes: int,
    approach_distance_weight: float,
    max_extra_cells: float = 2.0,
) -> dict[str, Any]:
    if refinement_passes < 0:
        raise ValueError("refinement_passes must be non-negative")
    if approach_distance_weight < 0.0:
        raise ValueError("approach_distance_weight must be non-negative")
    if max_extra_cells < 0.0:
        raise ValueError("max_extra_cells must be non-negative")
    if len(accepted_nodes) < 2 or refinement_passes == 0:
        return {"enabled": False, "changes": 0, "passes": 0}

    path_cost_cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    def terrain_path_cost(start: tuple[int, int], end: tuple[int, int]) -> float:
        if start == end:
            return 0.0
        key = tuple(sorted((start, end)))
        if key not in path_cost_cache:
            try:
                path_cost_cache[key] = float(route_segment(cost_map, start, end)[0].total_cost)
            except ValueError:
                path_cost_cache[key] = float("inf")
        return path_cost_cache[key]

    changes = 0
    completed_passes = 0
    for _pass_index in range(refinement_passes):
        pass_changes = 0
        for index, node in enumerate(accepted_nodes):
            candidates = node.get("terrain_approach_candidates", [])
            if not candidates:
                continue
            nearest_snap = min(float(candidate["snap_cells"]) for candidate in candidates)
            eligible_candidates = [
                (candidate_index, candidate)
                for candidate_index, candidate in enumerate(candidates)
                if float(candidate["snap_cells"]) <= nearest_snap + max_extra_cells + 1.0e-9
            ]
            previous_point = tuple(accepted_nodes[index - 1]["terrain_approach_grid"])
            next_point = tuple(accepted_nodes[(index + 1) % len(accepted_nodes)]["terrain_approach_grid"])
            ranked: list[tuple[float, float, int, tuple[int, int]]] = []
            for candidate_index, candidate in eligible_candidates:
                point = tuple(int(value) for value in candidate["grid"])
                objective = (
                    terrain_path_cost(previous_point, point)
                    + terrain_path_cost(point, next_point)
                    + float(candidate["snap_cells"]) * approach_distance_weight
                )
                ranked.append((objective, float(candidate["local_score"]), candidate_index, point))
            objective, _local_score, selected_index, selected_point = min(ranked)
            old_point = tuple(node["terrain_approach_grid"])
            if selected_point != old_point:
                pass_changes += 1
                changes += 1
            node["terrain_approach_grid"] = list(selected_point)
            node["terrain_selected_approach_index"] = selected_index
            node["terrain_snap_cells"] = float(candidates[selected_index]["snap_cells"])
            node["terrain_min_snap_cells"] = round(nearest_snap, 4)
            node["terrain_approach_objective"] = round(objective, 4)
        completed_passes += 1
        if pass_changes == 0:
            break

    return {
        "enabled": True,
        "method": "cyclic_neighbor_terrain_cost",
        "passes": completed_passes,
        "changes": changes,
        "approach_distance_weight": float(approach_distance_weight),
        "max_extra_cells": float(max_extra_cells),
        "path_cost_cache_entries": len(path_cost_cache),
    }


def route_segment(
    cost_map: TerrainCostMap,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[TerrainPath, str]:
    rows, cols = cost_map.cost.shape
    for margin in (96, 192):
        bounds = (
            max(0, min(start[0], goal[0]) - margin),
            min(rows - 1, max(start[0], goal[0]) + margin),
            max(0, min(start[1], goal[1]) - margin),
            min(cols - 1, max(start[1], goal[1]) + margin),
        )
        path = astar(
            cost_map,
            start,
            goal,
            search_bounds=bounds,
            max_expansions=500_000,
        )
        if path is not None:
            return path, f"margin_{margin}"

    path = astar(cost_map, start, goal, max_expansions=1_500_000)
    if path is None:
        raise ValueError(f"No terrain path from {start} to {goal}")
    return path, "full_raster"


def validate_pairwise_costs(pairwise_costs: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pairwise_costs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pairwise_costs must be a square 2D matrix")
    if matrix.shape[0] < 2:
        raise ValueError("pairwise_costs must contain at least two nodes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pairwise_costs must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("pairwise_costs must be non-negative")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-9, rtol=0.0):
        raise ValueError("pairwise_costs diagonal must be zero")
    if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=1e-9):
        raise ValueError("pairwise_costs must be symmetric")
    return matrix


def cycle_cost(order: list[int], pairwise_costs: np.ndarray) -> float:
    matrix = validate_pairwise_costs(pairwise_costs)
    node_count = matrix.shape[0]
    if (
        len(order) != node_count
        or any(isinstance(index, bool) or not isinstance(index, int) for index in order)
        or sorted(order) != list(range(node_count))
    ):
        raise ValueError("order must be an exact integer permutation of matrix nodes")
    return float(
        sum(
            matrix[order[index], order[(index + 1) % node_count]]
            for index in range(node_count)
        )
    )


def optimize_cycle_order(
    pairwise_costs: np.ndarray,
    *,
    max_two_opt_iterations: int,
) -> list[int]:
    if max_two_opt_iterations < 0:
        raise ValueError("max_two_opt_iterations must be non-negative")
    matrix = validate_pairwise_costs(pairwise_costs)
    node_count = matrix.shape[0]

    order = [0]
    remaining = set(range(1, node_count))
    while remaining:
        current = order[-1]
        selected = min(remaining, key=lambda index: (float(matrix[current, index]), index))
        order.append(selected)
        remaining.remove(selected)

    current_cost = cycle_cost(order, matrix)
    for _ in range(max_two_opt_iterations):
        improved = False
        for start in range(1, node_count - 1):
            for end in range(start + 1, node_count):
                candidate = (
                    order[:start]
                    + list(reversed(order[start : end + 1]))
                    + order[end + 1 :]
                )
                candidate_cost = cycle_cost(candidate, matrix)
                if candidate_cost < current_cost - 1e-9:
                    order = candidate
                    current_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return order


def build_pairwise_terrain_costs(
    cost_map: TerrainCostMap,
    accepted_nodes: list[dict[str, Any]],
) -> np.ndarray:
    if len(accepted_nodes) < 2:
        raise ValueError("At least two accepted nodes are required")
    approaches = [
        (int(node["terrain_approach_grid"][0]), int(node["terrain_approach_grid"][1]))
        for node in accepted_nodes
    ]
    costs = np.zeros((len(approaches), len(approaches)), dtype=np.float64)
    for first in range(len(approaches)):
        for second in range(first + 1, len(approaches)):
            path, _search_mode = route_segment(
                cost_map,
                approaches[first],
                approaches[second],
            )
            value = float(path.total_cost)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Invalid terrain path cost for nodes {first} and {second}: {value}"
                )
            costs[first, second] = value
            costs[second, first] = value
    return costs


def plan_cycle(
    *,
    cost_map: TerrainCostMap,
    accepted_nodes: list[dict[str, Any]],
    max_waypoint_spacing: float,
    min_clearance: float,
) -> tuple[list[PlannedSegment], list[tuple[int, int]], list[int], list[int]]:
    if len(accepted_nodes) < 2:
        raise ValueError("A cyclic route requires at least two accepted nodes")

    approaches = [
        (int(node["terrain_approach_grid"][0]), int(node["terrain_approach_grid"][1]))
        for node in accepted_nodes
    ]
    segments: list[PlannedSegment] = []
    for index, (start, goal) in enumerate(
        zip(approaches, approaches[1:] + approaches[:1])
    ):
        raw_path, search_mode = route_segment(cost_map, start, goal)
        simplified = simplify_path(
            cost_map,
            raw_path.points,
            max_waypoint_spacing,
            min_clearance=min_clearance,
        )
        for edge_start, edge_end in zip(simplified, simplified[1:]):
            if not line_is_walkable(
                cost_map,
                edge_start,
                edge_end,
                min_clearance=min_clearance,
            ):
                raise ValueError(
                    f"Simplified segment {index} crosses blocked terrain: "
                    f"{edge_start} -> {edge_end}"
                )
        segments.append(
            PlannedSegment(
                index=index,
                start_node_index=index,
                end_node_index=(index + 1) % len(accepted_nodes),
                raw_path=raw_path,
                simplified_points=simplified,
                search_mode=search_mode,
            )
        )

    route_points: list[tuple[int, int]] = []
    point_segments: list[int] = []
    node_route_indexes: list[int] = []
    for segment in segments:
        if not route_points:
            node_route_indexes.append(0)
            route_points.extend(segment.simplified_points)
            point_segments.extend([segment.index] * len(segment.simplified_points))
        else:
            node_route_indexes.append(len(route_points) - 1)
            route_points.extend(segment.simplified_points[1:])
            point_segments.extend(
                [segment.index] * (len(segment.simplified_points) - 1)
            )

    if route_points[-1] != route_points[0]:
        raise ValueError("Closing terrain segment did not return to the first approach")
    route_points.pop()
    point_segments.pop()
    if len(route_points) < 2:
        raise ValueError("Terrain route collapsed below two navigation points")

    cyclic_edges = zip(route_points, route_points[1:] + route_points[:1])
    for edge_start, edge_end in cyclic_edges:
        if not line_is_walkable(
            cost_map,
            edge_start,
            edge_end,
            min_clearance=min_clearance,
        ):
            raise ValueError(
                f"Final cyclic route crosses blocked terrain: {edge_start} -> {edge_end}"
            )
    return segments, route_points, point_segments, node_route_indexes


def build_node_access_plans(
    *,
    cost_map: TerrainCostMap,
    accepted_nodes: list[dict[str, Any]],
    route_points: list[tuple[int, int]],
    node_route_indexes: list[int],
    raster: TerrainRaster,
    zone: ZoneTransform,
    max_waypoint_spacing: float,
    min_clearance: float,
    max_options: int,
    approach_distance_weight: float,
    max_extra_cells: float,
) -> dict[str, Any]:
    if max_options <= 0:
        raise ValueError("max_options must be positive")
    if len(accepted_nodes) != len(node_route_indexes):
        raise ValueError("Every accepted node must have a route index")
    if len(route_points) < 3:
        raise ValueError("Node access planning requires at least three route points")

    total_options = 0
    fallback_options = 0
    for node, route_index in zip(accepted_nodes, node_route_indexes):
        attachment_index = (int(route_index) - 1) % len(route_points)
        resume_index = (int(route_index) + 1) % len(route_points)
        attachment = route_points[attachment_index]
        resume = route_points[resume_index]
        candidates = list(node.get("terrain_approach_candidates", []))
        if not candidates:
            raise ValueError("Accepted ore node has no terrain approach candidates")

        nearest_snap = min(float(candidate["snap_cells"]) for candidate in candidates)
        selected_candidate_index = int(node.get("terrain_selected_approach_index", 0))
        planned_options: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for candidate_index, candidate in enumerate(candidates):
            snap_cells = float(candidate["snap_cells"])
            if snap_cells > nearest_snap + max_extra_cells + 1.0e-9:
                continue
            approach = tuple(int(value) for value in candidate["grid"])
            try:
                inbound_points, inbound_cost, inbound_mode = _plan_node_access_leg(
                    cost_map,
                    attachment,
                    approach,
                    max_waypoint_spacing=max_waypoint_spacing,
                    min_clearance=min_clearance,
                )
                resume_points, resume_cost, resume_mode = _plan_node_access_leg(
                    cost_map,
                    approach,
                    resume,
                    max_waypoint_spacing=max_waypoint_spacing,
                    min_clearance=min_clearance,
                )
            except ValueError:
                continue

            objective = (
                inbound_cost
                + resume_cost
                + snap_cells * max(0.0, approach_distance_weight)
            )
            option = {
                "candidate_index": candidate_index,
                "sector": int(candidate.get("sector", -1)),
                "bearing_degrees": float(candidate.get("bearing_degrees", 0.0)),
                "approach_grid": [approach[0], approach[1]],
                "approach_coord": _grid_point_to_waypoint(approach, raster, zone)["coord"],
                "snap_cells": snap_cells,
                "objective": round(objective, 4),
                "inbound": _node_access_leg_payload(
                    inbound_points,
                    inbound_cost,
                    inbound_mode,
                    raster,
                    zone,
                ),
                "return": {
                    "method": "reverse_inbound",
                    "terrain_cost": round(inbound_cost, 4),
                    "world_length": round(path_length(inbound_points), 4),
                    "waypoints": [
                        _grid_point_to_waypoint(point, raster, zone)
                        for point in reversed(inbound_points)
                    ],
                },
                "resume": _node_access_leg_payload(
                    resume_points,
                    resume_cost,
                    resume_mode,
                    raster,
                    zone,
                ),
            }
            primary_rank = 0.0 if candidate_index == selected_candidate_index else 1.0
            rank = (
                primary_rank,
                objective,
                float(candidate.get("local_score", 0.0)),
                float(candidate_index),
            )
            planned_options.append((rank, option))

        planned_options.sort(key=lambda item: item[0])
        selected_options = [option for _rank, option in planned_options[:max_options]]
        if not selected_options or selected_options[0]["candidate_index"] != selected_candidate_index:
            raise ValueError("Selected ore approach has no valid access plan")
        for rank, option in enumerate(selected_options):
            option["rank"] = rank
            option["primary"] = rank == 0

        node["terrain_access_plan"] = {
            "method": "route_attachment_angular_candidates",
            "attachment_route_index": attachment_index,
            "resume_route_index": resume_index,
            "primary_option_rank": 0,
            "options": selected_options,
        }
        total_options += len(selected_options)
        fallback_options += max(0, len(selected_options) - 1)

    return {
        "enabled": True,
        "method": "route_attachment_angular_candidates",
        "node_count": len(accepted_nodes),
        "option_count": total_options,
        "fallback_option_count": fallback_options,
        "max_options_per_node": int(max_options),
    }


def _plan_node_access_leg(
    cost_map: TerrainCostMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_waypoint_spacing: float,
    min_clearance: float,
) -> tuple[list[tuple[int, int]], float, str]:
    if start == goal:
        return [start], 0.0, "same_point"
    raw_path, search_mode = route_segment(cost_map, start, goal)
    simplified = simplify_path(
        cost_map,
        raw_path.points,
        max_waypoint_spacing,
        min_clearance=min_clearance,
    )
    for edge_start, edge_end in zip(simplified, simplified[1:]):
        if not line_is_walkable(
            cost_map,
            edge_start,
            edge_end,
            min_clearance=min_clearance,
        ):
            raise ValueError(
                f"Node access leg crosses blocked terrain: {edge_start} -> {edge_end}"
            )
    return simplified, float(raw_path.total_cost), search_mode


def _node_access_leg_payload(
    points: list[tuple[int, int]],
    total_cost: float,
    search_mode: str,
    raster: TerrainRaster,
    zone: ZoneTransform,
) -> dict[str, Any]:
    return {
        "search_mode": search_mode,
        "terrain_cost": round(total_cost, 4),
        "world_length": round(path_length(points), 4),
        "waypoints": [_grid_point_to_waypoint(point, raster, zone) for point in points],
    }


def _grid_point_to_waypoint(
    point: tuple[int, int],
    raster: TerrainRaster,
    zone: ZoneTransform,
) -> dict[str, Any]:
    row, col = point
    ui_x, ui_y = raster.grid_to_ui(row, col, zone)
    return {
        "grid": [row, col],
        "coord": xy_to_coord(ui_x, ui_y),
        "x": round(float(ui_x), 6),
        "y": round(float(ui_y), 6),
    }


def build_route_loop(
    *,
    points: list[tuple[int, int]],
    point_segments: list[int],
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
) -> list[dict[str, Any]]:
    route_loop: list[dict[str, Any]] = []
    for index, ((row, col), segment_index) in enumerate(zip(points, point_segments)):
        ui_x, ui_y = raster.grid_to_ui(row, col, zone)
        height = raster.sample_height(row, col)
        slope = float(cost_map.slope_degrees[row, col])
        clearance = float(cost_map.clearance[row, col])
        if height is None or not all(
            math.isfinite(value) for value in (height, slope, clearance)
        ):
            raise ValueError(f"Non-finite route sample at grid point {(row, col)}")
        route_loop.append(
            {
                "index": index,
                "coord": xy_to_coord(ui_x, ui_y),
                "x": round(float(ui_x), 6),
                "y": round(float(ui_y), 6),
                "source": "terrain_astar",
                "height": round(float(height), 4),
                "slope_degrees": round(slope, 4),
                "clearance_cells": round(clearance, 4),
                "source_segment_index": int(segment_index),
            }
        )
    return route_loop


def build_anchor_entry_route(
    *,
    anchor_point: tuple[int, int],
    route_points: list[tuple[int, int]],
    raster: TerrainRaster,
    zone: ZoneTransform,
    cost_map: TerrainCostMap,
    max_waypoint_spacing: float,
    min_clearance: float,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    if not route_points:
        raise ValueError("Cannot build an anchor entry without a cyclic route")
    target_route_index = min(
        range(len(route_points)),
        key=lambda index: (math.dist(anchor_point, route_points[index]), index),
    )
    target = route_points[target_route_index]
    raw_path, search_mode = route_segment(cost_map, anchor_point, target)
    simplified = simplify_path(
        cost_map,
        raw_path.points,
        max_waypoint_spacing,
        min_clearance=min_clearance,
    )
    for edge_start, edge_end in zip(simplified, simplified[1:]):
        if not line_is_walkable(
            cost_map,
            edge_start,
            edge_end,
            min_clearance=min_clearance,
        ):
            raise ValueError(
                f"Anchor entry crosses blocked terrain: {edge_start} -> {edge_end}"
            )

    waypoints: list[dict[str, Any]] = []
    for index, (row, col) in enumerate(simplified):
        ui_x, ui_y = raster.grid_to_ui(row, col, zone)
        waypoints.append(
            {
                "index": index,
                "coord": xy_to_coord(ui_x, ui_y),
                "x": round(float(ui_x), 6),
                "y": round(float(ui_y), 6),
                "source": "terrain_astar_entry",
            }
        )
    return (
        {
            "status": "candidate_anchor_entry",
            "target_route_index": int(target_route_index),
            "search_mode": search_mode,
            "raw_point_count": len(raw_path.points),
            "waypoint_count": len(waypoints),
            "path_world_length": round(path_length(simplified), 4),
            "waypoints": waypoints,
        },
        simplified,
    )


def build_metrics(
    *,
    segments: list[PlannedSegment],
    route_points: list[tuple[int, int]],
    cost_map: TerrainCostMap,
    primary_component_area: int,
    accepted_count: int,
    terrain_rejected_count: int,
    source_paths: dict[str, str],
    route_bounds: dict[str, float] | None = None,
    route_bound_margin: float | None = None,
    ordering: dict[str, Any] | None = None,
    component_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    straight_length = 0.0
    final_path_length = 0.0
    segment_metrics: list[dict[str, Any]] = []
    for segment in segments:
        raw_length = path_length(segment.raw_path.points)
        simplified_length = path_length(segment.simplified_points)
        direct_length = (
            math.dist(
                segment.simplified_points[0],
                segment.simplified_points[-1],
            )
            * GRID_SPACING
        )
        straight_length += direct_length
        final_path_length += simplified_length
        segment_metrics.append(
            {
                "index": segment.index,
                "start_node_index": segment.start_node_index,
                "end_node_index": segment.end_node_index,
                "search_mode": segment.search_mode,
                "raw_point_count": len(segment.raw_path.points),
                "simplified_point_count": len(segment.simplified_points),
                "astar_total_cost": round(float(segment.raw_path.total_cost), 4),
                "straight_world_length": round(direct_length, 4),
                "raw_world_length": round(raw_length, 4),
                "simplified_world_length": round(simplified_length, 4),
            }
        )

    route_cells: list[tuple[int, int]] = []
    for start, end in zip(route_points, route_points[1:] + route_points[:1]):
        cells = supercover_line(start, end)
        route_cells.extend(cells if not route_cells else cells[1:])
    rows = np.asarray([point[0] for point in route_cells], dtype=np.intp)
    cols = np.asarray([point[1] for point in route_cells], dtype=np.intp)
    route_slopes = cost_map.slope_degrees[rows, cols]
    route_clearance = cost_map.clearance[rows, cols]
    return {
        "cost_config": cost_map.config.to_dict(),
        "primary_component_area_cells": int(primary_component_area),
        "accepted_node_count": int(accepted_count),
        "terrain_rejected_node_count": int(terrain_rejected_count),
        "segment_count": len(segments),
        "raw_point_count": int(
            sum(len(segment.raw_path.points) for segment in segments)
        ),
        "route_waypoint_count": len(route_points),
        "route_crossed_cell_count": len(route_cells),
        "straight_world_length": round(straight_length, 4),
        "path_world_length": round(final_path_length, 4),
        "detour_ratio": round(
            final_path_length / straight_length if straight_length > 0.0 else 1.0,
            6,
        ),
        "max_route_slope_degrees": round(float(np.max(route_slopes)), 4),
        "mean_route_slope_degrees": round(float(np.mean(route_slopes)), 4),
        "min_route_clearance_cells": round(float(np.min(route_clearance)), 4),
        "source_paths": source_paths,
        "route_bounds": route_bounds,
        "route_bound_margin": route_bound_margin,
        "ordering": ordering
        or {
            "enabled": False,
            "method": "preserve_source_order",
        },
        "component_selection": component_selection
        or {
            "method": "largest_walkable_component",
        },
        "segments": segment_metrics,
    }


def build_blocked_metrics(
    *,
    cost_map: TerrainCostMap,
    component_area: int,
    accepted_count: int,
    terrain_rejected_count: int,
    source_paths: dict[str, str],
    reason: str,
    route_bounds: dict[str, float] | None,
    route_bound_margin: float | None,
    component_selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "cost_config": cost_map.config.to_dict(),
        "primary_component_area_cells": int(component_area),
        "accepted_node_count": int(accepted_count),
        "terrain_rejected_node_count": int(terrain_rejected_count),
        "segment_count": 0,
        "raw_point_count": 0,
        "route_waypoint_count": 0,
        "route_crossed_cell_count": 0,
        "straight_world_length": 0.0,
        "path_world_length": 0.0,
        "detour_ratio": 1.0,
        "max_route_slope_degrees": 0.0,
        "mean_route_slope_degrees": 0.0,
        "min_route_clearance_cells": 0.0,
        "source_paths": source_paths,
        "route_bounds": route_bounds,
        "route_bound_margin": route_bound_margin,
        "ordering": {"enabled": False, "method": "blocked"},
        "component_selection": component_selection,
        "segments": [],
    }


def compose_output_route(
    *,
    source_route: dict[str, Any],
    accepted_nodes: list[dict[str, Any]],
    terrain_rejected: list[dict[str, Any]],
    node_route_indexes: list[int],
    route_loop: list[dict[str, Any]],
    metrics: dict[str, Any],
    entry_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for node, route_index in zip(accepted_nodes, node_route_indexes):
        node["route_index"] = int(route_index)

    output = dict(source_route)
    output["schema_version"] = 2
    output["mode"] = "cyclic"
    output["route_nodes"] = accepted_nodes
    output["route_loop"] = route_loop
    if entry_route is not None:
        output["entry_route"] = entry_route
    output["rejected_nodes"] = [
        dict(node) for node in source_route.get("rejected_nodes", [])
    ] + terrain_rejected

    generation = dict(source_route.get("generation", {}))
    generation.update(
        {
            "route_waypoints": len(route_loop),
            "accepted_nodes": len(accepted_nodes),
            "rejected_nodes": len(output["rejected_nodes"]),
            "topographic": metrics,
        }
    )
    output["generation"] = generation
    return output


def compose_blocked_output_route(
    *,
    source_route: dict[str, Any],
    accepted_nodes: list[dict[str, Any]],
    terrain_rejected: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    output = dict(source_route)
    output["schema_version"] = 3
    output["mode"] = "blocked"
    output["live_status"] = "rejected_no_anchor_reachable_cycle"
    output["route_nodes"] = accepted_nodes
    output["route_loop"] = []
    output["rejected_nodes"] = [
        dict(node) for node in source_route.get("rejected_nodes", [])
    ] + terrain_rejected
    generation = dict(source_route.get("generation", {}))
    generation.update(
        {
            "route_waypoints": 0,
            "accepted_nodes": len(accepted_nodes),
            "rejected_nodes": len(output["rejected_nodes"]),
            "topographic": metrics,
        }
    )
    output["generation"] = generation
    return output


def derive_overlay_title(source_route: dict[str, Any]) -> str:
    zone = source_route.get("zone")
    zone_name = zone.get("name") if isinstance(zone, dict) else None
    route_name = source_route.get("name")
    selected = next(
        (
            value.strip()
            for value in (zone_name, route_name, "Zone")
            if isinstance(value, str) and value.strip()
        ),
        "Zone",
    )
    suffix = " topographic route"
    return selected if selected.lower().endswith(suffix) else f"{selected}{suffix}"


def render_overlay(
    *,
    raster: TerrainRaster,
    cost_map: TerrainCostMap,
    route_points: list[tuple[int, int]],
    accepted_nodes: list[dict[str, Any]],
    rejected_nodes: list[dict[str, Any]],
    zone: ZoneTransform,
    metrics: dict[str, Any],
    title: str,
    output_path: str | Path,
    entry_points: list[tuple[int, int]] | None = None,
) -> None:
    heights = np.asarray(raster.heights, dtype=np.float32)
    valid = np.isfinite(heights)
    finite_heights = heights[valid]
    if finite_heights.size == 0:
        raise ValueError("Cannot render terrain without finite heights")
    low, high = np.percentile(finite_heights, [2.0, 98.0])
    if math.isclose(float(low), float(high)):
        high = low + 1.0

    normalized = np.zeros(heights.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (heights[valid] - low) / (high - low),
        0.0,
        1.0,
    )
    fill_height = float(np.median(finite_heights))
    filled = np.where(valid, heights, fill_height)
    gradient_row, gradient_col = np.gradient(filled, GRID_SPACING)
    light = 0.55 - 0.35 * gradient_col - 0.25 * gradient_row
    light = cv2.GaussianBlur(light.astype(np.float32), (0, 0), 1.2)
    light = cv2.normalize(light, None, 0.0, 1.0, cv2.NORM_MINMAX)
    gray = np.clip((0.65 * normalized + 0.35 * light) * 255.0, 0, 255)
    image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image[~valid] = (8, 8, 8)

    def blend(mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if not np.any(mask):
            return
        source = image[mask].astype(np.float32)
        target = np.asarray(color, dtype=np.float32)
        image[mask] = np.clip(source * (1.0 - alpha) + target * alpha, 0, 255)

    finite_cost = cost_map.cost[np.isfinite(cost_map.cost)]
    high_cost_threshold = (
        max(1.5, float(np.percentile(finite_cost, 85.0)))
        if finite_cost.size
        else math.inf
    )
    high_cost = np.isfinite(cost_map.cost) & (cost_map.cost >= high_cost_threshold)
    blend(high_cost, (0, 145, 255), 0.45)
    blend(cost_map.blocked, (20, 20, 225), 0.58)
    image[~valid] = (8, 8, 8)

    cyclic_pairs = zip(route_points, route_points[1:] + route_points[:1])
    for start, end in cyclic_pairs:
        start_xy = int(start[1]), int(start[0])
        end_xy = int(end[1]), int(end[0])
        cv2.line(image, start_xy, end_xy, (255, 230, 0), 5, cv2.LINE_AA)
        cv2.line(image, start_xy, end_xy, (255, 255, 255), 1, cv2.LINE_AA)

    for start, end in zip(entry_points or [], (entry_points or [])[1:]):
        start_xy = int(start[1]), int(start[0])
        end_xy = int(end[1]), int(end[0])
        cv2.line(image, start_xy, end_xy, (70, 255, 70), 6, cv2.LINE_AA)
        cv2.line(image, start_xy, end_xy, (255, 255, 255), 1, cv2.LINE_AA)

    for node_index, node in enumerate(accepted_nodes):
        raw_row, raw_col = node["terrain_raw_grid"]
        row, col = node["terrain_approach_grid"]
        cv2.line(
            image,
            (int(raw_col), int(raw_row)),
            (int(col), int(row)),
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(image, (int(col), int(row)), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (int(col), int(row)), 6, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(node_index),
            (int(col) + 7, int(row) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for node in rejected_nodes:
        if "terrain_raw_grid" in node:
            row, col = node["terrain_raw_grid"]
        elif "x" in node and "y" in node:
            row_float, col_float = raster.ui_to_grid(
                float(node["x"]),
                float(node["y"]),
                zone,
            )
            row, col = int(round(row_float)), int(round(col_float))
        else:
            continue
        if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
            cv2.drawMarker(
                image,
                (int(col), int(row)),
                (255, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                9,
                2,
                cv2.LINE_AA,
            )

    panel_width = min(390, image.shape[1])
    panel_height = min(166, image.shape[0])
    image[0:panel_height, 0:panel_width] = (22, 22, 22)
    legend = (
        ("Blocked / outside primary", (20, 20, 225)),
        ("High terrain cost", (0, 145, 255)),
        ("Accepted ore approach", (0, 255, 255)),
        ("Rejected ore node", (255, 0, 255)),
        ("Final terrain route", (255, 230, 0)),
        ("Anchor entry route", (70, 255, 70)),
    )
    cv2.putText(
        image,
        title,
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, (label, color) in enumerate(legend):
        y = 40 + index * 17
        cv2.rectangle(image, (12, y - 7), (24, y + 5), color, -1)
        cv2.putText(
            image,
            label,
            (31, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    summary = (
        f"nodes {metrics['accepted_node_count']} accepted / "
        f"{metrics['terrain_rejected_node_count']} terrain-rejected; "
        f"detour {metrics['detour_ratio']:.3f}x"
    )
    cv2.putText(
        image,
        summary,
        (12, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Failed to write route overlay to {destination}")


def main() -> None:
    args = parse_args()
    if args.max_snap_radius < 0.0:
        raise ValueError("--max-snap-radius must be non-negative")
    if args.approach_candidate_sectors <= 0:
        raise ValueError("--approach-candidate-sectors must be positive")
    if args.approach_refinement_passes < 0:
        raise ValueError("--approach-refinement-passes must be non-negative")
    if args.approach_distance_weight < 0.0:
        raise ValueError("--approach-distance-weight must be non-negative")
    if args.approach_max_extra_cells < 0.0:
        raise ValueError("--approach-max-extra-cells must be non-negative")
    if args.approach_access_options <= 0:
        raise ValueError("--approach-access-options must be positive")
    if (args.anchor_x is None) != (args.anchor_y is None):
        raise ValueError("--anchor-x and --anchor-y must be supplied together")
    if args.anchor_snap_radius < 0.0:
        raise ValueError("--anchor-snap-radius must be non-negative")
    if args.min_clearance < 0.0:
        raise ValueError("--min-clearance must be non-negative")
    if args.max_waypoint_spacing <= 0.0:
        raise ValueError("--max-waypoint-spacing must be positive")
    if args.route_bound_margin < 0.0:
        raise ValueError("--route-bound-margin must be non-negative")
    if args.max_two_opt_iterations < 0:
        raise ValueError("--max-two-opt-iterations must be non-negative")

    manifest = load_json(args.manifest)
    source_route = load_json(args.input_route)
    zone = zone_from_manifest(manifest)
    adt_paths = sorted(Path(args.adt_dir).glob("*.adt"))
    if not adt_paths:
        raise ValueError(f"No ADT files found in {args.adt_dir}")

    raster = build_terrain_raster(adt_paths)
    terrain_config = TerrainCostConfig(
        soft_slope_degrees=args.soft_slope_degrees,
        hard_slope_degrees=args.hard_slope_degrees,
        slope_weight=args.slope_weight,
        roughness_weight=args.roughness_weight,
        clearance_cells=args.clearance_cells,
        clearance_weight=args.clearance_weight,
    )
    base_cost_map = build_terrain_cost_map(raster, terrain_config)
    route_bounds = source_route.get("generation", {}).get("bounds")
    bounded_grid: tuple[int, int, int, int] | None = None
    if route_bounds is not None:
        base_cost_map, bounded_grid = restrict_to_ui_bounds(
            base_cost_map,
            raster=raster,
            zone=zone,
            bounds=route_bounds,
            margin=args.route_bound_margin,
        )
    component_selection: dict[str, Any]
    if args.anchor_x is not None and args.anchor_y is not None:
        anchor_grid_float = raster.ui_to_grid(args.anchor_x, args.anchor_y, zone)
        anchor_grid = int(round(anchor_grid_float[0])), int(round(anchor_grid_float[1]))
        cost_map, primary_mask, primary_area, snapped_anchor = (
            restrict_to_anchor_component(
                base_cost_map,
                anchor_grid,
                max_snap_radius=args.anchor_snap_radius,
                min_clearance=args.min_clearance,
            )
        )
        snapped_anchor_ui = raster.grid_to_ui(*snapped_anchor, zone)
        component_selection = {
            "method": "anchor_connected_component",
            "anchor_ui": [float(args.anchor_x), float(args.anchor_y)],
            "anchor_grid": list(anchor_grid),
            "snapped_anchor_grid": list(snapped_anchor),
            "snapped_anchor_ui": [
                round(float(snapped_anchor_ui[0]), 6),
                round(float(snapped_anchor_ui[1]), 6),
            ],
            "anchor_snap_cells": round(math.dist(anchor_grid, snapped_anchor), 4),
            "anchor_snap_limit_cells": float(args.anchor_snap_radius),
        }
    else:
        cost_map, primary_mask, primary_area = restrict_to_primary_component(
            base_cost_map,
            min_clearance=args.min_clearance,
        )
        component_selection = {"method": "largest_walkable_component"}
    accepted_nodes, terrain_rejected = snap_route_nodes(
        nodes=[dict(node) for node in source_route.get("route_nodes", [])],
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        max_snap_radius=args.max_snap_radius,
        min_clearance=args.min_clearance,
        approach_candidate_sectors=args.approach_candidate_sectors,
    )
    source_paths = {
        "input_route": str(Path(args.input_route)),
        "manifest": str(Path(args.manifest)),
        "adt_dir": str(Path(args.adt_dir)),
    }
    normalized_bounds = (
        {key: float(value) for key, value in route_bounds.items()}
        if route_bounds is not None
        else None
    )
    normalized_margin = float(args.route_bound_margin) if bounded_grid is not None else None
    if len(accepted_nodes) < 2:
        reason = "anchor_component_contains_fewer_than_two_reachable_ore_nodes"
        metrics = build_blocked_metrics(
            cost_map=cost_map,
            component_area=primary_area,
            accepted_count=len(accepted_nodes),
            terrain_rejected_count=len(terrain_rejected),
            source_paths=source_paths,
            reason=reason,
            route_bounds=normalized_bounds,
            route_bound_margin=normalized_margin,
            component_selection=component_selection,
        )
        output_route = compose_blocked_output_route(
            source_route=source_route,
            accepted_nodes=accepted_nodes,
            terrain_rejected=terrain_rejected,
            metrics=metrics,
        )
        write_json(output_route, args.output_route)
        write_json(metrics, args.output_metrics)
        render_overlay(
            raster=raster,
            cost_map=cost_map,
            route_points=[],
            accepted_nodes=accepted_nodes,
            rejected_nodes=[
                *[dict(node) for node in source_route.get("rejected_nodes", [])],
                *terrain_rejected,
            ],
            zone=zone,
            metrics=metrics,
            title=derive_overlay_title(source_route),
            output_path=args.output_overlay,
        )
        print(
            f"Blocked topographic route: {args.output_route}\n"
            f"reason={reason} accepted={len(accepted_nodes)} "
            f"terrain_rejected={len(terrain_rejected)} component_cells={primary_area}"
        )
        return

    ordering: dict[str, Any] = {
        "enabled": False,
        "method": "preserve_source_order",
        "source_route_orders": [
            int(node["terrain_source_route_order"]) for node in accepted_nodes
        ],
    }
    if args.optimize_node_order:
        pairwise_costs = build_pairwise_terrain_costs(cost_map, accepted_nodes)
        source_index_order = list(range(len(accepted_nodes)))
        optimized_index_order = optimize_cycle_order(
            pairwise_costs,
            max_two_opt_iterations=args.max_two_opt_iterations,
        )
        source_cost = cycle_cost(source_index_order, pairwise_costs)
        optimized_cost = cycle_cost(optimized_index_order, pairwise_costs)
        original_nodes = accepted_nodes
        accepted_nodes = []
        for route_order, source_index in enumerate(optimized_index_order):
            node = dict(original_nodes[source_index])
            node["route_order"] = route_order
            accepted_nodes.append(node)
        ordering = {
            "enabled": True,
            "method": "terrain_pairwise_nearest_neighbor_2opt",
            "max_two_opt_iterations": int(args.max_two_opt_iterations),
            "source_cycle_cost": round(source_cost, 4),
            "optimized_cycle_cost": round(optimized_cost, 4),
            "improvement_ratio": round(
                source_cost / optimized_cost if optimized_cost > 0.0 else 1.0,
                6,
            ),
            "source_route_orders": [
                int(node["terrain_source_route_order"]) for node in original_nodes
            ],
            "optimized_source_route_orders": [
                int(node["terrain_source_route_order"]) for node in accepted_nodes
            ],
        }
    ordering["approach_selection"] = optimize_ore_node_approaches(
        cost_map,
        accepted_nodes,
        refinement_passes=args.approach_refinement_passes,
        approach_distance_weight=args.approach_distance_weight,
        max_extra_cells=args.approach_max_extra_cells,
    )
    segments, route_points, point_segments, node_route_indexes = plan_cycle(
        cost_map=cost_map,
        accepted_nodes=accepted_nodes,
        max_waypoint_spacing=args.max_waypoint_spacing,
        min_clearance=args.min_clearance,
    )
    ordering["node_access_plans"] = build_node_access_plans(
        cost_map=cost_map,
        accepted_nodes=accepted_nodes,
        route_points=route_points,
        node_route_indexes=node_route_indexes,
        raster=raster,
        zone=zone,
        max_waypoint_spacing=args.max_waypoint_spacing,
        min_clearance=args.min_clearance,
        max_options=args.approach_access_options,
        approach_distance_weight=args.approach_distance_weight,
        max_extra_cells=args.approach_max_extra_cells,
    )
    route_loop = build_route_loop(
        points=route_points,
        point_segments=point_segments,
        raster=raster,
        zone=zone,
        cost_map=cost_map,
    )
    entry_route: dict[str, Any] | None = None
    entry_points: list[tuple[int, int]] = []
    if args.anchor_x is not None and args.anchor_y is not None:
        entry_route, entry_points = build_anchor_entry_route(
            anchor_point=snapped_anchor,
            route_points=route_points,
            raster=raster,
            zone=zone,
            cost_map=cost_map,
            max_waypoint_spacing=args.max_waypoint_spacing,
            min_clearance=args.min_clearance,
        )
        component_selection["entry_target_route_index"] = int(
            entry_route["target_route_index"]
        )
        component_selection["entry_waypoint_count"] = int(entry_route["waypoint_count"])
    metrics = build_metrics(
        segments=segments,
        route_points=route_points,
        cost_map=cost_map,
        primary_component_area=primary_area,
        accepted_count=len(accepted_nodes),
        terrain_rejected_count=len(terrain_rejected),
        source_paths=source_paths,
        route_bounds=normalized_bounds,
        route_bound_margin=normalized_margin,
        ordering=ordering,
        component_selection=component_selection,
    )
    output_route = compose_output_route(
        source_route=source_route,
        accepted_nodes=accepted_nodes,
        terrain_rejected=terrain_rejected,
        node_route_indexes=node_route_indexes,
        route_loop=route_loop,
        metrics=metrics,
        entry_route=entry_route,
    )
    overlay_title = derive_overlay_title(source_route)

    write_json(output_route, args.output_route)
    write_json(metrics, args.output_metrics)
    render_overlay(
        raster=raster,
        cost_map=cost_map,
        route_points=route_points,
        accepted_nodes=accepted_nodes,
        rejected_nodes=[
            *[dict(node) for node in source_route.get("rejected_nodes", [])],
            *terrain_rejected,
        ],
        zone=zone,
        metrics=metrics,
        title=overlay_title,
        output_path=args.output_overlay,
        entry_points=entry_points,
    )
    print(
        f"Topographic route: {args.output_route}\n"
        f"Metrics: {args.output_metrics}\n"
        f"Overlay: {args.output_overlay}\n"
        f"accepted={len(accepted_nodes)} terrain_rejected={len(terrain_rejected)} "
        f"waypoints={len(route_loop)} detour={metrics['detour_ratio']:.3f}x "
        f"primary_cells={int(np.count_nonzero(primary_mask))}"
    )


if __name__ == "__main__":
    main()
