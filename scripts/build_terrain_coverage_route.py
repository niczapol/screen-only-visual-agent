from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.build_topographic_route import (
        find_ore_approach_candidates,
        restrict_to_anchor_component,
        restrict_to_ui_bounds,
        route_segment,
        zone_from_manifest,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from build_topographic_route import (
        find_ore_approach_candidates,
        restrict_to_anchor_component,
        restrict_to_ui_bounds,
        route_segment,
        zone_from_manifest,
    )
from vision_bot.coords import xy_to_coord
from vision_bot.terrain_routing import (
    GRID_SPACING,
    TerrainCostConfig,
    TerrainCostMap,
    TerrainRaster,
    build_terrain_cost_map,
    build_terrain_raster,
    line_is_walkable,
    nearest_walkable,
    path_length,
    simplify_path,
)


GridPoint = tuple[int, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a terrain-aware minimap-coverage cycle from ore records and a "
            "human reference trajectory"
        )
    )
    parser.add_argument(
        "--nodes",
        default="external_data/routes/tanaris_spawn_nodes.json",
    )
    parser.add_argument(
        "--trajectory",
        default=(
            "data/human_demo_tanaris_20260802_114252/analysis/trajectory.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="data/extracted_client_data/tanaris_corrected/manifest.json",
    )
    parser.add_argument(
        "--adt-dir",
        default="data/extracted_client_data/tanaris_corrected/world/maps/Kalimdor",
    )
    parser.add_argument(
        "--output-route",
        default="data/routes/generated/tanaris_terrain_coverage_cycle_v4.json",
    )
    parser.add_argument(
        "--output-metrics",
        default="data/routes/generated/tanaris_terrain_coverage_cycle_v4_metrics.json",
    )
    parser.add_argument(
        "--output-overlay",
        default="data/routes/generated/tanaris_terrain_coverage_cycle_v4.png",
    )
    parser.add_argument(
        "--live-hazards",
        default="data/routes/live_hazards/tanaris.json",
        help="Optional validated live hazard circles or polygons to block before pathfinding",
    )
    parser.add_argument("--reveal-radius-yards", type=float, default=50.0)
    parser.add_argument("--human-anchor-spacing-yards", type=float, default=30.0)
    parser.add_argument("--human-snap-radius-cells", type=float, default=8.0)
    parser.add_argument("--node-candidate-sectors", type=int, default=8)
    parser.add_argument("--route-bound-margin", type=float, default=2.0)
    parser.add_argument("--anchor-snap-radius-cells", type=float, default=30.0)
    parser.add_argument("--min-clearance-cells", type=float, default=1.0)
    parser.add_argument("--max-waypoint-spacing-cells", type=float, default=16.0)
    parser.add_argument("--node-access-radius-yards", type=float, default=18.0)
    parser.add_argument("--node-access-max-options", type=int, default=3)
    parser.add_argument("--node-access-max-extra-cells", type=float, default=5.0)
    parser.add_argument("--node-access-distance-weight", type=float, default=1.0)
    parser.add_argument("--soft-slope-degrees", type=float, default=15.0)
    parser.add_argument("--hard-slope-degrees", type=float, default=35.0)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def build_live_hazard_mask(
    shape: tuple[int, ...],
    *,
    raster: TerrainRaster,
    zone: Any,
    hazards: Sequence[dict[str, Any]],
) -> np.ndarray:
    combined_mask = np.zeros(shape[:2], dtype=np.uint8)
    rows, cols = np.indices(combined_mask.shape)
    for hazard in hazards:
        hazard_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
        points = hazard.get("points")
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
            grid_points: list[tuple[int, int]] = []
            for point in points:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    ui_x, ui_y = float(point["x"]), float(point["y"])
                elif isinstance(point, Sequence) and len(point) >= 2:
                    ui_x, ui_y = float(point[0]), float(point[1])
                else:
                    continue
                row, col = raster.ui_to_grid(ui_x, ui_y, zone)
                grid_points.append((int(round(col)), int(round(row))))
            if len(grid_points) >= 3:
                cv2.fillPoly(
                    hazard_mask,
                    [np.asarray(grid_points, dtype=np.int32)],
                    1,
                )
        elif all(key in hazard for key in ("x", "y", "radius_yards")):
            center_row, center_col = raster.ui_to_grid(
                float(hazard["x"]),
                float(hazard["y"]),
                zone,
            )
            radius_cells = max(0.0, float(hazard["radius_yards"]) / GRID_SPACING)
            hazard_mask[
                np.square(rows - center_row) + np.square(cols - center_col)
                <= radius_cells * radius_cells
            ] = 1
        margin_cells = int(
            round(max(0.0, float(hazard.get("margin_yards", 0.0))) / GRID_SPACING)
        )
        if margin_cells > 0 and np.any(hazard_mask):
            kernel_size = margin_cells * 2 + 1
            hazard_mask = cv2.dilate(
                hazard_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            )
        combined_mask |= hazard_mask
    return combined_mask.astype(bool)


def node_indexes_in_mask(
    node_points: Sequence[tuple[float, float]],
    mask: np.ndarray,
) -> set[int]:
    """Return nodes whose spawn coordinate is inside a validated exclusion."""
    height, width = mask.shape[:2]
    excluded: set[int] = set()
    for index, (row_value, col_value) in enumerate(node_points):
        row = int(round(row_value))
        col = int(round(col_value))
        if 0 <= row < height and 0 <= col < width and bool(mask[row, col]):
            excluded.add(index)
    return excluded


def apply_live_hazard_exclusions(
    cost_map: TerrainCostMap,
    *,
    raster: TerrainRaster,
    zone: Any,
    hazards: Sequence[dict[str, Any]],
) -> tuple[TerrainCostMap, int]:
    blocked = cost_map.blocked.copy()
    blocked |= build_live_hazard_mask(
        blocked.shape,
        raster=raster,
        zone=zone,
        hazards=hazards,
    )

    added = int(np.count_nonzero(blocked & ~cost_map.blocked))
    if added == 0:
        return cost_map, 0

    clearance = cv2.distanceTransform(
        (~blocked).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    old_clearance = cost_map.clearance
    config = cost_map.config
    if config.clearance_cells > 0.0:
        old_fraction = np.clip(
            (config.clearance_cells - old_clearance) / config.clearance_cells,
            0.0,
            1.0,
        )
        new_fraction = np.clip(
            (config.clearance_cells - clearance) / config.clearance_cells,
            0.0,
            1.0,
        )
        cost = (
            cost_map.cost
            - config.clearance_weight * np.square(old_fraction)
            + config.clearance_weight * np.square(new_fraction)
        ).astype(np.float32)
    else:
        cost = cost_map.cost.copy()
    cost[blocked] = np.inf
    return (
        TerrainCostMap(
            cost=cost,
            slope_degrees=cost_map.slope_degrees.copy(),
            roughness=cost_map.roughness.copy(),
            clearance=clearance,
            blocked=blocked,
            config=config,
        ),
        added,
    )


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_row = end[0] - start[0]
    delta_col = end[1] - start[1]
    denominator = delta_row * delta_row + delta_col * delta_col
    if denominator <= 1.0e-12:
        return math.dist(point, start)
    projection = (
        (point[0] - start[0]) * delta_row
        + (point[1] - start[1]) * delta_col
    ) / denominator
    projection = min(1.0, max(0.0, projection))
    closest = (
        start[0] + projection * delta_row,
        start[1] + projection * delta_col,
    )
    return math.dist(point, closest)


def cyclic_polyline_distance(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return math.dist(point, polyline[0])
    return min(
        point_to_segment_distance(point, start, end)
        for start, end in zip(polyline, polyline[1:] + polyline[:1])
    )


def sparsify_human_trajectory(
    trajectory: Iterable[dict[str, Any]],
    *,
    raster: TerrainRaster,
    zone: Any,
    cost_map: TerrainCostMap,
    spacing_yards: float,
    snap_radius_cells: float,
    min_clearance_cells: float,
) -> tuple[list[dict[str, Any]], list[GridPoint]]:
    if spacing_yards <= 0.0:
        raise ValueError("spacing_yards must be positive")

    valid_samples: list[dict[str, Any]] = []
    full_points: list[GridPoint] = []
    last_sample: dict[str, Any] | None = None
    for sample in trajectory:
        if sample.get("x") is None or sample.get("y") is None:
            continue
        raw = tuple(
            int(round(value))
            for value in raster.ui_to_grid(float(sample["x"]), float(sample["y"]), zone)
        )
        snapped = nearest_walkable(
            cost_map,
            raw,
            snap_radius_cells,
            min_clearance=min_clearance_cells,
        )
        if snapped is None:
            continue
        last_sample = sample
        if not full_points or snapped != full_points[-1]:
            full_points.append(snapped)
        if valid_samples and (
            math.dist(tuple(valid_samples[-1]["grid"]), snapped) * GRID_SPACING
            < spacing_yards
        ):
            continue
        valid_samples.append(
            {
                "grid": [snapped[0], snapped[1]],
                "capture_offset": float(sample.get("capture_offset", 0.0)),
                "sample_index": int(sample.get("sample_index", -1)),
                "source": "human_reference",
            }
        )

    if not valid_samples:
        raise ValueError("Human trajectory contains no walkable coordinate samples")
    if (
        full_points
        and last_sample is not None
        and tuple(valid_samples[-1]["grid"]) != full_points[-1]
    ):
        valid_samples.append(
            {
                "grid": [full_points[-1][0], full_points[-1][1]],
                "capture_offset": float(last_sample.get("capture_offset", 0.0)),
                "sample_index": int(last_sample.get("sample_index", -1)),
                "source": "human_reference_end",
            }
        )
    if (
        len(valid_samples) > 2
        and math.dist(
            tuple(valid_samples[0]["grid"]),
            tuple(valid_samples[-1]["grid"]),
        )
        * GRID_SPACING
        < spacing_yards * 0.5
    ):
        valid_samples.pop()
    return valid_samples, full_points


def build_node_candidates(
    *,
    node_points: Sequence[tuple[float, float]],
    cost_map: TerrainCostMap,
    radius_cells: float,
    min_clearance_cells: float,
    sector_count: int,
) -> tuple[dict[GridPoint, set[int]], list[int]]:
    candidates: dict[GridPoint, set[int]] = {}
    unreachable: list[int] = []
    for node_index, node_point in enumerate(node_points):
        raw = tuple(int(round(value)) for value in node_point)
        approaches = find_ore_approach_candidates(
            cost_map,
            raw,
            radius_cells,
            min_clearance=min_clearance_cells,
            sector_count=sector_count,
        )
        if not approaches:
            unreachable.append(node_index)
            continue
        for approach in approaches:
            point = tuple(int(value) for value in approach["grid"])
            candidates.setdefault(point, set())

    for point in candidates:
        candidates[point] = {
            node_index
            for node_index, node_point in enumerate(node_points)
            if math.dist(point, node_point) <= radius_cells + 1.0e-9
        }
    return candidates, unreachable


def greedy_coverage_points(
    candidates: dict[GridPoint, set[int]],
    *,
    initially_covered: set[int],
    target_indexes: set[int],
    cost_map: TerrainCostMap,
    candidate_metadata: dict[GridPoint, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    metadata = candidate_metadata or {}
    uncovered = set(target_indexes) - set(initially_covered)
    selected: list[dict[str, Any]] = []
    remaining = dict(candidates)
    while uncovered:
        ranked = [
            (
                -len(covered_indexes & uncovered),
                0 if point in metadata else 1,
                float(cost_map.cost[point]),
                point[0],
                point[1],
                point,
                covered_indexes & uncovered,
            )
            for point, covered_indexes in remaining.items()
            if covered_indexes & uncovered
        ]
        if not ranked:
            break
        _negative_gain, _preference, _cost, _row, _col, point, gained = min(ranked)
        item = dict(
            metadata.get(
                point,
                {"grid": [point[0], point[1]], "source": "terrain_coverage"},
            )
        )
        item["newly_covered_node_indexes"] = sorted(gained)
        selected.append(item)
        uncovered -= gained
        remaining.pop(point, None)
    return selected, uncovered


def cheapest_insertion_order(
    backbone: Sequence[dict[str, Any]],
    additions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = [dict(item) for item in backbone]
    if not ordered:
        if not additions:
            return []
        ordered.append(dict(additions[0]))
        additions = additions[1:]
    remaining = [dict(item) for item in additions]
    while remaining:
        best: tuple[float, int, int] | None = None
        for addition_index, item in enumerate(remaining):
            point = tuple(item["grid"])
            for insert_after, start_item in enumerate(ordered):
                end_item = ordered[(insert_after + 1) % len(ordered)]
                start = tuple(start_item["grid"])
                end = tuple(end_item["grid"])
                increase = (
                    math.dist(start, point)
                    + math.dist(point, end)
                    - math.dist(start, end)
                )
                candidate = (increase, addition_index, insert_after)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _increase, addition_index, insert_after = best
        ordered.insert(insert_after + 1, remaining.pop(addition_index))
    return ordered


def plan_cycle(
    ordered_scan_points: Sequence[dict[str, Any]],
    *,
    cost_map: TerrainCostMap,
    max_waypoint_spacing_cells: float,
    min_clearance_cells: float,
) -> tuple[list[GridPoint], list[dict[str, Any]]]:
    if len(ordered_scan_points) < 2:
        raise ValueError("Coverage cycle requires at least two scan points")
    route_points: list[GridPoint] = []
    segment_metrics: list[dict[str, Any]] = []
    for segment_index, (start_item, end_item) in enumerate(
        zip(ordered_scan_points, ordered_scan_points[1:] + ordered_scan_points[:1])
    ):
        start = tuple(int(value) for value in start_item["grid"])
        end = tuple(int(value) for value in end_item["grid"])
        raw_path, search_mode = route_segment(cost_map, start, end)
        simplified = simplify_path(
            cost_map,
            raw_path.points,
            max_waypoint_spacing_cells,
            min_clearance=min_clearance_cells,
        )
        if segment_index == 0:
            route_points.extend(simplified)
        else:
            route_points.extend(simplified[1:])
        segment_metrics.append(
            {
                "index": segment_index,
                "start_scan_index": segment_index,
                "end_scan_index": (segment_index + 1) % len(ordered_scan_points),
                "search_mode": search_mode,
                "raw_point_count": len(raw_path.points),
                "simplified_point_count": len(simplified),
                "world_length_yards": round(path_length(simplified), 3),
                "terrain_cost": round(float(raw_path.total_cost), 3),
            }
        )
        if (segment_index + 1) % 20 == 0:
            print(
                f"Planned {segment_index + 1}/{len(ordered_scan_points)} "
                "coverage segments",
                flush=True,
            )

    if route_points[-1] == route_points[0]:
        route_points.pop()
    for start, end in zip(route_points, route_points[1:] + route_points[:1]):
        if not line_is_walkable(
            cost_map,
            start,
            end,
            min_clearance=min_clearance_cells,
        ):
            raise ValueError(f"Final route crosses blocked terrain: {start} -> {end}")
    return route_points, segment_metrics


def route_waypoint(
    point: GridPoint,
    *,
    index: int,
    raster: TerrainRaster,
    zone: Any,
    cost_map: TerrainCostMap,
) -> dict[str, Any]:
    row, col = point
    x, y = raster.grid_to_ui(row, col, zone)
    height = raster.sample_height(row, col)
    if height is None:
        raise ValueError(f"Route point {point} has no terrain height")
    return {
        "index": index,
        "coord": xy_to_coord(x, y),
        "x": round(float(x), 6),
        "y": round(float(y), 6),
        "source": "terrain_coverage_astar",
        "height": round(float(height), 4),
        "slope_degrees": round(float(cost_map.slope_degrees[row, col]), 4),
        "clearance_cells": round(float(cost_map.clearance[row, col]), 4),
    }


def build_coverage_node_access_plans(
    *,
    nodes: list[dict[str, Any]],
    node_points: Sequence[tuple[float, float]],
    covered_indexes: set[int],
    route_points: Sequence[GridPoint],
    raster: TerrainRaster,
    zone: Any,
    cost_map: TerrainCostMap,
    access_radius_cells: float,
    min_clearance_cells: float,
    max_waypoint_spacing_cells: float,
    sector_count: int,
    max_options: int,
    max_extra_cells: float,
    distance_weight: float,
) -> dict[str, Any]:
    """Attach reusable terrain-safe side approaches to covered ore records."""
    if max_options <= 0:
        raise ValueError("max_options must be positive")
    planned_nodes = 0
    option_count = 0
    facing_aligned_option_count = 0
    failures: list[dict[str, Any]] = []
    for node_index in sorted(covered_indexes):
        node = nodes[node_index]
        raw_point = tuple(int(round(value)) for value in node_points[node_index])
        candidates = find_ore_approach_candidates(
            cost_map,
            raw_point,
            access_radius_cells,
            min_clearance=min_clearance_cells,
            sector_count=sector_count,
        )
        if not candidates:
            failures.append({"index": node_index, "reason": "no_walkable_approach"})
            continue
        attachment_index = min(
            range(len(route_points)),
            key=lambda index: (math.dist(route_points[index], raw_point), index),
        )
        resume_index = (attachment_index + 1) % len(route_points)
        attachment = route_points[attachment_index]
        resume = route_points[resume_index]
        nearest_snap = min(float(candidate["snap_cells"]) for candidate in candidates)
        planned_options: list[
            tuple[tuple[float, float, float, int], dict[str, Any]]
        ] = []
        for candidate_index, candidate in enumerate(candidates):
            snap_cells = float(candidate["snap_cells"])
            if snap_cells > nearest_snap + max_extra_cells + 1.0e-9:
                continue
            approach = tuple(int(value) for value in candidate["grid"])
            facing_stage = _node_facing_staging_point(
                cost_map,
                raw_point,
                approach,
                min_clearance_cells=min_clearance_cells,
            )
            try:
                inbound, inbound_cost, inbound_mode = _plan_access_leg(
                    cost_map,
                    attachment,
                    facing_stage or approach,
                    max_waypoint_spacing_cells=max_waypoint_spacing_cells,
                    min_clearance_cells=min_clearance_cells,
                )
                if facing_stage is not None and inbound[-1] != approach:
                    inbound.append(approach)
                    inbound_cost += math.dist(facing_stage, approach) * float(
                        cost_map.cost[approach]
                    )
                    inbound_mode = f"{inbound_mode}+node_facing_final"
                onward, onward_cost, onward_mode = _plan_access_leg(
                    cost_map,
                    approach,
                    resume,
                    max_waypoint_spacing_cells=max_waypoint_spacing_cells,
                    min_clearance_cells=min_clearance_cells,
                )
            except ValueError:
                continue
            objective = inbound_cost + onward_cost + snap_cells * max(0.0, distance_weight)
            option = {
                "candidate_index": candidate_index,
                "primary": False,
                "approach_coord": _access_waypoint(approach, raster, zone)["coord"],
                "sector": int(candidate.get("sector", -1)),
                "bearing_degrees": float(candidate.get("bearing_degrees", 0.0)),
                "approach_distance_yards": round(snap_cells * GRID_SPACING, 3),
                "node_facing_aligned": facing_stage is not None,
                "facing_stage_coord": (
                    _access_waypoint(facing_stage, raster, zone)["coord"]
                    if facing_stage is not None
                    else None
                ),
                "objective": round(float(objective), 4),
                "inbound": _access_leg_payload(
                    inbound,
                    inbound_cost,
                    inbound_mode,
                    raster,
                    zone,
                ),
                "return": {
                    "method": "reverse_inbound",
                    "terrain_cost": round(float(inbound_cost), 4),
                    "world_length": round(path_length(inbound), 4),
                    "waypoints": [
                        _access_waypoint(point, raster, zone)
                        for point in reversed(inbound)
                    ],
                },
                "resume": _access_leg_payload(
                    onward,
                    onward_cost,
                    onward_mode,
                    raster,
                    zone,
                ),
            }
            planned_options.append(
                (
                    (
                        snap_cells,
                        float(objective),
                        float(candidate.get("local_score", 0.0)),
                        candidate_index,
                    ),
                    option,
                )
            )

        selected = [option for _rank, option in sorted(planned_options)[:max_options]]
        if not selected:
            failures.append({"index": node_index, "reason": "no_route_to_approach"})
            continue
        for rank, option in enumerate(selected):
            option["rank"] = rank
            option["primary"] = rank == 0
        node["terrain_access_plan"] = {
            "method": "coverage_route_attachment_angular_candidates",
            "attachment_route_index": attachment_index,
            "resume_route_index": resume_index,
            "primary_option_rank": 0,
            "options": selected,
        }
        planned_nodes += 1
        option_count += len(selected)
        facing_aligned_option_count += sum(
            bool(option.get("node_facing_aligned")) for option in selected
        )

    return {
        "enabled": True,
        "method": "coverage_route_attachment_angular_candidates",
        "planned_node_count": planned_nodes,
        "failure_count": len(failures),
        "option_count": option_count,
        "facing_aligned_option_count": facing_aligned_option_count,
        "max_options_per_node": max_options,
        "access_radius_cells": round(float(access_radius_cells), 4),
        "failures": failures,
    }


def _node_facing_staging_point(
    cost_map: TerrainCostMap,
    node: GridPoint,
    approach: GridPoint,
    *,
    min_clearance_cells: float,
) -> GridPoint | None:
    """Find an outward staging point so the final movement faces the ore node."""
    delta_row = approach[0] - node[0]
    delta_col = approach[1] - node[1]
    length = math.hypot(delta_row, delta_col)
    if length <= 1.0e-9:
        return None
    unit_row = delta_row / length
    unit_col = delta_col / length
    for distance in (3.0, 2.0, 1.0):
        stage = (
            int(round(approach[0] + unit_row * distance)),
            int(round(approach[1] + unit_col * distance)),
        )
        if stage == approach or not cost_map.is_walkable(
            stage,
            min_clearance=min_clearance_cells,
        ):
            continue
        if line_is_walkable(
            cost_map,
            stage,
            approach,
            min_clearance=min_clearance_cells,
        ):
            return stage
    return None


def _plan_access_leg(
    cost_map: TerrainCostMap,
    start: GridPoint,
    goal: GridPoint,
    *,
    max_waypoint_spacing_cells: float,
    min_clearance_cells: float,
) -> tuple[list[GridPoint], float, str]:
    if start == goal:
        return [start], 0.0, "same_point"
    raw_path, search_mode = route_segment(cost_map, start, goal)
    simplified = simplify_path(
        cost_map,
        raw_path.points,
        max_waypoint_spacing_cells,
        min_clearance=min_clearance_cells,
    )
    if any(
        not line_is_walkable(
            cost_map,
            first,
            second,
            min_clearance=min_clearance_cells,
        )
        for first, second in zip(simplified, simplified[1:])
    ):
        raise ValueError("Simplified access leg crosses blocked terrain")
    return simplified, float(raw_path.total_cost), search_mode


def _access_leg_payload(
    points: Sequence[GridPoint],
    terrain_cost: float,
    search_mode: str,
    raster: TerrainRaster,
    zone: Any,
) -> dict[str, Any]:
    return {
        "search_mode": search_mode,
        "terrain_cost": round(float(terrain_cost), 4),
        "world_length": round(path_length(points), 4),
        "waypoints": [_access_waypoint(point, raster, zone) for point in points],
    }


def _access_waypoint(point: GridPoint, raster: TerrainRaster, zone: Any) -> dict[str, Any]:
    x, y = raster.grid_to_ui(point[0], point[1], zone)
    return {
        "grid": [point[0], point[1]],
        "coord": xy_to_coord(x, y),
        "x": round(float(x), 6),
        "y": round(float(y), 6),
    }


def colorize_terrain(raster: TerrainRaster, cost_map: TerrainCostMap) -> np.ndarray:
    heights = np.asarray(raster.heights, dtype=np.float32)
    valid = np.isfinite(heights)
    normalized = np.zeros(heights.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(heights[valid], [2.0, 98.0])
        span = max(1.0, float(high - low))
        normalized[valid] = np.clip(
            (heights[valid] - low) / span * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    image = cv2.applyColorMap(normalized, color_map)
    image[~valid] = (16, 16, 16)
    image[cost_map.blocked & valid] = (
        image[cost_map.blocked & valid] * 0.3
    ).astype(np.uint8)
    return image


def render_overlay(
    *,
    raster: TerrainRaster,
    cost_map: TerrainCostMap,
    node_points: Sequence[tuple[float, float]],
    human_points: Sequence[GridPoint],
    scan_points: Sequence[dict[str, Any]],
    route_points: Sequence[GridPoint],
    covered_indexes: set[int],
    live_hazard_mask: np.ndarray | None,
    reveal_radius_cells: float,
    output_path: str | Path,
) -> None:
    image = colorize_terrain(raster, cost_map)
    if live_hazard_mask is not None and np.any(live_hazard_mask):
        contours, _hierarchy = cv2.findContours(
            live_hazard_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, contours, -1, (210, 40, 210), 3, cv2.LINE_AA)
    for index, point in enumerate(node_points):
        center = int(round(point[1])), int(round(point[0]))
        color = (40, 220, 255) if index in covered_indexes else (30, 30, 230)
        cv2.circle(image, center, 3, color, -1, lineType=cv2.LINE_AA)
    if len(human_points) > 1:
        polyline = np.asarray(
            [[point[1], point[0]] for point in human_points],
            dtype=np.int32,
        )
        cv2.polylines(image, [polyline], False, (255, 180, 30), 2, cv2.LINE_AA)
    if len(route_points) > 1:
        route_polyline = np.asarray(
            [[point[1], point[0]] for point in route_points + route_points[:1]],
            dtype=np.int32,
        )
        cv2.polylines(image, [route_polyline], False, (245, 245, 245), 2, cv2.LINE_AA)
    radius = int(round(reveal_radius_cells))
    for item in scan_points:
        row, col = item["grid"]
        color = (255, 120, 20) if item["source"].startswith("human") else (255, 40, 210)
        cv2.circle(image, (col, row), radius, color, 1, cv2.LINE_AA)
        cv2.circle(image, (col, row), 3, color, -1, cv2.LINE_AA)

    legend = [
        ("planned terrain route", (245, 245, 245)),
        ("human reference / scan radius", (255, 120, 20)),
        ("added scan point / radius", (255, 40, 210)),
        ("covered ore record", (40, 220, 255)),
        ("unresolved ore record", (30, 30, 230)),
        ("validated live exclusion", (210, 40, 210)),
    ]
    cv2.rectangle(image, (12, 12), (390, 186), (8, 8, 8), -1)
    cv2.putText(
        image,
        "Tanaris coverage V4 - 50 yd conservative radius",
        (28, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    for legend_index, (label, color) in enumerate(legend):
        y = 65 + legend_index * 20
        cv2.line(image, (28, y), (50, y), color, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (60, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Failed to write route overlay to {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reveal_radius_yards <= 0.0:
        raise ValueError("reveal radius must be positive")
    if args.node_access_radius_yards <= 0.0:
        raise ValueError("node access radius must be positive")
    node_source = load_json(args.nodes)
    nodes = [dict(node) for node in node_source["records"]]
    trajectory = load_jsonl(args.trajectory)
    manifest = load_json(args.manifest)
    zone = zone_from_manifest(manifest)
    adt_paths = sorted(Path(args.adt_dir).glob("*.adt"))
    raster = build_terrain_raster(adt_paths)
    cost_map = build_terrain_cost_map(
        raster,
        TerrainCostConfig(
            soft_slope_degrees=args.soft_slope_degrees,
            hard_slope_degrees=args.hard_slope_degrees,
        ),
    )
    live_hazard_path = Path(args.live_hazards) if args.live_hazards else None
    live_hazards = []
    if live_hazard_path is not None and live_hazard_path.exists():
        live_hazards = list(load_json(live_hazard_path).get("hazards", []))
    live_hazard_mask = build_live_hazard_mask(
        cost_map.blocked.shape,
        raster=raster,
        zone=zone,
        hazards=live_hazards,
    )
    cost_map, live_hazard_blocked_cells = apply_live_hazard_exclusions(
        cost_map,
        raster=raster,
        zone=zone,
        hazards=live_hazards,
    )
    bounds = {
        "min_x": min(float(node["x"]) for node in nodes),
        "max_x": max(float(node["x"]) for node in nodes),
        "min_y": min(float(node["y"]) for node in nodes),
        "max_y": max(float(node["y"]) for node in nodes),
    }
    cost_map, grid_bounds = restrict_to_ui_bounds(
        cost_map,
        raster=raster,
        zone=zone,
        bounds=bounds,
        margin=args.route_bound_margin,
    )
    first_sample = next(
        sample
        for sample in trajectory
        if sample.get("x") is not None and sample.get("y") is not None
    )
    raw_anchor = tuple(
        int(round(value))
        for value in raster.ui_to_grid(first_sample["x"], first_sample["y"], zone)
    )
    cost_map, _component, component_area, snapped_anchor = restrict_to_anchor_component(
        cost_map,
        raw_anchor,
        max_snap_radius=args.anchor_snap_radius_cells,
        min_clearance=args.min_clearance_cells,
    )

    human_scan_points, full_human_points = sparsify_human_trajectory(
        trajectory,
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        spacing_yards=args.human_anchor_spacing_yards,
        snap_radius_cells=args.human_snap_radius_cells,
        min_clearance_cells=args.min_clearance_cells,
    )
    node_points = [
        raster.ui_to_grid(float(node["x"]), float(node["y"]), zone)
        for node in nodes
    ]
    node_exclusion_mask = build_live_hazard_mask(
        cost_map.blocked.shape,
        raster=raster,
        zone=zone,
        hazards=[
            hazard
            for hazard in live_hazards
            if bool(hazard.get("exclude_nodes", False))
        ],
    )
    live_hazard_excluded_nodes = node_indexes_in_mask(
        node_points,
        node_exclusion_mask,
    )
    radius_cells = args.reveal_radius_yards / GRID_SPACING
    human_covered = {
        index
        for index, node_point in enumerate(node_points)
        if index not in live_hazard_excluded_nodes
        and cyclic_polyline_distance(node_point, full_human_points) <= radius_cells
    }
    candidates, no_terrain_candidate = build_node_candidates(
        node_points=node_points,
        cost_map=cost_map,
        radius_cells=radius_cells,
        min_clearance_cells=args.min_clearance_cells,
        sector_count=args.node_candidate_sectors,
    )
    no_terrain_candidate = [
        index
        for index in no_terrain_candidate
        if index not in live_hazard_excluded_nodes
    ]
    human_candidate_metadata: dict[GridPoint, dict[str, Any]] = {}
    for point in list(candidates):
        candidates[point] -= live_hazard_excluded_nodes
        if not candidates[point]:
            del candidates[point]
    for item in human_scan_points:
        point = tuple(item["grid"])
        candidates[point] = {
            index
            for index, node_point in enumerate(node_points)
            if index not in live_hazard_excluded_nodes
            and math.dist(point, node_point) <= radius_cells + 1.0e-9
        }
        existing = human_candidate_metadata.get(point)
        if existing is None or float(item["capture_offset"]) < float(
            existing["capture_offset"]
        ):
            human_candidate_metadata[point] = item
    target_indexes = set().union(*candidates.values()) if candidates else set()
    no_coverage_candidate = (
        set(range(len(nodes))) - target_indexes - live_hazard_excluded_nodes
    )
    selected_points, set_cover_unresolved = greedy_coverage_points(
        candidates,
        initially_covered=set(),
        target_indexes=target_indexes,
        cost_map=cost_map,
        candidate_metadata=human_candidate_metadata,
    )
    selected_human_points = sorted(
        (
            item
            for item in selected_points
            if item["source"].startswith("human_reference")
        ),
        key=lambda item: float(item["capture_offset"]),
    )
    additions = [
        item
        for item in selected_points
        if not item["source"].startswith("human_reference")
    ]
    ordered_scan_points = cheapest_insertion_order(selected_human_points, additions)
    print(
        f"Coverage targets: {len(target_indexes)}/{len(nodes)}; "
        f"human candidates={len(human_scan_points)} "
        f"selected={len(selected_human_points)} additions={len(additions)}",
        flush=True,
    )
    route_points: list[GridPoint] = []
    segment_metrics: list[dict[str, Any]] = []
    repair_addition_count = 0
    planning_unresolved = set(set_cover_unresolved)
    for repair_pass in range(3):
        route_points, segment_metrics = plan_cycle(
            ordered_scan_points,
            cost_map=cost_map,
            max_waypoint_spacing_cells=args.max_waypoint_spacing_cells,
            min_clearance_cells=args.min_clearance_cells,
        )
        route_target_covered = {
            index
            for index in target_indexes
            if cyclic_polyline_distance(node_points[index], route_points) <= radius_cells
        }
        missing_targets = target_indexes - route_target_covered
        if not missing_targets:
            break
        used_points = {tuple(item["grid"]) for item in ordered_scan_points}
        repair_candidates = {
            point: covered
            for point, covered in candidates.items()
            if point not in used_points
        }
        repairs, unresolved = greedy_coverage_points(
            repair_candidates,
            initially_covered=route_target_covered,
            target_indexes=target_indexes,
            cost_map=cost_map,
        )
        planning_unresolved |= unresolved
        if not repairs:
            break
        for repair in repairs:
            repair["source"] = f"terrain_coverage_repair_{repair_pass + 1}"
        additions.extend(repairs)
        repair_addition_count += len(repairs)
        ordered_scan_points = cheapest_insertion_order(ordered_scan_points, repairs)
        print(
            f"Coverage repair pass {repair_pass + 1}: added {len(repairs)} scan points",
            flush=True,
        )

    route_covered: set[int] = set()
    enriched_nodes: list[dict[str, Any]] = []
    for node_index, (node, node_point) in enumerate(zip(nodes, node_points)):
        distance_yards = cyclic_polyline_distance(node_point, route_points) * GRID_SPACING
        covered = (
            node_index not in live_hazard_excluded_nodes
            and distance_yards <= args.reveal_radius_yards + 1.0e-6
        )
        if covered:
            route_covered.add(node_index)
        item = dict(node)
        item.update(
            {
                "coverage_status": (
                    "live_hazard_excluded"
                    if node_index in live_hazard_excluded_nodes
                    else "covered"
                    if covered
                    else "no_walkable_scan_point_within_radius"
                    if node_index in no_coverage_candidate
                    else "route_planning_unresolved"
                ),
                "distance_to_coverage_route_yards": round(distance_yards, 3),
                "human_reference_covered": node_index in human_covered,
            }
        )
        enriched_nodes.append(item)

    route_loop = [
        route_waypoint(
            point,
            index=index,
            raster=raster,
            zone=zone,
            cost_map=cost_map,
        )
        for index, point in enumerate(route_points)
    ]
    node_access_metrics = build_coverage_node_access_plans(
        nodes=enriched_nodes,
        node_points=node_points,
        covered_indexes=route_covered,
        route_points=route_points,
        raster=raster,
        zone=zone,
        cost_map=cost_map,
        access_radius_cells=args.node_access_radius_yards / GRID_SPACING,
        min_clearance_cells=args.min_clearance_cells,
        max_waypoint_spacing_cells=args.max_waypoint_spacing_cells,
        sector_count=args.node_candidate_sectors,
        max_options=args.node_access_max_options,
        max_extra_cells=args.node_access_max_extra_cells,
        distance_weight=args.node_access_distance_weight,
    )
    scan_output: list[dict[str, Any]] = []
    for index, item in enumerate(ordered_scan_points):
        point = tuple(item["grid"])
        x, y = raster.grid_to_ui(point[0], point[1], zone)
        scan_output.append(
            {
                **item,
                "index": index,
                "coord": xy_to_coord(x, y),
                "x": round(float(x), 6),
                "y": round(float(y), 6),
            }
        )

    route_length = path_length(route_points + route_points[:1])
    human_path_length = path_length(full_human_points)
    human_offsets = [
        float(sample["capture_offset"])
        for sample in trajectory
        if sample.get("capture_offset") is not None
    ]
    human_duration = max(human_offsets) - min(human_offsets)
    human_average_speed = (
        human_path_length / human_duration if human_duration > 0.0 else 0.0
    )
    estimated_cycle_minutes = (
        route_length / human_average_speed / 60.0
        if human_average_speed > 0.0
        else None
    )
    slopes = [float(cost_map.slope_degrees[point]) for point in route_points]
    metrics = {
        "schema_version": 1,
        "status": "offline_analysis_candidate_not_live_validated",
        "zone": node_source.get("zone_name", "Tanaris"),
        "source_node_count": len(nodes),
        "eligible_node_count": len(nodes) - len(live_hazard_excluded_nodes),
        "human_reference_covered_count": len(human_covered),
        "human_reference_coverage_fraction": round(len(human_covered) / len(nodes), 6),
        "terrain_candidate_count": len(target_indexes),
        "live_hazard_excluded_node_count": len(live_hazard_excluded_nodes),
        "live_hazard_excluded_node_indexes": sorted(live_hazard_excluded_nodes),
        "no_terrain_candidate_count": len(no_terrain_candidate),
        "no_coverage_candidate_count": len(no_coverage_candidate),
        "set_cover_unresolved_count": len(planning_unresolved),
        "final_covered_count": len(route_covered),
        "final_coverage_fraction": round(len(route_covered) / len(nodes), 6),
        "eligible_coverage_fraction": round(
            len(route_covered) / max(1, len(nodes) - len(live_hazard_excluded_nodes)),
            6,
        ),
        "reveal_radius_yards": args.reveal_radius_yards,
        "reveal_radius_cells": round(radius_cells, 6),
        "human_candidate_count": len(human_scan_points),
        "human_anchor_count": len(selected_human_points),
        "added_scan_point_count": len(additions),
        "repair_scan_point_count": repair_addition_count,
        "total_scan_point_count": len(ordered_scan_points),
        "route_waypoint_count": len(route_points),
        "route_length_yards": round(route_length, 3),
        "human_reference_path_length_yards": round(human_path_length, 3),
        "human_reference_duration_seconds": round(human_duration, 3),
        "human_reference_average_speed_yards_per_second": round(
            human_average_speed,
            4,
        ),
        "estimated_cycle_minutes_at_human_average_speed": (
            round(estimated_cycle_minutes, 3)
            if estimated_cycle_minutes is not None
            else None
        ),
        "route_slope_mean_degrees": round(float(np.mean(slopes)), 4),
        "route_slope_max_degrees": round(float(np.max(slopes)), 4),
        "walkable_component_cells": component_area,
        "grid_bounds": list(grid_bounds),
        "anchor_grid": list(snapped_anchor),
        "terrain_config": cost_map.config.to_dict(),
        "live_hazards": {
            "path": str(live_hazard_path) if live_hazard_path is not None else None,
            "count": len(live_hazards),
            "blocked_cells": live_hazard_blocked_cells,
            "excluded_node_count": len(live_hazard_excluded_nodes),
        },
        "node_access_plans": node_access_metrics,
        "segments": segment_metrics,
        "source_paths": {
            "nodes": str(args.nodes),
            "trajectory": str(args.trajectory),
            "manifest": str(args.manifest),
            "adt_dir": str(args.adt_dir),
            "live_hazards": str(live_hazard_path) if live_hazard_path is not None else None,
        },
        "limitations": [
            "ADT height/slope routing does not yet model doodads, fences, buildings, or liquid.",
            "GatherMate records do not identify underground nodes; dark live minimap icons remain ignored.",
            "The extended portions outside the human recording require live validation.",
        ],
    }
    output = {
        "schema_version": 1,
        "name": "Tanaris terrain-aware minimap coverage cycle v4",
        "zone_id": node_source.get("gathermate_zone_id", 162),
        "world_map_area_id": node_source.get("world_map_area_id", 161),
        "status": metrics["status"],
        "coverage_policy": {
            "reveal_radius_yards": args.reveal_radius_yards,
            "behavior": (
                "Follow the scan route; intercept only a confirmed bright live ore icon. "
                "Ignore dark underground icons and historical GatherMate pins."
            ),
        },
        "coverage_scan_points": scan_output,
        "route_loop": route_loop,
        "route_nodes": enriched_nodes,
        "metrics": metrics,
    }
    write_json(output, args.output_route)
    write_json(metrics, args.output_metrics)
    render_overlay(
        raster=raster,
        cost_map=cost_map,
        node_points=node_points,
        human_points=full_human_points,
        scan_points=ordered_scan_points,
        route_points=route_points,
        covered_indexes=route_covered,
        live_hazard_mask=live_hazard_mask,
        reveal_radius_cells=radius_cells,
        output_path=args.output_overlay,
    )
    print(
        f"Route: {args.output_route}\n"
        f"Metrics: {args.output_metrics}\n"
        f"Overlay: {args.output_overlay}\n"
        f"Coverage: {len(route_covered)}/{len(nodes)}; "
        f"length={route_length:.1f} yd; waypoints={len(route_points)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
