from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.build_topographic_route import restrict_to_anchor_component, zone_from_manifest
from vision_bot.terrain_routing import (
    GRID_SPACING,
    TerrainCostConfig,
    build_terrain_cost_map,
    build_terrain_raster,
)


REFERENCE_SIZE = (2560, 1440)
REFERENCE_MAP_BOUNDS = (337, 129, 2218, 1380)
DEFAULT_UI_CROP = (24.0, 82.0, 55.0, 84.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a readable route/rejection briefing over the in-game zone map"
    )
    parser.add_argument("--route", required=True)
    parser.add_argument("--world-map-frame", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adt-dir", required=True)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-analysis", required=True)
    parser.add_argument("--crop", nargs=4, type=float, default=DEFAULT_UI_CROP)
    parser.add_argument("--spur-review-distance-cells", type=float, default=16.0)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rejection_counts(route: dict[str, Any]) -> Counter[str]:
    return Counter(str(item.get("reason", "unknown")) for item in route.get("rejected_nodes", []))


def analyze_spur_candidates(
    route: dict[str, Any],
    *,
    manifest_path: str | Path,
    adt_dir: str | Path,
    review_distance_cells: float,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(sorted(Path(adt_dir).glob("*.adt")))
    topographic = route["generation"]["topographic"]
    cost_config = TerrainCostConfig(**topographic["cost_config"])
    base_cost_map = build_terrain_cost_map(raster, cost_config)
    selection = topographic["component_selection"]
    anchor_grid = tuple(int(value) for value in selection["anchor_grid"])
    _restricted, primary_mask, _area, _snapped = restrict_to_anchor_component(
        base_cost_map,
        anchor_grid,
        max_snap_radius=float(selection.get("anchor_snap_limit_cells", 20.0)),
        min_clearance=1.0,
    )

    distance_input = np.where(primary_mask, 0, 255).astype(np.uint8)
    distance_to_component = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    terrain_rejected = [
        item
        for item in route.get("rejected_nodes", [])
        if item.get("reason") == "terrain_no_primary_component_approach"
        and "terrain_raw_grid" in item
    ]
    reviewed: list[dict[str, Any]] = []
    distances: list[float] = []
    for item in terrain_rejected:
        row, col = (int(value) for value in item["terrain_raw_grid"])
        distance_cells = float(distance_to_component[row, col])
        distances.append(distance_cells)
        if distance_cells <= review_distance_cells:
            reviewed.append(
                {
                    "coord": int(item["coord"]),
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "ore_type": str(item.get("ore_type", "Ore")),
                    "distance_to_active_component_cells": round(distance_cells, 4),
                    "distance_to_active_component_world": round(distance_cells * GRID_SPACING, 2),
                    "status": "live_spur_review_only",
                }
            )

    return {
        "automatic_safe_backtracking_spurs": 0,
        "reason": (
            "Every terrain-rejected node lies beyond the accepted eight-cell approach radius. "
            "Exact reverse traversal cannot make an unsafe outbound leg safe."
        ),
        "live_spur_review_distance_cells": float(review_distance_cells),
        "live_spur_review_candidates": sorted(
            reviewed,
            key=lambda item: item["distance_to_active_component_cells"],
        ),
        "terrain_rejected_distance_bands": {
            "8_to_12_cells": sum(8.0 < value <= 12.0 for value in distances),
            "12_to_16_cells": sum(12.0 < value <= 16.0 for value in distances),
            "16_to_32_cells": sum(16.0 < value <= 32.0 for value in distances),
            "over_32_cells": sum(value > 32.0 for value in distances),
        },
    }


def render_briefing_map(
    route: dict[str, Any],
    world_map_frame: np.ndarray,
    analysis: dict[str, Any],
    *,
    ui_crop: tuple[float, float, float, float],
) -> np.ndarray:
    map_image = _crop_world_map(world_map_frame)
    min_x, max_x, min_y, max_y = ui_crop
    source_height, source_width = map_image.shape[:2]
    x0 = int(round(min_x / 100.0 * source_width))
    x1 = int(round(max_x / 100.0 * source_width))
    y0 = int(round(min_y / 100.0 * source_height))
    y1 = int(round(max_y / 100.0 * source_height))
    cropped = map_image[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError("Requested UI crop does not intersect the map image")

    map_width, map_height = 2100, 760
    rendered = cv2.resize(cropped, (map_width, map_height), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((1080, 2200, 3), (20, 20, 20), dtype=np.uint8)
    canvas[80 : 80 + map_height, 50 : 50 + map_width] = rendered

    def point(x: float, y: float) -> tuple[int, int]:
        px = 50 + int(round((x - min_x) / (max_x - min_x) * map_width))
        py = 80 + int(round((y - min_y) / (max_y - min_y) * map_height))
        return px, py

    terrain_review_coords = {
        int(item["coord"]) for item in analysis.get("live_spur_review_candidates", [])
    }
    for item in route.get("rejected_nodes", []):
        if "x" not in item or "y" not in item:
            continue
        x, y = float(item["x"]), float(item["y"])
        if not _inside_ui_crop(x, y, ui_crop):
            continue
        center = point(x, y)
        reason = item.get("reason")
        if reason == "too_close_to_accepted_route_node":
            cv2.circle(canvas, center, 3, (170, 170, 170), -1, cv2.LINE_AA)
        elif int(item.get("coord", -1)) in terrain_review_coords:
            cv2.drawMarker(canvas, center, (255, 80, 255), cv2.MARKER_DIAMOND, 14, 2)
        else:
            cv2.drawMarker(canvas, center, (70, 70, 235), cv2.MARKER_TILTED_CROSS, 9, 2)

    entry = route.get("entry_route", {})
    entry_waypoints = entry.get("waypoints", []) if isinstance(entry, dict) else []
    staging_points = [
        item for item in entry_waypoints if item.get("source") == "reviewed_world_map_road_entry"
    ]
    generated_points = [
        item for item in entry_waypoints if item.get("source") != "reviewed_world_map_road_entry"
    ]
    _draw_polyline(canvas, staging_points, point, (0, 145, 255), 7)
    if staging_points and generated_points:
        _draw_polyline(canvas, [staging_points[-1], generated_points[0]], point, (70, 255, 70), 7)
    _draw_polyline(canvas, generated_points, point, (70, 255, 70), 7)
    _draw_polyline(canvas, route.get("route_loop", []), point, (255, 230, 0), 7, closed=True)

    for index, node in enumerate(route.get("route_nodes", []), start=1):
        center = point(float(node["x"]), float(node["y"]))
        cv2.circle(canvas, center, 11, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 13, (20, 20, 20), 2, cv2.LINE_AA)
        label = f"{index}: {node.get('ore_type', 'Ore')} {float(node['x']):.1f},{float(node['y']):.1f}"
        cv2.putText(
            canvas,
            label,
            (center[0] + 15, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if staging_points:
        start = point(float(staging_points[0]["x"]), float(staging_points[0]["y"]))
        cv2.circle(canvas, start, 16, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "LIVE START",
            (start[0] + 20, start[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Desolace: Shadowprey road entry + north-Kodo six-node cycle",
        (50, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    counts = rejection_counts(route)
    source_count = len(route.get("route_nodes", [])) + sum(counts.values())
    lines = [
        f"Source spawns: {source_count} | accepted: {len(route.get('route_nodes', []))}",
        f"Spacing duplicates: {counts.get('too_close_to_accepted_route_node', 0)} | terrain rejected: {counts.get('terrain_no_primary_component_approach', 0)}",
        f"Automatically safe return spurs: {analysis.get('automatic_safe_backtracking_spurs', 0)} | live-review near-component candidates: {len(analysis.get('live_spur_review_candidates', []))}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (50, 890 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    legend = [
        ("road entry", (0, 145, 255)),
        ("terrain entry", (70, 255, 70)),
        ("cycle", (255, 230, 0)),
        ("terrain reject", (70, 70, 235)),
        ("spur review", (255, 80, 255)),
        ("spacing duplicate", (170, 170, 170)),
    ]
    legend_x = 1270
    for index, (label, color) in enumerate(legend):
        x = legend_x + (index % 3) * 295
        y = 900 + (index // 3) * 50
        cv2.line(canvas, (x, y), (x + 34, y), color, 7, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x + 45, y + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _crop_world_map(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top, right, bottom = REFERENCE_MAP_BOUNDS
    sx = width / REFERENCE_SIZE[0]
    sy = height / REFERENCE_SIZE[1]
    x0, x1 = int(round(left * sx)), int(round(right * sx))
    y0, y1 = int(round(top * sy)), int(round(bottom * sy))
    return frame[y0:y1, x0:x1]


def _inside_ui_crop(
    x: float,
    y: float,
    crop: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = crop
    return min_x <= x <= max_x and min_y <= y <= max_y


def _draw_polyline(
    image: np.ndarray,
    items: list[dict[str, Any]],
    point,
    color: tuple[int, int, int],
    thickness: int,
    *,
    closed: bool = False,
) -> None:
    if len(items) < 2:
        return
    points = [point(float(item["x"]), float(item["y"])) for item in items]
    if closed:
        points.append(points[0])
    for start, end in zip(points, points[1:]):
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        cv2.line(image, start, end, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    route = load_json(args.route)
    analysis = analyze_spur_candidates(
        route,
        manifest_path=args.manifest,
        adt_dir=args.adt_dir,
        review_distance_cells=max(0.0, args.spur_review_distance_cells),
    )
    counts = rejection_counts(route)
    analysis.update(
        {
            "route": str(Path(args.route)),
            "source_spawn_count": len(route.get("route_nodes", [])) + sum(counts.values()),
            "accepted_node_count": len(route.get("route_nodes", [])),
            "spacing_rejected_count": counts.get("too_close_to_accepted_route_node", 0),
            "terrain_rejected_count": counts.get("terrain_no_primary_component_approach", 0),
        }
    )
    frame = cv2.imread(str(args.world_map_frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(args.world_map_frame)
    output = render_briefing_map(route, frame, analysis, ui_crop=tuple(args.crop))
    output_path = Path(args.output_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to write {output_path}")
    analysis_path = Path(args.output_analysis)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
