from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.build_topographic_route import zone_from_manifest
from vision_bot.terrain_routing import (
    GRID_SPACING,
    TerrainCostConfig,
    TerrainRaster,
    ZoneTransform,
    build_terrain_cost_map,
    build_terrain_raster,
)


DEFAULT_UI_CROP = (24.0, 82.0, 55.0, 84.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a route briefing directly over extracted ADT elevation and slope"
    )
    parser.add_argument("--route", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adt-dir", required=True)
    parser.add_argument("--spur-analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop", nargs=4, type=float, default=DEFAULT_UI_CROP)
    parser.add_argument("--contour-interval", type=float, default=20.0)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def crop_grid_bounds(
    raster: TerrainRaster,
    zone: ZoneTransform,
    ui_crop: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    min_x, max_x, min_y, max_y = ui_crop
    corners = [
        raster.ui_to_grid(x, y, zone)
        for x in (min_x, max_x)
        for y in (min_y, max_y)
    ]
    rows = [point[0] for point in corners]
    cols = [point[1] for point in corners]
    row0 = max(0, int(math.floor(min(rows))))
    row1 = min(raster.heights.shape[0], int(math.ceil(max(rows))) + 1)
    col0 = max(0, int(math.floor(min(cols))))
    col1 = min(raster.heights.shape[1], int(math.ceil(max(cols))) + 1)
    if row0 >= row1 or col0 >= col1:
        raise ValueError("UI crop does not intersect the extracted terrain raster")
    return row0, row1, col0, col1


def render_topographic_map(
    route: dict[str, Any],
    spur_analysis: dict[str, Any],
    *,
    raster: TerrainRaster,
    zone: ZoneTransform,
    ui_crop: tuple[float, float, float, float],
    contour_interval: float,
) -> np.ndarray:
    topographic = route["generation"]["topographic"]
    cost_config = TerrainCostConfig(**topographic["cost_config"])
    cost_map = build_terrain_cost_map(raster, cost_config)
    row0, row1, col0, col1 = crop_grid_bounds(raster, zone, ui_crop)

    height_crop = np.asarray(raster.heights[row0:row1, col0:col1], dtype=np.float32)
    slope_crop = np.asarray(cost_map.slope_degrees[row0:row1, col0:col1], dtype=np.float32)
    terrain = _render_terrain(
        height_crop,
        slope_crop,
        soft_slope=cost_config.soft_slope_degrees,
        hard_slope=cost_config.hard_slope_degrees,
        contour_interval=max(1.0, contour_interval),
    )

    source_height, source_width = terrain.shape[:2]
    map_height = 960
    map_width = int(round(map_height * source_width / source_height))
    sidebar_width = 650
    canvas = np.full((1080, map_width + sidebar_width + 100, 3), (20, 20, 20), dtype=np.uint8)
    resized = cv2.resize(terrain, (map_width, map_height), interpolation=cv2.INTER_CUBIC)
    origin_x, origin_y = 50, 78
    canvas[origin_y : origin_y + map_height, origin_x : origin_x + map_width] = resized

    def grid_point(row: float, col: float) -> tuple[int, int]:
        x = origin_x + int(round((float(col) - col0) / max(1, col1 - col0 - 1) * (map_width - 1)))
        y = origin_y + int(round((float(row) - row0) / max(1, row1 - row0 - 1) * (map_height - 1)))
        return x, y

    def ui_point(x: float, y: float) -> tuple[int, int]:
        row, col = raster.ui_to_grid(x, y, zone)
        return grid_point(row, col)

    review_coords = {
        int(item["coord"]) for item in spur_analysis.get("live_spur_review_candidates", [])
    }
    for node in route.get("rejected_nodes", []):
        if "x" not in node or "y" not in node:
            continue
        x, y = float(node["x"]), float(node["y"])
        if not _inside_ui_crop(x, y, ui_crop):
            continue
        center = ui_point(x, y)
        reason = str(node.get("reason", "unknown"))
        if reason == "too_close_to_accepted_route_node":
            cv2.circle(canvas, center, 3, (190, 190, 190), -1, cv2.LINE_AA)
        elif int(node.get("coord", -1)) in review_coords:
            cv2.drawMarker(canvas, center, (255, 80, 255), cv2.MARKER_DIAMOND, 14, 2)
        else:
            cv2.drawMarker(canvas, center, (40, 40, 245), cv2.MARKER_TILTED_CROSS, 10, 2)

    entry = route.get("entry_route", {})
    entry_points = entry.get("waypoints", []) if isinstance(entry, dict) else []
    staging = [item for item in entry_points if item.get("source") == "reviewed_world_map_road_entry"]
    terrain_entry = [item for item in entry_points if item.get("source") != "reviewed_world_map_road_entry"]
    _draw_ui_polyline(canvas, staging, ui_point, (0, 145, 255), 7)
    if staging and terrain_entry:
        _draw_ui_polyline(canvas, [staging[-1], terrain_entry[0]], ui_point, (70, 255, 70), 7)
    _draw_ui_polyline(canvas, terrain_entry, ui_point, (70, 255, 70), 7)
    _draw_ui_polyline(canvas, route.get("route_loop", []), ui_point, (255, 230, 0), 7, closed=True)

    accepted = route.get("route_nodes", [])
    for index, node in enumerate(accepted, start=1):
        center = ui_point(float(node["x"]), float(node["y"]))
        cv2.circle(canvas, center, 12, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 14, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (center[0] - 5, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    if staging:
        start = ui_point(float(staging[0]["x"]), float(staging[0]["y"]))
        cv2.circle(canvas, start, 16, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(canvas, start, 8, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "Desolace ADT topography: Shadowprey entry + six-node cycle",
        (50, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    _draw_sidebar(
        canvas,
        x=origin_x + map_width + 35,
        route=route,
        spur_analysis=spur_analysis,
        accepted=accepted,
        height_crop=height_crop,
        cost_config=cost_config,
        contour_interval=max(1.0, contour_interval),
    )
    return canvas


def _render_terrain(
    heights: np.ndarray,
    slopes: np.ndarray,
    *,
    soft_slope: float,
    hard_slope: float,
    contour_interval: float,
) -> np.ndarray:
    valid = np.isfinite(heights)
    finite = heights[valid]
    if finite.size == 0:
        raise ValueError("Terrain crop contains no finite heights")
    low, high = (float(value) for value in np.percentile(finite, [2.0, 98.0]))
    if math.isclose(low, high):
        high = low + 1.0
    normalized = np.zeros_like(heights, dtype=np.float32)
    normalized[valid] = np.clip((heights[valid] - low) / (high - low), 0.0, 1.0)
    image = _hypsometric_colors(normalized)

    fill = float(np.median(finite))
    filled = np.where(valid, heights, fill)
    gradient_row, gradient_col = np.gradient(filled, GRID_SPACING)
    light = 0.62 - 0.42 * gradient_col - 0.30 * gradient_row
    light = cv2.GaussianBlur(light.astype(np.float32), (0, 0), 1.0)
    light = cv2.normalize(light, None, 0.55, 1.18, cv2.NORM_MINMAX)
    image = np.clip(image.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)

    soft_mask = valid & np.isfinite(slopes) & (slopes >= soft_slope) & (slopes < hard_slope)
    hard_mask = valid & np.isfinite(slopes) & (slopes >= hard_slope)
    _blend(image, soft_mask, (0, 150, 255), 0.20)
    _blend(image, hard_mask, (30, 30, 235), 0.42)

    contour_level = np.floor((filled - low) / contour_interval).astype(np.int32)
    contour_edges = valid & (
        (contour_level != np.roll(contour_level, 1, axis=0))
        | (contour_level != np.roll(contour_level, 1, axis=1))
    )
    image[contour_edges] = (22, 22, 22)
    image[~valid] = (5, 5, 5)
    return image


def _hypsometric_colors(normalized: np.ndarray) -> np.ndarray:
    stops = np.asarray([0.0, 0.20, 0.43, 0.68, 0.86, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            (85, 70, 30),
            (95, 125, 55),
            (85, 155, 105),
            (110, 150, 170),
            (165, 185, 210),
            (235, 235, 235),
        ],
        dtype=np.float32,
    )
    result = np.empty((*normalized.shape, 3), dtype=np.float32)
    for channel in range(3):
        result[..., channel] = np.interp(normalized, stops, colors[:, channel])
    return np.clip(result, 0, 255).astype(np.uint8)


def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    source = image[mask].astype(np.float32)
    target = np.asarray(color, dtype=np.float32)
    image[mask] = np.clip(source * (1.0 - alpha) + target * alpha, 0, 255)


def _draw_ui_polyline(
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
        cv2.line(image, start, end, (20, 20, 20), thickness + 4, cv2.LINE_AA)
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        cv2.line(image, start, end, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_sidebar(
    image: np.ndarray,
    *,
    x: int,
    route: dict[str, Any],
    spur_analysis: dict[str, Any],
    accepted: list[dict[str, Any]],
    height_crop: np.ndarray,
    cost_config: TerrainCostConfig,
    contour_interval: float,
) -> None:
    valid_heights = height_crop[np.isfinite(height_crop)]
    counts = Counter(str(item.get("reason", "unknown")) for item in route.get("rejected_nodes", []))
    title_lines = [
        "TOPOGRAPHIC EVIDENCE",
        f"ADT elevation: {float(np.min(valid_heights)):.0f} to {float(np.max(valid_heights)):.0f} m",
        f"Contours: {contour_interval:.0f} m",
        f"Amber slope: {cost_config.soft_slope_degrees:.0f}-{cost_config.hard_slope_degrees:.0f} deg",
        f"Red slope: >= {cost_config.hard_slope_degrees:.0f} deg",
    ]
    y = 112
    for index, text in enumerate(title_lines):
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if index == 0 else 0.52,
            (250, 250, 250),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 36

    y += 16
    legend = [
        ("Road entry", (0, 145, 255), "line"),
        ("Terrain entry", (70, 255, 70), "line"),
        ("Six-node cycle", (255, 230, 0), "line"),
        ("Accepted ore node", (0, 255, 255), "circle"),
        ("Terrain excluded", (40, 40, 245), "cross"),
        ("Spur review only", (255, 80, 255), "diamond"),
        ("Spacing duplicate", (190, 190, 190), "circle"),
    ]
    for label, color, shape in legend:
        if shape == "line":
            cv2.line(image, (x, y - 6), (x + 42, y - 6), color, 7, cv2.LINE_AA)
        elif shape == "cross":
            cv2.drawMarker(image, (x + 20, y - 6), color, cv2.MARKER_TILTED_CROSS, 14, 2)
        elif shape == "diamond":
            cv2.drawMarker(image, (x + 20, y - 6), color, cv2.MARKER_DIAMOND, 14, 2)
        else:
            cv2.circle(image, (x + 20, y - 6), 7, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x + 55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        y += 42

    y += 12
    stats = [
        f"Source spawns: {len(accepted) + sum(counts.values())}",
        f"Accepted: {len(accepted)}",
        f"Spacing duplicates: {counts.get('too_close_to_accepted_route_node', 0)}",
        f"Terrain excluded: {counts.get('terrain_no_primary_component_approach', 0)}",
        f"Automatic return spurs: {spur_analysis.get('automatic_safe_backtracking_spurs', 0)}",
        f"Review-only spurs: {len(spur_analysis.get('live_spur_review_candidates', []))}",
    ]
    for text in stats:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (240, 240, 240), 1, cv2.LINE_AA)
        y += 33

    y += 12
    cv2.putText(image, "ROUTE ORE NODES", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 2, cv2.LINE_AA)
    y += 38
    for index, node in enumerate(accepted, start=1):
        label = (
            f"{index}. {node.get('ore_type', 'Ore')}  "
            f"{float(node['x']):.1f}, {float(node['y']):.1f}"
        )
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
        y += 32


def _inside_ui_crop(
    x: float,
    y: float,
    crop: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = crop
    return min_x <= x <= max_x and min_y <= y <= max_y


def main() -> None:
    args = parse_args()
    route = load_json(args.route)
    manifest = load_json(args.manifest)
    raster = build_terrain_raster(sorted(Path(args.adt_dir).glob("*.adt")))
    output = render_topographic_map(
        route,
        load_json(args.spur_analysis),
        raster=raster,
        zone=zone_from_manifest(manifest),
        ui_crop=tuple(args.crop),
        contour_interval=args.contour_interval,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise RuntimeError(f"Failed to write {destination}")


if __name__ == "__main__":
    main()
