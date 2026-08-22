from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.build_tanaris_rail_route_v17 import _option_coords
    from scripts.build_terrain_coverage_route import colorize_terrain
    from scripts.build_topographic_route import zone_from_manifest
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_tanaris_rail_route_v17 import _option_coords
    from build_terrain_coverage_route import colorize_terrain
    from build_topographic_route import zone_from_manifest
from vision_bot.config import load_config
from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.navmesh_route_entry import (
    NavMeshRouteEntryPlanner,
    build_hazard_mask,
)
from vision_bot.terrain_routing import GRID_SPACING, build_terrain_cost_map, build_terrain_raster, supercover_line


ROOT = Path(__file__).resolve().parents[1]
V16 = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v16_rail.json"
V17 = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v17_rail.json"
HAZARDS = ROOT / "data/routes/live_hazards/tanaris.json"
MANIFEST = ROOT / "data/extracted_client_data/tanaris_corrected/manifest.json"
ADT_DIR = ROOT / "data/extracted_client_data/tanaris_corrected/world/maps/Kalimdor"
DEFAULT_OUTPUT = ROOT / "data/offline_v0826_caverns_route_review.png"
DEFAULT_AUDIT = ROOT / "data/offline_v0826_caverns_route_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and audit the V17 Caverns exclusion.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_point(raster, zone, coord: int) -> tuple[int, int]:
    x, y = coord_to_xy(int(coord))
    return tuple(int(round(value)) for value in raster.ui_to_grid(x, y, zone))


def _point_masked(mask: np.ndarray, point: tuple[int, int]) -> bool:
    row, col = point
    return bool(0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col])


def _segment_masked(
    mask: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    return any(_point_masked(mask, point) for point in supercover_line(first, second))


def _route_audit(route: dict[str, Any], mask: np.ndarray, raster, zone) -> dict[str, Any]:
    points = [_grid_point(raster, zone, int(item["coord"])) for item in route["route_loop"]]
    return {
        "waypoint_count": len(points),
        "hazard_waypoint_indexes": [
            index for index, point in enumerate(points) if _point_masked(mask, point)
        ],
        "hazard_segment_indexes": [
            index
            for index, (first, second) in enumerate(zip(points, points[1:]))
            if _segment_masked(mask, first, second)
        ],
    }


def _polyline(image, route, raster, zone, offset, color, thickness) -> None:
    points = []
    for item in route:
        row, col = _grid_point(raster, zone, int(item["coord"]))
        points.append((col - offset[1], row - offset[0]))
    if len(points) >= 2:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    v16 = _load(V16)
    v17 = _load(V17)
    hazard_source = _load(HAZARDS)
    caverns = next(
        item
        for item in hazard_source["hazards"]
        if item.get("kind") == "caverns_of_time_full_surface_exclusion"
    )
    manifest = _load(MANIFEST)
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(sorted(ADT_DIR.glob("*.adt")))
    costs = build_terrain_cost_map(raster)
    image = colorize_terrain(raster, costs)
    mask = build_hazard_mask(
        image.shape,
        raster=raster,
        zone=zone,
        hazards=[caverns],
    )

    corners = [raster.ui_to_grid(x, y, zone) for x, y in ((0, 0), (100, 100))]
    row_min = max(0, int(min(point[0] for point in corners)))
    row_max = min(image.shape[0] - 1, int(max(point[0] for point in corners)))
    col_min = max(0, int(min(point[1] for point in corners)))
    col_max = min(image.shape[1] - 1, int(max(point[1] for point in corners)))
    canvas = image[row_min : row_max + 1, col_min : col_max + 1].copy()
    local_mask = mask[row_min : row_max + 1, col_min : col_max + 1]
    red = np.zeros_like(canvas)
    red[:, :] = (30, 30, 230)
    canvas[local_mask] = cv2.addWeighted(canvas, 0.38, red, 0.62, 0)[local_mask]

    offset = (row_min, col_min)
    _polyline(canvas, v16["route_loop"], raster, zone, offset, (125, 125, 125), 2)
    _polyline(canvas, v16["route_loop"][336:434], raster, zone, offset, (20, 90, 255), 5)
    _polyline(canvas, v17["route_loop"], raster, zone, offset, (255, 220, 40), 4)
    bridge = [
        item
        for item in v17["route_loop"]
        if item.get("source") == "caverns_of_time_v17_western_exclusion_bridge"
    ]
    _polyline(canvas, bridge, raster, zone, offset, (80, 255, 80), 6)

    planner = NavMeshRouteEntryPlanner.from_config(load_config("config.yaml"), zone_id=162)
    excluded_nodes = [
        node for node in v16["route_nodes"] if planner.coord_is_hazard(int(node["coord"]))
    ]
    for node in excluded_nodes:
        row, col = _grid_point(raster, zone, int(node["coord"]))
        cv2.circle(canvas, (col - col_min, row - row_min), 5, (0, 170, 255), -1, cv2.LINE_AA)

    death_coord = xy_to_coord(63.53, 49.70)
    death_row, death_col = _grid_point(raster, zone, death_coord)
    death_point = (death_col - col_min, death_row - row_min)
    cv2.drawMarker(canvas, death_point, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 5, cv2.LINE_AA)
    cv2.putText(canvas, "DEATH 63.53,49.70", (death_point[0] + 16, death_point[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (18, 18), (570, 150), (15, 15, 15), -1)
    cv2.putText(canvas, "Tanaris V17 - Caverns of Time exclusion", (34, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (38, 78), (102, 78), (80, 255, 80), 6, cv2.LINE_AA)
    cv2.putText(canvas, "V17 safe western bridge", (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(canvas, (38, 108), (102, 108), (20, 90, 255), 5, cv2.LINE_AA)
    cv2.putText(canvas, "removed V16 eastern lobe", (120, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.circle(canvas, (70, 137), 5, (0, 170, 255), -1)
    cv2.putText(canvas, "excluded ore/access nodes", (120, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)

    target_width = 1800
    scale = target_width / canvas.shape[1]
    canvas = cv2.resize(canvas, (target_width, int(canvas.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"Unable to write {args.output}")

    v16_audit = _route_audit(v16, mask, raster, zone)
    v17_audit = _route_audit(v17, mask, raster, zone)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    v17_points = [_grid_point(raster, zone, int(item["coord"])) for item in v17["route_loop"]]
    min_extra_clearance = min(float(distance[row, col]) for row, col in v17_points)

    unsafe_new_nodes = [
        int(node["node_id"])
        for node in v17["route_nodes"]
        if planner.coord_is_hazard(int(node["coord"]))
    ]
    unsafe_options = []
    for node in v17["route_nodes"]:
        plan = node.get("terrain_access_plan")
        if not isinstance(plan, dict):
            continue
        for option in plan.get("options", []):
            coords = _option_coords(option)
            if any(planner.coord_is_hazard(coord) for coord in coords):
                unsafe_options.append([int(node["node_id"]), int(option.get("rank", -1))])

    audit = {
        "schema_version": 1,
        "scope": "offline_only_no_live_client_started",
        "hazard_kind": caverns["kind"],
        "configured_margin_yards": float(caverns["margin_yards"]),
        "death_coordinate": {"x": 63.53, "y": 49.70, "inside_exclusion": _point_masked(mask, _grid_point(raster, zone, death_coord))},
        "stale_filtered_coordinate": {"x": 61.03, "y": 50.27, "inside_exclusion": _point_masked(mask, _grid_point(raster, zone, xy_to_coord(61.03, 50.27)))},
        "v16": v16_audit,
        "v17": v17_audit,
        "v17_minimum_clearance_beyond_configured_margin_yards": round(min_extra_clearance * GRID_SPACING, 3),
        "excluded_v16_runtime_node_ids": [int(node["node_id"]) for node in excluded_nodes],
        "excluded_v16_runtime_node_count": len(excluded_nodes),
        "unsafe_v17_runtime_node_ids": unsafe_new_nodes,
        "unsafe_v17_access_options": unsafe_options,
        "output_map": str(args.output),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
