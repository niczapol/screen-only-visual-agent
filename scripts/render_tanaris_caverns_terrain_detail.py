from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.build_terrain_coverage_route import colorize_terrain
    from scripts.build_topographic_route import zone_from_manifest
except ModuleNotFoundError:
    from build_terrain_coverage_route import colorize_terrain
    from build_topographic_route import zone_from_manifest
from vision_bot.coords import coord_to_xy
from vision_bot.navmesh_route_entry import build_hazard_mask
from vision_bot.terrain_routing import build_terrain_cost_map, build_terrain_raster


ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v18_rail.json"
MANIFEST = ROOT / "data/extracted_client_data/tanaris_corrected/manifest.json"
ADT_DIR = ROOT / "data/extracted_client_data/tanaris_corrected/world/maps/Kalimdor"
OUTPUT = ROOT / "data/offline_v0827_caverns_terrain_detail.png"
HAZARDS = ROOT / "data/routes/live_hazards/tanaris.json"


def main() -> int:
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(sorted(ADT_DIR.glob("*.adt")))
    image = colorize_terrain(raster, build_terrain_cost_map(raster))
    hazards = json.loads(HAZARDS.read_text(encoding="utf-8"))
    gorge = next(
        item
        for item in hazards["hazards"]
        if item.get("kind") == "caverns_of_time_gorge_interior"
    )
    gorge_mask = build_hazard_mask(
        image.shape,
        raster=raster,
        zone=zone,
        hazards=[gorge],
    )

    x_min, x_max, y_min, y_max = 52.0, 78.0, 27.0, 64.0
    grid_corners = [
        raster.ui_to_grid(x, y, zone)
        for x, y in ((x_min, y_min), (x_max, y_max))
    ]
    row_min = int(min(point[0] for point in grid_corners))
    row_max = int(max(point[0] for point in grid_corners))
    col_min = int(min(point[1] for point in grid_corners))
    col_max = int(max(point[1] for point in grid_corners))
    canvas = image[row_min : row_max + 1, col_min : col_max + 1].copy()
    local_gorge = gorge_mask[row_min : row_max + 1, col_min : col_max + 1]
    overlay = np.zeros_like(canvas)
    overlay[:, :] = (20, 20, 230)
    blended = cv2.addWeighted(canvas, 0.42, overlay, 0.58, 0)
    canvas[local_gorge] = blended[local_gorge]

    def point_from_coord(coord: int) -> tuple[int, int]:
        x, y = coord_to_xy(int(coord))
        row, col = raster.ui_to_grid(x, y, zone)
        return int(round(col)) - col_min, int(round(row)) - row_min

    route_points = [
        point_from_coord(int(item["coord"]))
        for item in route["route_loop"]
        if x_min <= float(item["x"]) <= x_max
        and y_min <= float(item["y"]) <= y_max
    ]
    cv2.polylines(
        canvas,
        [np.asarray(route_points, dtype=np.int32)],
        False,
        (255, 235, 40),
        3,
        cv2.LINE_AA,
    )
    for node in route["route_nodes"]:
        x, y = float(node["x"]), float(node["y"])
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        point = point_from_coord(int(node["coord"]))
        cv2.circle(canvas, point, 3, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(node["node_id"]),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.15,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    death = point_from_coord(6353497000)
    cv2.drawMarker(canvas, death, (40, 40, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.putText(
        canvas,
        "death 63.53,49.70",
        (death[0] + 6, death[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.20,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "V18 restored rail + DB nodes; red = gorge only",
        (6, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.22,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    scale = 1600 / canvas.shape[1]
    canvas = cv2.resize(
        canvas,
        (1600, int(canvas.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), canvas):
        raise OSError(f"Unable to write {OUTPUT}")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
