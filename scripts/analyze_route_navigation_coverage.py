from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Check route tile coverage in extracted server navigation data")
    parser.add_argument("--route-db", default="data/routes/generated/barrens_mining_75_125_safe_cycle.json")
    parser.add_argument("--manifest", default="data/extracted_client_data/barrens_v2/manifest.json")
    parser.add_argument("--navigation-dir", default="data/server_navigation/ac_data_v20")
    parser.add_argument("--output-json", default="data/routes/generated/barrens_navigation_coverage.json")
    parser.add_argument("--output-csv", default="data/routes/generated/barrens_navigation_coverage.csv")
    args = parser.parse_args()

    route_data = json.loads(Path(args.route_db).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    area = manifest["world_map_area"]
    map_id = int(area["map_id"])
    navigation_dir = Path(args.navigation_dir)

    rows = []
    seen_tiles: set[tuple[int, int]] = set()
    for item in route_data.get("route_loop", []):
        tile = route_coord_to_tile(float(item["x"]), float(item["y"]), area)
        if tile in seen_tiles:
            continue
        seen_tiles.add(tile)
        rows.append(_coverage_row(navigation_dir, map_id, tile[0], tile[1]))

    summary = {
        "route_db": args.route_db,
        "manifest": args.manifest,
        "navigation_dir": args.navigation_dir,
        "map_id": map_id,
        "zone_name": manifest.get("zone_name"),
        "route_tile_count": len(rows),
        "maps_present": sum(1 for row in rows if row["map_present"]),
        "mmaps_present": sum(1 for row in rows if row["mmtile_present"]),
        "vmaps_present": sum(1 for row in rows if row["vmtile_present"]),
        "base_mmap_present": (navigation_dir / "mmaps" / f"{map_id:03d}.mmap").exists(),
        "base_vmtree_present": (navigation_dir / "vmaps" / f"{map_id:03d}.vmtree").exists(),
        "tiles": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rows, output_csv)
    print(f"Navigation coverage: {output_json}")
    print(f"Navigation coverage CSV: {output_csv}")
    print(
        "tiles={route_tile_count} maps={maps_present} mmaps={mmaps_present} vmaps={vmaps_present} "
        "base_mmap={base_mmap_present} base_vmtree={base_vmtree_present}".format(**summary)
    )


def route_coord_to_tile(x_percent: float, y_percent: float, area: dict[str, Any]) -> tuple[int, int]:
    left = float(area["left"])
    right = float(area["right"])
    top = float(area["top"])
    bottom = float(area["bottom"])
    world_x = left + (right - left) * (x_percent / 100.0)
    world_y = top + (bottom - top) * (y_percent / 100.0)
    tile_x = int(math.floor(32.0 - world_y / 533.3333333333))
    tile_y = int(math.floor(32.0 - world_x / 533.3333333333))
    return tile_x, tile_y


def _coverage_row(navigation_dir: Path, map_id: int, tile_x: int, tile_y: int) -> dict[str, Any]:
    map_name = f"{map_id:03d}{tile_x:02d}{tile_y:02d}.map"
    mmtile_name = f"{map_id:03d}{tile_x:02d}{tile_y:02d}.mmtile"
    vmtile_name = f"{map_id:03d}_{tile_x:02d}_{tile_y:02d}.vmtile"
    map_path = navigation_dir / "maps" / map_name
    mmtile_path = navigation_dir / "mmaps" / mmtile_name
    vmtile_path = navigation_dir / "vmaps" / vmtile_name
    return {
        "map_id": map_id,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "map_file": map_name,
        "mmtile_file": mmtile_name,
        "vmtile_file": vmtile_name,
        "map_present": map_path.exists(),
        "mmtile_present": mmtile_path.exists(),
        "vmtile_present": vmtile_path.exists(),
        "map_size": map_path.stat().st_size if map_path.exists() else 0,
        "mmtile_size": mmtile_path.stat().st_size if mmtile_path.exists() else 0,
        "vmtile_size": vmtile_path.stat().st_size if vmtile_path.exists() else 0,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "map_id",
        "tile_x",
        "tile_y",
        "map_file",
        "mmtile_file",
        "vmtile_file",
        "map_present",
        "mmtile_present",
        "vmtile_present",
        "map_size",
        "mmtile_size",
        "vmtile_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
