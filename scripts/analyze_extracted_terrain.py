from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from vision_bot.terrain_routing import load_adt_v9


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize extracted ADT terrain height chunks")
    parser.add_argument("--input-dir", default="data/extracted_client_data/barrens_v2")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    adt_dir = input_dir / "world" / "maps" / "Kalimdor"
    rows = [analyze_adt(path) for path in sorted(adt_dir.glob("*.adt"))]
    rows = [row for row in rows if row is not None]

    summary = {
        "input_dir": str(input_dir),
        "tiles": len(rows),
        "chunks": sum(int(row["height_chunks"]) for row in rows),
        "height_min": min((row["height_min"] for row in rows), default=None),
        "height_max": max((row["height_max"] for row in rows), default=None),
        "max_tile_relief": max((row["height_relief"] for row in rows), default=None),
        "avg_tile_relief": round(statistics.fmean(row["height_relief"] for row in rows), 4) if rows else None,
        "tiles_detail": rows,
    }

    output_json = Path(args.output_json) if args.output_json else input_dir / "terrain_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "terrain_summary.csv"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rows, output_csv)
    print(f"Terrain summary: {output_json}")
    print(f"Terrain CSV: {output_csv}")
    print(
        "tiles={tiles} chunks={chunks} height_min={height_min} height_max={height_max} max_relief={max_tile_relief}".format(
            **summary
        )
    )


def analyze_adt(path: Path) -> dict[str, Any] | None:
    heights = load_adt_v9(path)
    finite = heights[np.isfinite(heights)]
    if finite.size == 0:
        return None

    tile_x, tile_y = _tile_xy(path)
    height_min = float(np.min(finite))
    height_max = float(np.max(finite))
    return {
        "file": str(path),
        "tile_x": tile_x,
        "tile_y": tile_y,
        "height_chunks": 256,
        "height_samples": int(finite.size),
        "height_min": round(height_min, 4),
        "height_max": round(height_max, 4),
        "height_relief": round(height_max - height_min, 4),
        "height_mean": round(float(np.mean(finite)), 4),
    }


def _tile_xy(path: Path) -> tuple[int | None, int | None]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return int(parts[-2]), int(parts[-1])
    return None, None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "tile_x",
        "tile_y",
        "height_chunks",
        "height_samples",
        "height_min",
        "height_max",
        "height_relief",
        "height_mean",
        "file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
