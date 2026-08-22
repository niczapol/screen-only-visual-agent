from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vision_bot.coords import coord_to_xy
from vision_bot.route_planner import ore_name


ZONE_PATTERN = re.compile(r"^\s*\[(\d+)\]\s*=\s*\{\s*$")
NODE_PATTERN = re.compile(r"^\s*\[(\d{7,10})\]\s*=\s*(\d+)\s*,?\s*$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one zone from GatherMate2MineDB saved variables"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--zone-id", required=True, type=int)
    parser.add_argument("--zone-name", required=True)
    parser.add_argument("--world-map-area-id", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-output")
    return parser.parse_args(argv)


def parse_saved_mining_nodes(text: str) -> dict[int, list[tuple[int, int]]]:
    started = False
    depth = 0
    current_zone: int | None = None
    result: dict[int, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        if not started:
            if re.match(r"^\s*GatherMate2MineDB\s*=\s*\{\s*$", line):
                started = True
                depth = 1
            continue

        if depth == 1:
            zone_match = ZONE_PATTERN.match(line)
            if zone_match:
                current_zone = int(zone_match.group(1))
                result.setdefault(current_zone, [])
        elif depth == 2 and current_zone is not None:
            node_match = NODE_PATTERN.match(line)
            if node_match:
                result[current_zone].append(
                    (int(node_match.group(1)), int(node_match.group(2)))
                )

        depth += line.count("{") - line.count("}")
        if depth < 2:
            current_zone = None
        if depth <= 0:
            return result
    if not started:
        raise ValueError("GatherMate2MineDB table was not found")
    raise ValueError("GatherMate2MineDB table is not balanced")


def normalize_zone(
    records: Sequence[tuple[int, int]],
    *,
    zone_id: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for source_index, (coord, ore_id) in enumerate(records):
        key = coord, ore_id
        if key in seen:
            continue
        seen.add(key)
        x, y = coord_to_xy(coord)
        normalized.append(
            {
                "index": source_index,
                "zone_id": zone_id,
                "coord": coord,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "ore_type": ore_name(ore_id),
                "ore_id": ore_id,
            }
        )
    normalized.sort(
        key=lambda item: (
            float(item["x"]),
            float(item["y"]),
            int(item["ore_id"]),
            int(item["coord"]),
        )
    )
    return normalized


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source)
    tables = parse_saved_mining_nodes(source.read_text(encoding="utf-8"))
    if args.zone_id not in tables:
        raise ValueError(f"GatherMate mining zone {args.zone_id} was not found")
    records = normalize_zone(tables[args.zone_id], zone_id=args.zone_id)
    output = {
        "schema_version": 1,
        "zone_name": args.zone_name,
        "gathermate_zone_id": args.zone_id,
        "world_map_area_id": args.world_map_area_id,
        "source": str(source),
        "records": records,
    }
    write_json(output, args.output)
    if args.seed_output:
        seed = {
            "schema_version": 1,
            "name": f"{args.zone_name} GatherMate all-node extraction seed",
            "zone_id": args.zone_id,
            "source": str(args.output),
            "route_loop": [
                {"x": record["x"], "y": record["y"]}
                for record in records
            ],
            "route_nodes": records,
        }
        write_json(seed, args.seed_output)
    print(f"Exported {len(records)} nodes for zone {args.zone_id} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
