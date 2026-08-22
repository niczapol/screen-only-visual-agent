from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vision_bot.route_planner import MiningNode


def load_permanent_exclusions(path: str | Path) -> set[int]:
    exclusion_path = Path(path)
    if not exclusion_path.exists():
        return set()

    try:
        data = json.loads(exclusion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    if isinstance(data, list):
        return {int(coord) for coord in data}

    if not isinstance(data, dict):
        return set()

    coords: set[int] = set()
    for item in data.get("excluded_coords", []):
        if isinstance(item, dict) and "coord" in item:
            coords.add(int(item["coord"]))
        elif isinstance(item, int | str):
            coords.add(int(item))
    return coords


def add_permanent_exclusion(path: str | Path, node: MiningNode, reason: str) -> None:
    exclusion_path = Path(path)
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)

    existing_items = _read_items(exclusion_path)
    if any(int(item.get("coord", -1)) == node.coord for item in existing_items):
        return

    existing_items.append(
        {
            "coord": node.coord,
            "zone_id": node.zone_id,
            "node_id": node.node_id,
            "ore_type": node.ore_type,
            "reason": reason,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    exclusion_path.write_text(
        json.dumps({"excluded_coords": existing_items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(data, dict):
        items = data.get("excluded_coords", [])
    elif isinstance(data, list):
        items = data
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and "coord" in item:
            normalized.append(item)
        elif isinstance(item, int | str):
            normalized.append({"coord": int(item), "reason": "legacy"})
    return normalized
