from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vision_bot.coords import xy_to_coord


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend a reviewed staging connector to a generated route entry"
    )
    parser.add_argument("--base-route", required=True)
    parser.add_argument("--entry-spec", required=True)
    parser.add_argument("--output-route", required=True)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def compose_route_entry(
    base_route: dict[str, Any],
    entry_spec: dict[str, Any],
) -> dict[str, Any]:
    base_entry = base_route.get("entry_route")
    if not isinstance(base_entry, dict) or not base_entry.get("waypoints"):
        raise ValueError("Base route must contain a non-empty entry_route")

    zone_id = int(base_route.get("zone", {}).get("id", 0))
    spec_zone_id = int(entry_spec.get("zone", {}).get("id", zone_id))
    if zone_id != spec_zone_id:
        raise ValueError(f"Entry zone {spec_zone_id} does not match route zone {zone_id}")

    staging = _normalize_waypoints(
        entry_spec.get("waypoints", []),
        source=str(entry_spec.get("waypoint_source", "reviewed_staging_entry")),
    )
    if len(staging) < 2:
        raise ValueError("Entry spec must contain at least two staging waypoints")

    generated = _normalize_waypoints(
        base_entry.get("waypoints", []),
        source="terrain_astar_entry",
    )
    combined = _deduplicate_join(staging, generated)
    for index, waypoint in enumerate(combined):
        waypoint["index"] = index

    result = copy.deepcopy(base_route)
    result["name"] = str(entry_spec.get("output_name") or f"{base_route.get('name', 'route')}_staged")
    result["entry_route"] = {
        "status": str(entry_spec.get("status", "candidate_reviewed_staging_entry")),
        "target_route_index": int(base_entry.get("target_route_index", 0)),
        "raw_point_count": len(combined),
        "waypoint_count": len(combined),
        "path_ui_length": round(_path_ui_length(combined), 6),
        "waypoints": combined,
        "staging": {
            key: copy.deepcopy(value)
            for key, value in entry_spec.items()
            if key not in {"waypoints", "zone", "output_name", "waypoint_source"}
        },
        "base_entry": {
            "status": base_entry.get("status"),
            "waypoint_count": len(generated),
            "path_world_length": base_entry.get("path_world_length"),
        },
    }

    generation = result.setdefault("generation", {})
    generation["route_waypoints"] = len(result.get("route_loop", []))
    topographic = generation.setdefault("topographic", {})
    component_selection = topographic.setdefault("component_selection", {})
    component_selection["runtime_entry_anchor_ui"] = [
        float(staging[0]["x"]),
        float(staging[0]["y"]),
    ]
    component_selection["entry_waypoint_count"] = len(combined)
    component_selection["staging_waypoint_count"] = len(staging)
    return result


def _normalize_waypoints(
    waypoints: object,
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(waypoints, list):
        raise ValueError("waypoints must be a list")
    normalized: list[dict[str, Any]] = []
    for item in waypoints:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError("Every waypoint must contain x and y")
        x = float(item["x"])
        y = float(item["y"])
        normalized.append(
            {
                "coord": int(item.get("coord") or xy_to_coord(x, y)),
                "x": round(x, 6),
                "y": round(y, 6),
                "source": str(item.get("source") or source),
            }
        )
    return normalized


def _deduplicate_join(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for waypoint in [*first, *second]:
        if combined and int(combined[-1]["coord"]) == int(waypoint["coord"]):
            continue
        combined.append(dict(waypoint))
    return combined


def _path_ui_length(waypoints: list[dict[str, Any]]) -> float:
    return sum(
        math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
        for start, end in zip(waypoints, waypoints[1:])
    )


def main() -> None:
    args = parse_args()
    write_json(
        compose_route_entry(load_json(args.base_route), load_json(args.entry_spec)),
        args.output_route,
    )


if __name__ == "__main__":
    main()
