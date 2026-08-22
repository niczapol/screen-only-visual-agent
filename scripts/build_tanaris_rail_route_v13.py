from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.build_tanaris_rail_route_v12 import (
    DEFAULT_SOURCE,
    build_dunemaul_bypass_route,
    generate_dunemaul_detour,
)
from vision_bot.config import load_config


DEFAULT_OUTPUT = Path("data/routes/generated/tanaris_terrain_coverage_cycle_v13_rail.json")
DEFAULT_REMOVE_START = 36
DEFAULT_REMOVE_END = 83
DEFAULT_CONTROL_XY = (
    (47.70, 50.50),
    (47.70, 64.00),
    (42.00, 65.00),
    (34.00, 64.00),
    (32.71, 77.63),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tanaris rail V13 around Dunemaul and the live-blocked "
            "western rock pass."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def build_v13(source: dict, config: dict) -> dict:
    detour = generate_dunemaul_detour(config, control_xy=DEFAULT_CONTROL_XY)
    route = build_dunemaul_bypass_route(
        source,
        detour_coords=detour,
        remove_start=DEFAULT_REMOVE_START,
        remove_end=DEFAULT_REMOVE_END,
    )
    route["name"] = "Tanaris terrain-aware full rail cycle v13 wide Dunemaul bypass"
    route["route_override"]["reason"] = (
        "V11 died in the Dunemaul pack at 37.73,54.96; V12 then repeated "
        "bounded stuck recovery at the narrow western rock pass 30.69,63.18. "
        "V13 removes the entire unsafe transit pocket and rejoins the old rail "
        "at 32.71,77.63 through a terrain-validated wide corridor."
    )
    route["route_override"]["observed_stuck_coordinate"] = {
        "x": 30.69,
        "y": 63.18,
    }
    return route


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    route = build_v13(source, load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(route["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
