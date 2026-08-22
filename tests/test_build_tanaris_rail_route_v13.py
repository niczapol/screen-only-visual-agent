from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from scripts.build_tanaris_rail_route_v13 import build_v13
from vision_bot.config import load_config


def test_v13_removes_live_blocked_western_pass_and_rejoins_southern_rail() -> None:
    source = json.loads(
        open(
            "data/routes/generated/tanaris_terrain_coverage_cycle_v11_rail.json",
            encoding="utf-8",
        ).read()
    )

    # V13 is a historical geometry regression.  Later live iterations append
    # hazards which deliberately intersect the old rail, so replaying V13
    # against today's hazard registry makes this test depend on future route
    # revisions instead of the V13 control points it owns.
    config = copy.deepcopy(load_config("config.yaml"))
    config["route"]["entry"]["navmesh_profiles"]["162"].pop("hazards_path", None)
    manifest = Path(
        config["route"]["entry"]["navmesh_profiles"]["162"]["manifest_path"]
    )
    if not manifest.exists():
        pytest.skip("external terrain/navmesh fixtures are not distributed")
    route = build_v13(source, config)

    assert route["route_override"]["removed_v11_indexes"] == [36, 83]
    assert route["route_override"]["observed_stuck_coordinate"] == {
        "x": 30.69,
        "y": 63.18,
    }
    assert all(item["index"] == index for index, item in enumerate(route["route_loop"]))
    assert all(
        not (30.55 <= float(item["x"]) <= 30.83 and 63.0 <= float(item["y"]) <= 63.34)
        for item in route["route_loop"]
    )
    assert any(
        float(item["x"]) == 32.71 and float(item["y"]) == 77.63
        for item in route["route_loop"]
    )
