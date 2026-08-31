from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vision_bot.core.geometry import MapPoint
from vision_bot.core.hazards import PhysicalHazardGuard
from vision_bot.engine.route_compiler import compile_configured_route
from vision_bot.v09_config import V09Config


ROOT = Path(__file__).parents[1]


def _config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_v19_provenance_matches_exact_v18_source_and_has_no_hazard_hits() -> None:
    source_path = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v18_rail.json"
    compiled_path = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v19_rail.json"
    data = json.loads(compiled_path.read_text(encoding="utf-8"))

    assert data["source"]["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert data["route_override"]["all_hazard_audit"] == {
        "hazard_point_indexes": [],
        "hazard_segment_indexes": [],
    }
    assert data["metrics"]["runtime_node_count"] == 74
    assert all("terrain_access_plan" in node for node in data["route_nodes"])

    compiled = compile_configured_route(_config(), V09Config.from_mapping(_config()).zone)
    assert compiled.report.mining_node_count == 73
    assert compiled.report.permanently_excluded_node_count == 1
    assert all(node.node_id != 79 for node in compiled.nodes)


def test_compiler_rejects_v18_because_its_global_hazard_audit_was_incomplete() -> None:
    config = _config()
    config["route"]["database_path"] = (
        "data/routes/generated/tanaris_terrain_coverage_cycle_v18_rail.json"
    )
    geometry = V09Config.from_mapping(config).zone

    with pytest.raises(ValueError, match="intersects configured hazards"):
        compile_configured_route(config, geometry)


def test_physical_hazard_guard_enforces_yard_radius_and_margin() -> None:
    config = _config()
    geometry = V09Config.from_mapping(config).zone
    guard = PhysicalHazardGuard.from_config(config, geometry)

    assert guard.contains(MapPoint(30.65, 73.82))
    assert "southwest_wreckage_contact" in guard.containing_kinds(
        MapPoint(30.65, 73.82)
    )
