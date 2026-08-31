from __future__ import annotations

from pathlib import Path

import yaml

from vision_bot.core.commands import CommandGroup
from vision_bot.engine.factory import build_v09_kernel


ROOT = Path(__file__).parents[1]


def test_factory_builds_all_hazard_compiled_v19_kernel() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    bundle = build_v09_kernel(config)

    # V19 contracts the disconnected V18 southwest excursion that intersected
    # older live hazards, then normalizes one inherited adjacent duplicate at
    # the immutable-artifact adapter edge.
    assert len(bundle.route.points) == 303
    assert bundle.route.total_length_yards > 12_000.0
    assert bundle.route_report.source_status == "offline_compiled_candidate_not_live_validated"
    assert bundle.route_report.normalized_duplicate_count == 1
    assert bundle.route_report.mining_node_count == 73
    assert bundle.route_report.permanently_excluded_node_count == 1
    assert not bundle.route_report.hazard_point_indexes
    assert not bundle.route_report.hazard_segment_indexes
    assert set(bundle.kernel.controllers) == {
        CommandGroup.RECOVERY,
        CommandGroup.COMBAT,
        CommandGroup.LOOT,
        CommandGroup.MINING,
        CommandGroup.MOUNT,
        CommandGroup.TRAVEL,
    }
