from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from vision_bot.v09_config import V09Config, V09ConfigError


ROOT = Path(__file__).parents[1]


def _project_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_project_v09_config_is_valid_and_fail_closed() -> None:
    config = V09Config.from_mapping(_project_config())

    assert config.enabled
    assert config.default_mode == "replay"
    assert config.protocol_version == 2
    assert config.performance.max_frame_backlog == 1
    assert config.performance.idle_ore_scan_interval_frames == 2
    assert config.performance.shutdown_guard_quiet_seconds == 5.0
    assert config.performance.shutdown_guard_max_seconds == 600.0
    assert config.recovery.release_retry_seconds == 1.5
    assert config.recovery.max_release_attempts == 4
    assert config.zone.zone_id == 162
    assert config.mining.capture_radius_yards < config.mining.braking_radius_yards


def test_v09_config_rejects_live_as_default() -> None:
    raw = deepcopy(_project_config())
    raw["v09"]["default_mode"] = "live"

    with pytest.raises(V09ConfigError, match="cannot be live"):
        V09Config.from_mapping(raw)


def test_v09_config_rejects_unknown_keys() -> None:
    raw = deepcopy(_project_config())
    raw["v09"]["navigation"]["lookahead_magic"] = 123

    with pytest.raises(V09ConfigError, match="unknown keys"):
        V09Config.from_mapping(raw)


def test_v09_config_rejects_invalid_mining_radius_order() -> None:
    raw = deepcopy(_project_config())
    raw["v09"]["mining"]["capture_radius_yards"] = 10.0

    with pytest.raises(V09ConfigError, match="capture < braking"):
        V09Config.from_mapping(raw)
