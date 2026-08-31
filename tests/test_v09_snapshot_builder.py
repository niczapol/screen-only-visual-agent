from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from vision_bot.combat import CombatState
from vision_bot.core.evidence import EvidenceState
from vision_bot.core.world import LifeState, MiningSignal
from vision_bot.game_state import GameState
from vision_bot.perception.snapshot_builder import build_snapshot_from_route_observation
from vision_bot.perception.snapshot_builder import SnapshotBuilder
from vision_bot.route_probe_observation import RouteFrameObservation
from vision_bot.runtime_markers import RuntimeTelemetry


def _observation(
    *,
    telemetry: RuntimeTelemetry | None = None,
    game_state: GameState | None = None,
    combat: CombatState | None = None,
    bright_points: tuple[tuple[int, int], ...] = (),
) -> RouteFrameObservation:
    return RouteFrameObservation(
        timestamp=42.0,
        telemetry=telemetry,
        game_state=game_state or GameState(False, 0, False, alive_health_bar=True),
        combat=combat or CombatState(),
        bright_ore_points=bright_points,
        dark_ore_points=(),
        armor_critical=False,
        mounted=True,
        forbidden_subzone_visible=False,
    )


def test_builder_preserves_frame_identity_across_all_evidence() -> None:
    snapshot = build_snapshot_from_route_observation(
        _observation(
            telemetry=RuntimeTelemetry(
                coord=5000500000,
                x=50.0,
                y=50.0,
                heading_degrees=180.0,
            ),
            bright_points=((100, 120),),
        ),
        frame_id=12,
        input_ready=True,
        zone_id=162,
    )

    assert snapshot.frame_id == 12
    assert snapshot.pose.frame_id == 12
    assert snapshot.pose.value is not None
    assert snapshot.pose.value.position.x_percent == 50.0
    assert snapshot.life.value is LifeState.ALIVE
    assert snapshot.mining.value is not None
    assert snapshot.mining.value.signal is MiningSignal.CANDIDATE
    assert snapshot.mining.value.bright_minimap_points == ((100, 120),)


def test_missing_telemetry_is_unknown_not_absent() -> None:
    snapshot = build_snapshot_from_route_observation(
        _observation(telemetry=None),
        frame_id=13,
        input_ready=True,
        zone_id=162,
    )

    assert snapshot.pose.state is EvidenceState.UNKNOWN
    assert snapshot.pose.reason == "telemetry_not_decoded"


def test_window_guard_fails_closed_without_input_readiness() -> None:
    snapshot = build_snapshot_from_route_observation(
        _observation(),
        frame_id=14,
        input_ready=False,
        zone_id=162,
    )

    assert snapshot.input_ready.state is EvidenceState.ABSENT


def test_game_state_maps_ghost_to_recovery_life_state() -> None:
    snapshot = build_snapshot_from_route_observation(
        _observation(
            game_state=GameState(
                death_or_blocking_modal=True,
                red_button_count=0,
                ghost_visual=True,
                ghost_button_count=2,
            )
        ),
        frame_id=15,
        input_ready=True,
        zone_id=162,
    )

    assert snapshot.life.value is LifeState.GHOST


def test_live_builder_withdraws_pose_but_not_window_input_on_frozen_protocol(
    monkeypatch,
) -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8")
    )
    builder = SnapshotBuilder(config, zone_id=162)

    def observation(_frame, _config, *, timestamp, **_kwargs):
        return replace(
            _observation(
                telemetry=RuntimeTelemetry(
                    coord=5000500000,
                    x=50.0,
                    y=50.0,
                    heading_degrees=180.0,
                    protocol_version=2,
                    frame_sequence=7,
                )
            ),
            timestamp=timestamp,
        )

    monkeypatch.setattr(
        "vision_bot.perception.snapshot_builder.observe_route_frame", observation
    )
    monkeypatch.setattr(
        "vision_bot.perception.snapshot_builder.detect_combat_loot_pending_marker",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "vision_bot.perception.snapshot_builder.detect_dead_hostile_target_marker",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "vision_bot.perception.snapshot_builder.detect_loot_opened_marker",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "vision_bot.perception.snapshot_builder.read_minimap_ore_tooltip_telemetry",
        lambda *_args: None,
    )
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

    first = builder.observe(frame, frame_id=1, captured_at=1.0, input_ready=True)
    briefly_same = builder.observe(
        frame, frame_id=2, captured_at=1.4, input_ready=True
    )
    frozen = builder.observe(frame, frame_id=3, captured_at=1.6, input_ready=True)

    assert first.input_ready.value is True
    assert briefly_same.input_ready.value is True
    assert frozen.input_ready.value is True
    assert frozen.pose.state is EvidenceState.UNKNOWN
    assert frozen.metadata["addon_protocol_fresh"] is False
