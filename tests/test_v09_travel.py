from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vision_bot.controllers.travel import TravelController, TravelPhase, TravelState
from vision_bot.core.commands import CommandKind
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint, PhysicalRoute
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    WorldSnapshot,
)
from vision_bot.v09_config import V09Config


ROOT = Path(__file__).parents[1]
CONFIG = V09Config.from_mapping(
    yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
)


def _evidence(value, frame_id: int, now: float):
    return Evidence.present(
        value,
        observed_at=now,
        frame_id=frame_id,
        source="test",
    )


def _snapshot(
    position: MapPoint,
    heading: float,
    *,
    frame_id: int,
    now: float,
) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=_evidence(True, frame_id, now),
        pose=_evidence(PlayerPose(162, position, heading), frame_id, now),
        life=_evidence(LifeState.ALIVE, frame_id, now),
        modal=_evidence(ModalState.CLEAR, frame_id, now),
        combat=_evidence(CombatView(active=False), frame_id, now),
        mounted=_evidence(True, frame_id, now),
        mining=_evidence(MiningPerception(), frame_id, now),
        loot_pending=_evidence(False, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(RecoveryPerception(), frame_id, now),
    )


def _straight_controller() -> TravelController:
    route = PhysicalRoute(
        [MapPoint(10.0, 10.0), MapPoint(20.0, 10.0), MapPoint(30.0, 10.0)],
        CONFIG.zone,
        loop=False,
    )
    return TravelController(route, CONFIG.navigation)


def test_straight_aligned_route_keeps_forward_held_without_turn() -> None:
    controller = _straight_controller()

    result = controller.reduce(
        TravelState(),
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=1, now=1.0),
        now=1.0,
    )

    assert result.intent is not None
    assert result.reason == "route_follow"
    assert any(
        command.kind is CommandKind.HOLD_KEY and command.key == "W"
        for command in result.intent.commands
    )
    assert not any(
        command.kind is CommandKind.DRAG_RELATIVE
        for command in result.intent.commands
    )


def test_route_progress_never_regresses_on_small_projection_jitter() -> None:
    controller = _straight_controller()
    first = controller.reduce(
        TravelState(),
        _snapshot(MapPoint(15.0, 10.0), 270.0, frame_id=1, now=1.0),
        now=1.0,
    )
    second = controller.reduce(
        first.state,
        _snapshot(MapPoint(14.99, 10.0), 270.0, frame_id=2, now=1.1),
        now=1.1,
    )

    assert first.state.progress_yards is not None
    assert second.state.progress_yards == pytest.approx(first.state.progress_yards)


def test_off_corridor_reentry_uses_minimum_physical_lookahead() -> None:
    controller = _straight_controller()

    result = controller.reduce(
        TravelState(),
        _snapshot(MapPoint(12.0, 11.0), 270.0, frame_id=1, now=1.0),
        now=1.0,
    )

    assert result.projection is not None
    assert result.projection.cross_track_yards > CONFIG.navigation.corridor_radius_yards
    assert result.lookahead_yards == CONFIG.navigation.lookahead_min_yards
    assert result.reason == "route_reentry"


def test_soft_stall_is_one_observed_jump_probe_while_forward_remains_held() -> None:
    controller = _straight_controller()
    first = controller.reduce(
        TravelState(),
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=1, now=0.0),
        now=0.0,
    )
    stalled = controller.reduce(
        first.state,
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=2, now=3.1),
        now=3.1,
    )

    assert stalled.state.phase is TravelPhase.SOFT_STALL
    assert stalled.intent is not None
    assert [command.kind for command in stalled.intent.commands] == [
        CommandKind.HOLD_KEY,
        CommandKind.TAP_KEY,
    ]
    assert stalled.intent.commands[1].key == "SPACE"
    assert not stalled.intent.commands[1].exclusive


def test_hard_stall_recovery_is_split_into_observable_steps() -> None:
    controller = _straight_controller()
    first = controller.reduce(
        TravelState(),
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=1, now=0.0),
        now=0.0,
    )
    hard = controller.reduce(
        first.state,
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=2, now=12.1),
        now=12.1,
    )

    assert hard.state.phase is TravelPhase.HARD_STALL
    assert hard.intent is not None
    assert len(hard.intent.commands) == 1
    assert hard.intent.commands[0].kind is CommandKind.TAP_KEY
    assert hard.intent.commands[0].key == "S"

    observing = controller.reduce(
        hard.state,
        _snapshot(MapPoint(12.0, 10.0), 270.0, frame_id=3, now=12.4),
        now=12.4,
    )
    assert observing.intent is None
    assert observing.reason == "hard_stall_observing_previous_step"


def test_turn_reversal_guard_is_short_and_observation_bounded() -> None:
    controller = _straight_controller()
    state = TravelState(last_turn_sign=1, last_turn_at=1.0)

    guarded_state, guarded = controller._route_motion(state, -30.0, now=1.2)
    _, released = controller._route_motion(
        guarded_state,
        -30.0,
        now=1.0 + CONFIG.navigation.turn_reversal_guard_seconds + 0.01,
    )

    assert guarded.reason == "route_yaw_reversal_hysteresis"
    assert not any(command.kind is CommandKind.DRAG_RELATIVE for command in guarded.commands)
    assert any(command.kind is CommandKind.DRAG_RELATIVE for command in released.commands)
