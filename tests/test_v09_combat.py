from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vision_bot.controllers.combat import CombatController, CombatControllerState
from vision_bot.core.commands import CommandKind
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint
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


def _snapshot(combat: CombatView, *, frame_id: int, now: float) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=_evidence(True, frame_id, now),
        pose=_evidence(
            PlayerPose(162, MapPoint(50.0, 50.0), 180.0),
            frame_id,
            now,
        ),
        life=_evidence(LifeState.ALIVE, frame_id, now),
        modal=_evidence(ModalState.CLEAR, frame_id, now),
        combat=_evidence(combat, frame_id, now),
        mounted=_evidence(False, frame_id, now),
        mining=_evidence(MiningPerception(), frame_id, now),
        loot_pending=_evidence(False, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(RecoveryPerception(), frame_id, now),
    )


def _controller() -> CombatController:
    return CombatController(CONFIG.combat, CONFIG.navigation)


def test_combat_entry_runs_X_before_E() -> None:
    controller = _controller()
    result = controller.reduce(
        CombatControllerState(),
        _snapshot(CombatView(active=True, hp_fraction=1.0), frame_id=1, now=1.0),
        now=1.0,
    )

    assert result.action == "priority_X"
    assert result.intent is not None
    assert [command.key for command in result.intent.commands] == ["X"]

    attack = controller.reduce(
        result.state,
        _snapshot(CombatView(active=True, hp_fraction=1.0), frame_id=2, now=1.1),
        now=1.1,
    )
    assert attack.action == "attack_E"
    assert attack.intent is not None
    assert [command.key for command in attack.intent.commands] == ["E"]


def test_Q_is_impossible_before_outgoing_hit_lock() -> None:
    controller = _controller()
    state = CombatControllerState(
        engaged=True,
        entered_at=0.0,
        last_priority_at=0.0,
        last_attack_at=0.0,
    )

    result = controller.reduce(
        state,
        _snapshot(CombatView(active=True), frame_id=1, now=1.0),
        now=1.0,
    )

    assert result.action == "attack_E"
    assert result.intent is not None
    assert all(command.key != "Q" for command in result.intent.commands)


def test_fresh_hit_locks_bearing_and_schedules_Q() -> None:
    controller = _controller()
    state = CombatControllerState(
        engaged=True,
        entered_at=0.0,
        last_priority_at=0.0,
        last_attack_at=0.9,
    )

    result = controller.reduce(
        state,
        _snapshot(
            CombatView(active=True, outgoing_hit=True, outcome_sequence=4),
            frame_id=1,
            now=1.0,
        ),
        now=1.0,
    )

    assert result.action == "aligned_Q"
    assert result.state.bearing_locked_at == 1.0
    assert result.intent is not None
    assert [command.key for command in result.intent.commands] == ["Q"]


def test_critical_F_schedules_V_after_released_key_gap() -> None:
    controller = _controller()

    result = controller.reduce(
        CombatControllerState(),
        _snapshot(CombatView(active=True, hp_fraction=0.20), frame_id=1, now=10.0),
        now=10.0,
    )

    assert result.action == "emergency_F_V"
    assert result.intent is not None
    first, followup = result.intent.commands
    assert first.key == "F"
    assert followup.key == "V"
    assert followup.not_before - (first.not_before + first.duration) == pytest.approx(0.12)
    assert all(command.exclusive for command in result.intent.commands)

    waiting = controller.reduce(
        result.state,
        _snapshot(CombatView(active=True, hp_fraction=0.20), frame_id=2, now=10.10),
        now=10.10,
    )
    assert waiting.intent is None
    assert waiting.action == "awaiting_emergency_V_followup"


def test_confirmed_out_of_range_holds_forward_without_attack_overlap() -> None:
    controller = _controller()
    state = CombatControllerState(
        engaged=True,
        entered_at=0.0,
        last_priority_at=0.0,
        last_attack_at=0.9,
        previous_range_visible=False,
    )

    result = controller.reduce(
        state,
        _snapshot(
            CombatView(active=True, out_of_range=True, outcome_sequence=2),
            frame_id=1,
            now=1.0,
        ),
        now=1.0,
    )

    assert result.action == "range_approach"
    assert result.intent is not None
    assert len(result.intent.commands) == 1
    assert result.intent.commands[0].kind is CommandKind.HOLD_KEY
    assert result.intent.commands[0].key == "W"


def test_wrong_facing_uses_one_bounded_rmb_drag() -> None:
    controller = _controller()
    state = CombatControllerState(
        engaged=True,
        entered_at=0.0,
        last_priority_at=0.0,
        last_attack_at=0.9,
    )

    result = controller.reduce(
        state,
        _snapshot(
            CombatView(active=True, wrong_facing=True, outcome_sequence=3),
            frame_id=1,
            now=1.0,
        ),
        now=1.0,
    )

    assert result.action == "face_search"
    assert result.intent is not None
    assert len(result.intent.commands) == 1
    assert result.intent.commands[0].kind is CommandKind.DRAG_RELATIVE
    assert result.intent.commands[0].duration == CONFIG.combat.face_search_drag_seconds
