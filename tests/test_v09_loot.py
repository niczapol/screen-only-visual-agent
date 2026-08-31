from __future__ import annotations

from pathlib import Path

import yaml

from vision_bot.controllers.loot import LootController
from vision_bot.core.commands import CommandGroup
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
from vision_bot.engine.supervisor import Supervisor, SupervisorMemory


ROOT = Path(__file__).parents[1]
PROJECT_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _evidence(value, frame_id: int, now: float):
    return Evidence.present(
        value,
        observed_at=now,
        frame_id=frame_id,
        source="test",
    )


def _snapshot(
    combat: CombatView,
    *,
    loot_pending: bool,
    frame_id: int = 1,
    now: float = 1.0,
) -> WorldSnapshot:
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
        loot_pending=_evidence(loot_pending, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(RecoveryPerception(), frame_id, now),
    )


def test_loot_reuses_accepted_F9_then_G_state_machine_as_intents() -> None:
    controller = LootController(PROJECT_CONFIG)
    first_snapshot = _snapshot(CombatView(active=False), loot_pending=True)
    first_owner = Supervisor().reduce(SupervisorMemory(), first_snapshot, now=1.0)

    first = controller.plan(first_snapshot, first_owner, now=1.0)

    assert first is not None
    assert first.owner is CommandGroup.LOOT
    assert first.commands[0].key == "F9"

    second_snapshot = _snapshot(
        CombatView(active=False, dead_hostile_target=True),
        loot_pending=True,
        frame_id=2,
        now=1.4,
    )
    second_owner = Supervisor().reduce(
        first_owner.memory,
        second_snapshot,
        now=1.4,
        active_groups=frozenset({CommandGroup.LOOT}),
    )
    second = controller.plan(second_snapshot, second_owner, now=1.4)

    assert second is not None
    assert second.commands[0].key == "G"


def test_loot_preempts_noncritical_combat_but_not_critical_FV() -> None:
    supervisor = Supervisor()
    ordinary = supervisor.reduce(
        SupervisorMemory(),
        _snapshot(
            CombatView(active=True, hp_fraction=0.80),
            loot_pending=True,
        ),
        now=1.0,
    )
    critical = supervisor.reduce(
        SupervisorMemory(),
        _snapshot(
            CombatView(active=True, hp_fraction=0.20),
            loot_pending=True,
        ),
        now=1.0,
    )

    assert ordinary.owner is CommandGroup.LOOT
    assert critical.owner is CommandGroup.COMBAT
    assert critical.reason == "combat_critical_heal_priority"
