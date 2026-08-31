from __future__ import annotations

from pathlib import Path

import yaml

from vision_bot.controllers.mounting import MountingController, MountingPhase, MountingState
from vision_bot.core.commands import CommandGroup, CommandKind
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
from vision_bot.engine.supervisor import BotState, ControlDecision, SupervisorMemory
from vision_bot.v09_config import V09Config


ROOT = Path(__file__).parents[1]
CONFIG = V09Config.from_mapping(
    yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
)


def _snapshot(*, mounted: bool, now: float) -> WorldSnapshot:
    evidence = lambda value: Evidence.present(
        value, observed_at=now, frame_id=1, source="test"
    )
    return WorldSnapshot(
        frame_id=1,
        captured_at=now,
        input_ready=evidence(True),
        pose=evidence(PlayerPose(162, MapPoint(50.0, 50.0), 0.0)),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=False)),
        mounted=evidence(mounted),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )


def _decision(previous: BotState = BotState.TRAVEL) -> ControlDecision:
    return ControlDecision(
        SupervisorMemory(state=BotState.MOUNTING, previous_state=previous),
        CommandGroup.MOUNT,
        "test",
    )


def test_mounting_uses_one_nonblocking_cast_and_waits_for_visible_marker() -> None:
    controller = MountingController(CONFIG.mounting)
    cast = controller.reduce(
        MountingState(), _snapshot(mounted=False, now=1.0), _decision(), now=1.0
    )

    assert cast.state.phase is MountingPhase.CASTING
    assert cast.intent is not None
    assert cast.intent.commands[0].kind is CommandKind.TAP_KEY
    assert cast.intent.commands[0].key == "1"

    waiting = controller.reduce(
        cast.state, _snapshot(mounted=False, now=1.5), _decision(), now=1.5
    )
    assert waiting.intent is None
    assert waiting.state.phase is MountingPhase.CASTING

    confirmed = controller.reduce(
        cast.state, _snapshot(mounted=True, now=2.0), _decision(), now=2.0
    )
    assert confirmed.state.phase is MountingPhase.IDLE


def test_mounting_observes_post_combat_settle_before_cast() -> None:
    controller = MountingController(CONFIG.mounting)
    settled = controller.reduce(
        MountingState(),
        _snapshot(mounted=False, now=10.0),
        _decision(BotState.COMBAT),
        now=10.0,
    )

    assert settled.state.phase is MountingPhase.SETTLE
    assert settled.intent is None

    cast = controller.reduce(
        settled.state,
        _snapshot(mounted=False, now=13.0),
        _decision(BotState.COMBAT),
        now=13.0,
    )
    assert cast.intent is not None
