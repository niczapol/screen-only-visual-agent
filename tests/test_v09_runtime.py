from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vision_bot.controllers.combat import CombatController
from vision_bot.controllers.travel import TravelController
from vision_bot.core.commands import CommandGroup, MouseButton
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
from vision_bot.engine.input_executor import InputExecutor
from vision_bot.engine.kernel import DeterministicKernel
from vision_bot.engine.runtime import V09Runtime
from vision_bot.engine.supervisor import Supervisor
from vision_bot.v09_config import V09Config


ROOT = Path(__file__).parents[1]
CONFIG = V09Config.from_mapping(
    yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
)


@dataclass
class FakeBackend:
    calls: list[tuple] = field(default_factory=list)

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def move_cursor(self, x: int, y: int) -> None:
        self.calls.append(("move_cursor", x, y))

    def mouse_down(self, button: MouseButton) -> None:
        self.calls.append(("mouse_down", button.value))

    def mouse_up(self, button: MouseButton) -> None:
        self.calls.append(("mouse_up", button.value))

    def move_mouse_relative(self, delta_x: int, delta_y: int) -> None:
        self.calls.append(("mouse_move_relative", delta_x, delta_y))


def _evidence(value, frame_id: int, now: float):
    return Evidence.present(
        value,
        observed_at=now,
        frame_id=frame_id,
        source="test",
    )


def _snapshot(
    *,
    frame_id: int,
    now: float,
    combat: CombatView,
    life: LifeState = LifeState.ALIVE,
) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=_evidence(True, frame_id, now),
        pose=_evidence(
            PlayerPose(162, MapPoint(12.0, 10.0), 270.0),
            frame_id,
            now,
        ),
        life=_evidence(life, frame_id, now),
        modal=_evidence(ModalState.CLEAR, frame_id, now),
        combat=_evidence(combat, frame_id, now),
        mounted=_evidence(True, frame_id, now),
        mining=_evidence(MiningPerception(), frame_id, now),
        loot_pending=_evidence(False, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(RecoveryPerception(), frame_id, now),
    )


def _kernel() -> DeterministicKernel:
    route = PhysicalRoute(
        [MapPoint(10.0, 10.0), MapPoint(20.0, 10.0), MapPoint(30.0, 10.0)],
        CONFIG.zone,
        loop=False,
    )
    return DeterministicKernel(
        Supervisor(),
        {
            CommandGroup.TRAVEL: TravelController(route, CONFIG.navigation),
            CommandGroup.COMBAT: CombatController(CONFIG.combat, CONFIG.navigation),
        },
    )


def test_runtime_preempts_travel_before_combat_key() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    runtime = V09Runtime(_kernel(), executor=executor, live_input_enabled=True)

    runtime.step(
        _snapshot(frame_id=1, now=1.0, combat=CombatView(active=False)),
        now=1.0,
    )
    assert backend.calls == [("key_down", "W")]

    runtime.step(
        _snapshot(
            frame_id=2,
            now=1.1,
            combat=CombatView(active=True, hp_fraction=1.0, attacker_count=2),
        ),
        now=1.1,
    )

    assert ("key_up", "W") in backend.calls
    assert backend.calls[-1] == ("key_down", "X")
    assert executor.held_keys == {"X"}


def test_replay_runtime_never_executes_intents() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    runtime = V09Runtime(_kernel(), executor=executor, live_input_enabled=False)

    step = runtime.step(
        _snapshot(frame_id=1, now=1.0, combat=CombatView(active=False)),
        now=1.0,
    )

    assert step.kernel.intent is not None
    assert not step.live_input_enabled
    assert backend.calls == []


def test_runtime_scope_blocks_travel_but_allows_combat() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    runtime = V09Runtime(_kernel(), executor=executor, live_input_enabled=True)
    shutdown_scope = frozenset(
        {CommandGroup.RECOVERY, CommandGroup.COMBAT, CommandGroup.LOOT}
    )

    travel = runtime.step(
        _snapshot(frame_id=1, now=1.0, combat=CombatView(active=False)),
        now=1.0,
        permitted_groups=shutdown_scope,
    )

    assert travel.kernel.decision.owner is CommandGroup.TRAVEL
    assert not travel.live_input_enabled
    assert backend.calls == []

    combat = runtime.step(
        _snapshot(
            frame_id=2,
            now=1.1,
            combat=CombatView(active=True, hp_fraction=1.0, attacker_count=2),
        ),
        now=1.1,
        permitted_groups=shutdown_scope,
    )

    assert combat.kernel.decision.owner is CommandGroup.COMBAT
    assert combat.live_input_enabled
    assert backend.calls[-1] == ("key_down", "X")


def test_death_preempts_scheduled_emergency_V_followup() -> None:
    backend = FakeBackend()
    executor = InputExecutor(backend)
    runtime = V09Runtime(_kernel(), executor=executor, live_input_enabled=True)

    runtime.step(
        _snapshot(
            frame_id=1,
            now=10.0,
            combat=CombatView(active=True, hp_fraction=0.20),
        ),
        now=10.0,
    )
    assert backend.calls[-1] == ("key_down", "F")

    runtime.step(
        _snapshot(
            frame_id=2,
            now=10.06,
            combat=CombatView(active=False),
            life=LifeState.GHOST,
        ),
        now=10.06,
    )
    runtime.step(
        _snapshot(
            frame_id=3,
            now=10.30,
            combat=CombatView(active=False),
            life=LifeState.GHOST,
        ),
        now=10.30,
    )

    assert ("key_down", "V") not in backend.calls
    assert not executor.held_keys
