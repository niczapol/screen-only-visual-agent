from __future__ import annotations

from vision_bot.core.commands import CommandGroup
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint, PhysicalRoute, ZoneGeometry
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    MiningSignal,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    WorldSnapshot,
)
from vision_bot.engine.supervisor import BotState, Supervisor, SupervisorMemory


def _present(value, *, frame_id: int = 1, observed_at: float = 10.0):
    return Evidence.present(
        value,
        observed_at=observed_at,
        frame_id=frame_id,
        source="test",
    )


def _absent(*, frame_id: int = 1, observed_at: float = 10.0):
    return Evidence.absent(
        observed_at=observed_at,
        frame_id=frame_id,
        source="test",
    )


def _snapshot(
    *,
    input_ready=True,
    life=LifeState.ALIVE,
    modal=ModalState.CLEAR,
    combat=CombatView(active=False),
    mining=MiningPerception(),
    loot_pending=False,
    forbidden=False,
    mounted=True,
    position=MapPoint(50.0, 50.0),
    now=10.0,
) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=1,
        captured_at=now,
        input_ready=_present(input_ready, observed_at=now),
        pose=_present(PlayerPose(440, position, 90.0), observed_at=now),
        life=_present(life, observed_at=now),
        modal=_present(modal, observed_at=now),
        combat=_present(combat, observed_at=now),
        mounted=_present(mounted, observed_at=now),
        mining=_present(mining, observed_at=now),
        loot_pending=_present(loot_pending, observed_at=now),
        forbidden_subzone=_present(forbidden, observed_at=now),
        recovery=_present(RecoveryPerception(), observed_at=now),
    )


def test_supervisor_defaults_to_one_travel_owner() -> None:
    decision = Supervisor().reduce(SupervisorMemory(), _snapshot(), now=10.0)

    assert decision.state is BotState.TRAVEL
    assert decision.owner is CommandGroup.TRAVEL
    assert CommandGroup.COMBAT in decision.preempt_groups


def test_supervisor_death_preempts_combat_and_mining() -> None:
    snapshot = _snapshot(
        life=LifeState.GHOST,
        combat=CombatView(active=True),
        mining=MiningPerception(signal=MiningSignal.APPROACH),
    )

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.DEATH_RECOVERY
    assert decision.owner is CommandGroup.RECOVERY
    assert CommandGroup.COMBAT in decision.preempt_groups
    assert CommandGroup.MINING in decision.preempt_groups


def test_supervisor_combat_preempts_mining() -> None:
    snapshot = _snapshot(
        combat=CombatView(active=True, attacker_count=2),
        mining=MiningPerception(signal=MiningSignal.APPROACH),
    )

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.COMBAT
    assert decision.owner is CommandGroup.COMBAT


def test_supervisor_keeps_mounted_escape_until_timeout() -> None:
    supervisor = Supervisor(mounted_escape_seconds=12.0, mounted_escape_attacker_count=2)
    snapshot = _snapshot(combat=CombatView(active=True, attacker_count=1))

    first = supervisor.reduce(SupervisorMemory(), snapshot, now=10.0)
    before_timeout = supervisor.reduce(
        first.memory,
        _snapshot(combat=CombatView(active=True, attacker_count=1), now=21.9),
        now=21.9,
    )
    handoff = supervisor.reduce(
        first.memory,
        _snapshot(combat=CombatView(active=True, attacker_count=1), now=22.0),
        now=22.0,
    )

    assert first.state is BotState.HAZARD_EGRESS
    assert first.owner is CommandGroup.TRAVEL
    assert before_timeout.owner is CommandGroup.TRAVEL
    assert handoff.state is BotState.COMBAT
    assert handoff.owner is CommandGroup.COMBAT


def test_supervisor_elevated_threat_skips_mounted_escape() -> None:
    snapshot = _snapshot(combat=CombatView(active=True, attacker_count=2))

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.COMBAT
    assert decision.owner is CommandGroup.COMBAT


def test_supervisor_combat_remains_operable_when_pose_protocol_is_unknown() -> None:
    snapshot = _snapshot(combat=CombatView(active=True, attacker_count=1))
    snapshot = WorldSnapshot(
        **{
            **snapshot.__dict__,
            "pose": Evidence.unknown(
                observed_at=10.0,
                frame_id=1,
                source="runtime_telemetry",
                reason="telemetry_not_decoded",
            ),
        }
    )

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.COMBAT
    assert decision.owner is CommandGroup.COMBAT
    assert decision.reason == "combat_confirmed"


def test_supervisor_pauses_normal_route_when_pose_protocol_is_unknown() -> None:
    snapshot = _snapshot()
    snapshot = WorldSnapshot(
        **{
            **snapshot.__dict__,
            "pose": Evidence.unknown(
                observed_at=10.0,
                frame_id=1,
                source="runtime_telemetry",
                reason="telemetry_not_decoded",
            ),
        }
    )

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.PAUSED
    assert decision.owner is None
    assert decision.reason == "pose_unknown_or_stale_outside_combat"


def test_supervisor_uses_configured_emergency_hp_threshold() -> None:
    supervisor = Supervisor(emergency_hp_fraction=0.30)
    snapshot = _snapshot(
        combat=CombatView(active=True, hp_fraction=0.25, attacker_count=1),
    )

    decision = supervisor.reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.COMBAT
    assert decision.reason == "combat_critical_heal_priority"


def test_supervisor_mounts_before_normal_travel() -> None:
    decision = Supervisor().reduce(
        SupervisorMemory(),
        _snapshot(mounted=False),
        now=10.0,
    )

    assert decision.state is BotState.MOUNTING
    assert decision.owner is CommandGroup.MOUNT


def test_supervisor_pauses_off_corridor_instead_of_blind_route_reentry() -> None:
    geometry = ZoneGeometry(zone_id=440, width_yards=1000.0, height_yards=1000.0)
    route = PhysicalRoute(
        (MapPoint(40.0, 50.0), MapPoint(60.0, 50.0)),
        geometry,
        loop=False,
    )
    supervisor = Supervisor(
        route=route,
        corridor_radius_yards=18.0,
    )

    decision = supervisor.reduce(
        SupervisorMemory(),
        _snapshot(position=MapPoint(50.0, 55.0)),
        now=10.0,
    )

    assert decision.state is BotState.PAUSED
    assert decision.owner is None
    assert decision.reason == "outside_route_corridor_requires_entry_plan"


def test_active_mining_transaction_may_return_to_route_from_outside_corridor() -> None:
    geometry = ZoneGeometry(zone_id=440, width_yards=1000.0, height_yards=1000.0)
    route = PhysicalRoute(
        (MapPoint(40.0, 50.0), MapPoint(60.0, 50.0)),
        geometry,
        loop=False,
    )
    decision = Supervisor(
        route=route,
        corridor_radius_yards=18.0,
    ).reduce(
        SupervisorMemory(),
        _snapshot(position=MapPoint(50.0, 55.0)),
        now=10.0,
        active_groups=frozenset({CommandGroup.MINING}),
    )

    assert decision.state is BotState.MINING
    assert decision.owner is CommandGroup.MINING


def test_supervisor_fails_closed_when_input_evidence_is_stale() -> None:
    snapshot = _snapshot()
    stale = Evidence.present(
        True,
        observed_at=8.0,
        frame_id=0,
        source="window",
    )
    snapshot = WorldSnapshot(
        **{
            **snapshot.__dict__,
            "input_ready": stale,
        }
    )

    decision = Supervisor().reduce(SupervisorMemory(), snapshot, now=10.0)

    assert decision.state is BotState.WAIT_FOR_CLIENT
    assert decision.owner is None
    assert set(decision.preempt_groups) == set(CommandGroup)
