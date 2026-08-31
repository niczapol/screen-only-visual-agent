from __future__ import annotations

from dataclasses import replace

import numpy as np
import yaml

from vision_bot.controllers.combat import CombatControllerState
from vision_bot.controllers.mining_approach import (
    MiningApproachPhase,
    MiningApproachState,
)
from vision_bot.controllers.mounting import MountingPhase, MountingState
from vision_bot.controllers.recovery import (
    RecoveryControllerState,
    RecoveryPhase,
)
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
from vision_bot.engine.factory import build_v09_kernel
from vision_bot.engine.orchestrator import RuntimeEventLog, V09FrameOrchestrator
from vision_bot.engine.runtime import V09Runtime
from vision_bot.perception.mining_world import MiningWorldSensorResult
from vision_bot.replay.episode import read_episode


class _SnapshotBuilder:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.observe_kwargs = []

    def observe(self, _frame, **kwargs):
        self.observe_kwargs.append(dict(kwargs))
        return replace(
            self.snapshot,
            frame_id=kwargs["frame_id"],
            captured_at=kwargs["captured_at"],
        )


class _MiningSensor:
    def __init__(self):
        self.phase = None
        self.reset_count = 0

    def observe(self, _frame, perception, **kwargs):
        self.phase = kwargs["phase"]
        return MiningWorldSensorResult(perception, "test", 0.0, None)

    def close(self):
        pass

    def reset(self):
        self.reset_count += 1


class _RecoverySensor:
    def __init__(self):
        self.phase = None

    def observe(self, _frame, **kwargs):
        self.phase = kwargs["phase"]
        return RecoveryPerception()


def test_orchestrator_enriches_one_snapshot_before_one_kernel_step(tmp_path) -> None:
    project_config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bundle = build_v09_kernel(project_config)
    evidence = lambda value: Evidence.present(
        value, observed_at=1.0, frame_id=1, source="test"
    )
    base = WorldSnapshot(
        frame_id=1,
        captured_at=1.0,
        input_ready=evidence(False),
        pose=Evidence.unknown(
            observed_at=1.0, frame_id=1, source="test", reason="test_unknown"
        ),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=False)),
        mounted=evidence(False),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )
    mining_sensor = _MiningSensor()
    recovery_sensor = _RecoverySensor()
    episode_path = tmp_path / "shadow.jsonl"
    orchestrator = V09FrameOrchestrator(
        project_config,
        bundle,
        V09Runtime(bundle.kernel),
        snapshot_builder=_SnapshotBuilder(base),
        mining_sensor=mining_sensor,
        recovery_sensor=recovery_sensor,
        event_log=RuntimeEventLog(
            episode_path,
            name="orchestrator-test",
            config_fingerprint="abc",
        ),
    )
    try:
        outcome = orchestrator.process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=7,
            captured_at=2.0,
            input_ready=False,
            cursor_client_point=None,
        )
    finally:
        orchestrator.close()

    assert outcome.snapshot.frame_id == 7
    assert outcome.runtime.kernel.frame_id == 7
    assert outcome.snapshot.mining.source == "v09_mining_fused_sensor"
    assert outcome.snapshot.recovery.source == "v09_recovery_sensor"
    assert mining_sensor.phase is MiningApproachPhase.IDLE
    assert recovery_sensor.phase is RecoveryPhase.IDLE
    assert not outcome.runtime.live_input_enabled
    episode = read_episode(episode_path)
    assert episode.manifest.name == "orchestrator-test"
    assert episode.manifest.config_fingerprint == "abc"
    assert [item.frame_id for item in episode.snapshots] == [7]


def test_orchestrator_fuses_visible_pose_with_physical_hazard_guard() -> None:
    project_config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bundle = build_v09_kernel(project_config)
    evidence = lambda value: Evidence.present(
        value, observed_at=1.0, frame_id=1, source="test"
    )
    base = WorldSnapshot(
        frame_id=1,
        captured_at=1.0,
        input_ready=evidence(True),
        pose=evidence(PlayerPose(162, MapPoint(30.65, 73.82), 0.0)),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=False)),
        mounted=evidence(True),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )
    orchestrator = V09FrameOrchestrator(
        project_config,
        bundle,
        V09Runtime(bundle.kernel),
        snapshot_builder=_SnapshotBuilder(base),
        mining_sensor=_MiningSensor(),
        recovery_sensor=_RecoverySensor(),
    )
    try:
        outcome = orchestrator.process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=2,
            captured_at=1.1,
            input_ready=True,
            cursor_client_point=None,
        )
    finally:
        orchestrator.close()

    assert outcome.snapshot.forbidden_subzone.value is True
    assert outcome.snapshot.forbidden_subzone.source == "v09_physical_hazard_guard"
    assert "southwest_wreckage_contact" in outcome.snapshot.metadata[
        "physical_hazard_kinds"
    ]


def test_orchestrator_clears_completed_recovery_and_ended_combat_state() -> None:
    project_config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bundle = build_v09_kernel(project_config)
    evidence = lambda value: Evidence.present(
        value, observed_at=2.0, frame_id=1, source="test"
    )
    base = WorldSnapshot(
        frame_id=1,
        captured_at=2.0,
        input_ready=evidence(True),
        pose=evidence(PlayerPose(162, bundle.route.points[0], 0.0)),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=False)),
        mounted=evidence(True),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )
    recovery = bundle.controllers[CommandGroup.RECOVERY]
    combat = bundle.controllers[CommandGroup.COMBAT]
    recovery.state = RecoveryControllerState(phase=RecoveryPhase.ACCEPT_RESURRECTION)
    combat.state = CombatControllerState(engaged=True, entered_at=1.0)
    orchestrator = V09FrameOrchestrator(
        project_config,
        bundle,
        V09Runtime(bundle.kernel),
        snapshot_builder=_SnapshotBuilder(base),
        mining_sensor=_MiningSensor(),
        recovery_sensor=_RecoverySensor(),
    )
    try:
        orchestrator.process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=2,
            captured_at=2.0,
            input_ready=True,
            cursor_client_point=None,
        )
    finally:
        orchestrator.close()

    assert recovery.state.phase is RecoveryPhase.IDLE
    assert not combat.state.engaged


def test_orchestrator_aborts_mining_and_mounting_when_combat_interrupts() -> None:
    project_config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bundle = build_v09_kernel(project_config)
    evidence = lambda value: Evidence.present(
        value, observed_at=3.0, frame_id=1, source="test"
    )
    base = WorldSnapshot(
        frame_id=1,
        captured_at=3.0,
        input_ready=evidence(True),
        pose=evidence(PlayerPose(162, bundle.route.points[0], 0.0)),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=True, attacker_count=2)),
        mounted=evidence(False),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )
    mining = bundle.controllers[CommandGroup.MINING]
    mounting = bundle.controllers[CommandGroup.MOUNT]
    mining.state = MiningApproachState(
        phase=MiningApproachPhase.VERIFY,
        clicked_at=2.0,
    )
    mounting.state = MountingState(
        phase=MountingPhase.CASTING,
        attempts=1,
        phase_entered_at=2.0,
    )
    mining_sensor = _MiningSensor()
    orchestrator = V09FrameOrchestrator(
        project_config,
        bundle,
        V09Runtime(bundle.kernel),
        snapshot_builder=_SnapshotBuilder(base),
        mining_sensor=mining_sensor,
        recovery_sensor=_RecoverySensor(),
    )
    try:
        orchestrator.process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=2,
            captured_at=3.0,
            input_ready=True,
            cursor_client_point=None,
        )
    finally:
        orchestrator.close()

    assert mining.state.phase is MiningApproachPhase.IDLE
    assert mounting.state.phase is MountingPhase.IDLE
    assert mining_sensor.reset_count == 1


def test_orchestrator_throttles_only_idle_ore_scans() -> None:
    project_config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    project_config["v09"]["performance"]["idle_ore_scan_interval_frames"] = 2
    bundle = build_v09_kernel(project_config)
    evidence = lambda value: Evidence.present(
        value, observed_at=4.0, frame_id=1, source="test"
    )
    base = WorldSnapshot(
        frame_id=1,
        captured_at=4.0,
        input_ready=evidence(True),
        pose=evidence(PlayerPose(162, bundle.route.points[0], 0.0)),
        life=evidence(LifeState.ALIVE),
        modal=evidence(ModalState.CLEAR),
        combat=evidence(CombatView(active=False)),
        mounted=evidence(True),
        mining=evidence(MiningPerception()),
        loot_pending=evidence(False),
        forbidden_subzone=evidence(False),
        recovery=evidence(RecoveryPerception()),
    )
    snapshot_builder = _SnapshotBuilder(base)
    orchestrator = V09FrameOrchestrator(
        project_config,
        bundle,
        V09Runtime(bundle.kernel),
        snapshot_builder=snapshot_builder,
        mining_sensor=_MiningSensor(),
        recovery_sensor=_RecoverySensor(),
    )
    try:
        for frame_id in (1, 2):
            orchestrator.process_frame(
                np.zeros((10, 10, 3), dtype=np.uint8),
                frame_id=frame_id,
                captured_at=4.0 + frame_id / 10,
                input_ready=True,
                cursor_client_point=None,
            )
        orchestrator.mining_controller.state = MiningApproachState(
            phase=MiningApproachPhase.CONTINUOUS_APPROACH,
            started_at=4.0,
        )
        orchestrator.process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=3,
            captured_at=4.3,
            input_ready=True,
            cursor_client_point=None,
        )
    finally:
        orchestrator.close()

    assert [call["recognize_ore"] for call in snapshot_builder.observe_kwargs] == [
        True,
        False,
        True,
    ]
