from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np

from vision_bot.controllers.mining_approach import (
    MiningApproachController,
    MiningApproachPhase,
)
from vision_bot.controllers.recovery import RecoveryController
from vision_bot.core.commands import CommandGroup
from vision_bot.core.evidence import Evidence
from vision_bot.core.world import LifeState, MiningPerception, WorldSnapshot
from vision_bot.death_recovery import (
    read_resurrection_sickness_remaining,
    write_resurrection_sickness_state,
)
from vision_bot.config import runtime_state_path
from vision_bot.engine.factory import V09KernelBundle
from vision_bot.engine.runtime import RuntimeStep, V09Runtime
from vision_bot.perception.mining_world import MiningWorldSensor, MiningWorldSensorResult
from vision_bot.perception.recovery import RecoverySensor
from vision_bot.perception.snapshot_builder import SnapshotBuilder
from vision_bot.replay.codec import snapshot_to_dict


@dataclass(frozen=True)
class FrameOutcome:
    snapshot: WorldSnapshot
    runtime: RuntimeStep
    mining_sensor: MiningWorldSensorResult


class RuntimeEventLog:
    """Append-only decision log suitable for replay triage."""

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "v09-runtime",
        source: str = "v09-shadow",
        config_fingerprint: str | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = destination.open("w", encoding="utf-8", newline="\n")
        self._handle.write(
            json.dumps(
                {
                    "record": "manifest",
                    "name": name,
                    "schema_version": 1,
                    "source": source,
                    "config_fingerprint": config_fingerprint,
                },
                sort_keys=True,
            )
            + "\n"
        )

    def close(self) -> None:
        self._handle.close()

    def write(self, outcome: FrameOutcome) -> None:
        runtime = outcome.runtime
        step = runtime.kernel
        record = {
            "record": "snapshot",
            "snapshot": snapshot_to_dict(outcome.snapshot),
            "runtime": {
                "schema": 1,
                "frame_id": outcome.snapshot.frame_id,
                "captured_at": outcome.snapshot.captured_at,
                "state": step.decision.state.value,
                "owner": step.decision.owner.value if step.decision.owner else None,
                "reason": step.decision.reason,
                "intent": step.intent.reason if step.intent else None,
                "input_enabled": runtime.live_input_enabled,
                "executed_events": sum(event.executed for event in runtime.input_events),
                "pending_input_actions": runtime.pending_input_actions,
                "held_keys": runtime.held_keys,
                "held_buttons": runtime.held_buttons,
                "actuation_delay_seconds": runtime.actuation_delay_seconds,
                "permitted_groups": runtime.permitted_groups,
                "input_events": [
                    {
                        "operation": event.operation,
                        "value": event.value,
                        "reason": event.reason,
                        "executed": event.executed,
                    }
                    for event in runtime.input_events
                ],
                "mining_detector_reason": outcome.mining_sensor.detector_reason,
                "mining_detection_frame_id": outcome.mining_sensor.detection_frame_id,
            },
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()


class V09FrameOrchestrator:
    """One-frame composition root for all v0.9 perception and control.

    Every controller consumes evidence derived from the same captured frame.
    The only asynchronous component, world-ore inference, is a proposal sensor
    and cannot independently authorize an interaction.
    """

    def __init__(
        self,
        project_config: dict,
        bundle: V09KernelBundle,
        runtime: V09Runtime,
        *,
        snapshot_builder: SnapshotBuilder | None = None,
        mining_sensor: MiningWorldSensor | None = None,
        recovery_sensor: RecoverySensor | None = None,
        event_log: RuntimeEventLog | None = None,
    ) -> None:
        self.project_config = project_config
        self.bundle = bundle
        self.runtime = runtime
        self.snapshot_builder = snapshot_builder or SnapshotBuilder(
            project_config,
            zone_id=bundle.config.zone.zone_id,
        )
        detector_config = project_config.get("mining", {}).get("ore_world_detector", {})
        mining_config = project_config.get("mining", {})
        self.mining_sensor = mining_sensor or MiningWorldSensor(
            project_config,
            result_max_age_seconds=bundle.config.freshness.heavy_sensor_seconds,
            refresh_interval_seconds=float(
                detector_config.get("refresh_interval_seconds", 0.75)
            ),
            gather_wait_seconds=float(mining_config.get("gather_wait_seconds", 3.0)),
            clear_frames_required=int(mining_config.get("verify_clear_frames", 2)),
        )
        self.recovery_sensor = recovery_sensor or RecoverySensor(project_config)
        self.event_log = event_log
        mining = bundle.controllers.get(CommandGroup.MINING)
        recovery = bundle.controllers.get(CommandGroup.RECOVERY)
        if not isinstance(mining, MiningApproachController):
            raise TypeError("v0.9 bundle requires MiningApproachController")
        if not isinstance(recovery, RecoveryController):
            raise TypeError("v0.9 bundle requires RecoveryController")
        self.mining_controller = mining
        self.recovery_controller = recovery
        recovery_cfg = project_config.get("safety", {}).get("death_recovery", {})
        self._sickness_enabled = bool(
            runtime.live_input_enabled
            and recovery_cfg.get("resurrect_sickness_wait_enabled", True)
        )
        self._sickness_seconds = float(
            recovery_cfg.get("resurrect_sickness_wait_seconds", 600.0)
        )
        self._sickness_path = runtime_state_path(
            str(
                recovery_cfg.get(
                    "resurrection_sickness_state_path",
                    "data/runtime/resurrection_sickness.json",
                )
            )
        )
        self._sickness_recorded_for_recovery = False

    def close(self) -> None:
        self.mining_sensor.close()
        if self.event_log is not None:
            self.event_log.close()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        captured_at: float,
        input_ready: bool,
        cursor_client_point: tuple[int, int] | None,
        minimap: np.ndarray | None = None,
        permitted_groups: frozenset[CommandGroup] | None = None,
    ) -> FrameOutcome:
        snapshot = self.snapshot_builder.observe(
            frame,
            frame_id=frame_id,
            captured_at=captured_at,
            input_ready=input_ready,
            minimap=minimap,
            recognize_ore=(
                self.mining_controller.active
                or (frame_id - 1)
                % self.bundle.config.performance.idle_ore_scan_interval_frames
                == 0
            ),
        )
        snapshot = self._apply_resurrection_sickness(snapshot, captured_at=captured_at)
        self._synchronize_controller_lifecycles(snapshot)
        pose = snapshot.pose.value
        if pose is not None:
            hazard_kinds = self.bundle.hazards.containing_kinds(pose.position)
            if hazard_kinds:
                snapshot = replace(
                    snapshot,
                    forbidden_subzone=Evidence.present(
                        True,
                        observed_at=captured_at,
                        frame_id=frame_id,
                        source="v09_physical_hazard_guard",
                        reason=",".join(hazard_kinds),
                    ),
                    metadata={
                        **dict(snapshot.metadata),
                        "physical_hazard_kinds": hazard_kinds,
                    },
                )
        recovery_value = self.recovery_sensor.observe(
            frame,
            life=snapshot.life.value,
            phase=self.recovery_controller.state.phase,
        )
        snapshot = replace(
            snapshot,
            recovery=Evidence.present(
                recovery_value,
                observed_at=captured_at,
                frame_id=frame_id,
                source="v09_recovery_sensor",
            ),
        )
        mining_state = self.mining_controller.state
        combat = snapshot.combat.value
        filtered_mining = self.mining_controller.filter_perception(
            snapshot.mining.value or MiningPerception(),
            now=captured_at,
        )
        mining_result = self.mining_sensor.observe(
            frame,
            filtered_mining,
            phase=mining_state.phase,
            frame_id=frame_id,
            now=captured_at,
            cursor_client_point=cursor_client_point,
            target_marker_point=mining_state.pending_marker_point,
            clicked_at=mining_state.clicked_at,
            combat=combat,
        )
        snapshot = replace(
            snapshot,
            mining=Evidence.present(
                mining_result.perception,
                observed_at=captured_at,
                frame_id=frame_id,
                source="v09_mining_fused_sensor",
            ),
        )
        runtime_step = self.runtime.step(
            snapshot,
            now=captured_at,
            actuation_now=(
                time.monotonic()
                if self.runtime.live_input_enabled
                else captured_at
            ),
            permitted_groups=permitted_groups,
        )
        outcome = FrameOutcome(snapshot, runtime_step, mining_result)
        if self.event_log is not None:
            self.event_log.write(outcome)
        return outcome

    def _synchronize_controller_lifecycles(self, snapshot: WorldSnapshot) -> None:
        """Abort transactions whose visible world preconditions ended."""

        life = snapshot.life.value
        combat = snapshot.combat.value
        combat_active = bool(combat is not None and combat.active)

        if life not in {LifeState.DEAD, LifeState.GHOST} and self.recovery_controller.active:
            self.recovery_controller.reset()

        mining_interrupted = bool(life is not LifeState.ALIVE or combat_active)
        if mining_interrupted and self.mining_controller.state.phase is not MiningApproachPhase.IDLE:
            self.mining_controller.reset()
            reset_sensor = getattr(self.mining_sensor, "reset", None)
            if callable(reset_sensor):
                reset_sensor()

        combat_controller = self.bundle.controllers.get(CommandGroup.COMBAT)
        if (
            combat_controller is not None
            and bool(getattr(combat_controller, "active", False))
            and (life is not LifeState.ALIVE or not combat_active)
        ):
            combat_controller.reset()

        if life is not LifeState.ALIVE or combat_active:
            for group in (CommandGroup.MOUNT, CommandGroup.LOOT):
                controller = self.bundle.controllers.get(group)
                if controller is not None and bool(getattr(controller, "active", False)):
                    controller.reset()

    def _apply_resurrection_sickness(
        self,
        snapshot: WorldSnapshot,
        *,
        captured_at: float,
    ) -> WorldSnapshot:
        if not self._sickness_enabled or snapshot.life.value is not LifeState.ALIVE:
            return snapshot
        if (
            self.recovery_controller.active
            and not self._sickness_recorded_for_recovery
        ):
            write_resurrection_sickness_state(
                self._sickness_path,
                self._sickness_seconds,
            )
            self._sickness_recorded_for_recovery = True
        remaining = read_resurrection_sickness_remaining(self._sickness_path)
        if remaining is None:
            return snapshot
        return replace(
            snapshot,
            life=Evidence.present(
                LifeState.RESURRECTION_SICKNESS,
                observed_at=captured_at,
                frame_id=snapshot.frame_id,
                source="persisted_resurrection_sickness",
                reason=f"remaining_seconds={remaining:.1f}",
            ),
            metadata={
                **dict(snapshot.metadata),
                "resurrection_sickness_remaining_seconds": remaining,
            },
        )
