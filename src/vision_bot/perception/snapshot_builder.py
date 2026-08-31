from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    MiningSignal,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    RecoverySignal,
    WorldSnapshot,
)
from vision_bot.route_probe_observation import RouteFrameObservation, observe_route_frame
from vision_bot.regions import resolve_region
from vision_bot.runtime_markers import (
    RUNTIME_STATUS_COMBAT,
    RUNTIME_STATUS_LOOT_OPENED,
    RUNTIME_STATUS_LOOT_PENDING,
    RUNTIME_STATUS_MOUNTED,
    RUNTIME_STATUS_OUTGOING_HIT,
    RUNTIME_STATUS_OUT_OF_RANGE,
    RUNTIME_STATUS_TARGET_IS_ATTACKER,
    RUNTIME_STATUS_WRONG_FACING,
    detect_combat_loot_pending_marker,
    detect_dead_hostile_target_marker,
    detect_loot_opened_marker,
    read_minimap_ore_tooltip_telemetry,
)


@dataclass
class SnapshotBuilder:
    """Build one controller snapshot from exactly one captured frame."""

    config: dict[str, Any]
    zone_id: int
    _last_protocol_sequence: int | None = field(default=None, init=False, repr=False)
    _last_protocol_advanced_at: float | None = field(default=None, init=False, repr=False)

    def observe(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        captured_at: float,
        input_ready: bool,
        minimap: np.ndarray | None = None,
        recognize_ore: bool = False,
    ) -> WorldSnapshot:
        observation = observe_route_frame(
            frame,
            self.config,
            timestamp=captured_at,
            minimap=minimap,
            recognize_ore=recognize_ore,
        )
        protocol_fresh = self._protocol_is_fresh(observation.telemetry, captured_at)
        if not protocol_fresh:
            observation = replace(observation, telemetry=None)
        snapshot = build_snapshot_from_route_observation(
            observation,
            frame_id=frame_id,
            # Input readiness means the selected window/adapter is safe.  The
            # protocol separately owns pose/status freshness.  This distinction
            # lets authoritative visible combat/recovery remain operable when
            # map-position telemetry is unavailable, while route/mining still
            # fail closed on UNKNOWN pose.
            input_ready=input_ready,
            zone_id=self.zone_id,
            loot_pending=detect_combat_loot_pending_marker(frame, self.config),
            dead_hostile_target=detect_dead_hostile_target_marker(frame, self.config),
            loot_opened=detect_loot_opened_marker(frame, self.config),
        )
        tooltip = read_minimap_ore_tooltip_telemetry(frame, self.config)
        mining_value = snapshot.mining.value or MiningPerception()
        if tooltip is not None:
            mining_value = replace(
                mining_value,
                tooltip_ore_type=tooltip.ore_type,
            )
        mining_evidence = Evidence.present(
            mining_value,
            observed_at=captured_at,
            frame_id=frame_id,
            source="minimap_ore_detector_and_tooltip",
        )
        frame_height, frame_width = frame.shape[:2]
        minimap_x, minimap_y, minimap_width, minimap_height = resolve_region(
            self.config.get("minimap", {}),
            frame.shape,
            self.config,
        )
        recognition_cfg = self.config.get("recognition", {})
        center_cfg = recognition_cfg.get(
            "sensor_circle_center_fraction", {"x": 0.50, "y": 0.48}
        )
        metadata = {
            **dict(snapshot.metadata),
            "screen_size": (frame_width, frame_height),
            "minimap_origin": (minimap_x, minimap_y),
            "minimap_shape": (minimap_height, minimap_width),
            "minimap_center": (
                minimap_width * float(center_cfg.get("x", 0.50)),
                minimap_height * float(center_cfg.get("y", 0.48)),
            ),
            "addon_protocol_fresh": protocol_fresh,
            "addon_protocol_sequence": (
                observation.telemetry.frame_sequence
                if observation.telemetry is not None
                else None
            ),
        }
        return replace(snapshot, mining=mining_evidence, metadata=metadata)

    def _protocol_is_fresh(self, telemetry: Any, captured_at: float) -> bool:
        if telemetry is None or telemetry.frame_sequence is None:
            return False
        if telemetry.protocol_version != int(
            self.config.get("v09", {}).get("protocol_version", 2)
        ):
            return False
        if telemetry.frame_sequence != self._last_protocol_sequence:
            self._last_protocol_sequence = telemetry.frame_sequence
            self._last_protocol_advanced_at = captured_at
            return True
        if self._last_protocol_advanced_at is None:
            self._last_protocol_advanced_at = captured_at
        max_stall = float(
            self.config.get("v09", {})
            .get("freshness", {})
            .get("critical_seconds", 0.50)
        )
        return captured_at - self._last_protocol_advanced_at <= max_stall


def build_snapshot_from_route_observation(
    observation: RouteFrameObservation,
    *,
    frame_id: int,
    input_ready: bool,
    zone_id: int,
    loot_pending: bool = False,
    dead_hostile_target: bool = False,
    loot_opened: bool = False,
) -> WorldSnapshot:
    observed_at = observation.timestamp
    input_evidence = (
        Evidence.present(
            True,
            observed_at=observed_at,
            frame_id=frame_id,
            source="window_guard",
        )
        if input_ready
        else Evidence.absent(
            observed_at=observed_at,
            frame_id=frame_id,
            source="window_guard",
            reason="window_not_ready_for_input",
        )
    )

    telemetry = observation.telemetry
    if telemetry is None:
        pose = Evidence.unknown(
            observed_at=observed_at,
            frame_id=frame_id,
            source="runtime_telemetry",
            reason="telemetry_not_decoded",
        )
    else:
        pose = Evidence.present(
            PlayerPose(
                zone_id=int(zone_id),
                position=MapPoint(telemetry.x, telemetry.y),
                heading_degrees=telemetry.heading_degrees,
            ),
            observed_at=observed_at,
            frame_id=frame_id,
            source="runtime_telemetry",
        )

    game_state = observation.game_state
    if game_state.death_or_blocking_modal:
        life_value = (
            LifeState.GHOST
            if game_state.ghost_visual or game_state.ghost_button_count >= 2
            else LifeState.DEAD
        )
        life = Evidence.present(
            life_value,
            observed_at=observed_at,
            frame_id=frame_id,
            source="game_state",
        )
    elif game_state.alive_health_bar or observation.combat.active:
        life = Evidence.present(
            LifeState.ALIVE,
            observed_at=observed_at,
            frame_id=frame_id,
            source="game_state",
        )
    else:
        life = Evidence.unknown(
            observed_at=observed_at,
            frame_id=frame_id,
            source="game_state",
            reason="alive_and_death_evidence_both_absent",
        )

    if game_state.dismissable_modal_click is not None:
        modal_value = ModalState.DISMISSABLE
    elif game_state.death_or_blocking_modal and life.value not in {
        LifeState.DEAD,
        LifeState.GHOST,
    }:
        modal_value = ModalState.BLOCKING
    else:
        modal_value = ModalState.CLEAR
    modal = Evidence.present(
        modal_value,
        observed_at=observed_at,
        frame_id=frame_id,
        source="game_state",
        geometry=(
            {"click": game_state.dismissable_modal_click}
            if game_state.dismissable_modal_click is not None
            else None
        ),
    )

    legacy_combat = observation.combat
    telemetry_status = telemetry.status_bits if telemetry is not None else 0
    combat = Evidence.present(
        CombatView(
            active=legacy_combat.active or bool(telemetry_status & RUNTIME_STATUS_COMBAT),
            hp_fraction=legacy_combat.player_health_fraction,
            target_confirmed=legacy_combat.target_present,
            outgoing_hit=(
                legacy_combat.outgoing_hit_marker_visible
                or bool(telemetry_status & RUNTIME_STATUS_OUTGOING_HIT)
            ),
            wrong_facing=(
                legacy_combat.facing_error_marker_visible
                or legacy_combat.facing_error_visible
                or bool(telemetry_status & RUNTIME_STATUS_WRONG_FACING)
            ),
            out_of_range=(
                legacy_combat.out_of_range_marker_visible
                or bool(telemetry_status & RUNTIME_STATUS_OUT_OF_RANGE)
            ),
            attacker_count=legacy_combat.attacker_count,
            dead_hostile_target=dead_hostile_target,
            outcome_sequence=(telemetry.event_sequence if telemetry is not None else None),
            loot_opened=loot_opened or bool(telemetry_status & RUNTIME_STATUS_LOOT_OPENED),
            target_is_attacker=(
                legacy_combat.target_is_attacker
                or bool(telemetry_status & RUNTIME_STATUS_TARGET_IS_ATTACKER)
            ),
        ),
        observed_at=observed_at,
        frame_id=frame_id,
        source=legacy_combat.source or "combat_detector",
        geometry=(
            {"target_bbox": legacy_combat.bbox}
            if legacy_combat.bbox is not None
            else None
        ),
    )

    bright_points = observation.bright_ore_points
    dark_points = observation.dark_ore_points
    mining = Evidence.present(
        MiningPerception(
            # Raw points may acquire mining ownership only for a bounded native
            # tooltip identity probe. They still cannot authorize approach or
            # interaction without database/access and tooltip agreement.
            signal=(
                MiningSignal.CANDIDATE
                if bright_points or dark_points
                else MiningSignal.IDLE
            ),
            bright_minimap_points=bright_points,
            dark_minimap_points=dark_points,
        ),
        observed_at=observed_at,
        frame_id=frame_id,
        source="minimap_ore_detector",
    )

    mounted = _marker_evidence(
        observation.mounted or bool(telemetry_status & RUNTIME_STATUS_MOUNTED),
        observed_at=observed_at,
        frame_id=frame_id,
        source="mounted_marker",
    )
    forbidden = _marker_evidence(
        observation.forbidden_subzone_visible,
        observed_at=observed_at,
        frame_id=frame_id,
        source="forbidden_subzone_marker",
    )
    loot = _marker_evidence(
        loot_pending or bool(telemetry_status & RUNTIME_STATUS_LOOT_PENDING),
        observed_at=observed_at,
        frame_id=frame_id,
        source="combat_loot_pending_marker",
    )
    recovery_signal = (
        RecoverySignal.GHOST_SEARCH
        if life.value is LifeState.GHOST
        else RecoverySignal.NONE
    )
    recovery = Evidence.present(
        RecoveryPerception(signal=recovery_signal),
        observed_at=observed_at,
        frame_id=frame_id,
        source="recovery_perception_adapter",
    )

    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=observed_at,
        input_ready=input_evidence,
        pose=pose,
        life=life,
        modal=modal,
        combat=combat,
        mounted=mounted,
        mining=mining,
        loot_pending=loot,
        forbidden_subzone=forbidden,
        recovery=recovery,
        metadata={"armor_critical": observation.armor_critical},
    )


def _marker_evidence(
    visible: bool,
    *,
    observed_at: float,
    frame_id: int,
    source: str,
) -> Evidence[bool]:
    if visible:
        return Evidence.present(
            True,
            observed_at=observed_at,
            frame_id=frame_id,
            source=source,
        )
    return Evidence.absent(
        observed_at=observed_at,
        frame_id=frame_id,
        source=source,
        reason="marker_not_visible_in_valid_frame",
    )
