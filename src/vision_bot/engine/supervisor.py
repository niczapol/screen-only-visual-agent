from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from vision_bot.core.commands import CommandGroup
from vision_bot.core.evidence import Evidence, EvidenceState
from vision_bot.core.geometry import PhysicalRoute
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    MiningSignal,
    ModalState,
    WorldSnapshot,
)


class BotState(str, Enum):
    STARTUP = "startup"
    WAIT_FOR_CLIENT = "wait_for_client"
    PREFLIGHT = "preflight"
    TRAVEL = "travel"
    MOUNTING = "mounting"
    MINING = "mining"
    COMBAT = "combat"
    LOOT = "loot"
    HAZARD_EGRESS = "hazard_egress"
    DEATH_RECOVERY = "death_recovery"
    RESURRECTION_SICKNESS = "resurrection_sickness"
    PAUSED = "paused"
    FAULT = "fault"


@dataclass(frozen=True)
class SupervisorMemory:
    state: BotState = BotState.STARTUP
    entered_at: float = 0.0
    previous_state: BotState | None = None
    transition_count: int = 0
    mounted_escape_started_at: float | None = None


@dataclass(frozen=True)
class ControlDecision:
    memory: SupervisorMemory
    owner: CommandGroup | None
    reason: str
    preempt_groups: tuple[CommandGroup, ...] = ()

    @property
    def state(self) -> BotState:
        return self.memory.state


class Supervisor:
    """Pure top-level state and operational-owner reducer."""

    def __init__(
        self,
        *,
        critical_max_age_seconds: float = 0.50,
        operational_max_age_seconds: float = 0.75,
        mounted_escape_seconds: float = 12.0,
        mounted_escape_attacker_count: int = 2,
        emergency_hp_fraction: float = 0.20,
        route: PhysicalRoute | None = None,
        corridor_radius_yards: float = 18.0,
    ) -> None:
        self.critical_max_age_seconds = max(0.01, float(critical_max_age_seconds))
        self.operational_max_age_seconds = max(
            self.critical_max_age_seconds,
            float(operational_max_age_seconds),
        )
        self.mounted_escape_seconds = max(0.0, float(mounted_escape_seconds))
        self.mounted_escape_attacker_count = max(1, int(mounted_escape_attacker_count))
        self.emergency_hp_fraction = min(1.0, max(0.0, float(emergency_hp_fraction)))
        self.route = route
        self.corridor_radius_yards = max(0.0, float(corridor_radius_yards))

    def reduce(
        self,
        memory: SupervisorMemory,
        snapshot: WorldSnapshot,
        *,
        now: float,
        active_groups: frozenset[CommandGroup] = frozenset(),
    ) -> ControlDecision:
        input_ready = _observed_bool(
            snapshot.input_ready,
            now=now,
            max_age_seconds=self.critical_max_age_seconds,
        )
        if input_ready is not True:
            return self._decision(
                memory,
                BotState.WAIT_FOR_CLIENT,
                None,
                now=now,
                reason=(
                    "input_not_ready"
                    if input_ready is False
                    else "input_readiness_unknown_or_stale"
                ),
                preempt_groups=tuple(CommandGroup),
            )

        life = snapshot.life.present_value(
            now=now,
            max_age_seconds=self.critical_max_age_seconds,
        )
        if life in {LifeState.DEAD, LifeState.GHOST}:
            return self._decision(
                memory,
                BotState.DEATH_RECOVERY,
                CommandGroup.RECOVERY,
                now=now,
                reason=f"life_{life.value}",
                preempt_groups=_all_except(CommandGroup.RECOVERY),
            )
        if life is LifeState.RESURRECTION_SICKNESS:
            return self._decision(
                memory,
                BotState.RESURRECTION_SICKNESS,
                None,
                now=now,
                reason="resurrection_sickness_active",
                preempt_groups=tuple(CommandGroup),
            )
        if life is None:
            return self._decision(
                memory,
                BotState.PAUSED,
                None,
                now=now,
                reason="life_evidence_unknown_or_stale",
                preempt_groups=tuple(CommandGroup),
            )

        modal = snapshot.modal.present_value(
            now=now,
            max_age_seconds=self.critical_max_age_seconds,
        )
        if modal is ModalState.BLOCKING:
            return self._decision(
                memory,
                BotState.PAUSED,
                None,
                now=now,
                reason="blocking_modal",
                preempt_groups=tuple(CommandGroup),
            )
        if modal is ModalState.DISMISSABLE:
            return self._decision(
                memory,
                BotState.DEATH_RECOVERY,
                CommandGroup.RECOVERY,
                now=now,
                reason="dismissable_modal",
                preempt_groups=_all_except(CommandGroup.RECOVERY),
            )

        combat = snapshot.combat.present_value(
            now=now,
            max_age_seconds=self.critical_max_age_seconds,
        )
        mounted = _observed_bool(
            snapshot.mounted,
            now=now,
            max_age_seconds=self.operational_max_age_seconds,
        )
        critical_combat_heal = bool(
            isinstance(combat, CombatView)
            and combat.active
            and combat.hp_fraction is not None
            and combat.hp_fraction <= self.emergency_hp_fraction
        )
        if critical_combat_heal:
            return self._decision(
                memory,
                BotState.COMBAT,
                CommandGroup.COMBAT,
                now=now,
                reason="combat_critical_heal_priority",
                preempt_groups=_all_except(CommandGroup.COMBAT),
            )

        loot_pending = _observed_bool(
            snapshot.loot_pending,
            now=now,
            max_age_seconds=self.operational_max_age_seconds,
        )
        if loot_pending is True or CommandGroup.LOOT in active_groups:
            return self._decision(
                memory,
                BotState.LOOT,
                CommandGroup.LOOT,
                now=now,
                reason="post_combat_loot_pending",
                preempt_groups=_all_except(CommandGroup.LOOT),
            )

        if isinstance(combat, CombatView) and combat.active:
            pose = snapshot.pose.present_value(
                now=now,
                max_age_seconds=self.operational_max_age_seconds,
            )
            elevated_threat = bool(
                combat.attacker_count is not None
                and combat.attacker_count >= self.mounted_escape_attacker_count
            )
            if mounted is True and pose is not None and not elevated_threat:
                escape_started = (
                    memory.mounted_escape_started_at
                    if memory.mounted_escape_started_at is not None
                    else now
                )
                if now - escape_started < self.mounted_escape_seconds:
                    return self._decision(
                        memory,
                        BotState.HAZARD_EGRESS,
                        CommandGroup.TRAVEL,
                        now=now,
                        reason="mounted_escape_before_combat_handoff",
                        preempt_groups=_all_except(CommandGroup.TRAVEL),
                        mounted_escape_started_at=escape_started,
                    )
            return self._decision(
                memory,
                BotState.COMBAT,
                CommandGroup.COMBAT,
                now=now,
                reason="combat_confirmed",
                preempt_groups=_all_except(CommandGroup.COMBAT),
            )

        pose = snapshot.pose.present_value(
            now=now,
            max_age_seconds=self.operational_max_age_seconds,
        )
        if pose is None:
            return self._decision(
                memory,
                BotState.PAUSED,
                None,
                now=now,
                reason="pose_unknown_or_stale_outside_combat",
                preempt_groups=tuple(CommandGroup),
            )

        forbidden = _observed_bool(
            snapshot.forbidden_subzone,
            now=now,
            max_age_seconds=self.critical_max_age_seconds,
        )
        if forbidden is True:
            return self._decision(
                memory,
                BotState.HAZARD_EGRESS,
                CommandGroup.TRAVEL,
                now=now,
                reason="forbidden_subzone_visible",
                preempt_groups=_all_except(CommandGroup.TRAVEL),
            )

        mining = snapshot.mining.present_value(
            now=now,
            max_age_seconds=self.operational_max_age_seconds,
        )
        mining_requested = bool(
            CommandGroup.MINING in active_groups
            or (
                isinstance(mining, MiningPerception)
                and mining.signal is not MiningSignal.IDLE
            )
        )
        if mining_requested:
            return self._decision(
                memory,
                BotState.MINING,
                CommandGroup.MINING,
                now=now,
                reason=f"mining_{mining.signal.value}",
                preempt_groups=_all_except(CommandGroup.MINING),
            )

        if (
            self.route is not None
            and self.route.project(pose.position).cross_track_yards
            > self.corridor_radius_yards
        ):
            return self._decision(
                memory,
                BotState.PAUSED,
                None,
                now=now,
                reason="outside_route_corridor_requires_entry_plan",
                preempt_groups=tuple(CommandGroup),
            )

        if mounted is None:
            return self._decision(
                memory,
                BotState.PAUSED,
                None,
                now=now,
                reason="mounted_evidence_unknown_or_stale",
                preempt_groups=tuple(CommandGroup),
            )
        if mounted is False:
            return self._decision(
                memory,
                BotState.MOUNTING,
                CommandGroup.MOUNT,
                now=now,
                reason="visible_mounted_marker_absent",
                preempt_groups=_all_except(CommandGroup.MOUNT),
            )

        return self._decision(
            memory,
            BotState.TRAVEL,
            CommandGroup.TRAVEL,
            now=now,
            reason="normal_route_travel",
            preempt_groups=_all_except(CommandGroup.TRAVEL),
        )

    @staticmethod
    def _decision(
        memory: SupervisorMemory,
        state: BotState,
        owner: CommandGroup | None,
        *,
        now: float,
        reason: str,
        preempt_groups: tuple[CommandGroup, ...],
        mounted_escape_started_at: float | None = None,
    ) -> ControlDecision:
        if memory.state is state:
            next_memory = replace(
                memory,
                mounted_escape_started_at=mounted_escape_started_at,
            )
        else:
            next_memory = SupervisorMemory(
                state=state,
                entered_at=float(now),
                previous_state=memory.state,
                transition_count=memory.transition_count + 1,
                mounted_escape_started_at=mounted_escape_started_at,
            )
        return ControlDecision(
            memory=next_memory,
            owner=owner,
            reason=reason,
            preempt_groups=preempt_groups,
        )


def _observed_bool(
    evidence: Evidence[bool],
    *,
    now: float,
    max_age_seconds: float,
) -> bool | None:
    if not evidence.is_fresh(now=now, max_age_seconds=max_age_seconds):
        return None
    if evidence.state is EvidenceState.UNKNOWN:
        return None
    if evidence.state is EvidenceState.ABSENT:
        return False
    return bool(evidence.value)


def _all_except(owner: CommandGroup) -> tuple[CommandGroup, ...]:
    return tuple(group for group in CommandGroup if group is not owner)
