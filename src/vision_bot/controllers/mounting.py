from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from vision_bot.core.commands import Command, CommandGroup, CommandKind, ControlIntent
from vision_bot.core.world import WorldSnapshot
from vision_bot.engine.supervisor import BotState, ControlDecision
from vision_bot.v09_config import MountingConfig


class MountingPhase(str, Enum):
    IDLE = "idle"
    SETTLE = "settle"
    CASTING = "casting"
    RETRY_WAIT = "retry_wait"
    CYCLE_COOLDOWN = "cycle_cooldown"


@dataclass(frozen=True)
class MountingState:
    phase: MountingPhase = MountingPhase.IDLE
    attempts: int = 0
    phase_entered_at: float | None = None
    next_action_at: float = 0.0


@dataclass(frozen=True)
class MountingDecision:
    state: MountingState
    intent: ControlIntent | None
    reason: str


class MountingController:
    """Visible-marker, non-blocking mount scheduler.

    It never assumes that a cast succeeded.  Only the mounted marker ends the
    transaction; bounded attempt cycles are separated by an observable
    cooldown so a missing/blocked cast cannot become key spam.
    """

    def __init__(self, config: MountingConfig) -> None:
        self.config = config
        self.state = MountingState()

    @property
    def active(self) -> bool:
        return self.state.phase is not MountingPhase.IDLE

    def reset(self) -> None:
        self.state = MountingState()

    def plan(
        self,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> ControlIntent | None:
        result = self.reduce(self.state, snapshot, decision, now=now)
        self.state = result.state
        return result.intent

    def reduce(
        self,
        state: MountingState,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> MountingDecision:
        mounted = snapshot.mounted.present_value(now=now, max_age_seconds=0.75)
        if mounted is True:
            return MountingDecision(MountingState(), None, "mounted_confirmed")
        if not self.config.enabled:
            return MountingDecision(state, None, "mounting_disabled")
        if mounted is None:
            return MountingDecision(state, None, "mounted_evidence_unknown")

        if state.phase is MountingPhase.IDLE:
            settle = (
                self.config.post_combat_settle_seconds
                if decision.memory.previous_state in {BotState.COMBAT, BotState.LOOT}
                else 0.0
            )
            if settle > 0.0:
                next_state = MountingState(
                    phase=MountingPhase.SETTLE,
                    phase_entered_at=now,
                    next_action_at=now + settle,
                )
                return MountingDecision(next_state, None, "post_combat_mount_settle")
            return self._cast(state, now=now)

        if now < state.next_action_at:
            return MountingDecision(state, None, state.phase.value)
        if state.phase in {MountingPhase.SETTLE, MountingPhase.RETRY_WAIT}:
            return self._cast(state, now=now)
        if state.phase is MountingPhase.CASTING:
            attempts = state.attempts
            if attempts >= self.config.max_attempts_per_cycle:
                return MountingDecision(
                    replace(
                        state,
                        phase=MountingPhase.CYCLE_COOLDOWN,
                        phase_entered_at=now,
                        next_action_at=now + self.config.cycle_cooldown_seconds,
                    ),
                    None,
                    "mount_cycle_cooldown",
                )
            return MountingDecision(
                replace(
                    state,
                    phase=MountingPhase.RETRY_WAIT,
                    phase_entered_at=now,
                    next_action_at=max(
                        now,
                        (state.phase_entered_at or now) + self.config.retry_seconds,
                    ),
                ),
                None,
                "mount_retry_wait",
            )
        if state.phase is MountingPhase.CYCLE_COOLDOWN:
            return self._cast(replace(state, attempts=0), now=now)
        return MountingDecision(state, None, state.phase.value)

    def _cast(self, state: MountingState, *, now: float) -> MountingDecision:
        next_state = replace(
            state,
            phase=MountingPhase.CASTING,
            attempts=state.attempts + 1,
            phase_entered_at=now,
            next_action_at=now + self.config.cast_seconds,
        )
        intent = ControlIntent(
            CommandGroup.MOUNT,
            (
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.MOUNT,
                    key=self.config.key,
                    duration=0.06,
                    reason="mount_visible_marker_absent_cast",
                    exclusive=True,
                    deadline=now + 0.40,
                ),
            ),
            "mount_visible_marker_absent_cast",
        )
        return MountingDecision(next_state, intent, "mount_cast")
