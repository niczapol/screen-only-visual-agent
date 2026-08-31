from __future__ import annotations

from dataclasses import dataclass, replace

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.core.world import CombatView, WorldSnapshot
from vision_bot.engine.supervisor import ControlDecision
from vision_bot.v09_config import CombatConfig, NavigationConfig


@dataclass(frozen=True)
class CombatControllerState:
    engaged: bool = False
    entered_at: float | None = None
    last_attack_at: float | None = None
    last_priority_at: float | None = None
    last_aligned_at: float | None = None
    last_emergency_at: float | None = None
    last_heal_at: float | None = None
    last_face_search_at: float | None = None
    bearing_locked_at: float | None = None
    last_outcome_sequence: int | None = None
    previous_hit_visible: bool = False
    previous_facing_visible: bool = False
    previous_range_visible: bool = False
    emergency_followup_until: float | None = None


@dataclass(frozen=True)
class CombatControllerDecision:
    state: CombatControllerState
    intent: ControlIntent | None
    action: str


class CombatController:
    """Serialized combat transaction scheduler: F -> V > X > Q > E."""

    def __init__(self, config: CombatConfig, navigation: NavigationConfig) -> None:
        self.config = config
        self.navigation = navigation
        self.state = CombatControllerState()

    @property
    def active(self) -> bool:
        return self.state.engaged

    def reset(self) -> None:
        self.state = CombatControllerState()

    def plan(
        self,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> ControlIntent | None:
        result = self.reduce(self.state, snapshot, now=now)
        self.state = result.state
        return result.intent

    def reduce(
        self,
        state: CombatControllerState,
        snapshot: WorldSnapshot,
        *,
        now: float,
    ) -> CombatControllerDecision:
        combat = snapshot.combat.present_value(now=now, max_age_seconds=0.50)
        if not isinstance(combat, CombatView) or not combat.active:
            return CombatControllerDecision(
                CombatControllerState(),
                None,
                "combat_not_confirmed",
            )

        if not state.engaged:
            state = CombatControllerState(engaged=True, entered_at=now)
        state, fresh_hit, fresh_facing, fresh_range = self._observe_outcome(
            state,
            combat,
            now=now,
        )

        if (
            state.emergency_followup_until is not None
            and now < state.emergency_followup_until
        ):
            return CombatControllerDecision(
                state,
                None,
                "awaiting_emergency_V_followup",
            )
        if state.emergency_followup_until is not None:
            state = replace(state, emergency_followup_until=None)

        hp = combat.hp_fraction
        if (
            hp is not None
            and hp <= self.config.emergency_hp_fraction
            and _due(state.last_emergency_at, now, self.config.emergency_interval_seconds)
        ):
            followup_at = (
                now
                + self.config.key_tap_seconds
                + self.config.emergency_followup_gap_seconds
            )
            next_state = replace(
                state,
                last_emergency_at=now,
                last_heal_at=followup_at,
                emergency_followup_until=followup_at + self.config.key_tap_seconds,
            )
            return CombatControllerDecision(
                next_state,
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        _tap(
                            self.config.emergency_key,
                            now,
                            self.config.key_tap_seconds,
                            "combat_emergency_F",
                        ),
                        _tap(
                            self.config.emergency_followup_key,
                            followup_at,
                            self.config.key_tap_seconds,
                            "combat_emergency_V_followup",
                        ),
                    ),
                    "combat_emergency_F_V",
                ),
                "emergency_F_V",
            )

        if (
            hp is not None
            and hp <= self.config.heal_hp_fraction
            and _due(state.last_heal_at, now, self.config.heal_interval_seconds)
        ):
            return CombatControllerDecision(
                replace(state, last_heal_at=now),
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        _tap(
                            self.config.heal_key,
                            now,
                            self.config.key_tap_seconds,
                            "combat_confirmed_health_heal",
                        ),
                    ),
                    "combat_confirmed_health_heal",
                ),
                "heal_V",
            )

        if _due(state.last_priority_at, now, self.config.priority_interval_seconds):
            return CombatControllerDecision(
                replace(state, last_priority_at=now),
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        _tap(
                            self.config.priority_key,
                            now,
                            self.config.key_tap_seconds,
                            "combat_priority_X",
                        ),
                    ),
                    "combat_priority_X",
                ),
                "priority_X",
            )

        bearing_locked = bool(
            state.bearing_locked_at is not None
            and now - state.bearing_locked_at <= self.config.bearing_lock_seconds
        )
        if (
            bearing_locked
            and _due(state.last_aligned_at, now, self.config.aligned_interval_seconds)
        ):
            return CombatControllerDecision(
                replace(state, last_aligned_at=now),
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        _tap(
                            self.config.aligned_key,
                            now,
                            self.config.key_tap_seconds,
                            "combat_aligned_Q",
                        ),
                    ),
                    "combat_aligned_Q",
                ),
                "aligned_Q",
            )

        if (fresh_facing or combat.wrong_facing) and _due(
            state.last_face_search_at,
            now,
            self.config.face_search_cooldown_seconds,
        ):
            duration = self.config.face_search_drag_seconds
            pixels = max(2, round(self.navigation.yaw_pixels_per_second * duration))
            return CombatControllerDecision(
                replace(state, last_face_search_at=now),
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        Command(
                            kind=CommandKind.DRAG_RELATIVE,
                            group=CommandGroup.COMBAT,
                            mouse_button=MouseButton.RIGHT,
                            delta_x=pixels,
                            duration=duration,
                            reason="combat_wrong_facing_search",
                            exclusive=True,
                            deadline=now + duration + 0.25,
                        ),
                    ),
                    "combat_wrong_facing_search",
                ),
                "face_search",
            )

        if fresh_range or combat.out_of_range:
            return CombatControllerDecision(
                state,
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        Command(
                            kind=CommandKind.HOLD_KEY,
                            group=CommandGroup.COMBAT,
                            key="W",
                            reason="combat_out_of_range_approach",
                            deadline=now + 0.30,
                        ),
                    ),
                    "combat_out_of_range_approach",
                ),
                "range_approach",
            )

        if _due(state.last_attack_at, now, self.config.attack_interval_seconds):
            return CombatControllerDecision(
                replace(state, last_attack_at=now),
                ControlIntent(
                    CommandGroup.COMBAT,
                    (
                        _tap(
                            self.config.attack_key,
                            now,
                            self.config.key_tap_seconds,
                            "combat_attack_E",
                        ),
                    ),
                    "combat_attack_E",
                ),
                "attack_E",
            )
        return CombatControllerDecision(state, None, "awaiting_attack_cadence")

    def _observe_outcome(
        self,
        state: CombatControllerState,
        combat: CombatView,
        *,
        now: float,
    ) -> tuple[CombatControllerState, bool, bool, bool]:
        sequence_fresh = bool(
            combat.outcome_sequence is not None
            and combat.outcome_sequence != state.last_outcome_sequence
        )
        fresh_hit = combat.outgoing_hit and (
            sequence_fresh or not state.previous_hit_visible
        )
        fresh_facing = combat.wrong_facing and (
            sequence_fresh or not state.previous_facing_visible
        )
        fresh_range = combat.out_of_range and (
            sequence_fresh or not state.previous_range_visible
        )
        bearing_locked_at = now if fresh_hit else state.bearing_locked_at
        return (
            replace(
                state,
                bearing_locked_at=bearing_locked_at,
                last_outcome_sequence=(
                    combat.outcome_sequence
                    if combat.outcome_sequence is not None
                    else state.last_outcome_sequence
                ),
                previous_hit_visible=combat.outgoing_hit,
                previous_facing_visible=combat.wrong_facing,
                previous_range_visible=combat.out_of_range,
            ),
            fresh_hit,
            fresh_facing,
            fresh_range,
        )


def _due(previous: float | None, now: float, interval: float) -> bool:
    return previous is None or now - previous >= interval


def _tap(key: str, at: float, duration: float, reason: str) -> Command:
    return Command(
        kind=CommandKind.TAP_KEY,
        group=CommandGroup.COMBAT,
        key=key,
        duration=duration,
        not_before=at,
        deadline=at + duration + 0.30,
        reason=reason,
        exclusive=True,
    )
