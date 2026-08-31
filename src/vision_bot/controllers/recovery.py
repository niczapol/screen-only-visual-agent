from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.core.geometry import MapPoint, ZoneGeometry
from vision_bot.core.world import (
    LifeState,
    RecoveryPerception,
    RecoverySignal,
    WorldSnapshot,
)
from vision_bot.engine.supervisor import ControlDecision
from vision_bot.heading_navigation import signed_heading_error_degrees
from vision_bot.v09_config import NavigationConfig, RecoveryConfig


class RecoveryPhase(str, Enum):
    IDLE = "idle"
    RELEASE_SPIRIT = "release_spirit"
    WAIT_FOR_GHOST = "wait_for_ghost"
    SEEK_HEALER = "seek_healer"
    INTERACT_HEALER = "interact_healer"
    WAIT_FOR_RETURN = "wait_for_return"
    RETURN_TO_LIFE = "return_to_life"
    ACCEPT_RESURRECTION = "accept_resurrection"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class RecoveryControllerState:
    phase: RecoveryPhase = RecoveryPhase.IDLE
    started_at: float | None = None
    phase_entered_at: float | None = None
    last_action_at: float | None = None
    release_attempts: int = 0
    target_attempts: int = 0
    interact_attempts: int = 0
    approach_attempts: int = 0
    last_action_kind: str | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class RecoveryControllerDecision:
    state: RecoveryControllerState
    intent: ControlIntent | None
    action: str


class RecoveryController:
    """Non-blocking Spirit Healer recovery with visible C/G authority."""

    def __init__(
        self,
        config: RecoveryConfig,
        geometry: ZoneGeometry,
        navigation: NavigationConfig,
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.navigation = navigation
        self.state = RecoveryControllerState()

    @property
    def active(self) -> bool:
        return self.state.phase not in {
            RecoveryPhase.IDLE,
            RecoveryPhase.COMPLETE,
            RecoveryPhase.FAILED,
        }

    def reset(self) -> None:
        self.state = RecoveryControllerState()

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
        state: RecoveryControllerState,
        snapshot: WorldSnapshot,
        *,
        now: float,
    ) -> RecoveryControllerDecision:
        life = snapshot.life.present_value(now=now, max_age_seconds=0.50)
        perception = snapshot.recovery.present_value(now=now, max_age_seconds=0.75)
        if life is LifeState.RESURRECTION_SICKNESS:
            return RecoveryControllerDecision(
                replace(
                    state,
                    phase=RecoveryPhase.COMPLETE,
                    phase_entered_at=now,
                    outcome="resurrection_sickness",
                ),
                None,
                "sickness_supervisor_wait",
            )
        if life is LifeState.ALIVE:
            phase = RecoveryPhase.COMPLETE if state.phase is not RecoveryPhase.IDLE else RecoveryPhase.IDLE
            return RecoveryControllerDecision(
                replace(state, phase=phase, phase_entered_at=now),
                None,
                "alive_recovery_complete" if phase is RecoveryPhase.COMPLETE else "alive_idle",
            )
        if life not in {LifeState.DEAD, LifeState.GHOST}:
            return RecoveryControllerDecision(state, None, "life_unknown_wait")
        if state.phase is RecoveryPhase.FAILED:
            if life is LifeState.GHOST and state.outcome in {
                "release_attempts_exhausted",
                "release_button_timeout",
                "ghost_transition_timeout",
            }:
                # The visible dead -> ghost transition is authoritative forward
                # progress even if our Release clicks were not accepted and the
                # client eventually released automatically.  Resume at the
                # healer phase; do not reopen unrelated terminal ghost failures.
                state = RecoveryControllerState(
                    phase=RecoveryPhase.SEEK_HEALER,
                    started_at=state.started_at or now,
                    phase_entered_at=now,
                )
            else:
                return RecoveryControllerDecision(state, None, state.phase.value)
        if state.phase is RecoveryPhase.COMPLETE:
            return RecoveryControllerDecision(state, None, state.phase.value)

        if state.phase is RecoveryPhase.IDLE:
            state = RecoveryControllerState(
                phase=(
                    RecoveryPhase.RELEASE_SPIRIT
                    if life is LifeState.DEAD
                    else RecoveryPhase.SEEK_HEALER
                ),
                started_at=now,
                phase_entered_at=now,
            )

        signal = perception.signal if perception is not None else RecoverySignal.NONE
        if signal is RecoverySignal.ACCEPT_RESURRECTION:
            state = _enter(state, RecoveryPhase.ACCEPT_RESURRECTION, now)
        elif (
            signal is RecoverySignal.RETURN_TO_LIFE
            and state.phase is not RecoveryPhase.ACCEPT_RESURRECTION
        ):
            state = _enter(state, RecoveryPhase.RETURN_TO_LIFE, now)
        elif (
            signal is RecoverySignal.HEALER_DIALOG
            and state.phase
            not in {
                RecoveryPhase.RETURN_TO_LIFE,
                RecoveryPhase.ACCEPT_RESURRECTION,
            }
        ):
            state = _enter(state, RecoveryPhase.WAIT_FOR_RETURN, now)
        elif signal is RecoverySignal.HEALER_TARGETED:
            state = _enter(state, RecoveryPhase.INTERACT_HEALER, now)
        elif life is LifeState.GHOST and state.phase in {
            RecoveryPhase.RELEASE_SPIRIT,
            RecoveryPhase.WAIT_FOR_GHOST,
        }:
            state = _enter(state, RecoveryPhase.SEEK_HEALER, now)

        if state.phase is RecoveryPhase.RELEASE_SPIRIT:
            return self._release_spirit(state, perception, now=now)
        if state.phase is RecoveryPhase.WAIT_FOR_GHOST:
            if (
                life is LifeState.DEAD
                and perception is not None
                and perception.signal is RecoverySignal.RELEASE_SPIRIT
                and perception.click_point is not None
                and _due(
                    state.last_action_at,
                    now,
                    self.config.release_retry_seconds,
                )
            ):
                # The target client's Return to Graveyard flow may expose a second
                # visible Yes button before the ghost transition completes.
                return self._release_spirit(
                    replace(state, phase=RecoveryPhase.RELEASE_SPIRIT),
                    perception,
                    now=now,
                )
            if _phase_age(state, now) > self.config.release_timeout_seconds:
                return self._failed(state, now, "ghost_transition_timeout")
            return RecoveryControllerDecision(state, None, "awaiting_ghost_transition")
        if state.phase is RecoveryPhase.SEEK_HEALER:
            return self._seek_healer(state, now=now)
        if state.phase is RecoveryPhase.INTERACT_HEALER:
            return self._interact_healer(state, snapshot, now=now)
        if state.phase is RecoveryPhase.WAIT_FOR_RETURN:
            if _phase_age(state, now) > self.config.dialog_timeout_seconds:
                return self._failed(state, now, "return_to_life_not_observed")
            return RecoveryControllerDecision(state, None, "awaiting_return_to_life")
        if state.phase is RecoveryPhase.RETURN_TO_LIFE:
            return self._click_visible(
                state,
                perception,
                now=now,
                reason="recovery_return_to_life",
            )
        if state.phase is RecoveryPhase.ACCEPT_RESURRECTION:
            return self._click_visible(
                state,
                perception,
                now=now,
                reason="recovery_accept_resurrection",
            )
        return RecoveryControllerDecision(state, None, state.phase.value)

    def _release_spirit(
        self,
        state: RecoveryControllerState,
        perception: RecoveryPerception | None,
        *,
        now: float,
    ) -> RecoveryControllerDecision:
        if state.release_attempts >= self.config.max_release_attempts:
            return self._failed(state, now, "release_attempts_exhausted")
        if _phase_age(state, now) > self.config.release_timeout_seconds:
            return self._failed(state, now, "release_button_timeout")
        if (
            perception is None
            or perception.signal is not RecoverySignal.RELEASE_SPIRIT
            or perception.click_point is None
        ):
            return RecoveryControllerDecision(state, None, "release_button_not_visible")
        next_state = replace(
            state,
            phase=RecoveryPhase.WAIT_FOR_GHOST,
            phase_entered_at=now,
            last_action_at=now,
            release_attempts=state.release_attempts + 1,
        )
        return RecoveryControllerDecision(
            next_state,
            _visible_click(perception.click_point, now, "recovery_release_spirit"),
            "release_spirit_click",
        )

    def _seek_healer(
        self,
        state: RecoveryControllerState,
        *,
        now: float,
    ) -> RecoveryControllerDecision:
        if _phase_age(state, now) > self.config.healer_search_timeout_seconds:
            return self._failed(state, now, "healer_search_timeout")
        if state.target_attempts >= self.config.max_target_attempts:
            return self._failed(state, now, "healer_target_attempts_exhausted")
        if not _due(state.last_action_at, now, self.config.target_interval_seconds):
            return RecoveryControllerDecision(state, None, "healer_target_observing")
        next_state = replace(
            state,
            last_action_at=now,
            target_attempts=state.target_attempts + 1,
        )
        return RecoveryControllerDecision(
            next_state,
            _key_intent(
                self.config.target_key,
                now,
                self.config.key_tap_seconds,
                "recovery_target_spirit_healer_C",
            ),
            "target_spirit_healer_C",
        )

    def _interact_healer(
        self,
        state: RecoveryControllerState,
        snapshot: WorldSnapshot,
        *,
        now: float,
    ) -> RecoveryControllerDecision:
        pose = snapshot.pose.present_value(now=now, max_age_seconds=0.75)
        if pose is None:
            return RecoveryControllerDecision(state, None, "healer_approach_pose_unknown")
        anchor = min(
            self.config.healer_anchors,
            key=lambda point: self.geometry.distance(pose.position, point),
        )
        distance = self.geometry.distance(pose.position, anchor)
        if distance > self.config.healer_anchor_activation_radius_yards:
            return self._failed(state, now, "confirmed_healer_outside_anchor_radius")
        required_wait = (
            self.config.approach_settle_seconds
            if state.last_action_kind == "approach"
            else self.config.interact_interval_seconds
        )
        if not _due(state.last_action_at, now, required_wait):
            return RecoveryControllerDecision(state, None, "healer_action_observing")
        if distance > self.config.healer_interact_radius_yards:
            desired = self.geometry.heading_degrees(pose.position, anchor)
            if desired is None:
                return RecoveryControllerDecision(state, None, "healer_anchor_reached")
            error = signed_heading_error_degrees(pose.heading_degrees, desired)
            if abs(error) > self.navigation.turn_engage_degrees:
                max_duration = self.navigation.pivot_drag_max_seconds
                max_pixels = max(
                    2,
                    round(self.navigation.yaw_pixels_per_second * max_duration),
                )
                pixels = round(-error * self.navigation.yaw_pixels_per_degree)
                pixels = max(-max_pixels, min(max_pixels, pixels))
                return RecoveryControllerDecision(
                    replace(state, last_action_kind="turn"),
                    ControlIntent(
                        CommandGroup.RECOVERY,
                        (
                            Command(
                                kind=CommandKind.RELEASE_KEY,
                                group=CommandGroup.RECOVERY,
                                key=self.config.approach_key,
                                reason="recovery_face_confirmed_healer_anchor",
                                deadline=now + 0.30,
                            ),
                            Command(
                                kind=CommandKind.DRAG_RELATIVE,
                                group=CommandGroup.RECOVERY,
                                mouse_button=MouseButton.RIGHT,
                                delta_x=pixels,
                                duration=max(
                                    0.04,
                                    min(
                                        max_duration,
                                        abs(pixels) / self.navigation.yaw_pixels_per_second,
                                    ),
                                ),
                                reason="recovery_face_confirmed_healer_anchor",
                                deadline=now + 0.50,
                            ),
                        ),
                        "recovery_face_confirmed_healer_anchor",
                    ),
                    "face_confirmed_healer_anchor",
                )
            if state.approach_attempts >= self.config.max_approach_attempts:
                return self._failed(state, now, "healer_approach_attempts_exhausted")
            next_state = replace(
                state,
                last_action_at=now,
                last_action_kind="approach",
                approach_attempts=state.approach_attempts + 1,
            )
            return RecoveryControllerDecision(
                next_state,
                _key_intent(
                    self.config.approach_key,
                    now,
                    self.config.approach_seconds,
                    "recovery_approach_confirmed_healer_W",
                ),
                "approach_confirmed_healer_W",
            )
        if state.interact_attempts >= self.config.max_interact_attempts:
            return self._failed(state, now, "healer_interact_attempts_exhausted")
        next_state = replace(
            state,
            last_action_at=now,
            last_action_kind="interact",
            interact_attempts=state.interact_attempts + 1,
        )
        return RecoveryControllerDecision(
            next_state,
            _key_intent(
                self.config.interact_key,
                now,
                self.config.key_tap_seconds,
                "recovery_interact_spirit_healer_G",
            ),
            "interact_spirit_healer_G",
        )

    def _click_visible(
        self,
        state: RecoveryControllerState,
        perception: RecoveryPerception | None,
        *,
        now: float,
        reason: str,
    ) -> RecoveryControllerDecision:
        if state.interact_attempts >= self.config.max_interact_attempts:
            return self._failed(state, now, f"{reason}_attempts_exhausted")
        if _phase_age(state, now) > self.config.dialog_timeout_seconds:
            return self._failed(state, now, f"{reason}_timeout")
        if perception is None or perception.click_point is None:
            return RecoveryControllerDecision(state, None, f"{reason}_not_visible")
        if not _due(state.last_action_at, now, self.config.interact_interval_seconds):
            return RecoveryControllerDecision(state, None, f"{reason}_observing")
        next_state = replace(
            state,
            last_action_at=now,
            interact_attempts=state.interact_attempts + 1,
        )
        return RecoveryControllerDecision(
            next_state,
            _visible_click(perception.click_point, now, reason),
            reason,
        )

    @staticmethod
    def _failed(
        state: RecoveryControllerState,
        now: float,
        outcome: str,
    ) -> RecoveryControllerDecision:
        return RecoveryControllerDecision(
            replace(
                state,
                phase=RecoveryPhase.FAILED,
                phase_entered_at=now,
                outcome=outcome,
            ),
            None,
            outcome,
        )


def _enter(
    state: RecoveryControllerState,
    phase: RecoveryPhase,
    now: float,
) -> RecoveryControllerState:
    if state.phase is phase:
        return state
    reset_interactions = phase in {
        RecoveryPhase.INTERACT_HEALER,
        RecoveryPhase.RETURN_TO_LIFE,
        RecoveryPhase.ACCEPT_RESURRECTION,
    }
    return replace(
        state,
        phase=phase,
        phase_entered_at=now,
        last_action_at=None,
        interact_attempts=(0 if reset_interactions else state.interact_attempts),
    )


def _due(previous: float | None, now: float, interval: float) -> bool:
    return previous is None or now - previous >= interval


def _phase_age(state: RecoveryControllerState, now: float) -> float:
    return 0.0 if state.phase_entered_at is None else max(0.0, now - state.phase_entered_at)


def _key_intent(key: str, now: float, duration: float, reason: str) -> ControlIntent:
    return ControlIntent(
        CommandGroup.RECOVERY,
        (
            Command(
                kind=CommandKind.TAP_KEY,
                group=CommandGroup.RECOVERY,
                key=key,
                duration=duration,
                reason=reason,
                exclusive=True,
                deadline=now + duration + 0.30,
            ),
        ),
        reason,
    )


def _visible_click(point: tuple[int, int], now: float, reason: str) -> ControlIntent:
    return ControlIntent(
        CommandGroup.RECOVERY,
        (
            Command(
                kind=CommandKind.MOVE_CURSOR,
                group=CommandGroup.RECOVERY,
                point=point,
                reason=reason,
                deadline=now + 0.30,
            ),
            Command(
                kind=CommandKind.CLICK,
                group=CommandGroup.RECOVERY,
                mouse_button=MouseButton.LEFT,
                duration=0.12,
                reason=reason,
                exclusive=True,
                deadline=now + 0.30,
            ),
        ),
        reason,
    )
