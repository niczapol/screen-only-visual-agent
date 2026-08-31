from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.core.geometry import MapPoint, PhysicalRoute, RouteProjection
from vision_bot.core.world import PlayerPose, WorldSnapshot
from vision_bot.engine.supervisor import ControlDecision
from vision_bot.heading_navigation import signed_heading_error_degrees
from vision_bot.v09_config import NavigationConfig


class TravelPhase(str, Enum):
    ROUTE = "route"
    SOFT_STALL = "soft_stall"
    HARD_STALL = "hard_stall"


@dataclass(frozen=True)
class TravelState:
    phase: TravelPhase = TravelPhase.ROUTE
    progress_yards: float | None = None
    last_position: MapPoint | None = None
    last_observed_at: float | None = None
    speed_yards_per_second: float = 0.0
    stall_started_at: float | None = None
    last_motion_at: float | None = None
    last_soft_action_at: float | None = None
    hard_recovery_step: int = 0
    hard_step_at: float | None = None
    last_turn_sign: int = 0
    last_turn_at: float | None = None


@dataclass(frozen=True)
class TravelDecision:
    state: TravelState
    intent: ControlIntent | None
    projection: RouteProjection | None
    target: MapPoint | None
    lookahead_yards: float | None
    reason: str


class TravelController:
    """Physical-distance route following with bounded RMB yaw and observed recovery."""

    def __init__(self, route: PhysicalRoute, config: NavigationConfig) -> None:
        self.route = route
        self.config = config
        self.state = TravelState()

    @property
    def active(self) -> bool:
        # Travel is the supervisor default and never needs to latch ownership.
        return False

    def reset(self) -> None:
        self.state = TravelState()

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
        state: TravelState,
        snapshot: WorldSnapshot,
        *,
        now: float,
    ) -> TravelDecision:
        pose = snapshot.pose.present_value(now=now, max_age_seconds=0.75)
        if pose is None:
            return TravelDecision(
                state,
                _release_forward(now, "travel_pose_unknown"),
                None,
                None,
                None,
                "pose_unknown_or_stale",
            )

        projection = self.route.project(pose.position)
        state = self._observe_motion(state, pose, projection, now=now)
        stall_age = (
            0.0
            if state.stall_started_at is None
            else max(0.0, now - state.stall_started_at)
        )
        if stall_age >= self.config.stall_hard_seconds:
            return self._hard_stall_step(state, projection, now=now)
        if stall_age >= self.config.stall_soft_seconds:
            soft = self._soft_stall_step(state, projection, now=now)
            if soft is not None:
                return soft

        lookahead = self._lookahead(state, projection)
        progress = state.progress_yards or projection.distance_along_yards
        target = self.route.point_at(progress + lookahead)
        desired = self.route.geometry.heading_degrees(pose.position, target)
        if desired is None:
            return TravelDecision(
                state,
                _release_forward(now, "route_target_reached"),
                projection,
                target,
                lookahead,
                "target_reached",
            )
        error = signed_heading_error_degrees(pose.heading_degrees, desired)
        state, intent = self._route_motion(state, error, now=now)
        return TravelDecision(
            state,
            intent,
            projection,
            target,
            lookahead,
            (
                "route_reentry"
                if projection.cross_track_yards > self.config.corridor_radius_yards
                else "route_follow"
            ),
        )

    def _observe_motion(
        self,
        state: TravelState,
        pose: PlayerPose,
        projection: RouteProjection,
        *,
        now: float,
    ) -> TravelState:
        progress = _unwrap_progress(
            projection.distance_along_yards,
            state.progress_yards,
            self.route.total_length_yards,
        )
        projection_acceptable = (
            projection.cross_track_yards <= self.config.corridor_radius_yards
        )
        if state.progress_yards is not None:
            max_forward = max(25.0, state.speed_yards_per_second * 2.5 + 10.0)
            if progress < state.progress_yards - 3.0 or progress > state.progress_yards + max_forward:
                projection_acceptable = False
        accepted_progress = (
            max(state.progress_yards or progress, progress)
            if projection_acceptable
            else state.progress_yards
        )
        if accepted_progress is None:
            accepted_progress = progress

        speed = state.speed_yards_per_second
        moved = 0.0
        if state.last_position is not None and state.last_observed_at is not None:
            elapsed = now - state.last_observed_at
            if elapsed > 1.0e-6:
                moved = self.route.geometry.distance(state.last_position, pose.position)
                instant_speed = min(40.0, moved / elapsed)
                speed = speed * 0.65 + instant_speed * 0.35

        motion_confirmed = moved >= self.config.stall_min_progress_yards
        if state.last_position is None:
            last_motion_at = now
            stall_started_at = now
        elif motion_confirmed:
            last_motion_at = now
            stall_started_at = now
        else:
            last_motion_at = state.last_motion_at
            if state.stall_started_at is not None:
                stall_started_at = state.stall_started_at
            elif state.last_observed_at is not None:
                stall_started_at = state.last_observed_at
            else:
                stall_started_at = now

        phase = TravelPhase.ROUTE if motion_confirmed else state.phase
        hard_recovery_step = 0 if motion_confirmed else state.hard_recovery_step
        return replace(
            state,
            phase=phase,
            progress_yards=accepted_progress,
            last_position=pose.position,
            last_observed_at=now,
            speed_yards_per_second=speed,
            stall_started_at=stall_started_at,
            last_motion_at=last_motion_at,
            hard_recovery_step=hard_recovery_step,
            hard_step_at=None if motion_confirmed else state.hard_step_at,
        )

    def _lookahead(self, state: TravelState, projection: RouteProjection) -> float:
        if projection.cross_track_yards > self.config.corridor_radius_yards:
            return self.config.lookahead_min_yards
        speed_target = (
            state.speed_yards_per_second * 2.0
            if state.speed_yards_per_second >= 3.0
            else self.config.lookahead_cruise_yards
        )
        base = max(
            self.config.lookahead_min_yards,
            min(self.config.lookahead_max_yards, speed_target),
        )
        progress = state.progress_yards or projection.distance_along_yards
        near = self.route.point_at(progress + self.config.lookahead_min_yards)
        far = self.route.point_at(progress + base)
        start = self.route.point_at(progress)
        first_heading = self.route.geometry.heading_degrees(start, near)
        second_heading = self.route.geometry.heading_degrees(near, far)
        if first_heading is None or second_heading is None:
            return self.config.lookahead_min_yards
        curvature = abs(signed_heading_error_degrees(first_heading, second_heading))
        curvature_scale = max(0.25, 1.0 - curvature / 90.0)
        return max(
            self.config.lookahead_min_yards,
            min(self.config.lookahead_max_yards, base * curvature_scale),
        )

    def _route_motion(
        self,
        state: TravelState,
        heading_error: float,
        *,
        now: float,
    ) -> tuple[TravelState, ControlIntent]:
        magnitude = abs(heading_error)
        pivot = magnitude >= self.config.pivot_degrees
        turn_sign = 1 if heading_error > 0.0 else -1
        reversal_suppressed = bool(
            magnitude < 50.0
            and state.last_turn_sign not in {0, turn_sign}
            and state.last_turn_at is not None
            and now - state.last_turn_at < self.config.turn_reversal_guard_seconds
        )
        commands: list[Command] = [
            Command(
                kind=CommandKind.RELEASE_KEY if pivot else CommandKind.HOLD_KEY,
                group=CommandGroup.TRAVEL,
                key="W",
                reason="route_pivot" if pivot else "route_continuous_forward",
                deadline=now + 0.30,
            )
        ]
        if magnitude > self.config.turn_engage_degrees and not reversal_suppressed:
            max_duration = (
                self.config.pivot_drag_max_seconds
                if pivot
                else self.config.moving_drag_max_seconds
            )
            max_pixels = max(2, round(self.config.yaw_pixels_per_second * max_duration))
            pixels = round(-heading_error * self.config.yaw_pixels_per_degree)
            pixels = max(-max_pixels, min(max_pixels, pixels))
            if pixels:
                commands.append(
                    Command(
                        kind=CommandKind.DRAG_RELATIVE,
                        group=CommandGroup.TRAVEL,
                        mouse_button=MouseButton.RIGHT,
                        delta_x=pixels,
                        duration=max(
                            0.04,
                            min(max_duration, abs(pixels) / self.config.yaw_pixels_per_second),
                        ),
                        reason="route_physical_yaw",
                        deadline=now + 0.50,
                    )
                )
                state = replace(
                    state,
                    last_turn_sign=turn_sign,
                    last_turn_at=now,
                )
        return state, ControlIntent(
            CommandGroup.TRAVEL,
            tuple(commands),
            (
                "route_yaw_reversal_hysteresis"
                if reversal_suppressed
                else "route_physical_follow"
            ),
        )

    def _soft_stall_step(
        self,
        state: TravelState,
        projection: RouteProjection,
        *,
        now: float,
    ) -> TravelDecision | None:
        if state.last_soft_action_at is not None and now - state.last_soft_action_at < 2.0:
            return None
        next_state = replace(
            state,
            phase=TravelPhase.SOFT_STALL,
            last_soft_action_at=now,
        )
        intent = ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="soft_stall_keep_forward",
                    deadline=now + 0.30,
                ),
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.TRAVEL,
                    key="SPACE",
                    duration=0.08,
                    reason="soft_stall_jump_probe",
                    deadline=now + 0.30,
                ),
            ),
            "soft_stall_observed_probe",
        )
        return TravelDecision(
            next_state,
            intent,
            projection,
            None,
            None,
            "soft_stall_observed_probe",
        )

    def _hard_stall_step(
        self,
        state: TravelState,
        projection: RouteProjection,
        *,
        now: float,
    ) -> TravelDecision:
        if state.hard_step_at is not None and now - state.hard_step_at < 0.70:
            return TravelDecision(
                replace(state, phase=TravelPhase.HARD_STALL),
                None,
                projection,
                None,
                None,
                "hard_stall_observing_previous_step",
            )
        step = state.hard_recovery_step % 3
        if step == 0:
            commands = (
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.TRAVEL,
                    key="S",
                    duration=0.55,
                    reason="hard_stall_back_probe",
                    exclusive=True,
                    deadline=now + 0.85,
                ),
            )
            reason = "hard_stall_back_probe"
        elif step == 1:
            direction = -1 if (state.hard_recovery_step // 3) % 2 == 0 else 1
            commands = (
                Command(
                    kind=CommandKind.DRAG_RELATIVE,
                    group=CommandGroup.TRAVEL,
                    mouse_button=MouseButton.RIGHT,
                    delta_x=direction * round(self.config.yaw_pixels_per_second * 0.30),
                    duration=0.30,
                    reason="hard_stall_turn_probe",
                    exclusive=True,
                    deadline=now + 0.60,
                ),
            )
            reason = "hard_stall_turn_probe"
        else:
            commands = (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="hard_stall_forward_probe",
                    deadline=now + 0.30,
                ),
            )
            reason = "hard_stall_forward_probe"
        next_state = replace(
            state,
            phase=TravelPhase.HARD_STALL,
            hard_recovery_step=state.hard_recovery_step + 1,
            hard_step_at=now,
        )
        return TravelDecision(
            next_state,
            ControlIntent(CommandGroup.TRAVEL, commands, reason),
            projection,
            None,
            None,
            reason,
        )


def _unwrap_progress(
    projected: float,
    previous: float | None,
    route_length: float,
) -> float:
    if previous is None:
        return projected
    lap = math.floor(previous / route_length)
    candidates = (
        projected + (lap - 1) * route_length,
        projected + lap * route_length,
        projected + (lap + 1) * route_length,
    )
    return min(candidates, key=lambda value: abs(value - previous))


def _release_forward(now: float, reason: str) -> ControlIntent:
    return ControlIntent(
        CommandGroup.TRAVEL,
        (
            Command(
                kind=CommandKind.RELEASE_KEY,
                group=CommandGroup.TRAVEL,
                key="W",
                reason=reason,
                deadline=now + 0.25,
            ),
        ),
        reason,
    )
