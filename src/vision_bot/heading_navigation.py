from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from vision_bot.coords import coord_to_xy


TurnKey = Literal["A", "D"]


@dataclass(frozen=True)
class HeadingNavigationCommand:
    hold_forward: bool
    turn_key: TurnKey | None
    desired_heading_degrees: float | None
    heading_error_degrees: float | None
    braking: bool
    reason: str
    turn_hold_seconds: float | None = None

    @property
    def heading_available(self) -> bool:
        return self.desired_heading_degrees is not None


@dataclass
class VisibleHeadingTracker:
    max_age_seconds: float
    heading_degrees: float | None = None
    observed_at: float | None = None

    def observe(self, heading_degrees: float | None, *, now: float) -> None:
        if heading_degrees is None or not math.isfinite(heading_degrees):
            return
        self.heading_degrees = float(heading_degrees) % 360.0
        self.observed_at = float(now)

    def current(self, *, now: float) -> tuple[float | None, bool]:
        if self.heading_degrees is None or self.observed_at is None:
            return None, False
        fresh = float(now) - self.observed_at <= max(0.0, self.max_age_seconds)
        return self.heading_degrees, fresh

    def reset(self) -> None:
        self.heading_degrees = None
        self.observed_at = None


class HeadingNavigationController:
    """Convert visible player heading and a target coordinate into held key state."""

    def __init__(
        self,
        *,
        turn_engage_degrees: float = 9.0,
        turn_release_degrees: float = 4.0,
        pivot_degrees: float = 100.0,
        positive_error_turn_key: TurnKey = "A",
        turn_rate_degrees_per_second: float = 110.0,
        min_turn_pulse_seconds: float = 0.06,
        max_turn_pulse_seconds: float = 0.35,
        opposite_turn_lock_seconds: float = 2.50,
        opposite_turn_engage_degrees: float = 28.0,
    ) -> None:
        self.turn_engage_degrees = max(0.0, float(turn_engage_degrees))
        self.turn_release_degrees = min(
            self.turn_engage_degrees,
            max(0.0, float(turn_release_degrees)),
        )
        self.pivot_degrees = max(
            self.turn_engage_degrees,
            min(180.0, float(pivot_degrees)),
        )
        self.positive_error_turn_key = positive_error_turn_key
        self.turn_rate_degrees_per_second = max(
            1.0,
            float(turn_rate_degrees_per_second),
        )
        self.min_turn_pulse_seconds = max(0.0, float(min_turn_pulse_seconds))
        self.max_turn_pulse_seconds = max(
            self.min_turn_pulse_seconds,
            float(max_turn_pulse_seconds),
        )
        self.opposite_turn_lock_seconds = max(
            0.0,
            float(opposite_turn_lock_seconds),
        )
        self.opposite_turn_engage_degrees = max(
            self.turn_engage_degrees,
            min(180.0, float(opposite_turn_engage_degrees)),
        )
        self.held_turn_key: TurnKey | None = None
        self.last_turn_key: TurnKey | None = None
        self.last_turn_at: float | None = None

    def reset(self) -> HeadingNavigationCommand:
        self.held_turn_key = None
        self.last_turn_key = None
        self.last_turn_at = None
        return HeadingNavigationCommand(
            hold_forward=False,
            turn_key=None,
            desired_heading_degrees=None,
            heading_error_degrees=None,
            braking=False,
            reason="reset",
        )

    def plan(
        self,
        *,
        current_coord: int,
        target_coord: int,
        heading_degrees: float | None,
        input_available: bool = True,
        heading_fresh: bool = True,
        now: float | None = None,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> HeadingNavigationCommand:
        if not input_available:
            return self._unavailable("input_unavailable")
        if heading_degrees is None or not math.isfinite(heading_degrees):
            return self._unavailable("heading_unavailable")
        if not heading_fresh:
            return self._unavailable("heading_stale")

        desired = heading_to_coord_degrees(
            current_coord,
            target_coord,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
        )
        if desired is None:
            self.held_turn_key = None
            return HeadingNavigationCommand(
                hold_forward=False,
                turn_key=None,
                desired_heading_degrees=None,
                heading_error_degrees=0.0,
                braking=False,
                reason="target_reached",
            )

        error = signed_heading_error_degrees(heading_degrees, desired)
        magnitude = abs(error)
        turn_hold_seconds: float | None = None
        reversal_suppressed = False
        if magnitude <= self.turn_engage_degrees:
            self.held_turn_key = None
            turn_key = None
        else:
            # Every correction is bounded independently of the next perception
            # tick. Holding a pivot key until the next captured frame caused
            # large overshoots and immediate A/D reversals in live runs.
            candidate_turn_key = self._turn_key(error)
            reversal_suppressed = bool(
                now is not None
                and self.last_turn_at is not None
                and self.last_turn_key is not None
                and candidate_turn_key != self.last_turn_key
                and now - self.last_turn_at < self.opposite_turn_lock_seconds
                and magnitude < self.opposite_turn_engage_degrees
            )
            turn_key = None if reversal_suppressed else candidate_turn_key
            self.held_turn_key = None
            if turn_key is not None:
                turn_hold_seconds = max(
                    self.min_turn_pulse_seconds,
                    min(
                        self.max_turn_pulse_seconds,
                        (magnitude - self.turn_release_degrees)
                        / self.turn_rate_degrees_per_second,
                    ),
                )
                self.last_turn_key = turn_key
                self.last_turn_at = now

        braking = bool(turn_key is not None and magnitude >= self.pivot_degrees)
        if reversal_suppressed:
            reason = "heading_reversal_hysteresis"
        elif turn_key is None:
            reason = "heading_aligned"
        elif braking:
            reason = "pivot_to_heading"
        else:
            reason = "pulse_to_heading"
        return HeadingNavigationCommand(
            hold_forward=not braking,
            turn_key=turn_key,
            desired_heading_degrees=desired,
            heading_error_degrees=error,
            braking=braking,
            reason=reason,
            turn_hold_seconds=turn_hold_seconds,
        )

    def _unavailable(self, reason: str) -> HeadingNavigationCommand:
        self.held_turn_key = None
        return HeadingNavigationCommand(
            hold_forward=False,
            turn_key=None,
            desired_heading_degrees=None,
            heading_error_degrees=None,
            braking=False,
            reason=reason,
        )

    def _turn_key(self, error: float) -> TurnKey:
        if error > 0.0:
            return self.positive_error_turn_key
        return "D" if self.positive_error_turn_key == "A" else "A"


def heading_to_coord_degrees(
    current_coord: int,
    target_coord: int,
    *,
    coordinate_scale_x: float = 1.0,
    coordinate_scale_y: float = 1.0,
) -> float | None:
    """Return WoW `GetPlayerFacing` degrees for a UI-map coordinate target."""
    current_x, current_y = coord_to_xy(current_coord)
    target_x, target_y = coord_to_xy(target_coord)
    delta_x = (target_x - current_x) * max(1.0e-9, float(coordinate_scale_x))
    delta_y = (target_y - current_y) * max(1.0e-9, float(coordinate_scale_y))
    if math.hypot(delta_x, delta_y) <= 1.0e-9:
        return None
    return math.degrees(math.pi + math.atan2(delta_x, delta_y)) % 360.0


def signed_heading_error_degrees(current: float, desired: float) -> float:
    error = (float(desired) - float(current) + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(error, -180.0, abs_tol=1.0e-9) else error
