from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from vision_bot.heading_navigation import (
    HeadingNavigationCommand,
    HeadingNavigationController,
    VisibleHeadingTracker,
)
from vision_bot.movement import MovementNavigator


@dataclass(frozen=True)
class RouteMotionSnapshot:
    enabled: bool
    mode: str
    heading_degrees: float | None
    heading_fresh: bool
    command: HeadingNavigationCommand | None

    @property
    def forward_held(self) -> bool:
        return bool(self.command is not None and self.command.hold_forward)

    @property
    def aligned(self) -> bool:
        return bool(
            self.command is not None
            and self.command.turn_key is None
            and self.command.heading_error_degrees is not None
        )

    def to_dict(self) -> dict[str, Any]:
        command = self.command
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "heading_degrees": self.heading_degrees,
            "heading_fresh": self.heading_fresh,
            "aligned": self.aligned,
            "hold_forward": command.hold_forward if command is not None else None,
            "turn_key": command.turn_key if command is not None else None,
            "desired_heading_degrees": (
                command.desired_heading_degrees if command is not None else None
            ),
            "heading_error_degrees": (
                command.heading_error_degrees if command is not None else None
            ),
            "braking": command.braking if command is not None else None,
            "reason": command.reason if command is not None else None,
            "turn_hold_seconds": (
                command.turn_hold_seconds if command is not None else None
            ),
        }


class RouteMotionController:
    """Own normal route movement while preserving a tested legacy fallback."""

    def __init__(
        self,
        navigator: MovementNavigator,
        *,
        enabled: bool,
        fallback_to_legacy: bool,
        heading_tracker: VisibleHeadingTracker,
        heading_controller: HeadingNavigationController,
        reset_after_seconds: float,
    ) -> None:
        self.navigator = navigator
        self.enabled = bool(enabled)
        self.fallback_to_legacy = bool(fallback_to_legacy)
        self.heading_tracker = heading_tracker
        self.heading_controller = heading_controller
        self.reset_after_seconds = max(0.0, float(reset_after_seconds))
        self.last_applied_at: float | None = None
        self.last_snapshot = RouteMotionSnapshot(
            enabled=self.enabled,
            mode="idle",
            heading_degrees=None,
            heading_fresh=False,
            command=None,
        )

    @classmethod
    def from_config(
        cls,
        navigator: MovementNavigator,
        config: dict[str, Any],
    ) -> "RouteMotionController":
        movement_cfg = config.get("movement", {})
        cfg = movement_cfg.get("heading_control", {})
        if not isinstance(cfg, dict):
            cfg = {}
        positive_error_turn_key = str(
            cfg.get("positive_error_turn_key", "A")
        ).upper()
        if positive_error_turn_key not in {"A", "D"}:
            positive_error_turn_key = "A"
        return cls(
            navigator,
            enabled=bool(cfg.get("enabled", False)),
            fallback_to_legacy=bool(cfg.get("fallback_to_legacy", True)),
            heading_tracker=VisibleHeadingTracker(
                max_age_seconds=float(cfg.get("max_heading_age_seconds", 0.60))
            ),
            heading_controller=HeadingNavigationController(
                turn_engage_degrees=float(cfg.get("turn_engage_degrees", 9.0)),
                turn_release_degrees=float(cfg.get("turn_release_degrees", 4.0)),
                pivot_degrees=float(cfg.get("pivot_degrees", 100.0)),
                positive_error_turn_key=positive_error_turn_key,
                turn_rate_degrees_per_second=float(
                    cfg.get("turn_rate_degrees_per_second", 110.0)
                ),
                min_turn_pulse_seconds=float(
                    cfg.get("min_turn_pulse_seconds", 0.06)
                ),
                max_turn_pulse_seconds=float(
                    cfg.get("max_turn_pulse_seconds", 0.35)
                ),
                opposite_turn_lock_seconds=float(
                    cfg.get("opposite_turn_lock_seconds", 2.50)
                ),
                opposite_turn_engage_degrees=float(
                    cfg.get("opposite_turn_engage_degrees", 28.0)
                ),
            ),
            reset_after_seconds=float(cfg.get("reset_after_seconds", 0.75)),
        )

    def observe_heading(self, heading_degrees: float | None, *, now: float) -> None:
        self.heading_tracker.observe(heading_degrees, now=now)

    def move_towards(
        self,
        current_coord: int,
        target_coord: int,
        *,
        now: float,
    ) -> RouteMotionSnapshot:
        if not self.enabled:
            self.navigator.move_towards(current_coord, target_coord)
            return self._record("legacy_disabled", now=now, command=None)

        if (
            self.last_applied_at is None
            or now - self.last_applied_at > self.reset_after_seconds
        ):
            self.heading_controller.reset()

        heading, fresh = self.heading_tracker.current(now=now)
        command = self.heading_controller.plan(
            current_coord=current_coord,
            target_coord=target_coord,
            heading_degrees=heading,
            heading_fresh=fresh,
            now=now,
        )
        if command.heading_available and fresh:
            self.navigator.apply_heading_command(
                command,
                current_coord=current_coord,
                target_coord=target_coord,
            )
            return self._record("visible_heading", now=now, command=command)

        if self.fallback_to_legacy:
            self.heading_controller.reset()
            self.navigator.move_towards(current_coord, target_coord)
            return self._record("legacy_heading_unavailable", now=now, command=command)

        self.navigator.apply_heading_command(
            command,
            current_coord=current_coord,
            target_coord=target_coord,
        )
        return self._record("heading_fail_closed", now=now, command=command)

    def face_towards(
        self,
        current_coord: int,
        target_coord: int,
        *,
        now: float,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> RouteMotionSnapshot:
        """Turn toward a coordinate without moving forward."""
        if (
            self.last_applied_at is None
            or now - self.last_applied_at > self.reset_after_seconds
        ):
            self.heading_controller.reset()

        heading, fresh = self.heading_tracker.current(now=now)
        command = self.heading_controller.plan(
            current_coord=current_coord,
            target_coord=target_coord,
            heading_degrees=heading,
            heading_fresh=fresh,
            now=now,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
        )
        face_command = replace(
            command,
            hold_forward=False,
            braking=command.turn_key is not None,
            reason=f"face_{command.reason}",
        )
        self.navigator.apply_heading_command(
            face_command,
            current_coord=current_coord,
            target_coord=target_coord,
        )
        mode = "face_visible_heading" if command.heading_available and fresh else "face_heading_unavailable"
        return self._record(mode, now=now, command=face_command)

    def suspend(self) -> None:
        self.heading_controller.reset()
        self.navigator.release_heading_control()
        self.last_applied_at = None
        heading, fresh = self.heading_tracker.current(now=float("inf"))
        self.last_snapshot = RouteMotionSnapshot(
            enabled=self.enabled,
            mode="suspended",
            heading_degrees=heading,
            heading_fresh=fresh,
            command=None,
        )

    @property
    def forward_held(self) -> bool:
        if "W" in self.navigator.heading_held_keys:
            return True
        return self.navigator.held_key == "W"

    def _record(
        self,
        mode: str,
        *,
        now: float,
        command: HeadingNavigationCommand | None,
    ) -> RouteMotionSnapshot:
        heading, fresh = self.heading_tracker.current(now=now)
        self.last_applied_at = now
        self.last_snapshot = RouteMotionSnapshot(
            enabled=self.enabled,
            mode=mode,
            heading_degrees=heading,
            heading_fresh=fresh,
            command=command,
        )
        return self.last_snapshot
