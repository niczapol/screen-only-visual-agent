from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from vision_bot.mounting import MountTravelState
from vision_bot.movement import InputController, MovementNavigator


@dataclass(frozen=True)
class HardStuckEscapeSettings:
    enabled: bool = True
    dismount_settle_seconds: float = 0.25
    back_seconds: float = 0.75
    turn_seconds: float = 1.95
    turn_pulse_seconds: float = 0.39
    forward_seconds: float = 0.90
    remount_delay_seconds: float = 2.0
    forward_waypoint_padding: int = 6

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HardStuckEscapeSettings":
        raw = config.get("movement", {}).get("hard_stuck_escape", {})
        cfg = raw if isinstance(raw, dict) else {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            dismount_settle_seconds=max(
                0.0, float(cfg.get("dismount_settle_seconds", 0.25))
            ),
            back_seconds=max(0.0, float(cfg.get("back_seconds", 0.75))),
            turn_seconds=max(0.0, float(cfg.get("turn_seconds", 1.95))),
            turn_pulse_seconds=max(
                0.04, float(cfg.get("turn_pulse_seconds", 0.39))
            ),
            forward_seconds=max(0.0, float(cfg.get("forward_seconds", 0.90))),
            remount_delay_seconds=max(
                0.0, float(cfg.get("remount_delay_seconds", 2.0))
            ),
            forward_waypoint_padding=max(
                0, int(cfg.get("forward_waypoint_padding", 6))
            ),
        )


def perform_hard_stuck_escape(
    *,
    navigator: MovementNavigator,
    input_controller: InputController,
    mount_state: MountTravelState,
    mounted: bool,
    turn_key: str,
    settings: HardStuckEscapeSettings,
    now: float,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Perform one bounded, last-resort escape after a progress timeout.

    Ordinary obstacle recovery remains mounted and lightweight. This sequence is
    deliberately reserved for a confirmed hard stall: stop, dismount, create
    clearance behind the character, yaw about 180 degrees using bounded RMB
    pulses, and leave the contact point on foot.
    """

    if not settings.enabled:
        return None

    key = str(turn_key).strip().upper()
    if key not in {"A", "D"}:
        key = "D"

    navigator.stop()
    if mounted:
        input_controller.tap_key(mount_state.key, duration=0.06)
        if settings.dismount_settle_seconds > 0.0:
            sleep(settings.dismount_settle_seconds)

    if settings.back_seconds > 0.0:
        input_controller.tap_key("S", duration=settings.back_seconds)

    remaining_turn = settings.turn_seconds
    while remaining_turn > 1e-6:
        pulse = min(settings.turn_pulse_seconds, remaining_turn)
        navigator.turn_character(key, duration=pulse, context="recovery")
        remaining_turn -= pulse

    if settings.forward_seconds > 0.0:
        input_controller.tap_key("W", duration=settings.forward_seconds)

    mount_state.defer_mount(
        now=now,
        seconds=(
            settings.dismount_settle_seconds
            + settings.back_seconds
            + settings.turn_seconds
            + settings.forward_seconds
            + settings.remount_delay_seconds
        ),
    )
    return "hard_stuck_dismount_reverse_escape"
