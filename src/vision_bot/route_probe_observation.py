from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_bot.combat import CombatState, detect_combat_state
from vision_bot.game_state import GameState, detect_game_state
from vision_bot.recognition import recognize_ore_point_classes
from vision_bot.runtime_markers import (
    RuntimeTelemetry,
    detect_armor_critical_marker,
    detect_forbidden_subzone_marker,
    detect_mounted_marker,
    read_runtime_telemetry,
)


@dataclass(frozen=True)
class RouteFrameObservation:
    """Immutable perception result consumed by one route-control iteration."""

    timestamp: float
    telemetry: RuntimeTelemetry | None
    game_state: GameState
    combat: CombatState
    bright_ore_points: tuple[tuple[int, int], ...]
    dark_ore_points: tuple[tuple[int, int], ...]
    armor_critical: bool
    mounted: bool
    forbidden_subzone_visible: bool


def observe_route_frame(
    frame: np.ndarray,
    config: dict[str, Any],
    *,
    timestamp: float,
    minimap: np.ndarray | None = None,
    recognize_ore: bool = False,
) -> RouteFrameObservation:
    bright_ore_points: list[tuple[int, int]] = []
    dark_ore_points: list[tuple[int, int]] = []
    if recognize_ore and minimap is not None:
        bright_ore_points, dark_ore_points = recognize_ore_point_classes(minimap, config)

    return RouteFrameObservation(
        timestamp=timestamp,
        telemetry=read_runtime_telemetry(frame, config),
        game_state=detect_game_state(frame),
        combat=detect_combat_state(frame, config),
        bright_ore_points=tuple(bright_ore_points),
        dark_ore_points=tuple(dark_ore_points),
        armor_critical=detect_armor_critical_marker(frame, config),
        mounted=detect_mounted_marker(frame, config),
        forbidden_subzone_visible=detect_forbidden_subzone_marker(frame, config),
    )
