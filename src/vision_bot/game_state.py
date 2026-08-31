from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GameState:
    death_or_blocking_modal: bool
    red_button_count: int
    ghost_visual: bool
    ghost_button_count: int = 0
    alive_health_bar: bool = False
    dismissable_modal_click: tuple[int, int] | None = None


def detect_game_state(frame: np.ndarray) -> GameState:
    if frame.size == 0:
        return GameState(False, 0, False)

    height, width = frame.shape[:2]
    alive_health_bar = _detect_alive_player_health_bar(frame)
    if alive_health_bar:
        # A saturated visible player-health fill is authoritative live-state
        # evidence. Skip the large world-palette percentiles and ghost-button
        # masks on the normal path; they only disambiguate an empty health bar.
        return GameState(
            death_or_blocking_modal=False,
            red_button_count=0,
            ghost_visual=False,
            ghost_button_count=0,
            alive_health_bar=True,
            dismissable_modal_click=None,
        )
    ghost_visual = _detect_ghost_visual(frame)
    ghost_button_count = _count_ghost_action_buttons(frame)
    # Startup popups are intentionally ignored. Their button area overlaps the
    # action bar, so unattended dismissal is less safe than leaving them open.
    dismissable_modal_click = None
    top = int(height * 0.06)
    bottom = int(height * 0.25)
    left = int(width * 0.28)
    right = int(width * 0.72)
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return GameState(
            not alive_health_bar and (ghost_visual or ghost_button_count >= 2),
            0,
            ghost_visual,
            ghost_button_count,
            alive_health_bar,
            dismissable_modal_click,
        )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 90, 35]), np.array([12, 255, 210]))
    red_high = cv2.inRange(hsv, np.array([170, 90, 35]), np.array([179, 255, 210]))
    red_mask = cv2.bitwise_or(red_low, red_high)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    num_labels, _labels, stats, centers = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    button_count = 0
    for label_index in range(1, num_labels):
        _x, _y, component_width, component_height, area = stats[label_index]
        component_center_x = left + float(centers[label_index][0])
        component_center_y = top + float(centers[label_index][1])
        if (
            area >= 3000
            and 130 <= component_width <= 360
            and 22 <= component_height <= 55
            and component_width / max(1, component_height) >= 2.5
            and width * 0.36 <= component_center_x <= width * 0.64
            and height * 0.18 <= component_center_y <= height * 0.34
        ):
            button_count += 1

    return GameState(
        death_or_blocking_modal=(
            not alive_health_bar
            and (
                button_count >= 2
                or ghost_button_count >= 2
                or ghost_visual
            )
        ),
        red_button_count=button_count,
        ghost_visual=ghost_visual,
        ghost_button_count=ghost_button_count,
        alive_health_bar=alive_health_bar,
        dismissable_modal_click=dismissable_modal_click,
    )


def _detect_alive_player_health_bar(frame: np.ndarray) -> bool:
    """Veto palette-only ghost detection when the player has visible health."""
    height, width = frame.shape[:2]
    left = int(round(width * (155.0 / 2560.0)))
    top = int(round(height * (70.0 / 1440.0)))
    right = int(round(width * (345.0 / 2560.0)))
    bottom = int(round(height * (95.0 / 1440.0)))
    roi = frame[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Empty ghost bars are blue/gray. Their unstable low-saturation hue can fall
    # inside the green range, so only a genuinely saturated live-health fill may
    # veto otherwise conclusive ghost evidence.
    green = cv2.inRange(hsv, np.array([35, 140, 50]), np.array([95, 255, 255]))
    min_pixels = max(1, int(round(roi.shape[0] * (4.0 / 25.0))))
    filled_columns = np.count_nonzero(np.count_nonzero(green, axis=0) >= min_pixels)
    # A critically low but living player can have only a few filled columns.
    # Requiring 10% made sandy Tanaris frames look like the desaturated ghost
    # world shortly before the real death dialog appeared.
    return float(filled_columns) / max(1, roi.shape[1]) >= 0.02


def _detect_ghost_visual(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.15) : int(height * 0.75), int(width * 0.08) : int(width * 0.82)]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturation_p10 = float(np.percentile(saturation, 10))
    saturation_p90 = float(np.percentile(saturation, 90))
    return (
        float(np.mean(value)) > 95.0
        and float(np.percentile(value, 90)) > 130.0
        and float(np.mean(saturation)) < 130.0
        and saturation_p90 < 155.0
        and saturation_p90 - saturation_p10 < 25.0
        and float(np.mean((saturation >= 70) & (saturation <= 110))) > 0.80
        and float(np.mean(saturation > 170)) < 0.03
    )


def _count_ghost_action_buttons(frame: np.ndarray) -> int:
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.06) : int(height * 0.22), int(width * 0.28) : int(width * 0.72)]
    if roi.size == 0:
        return 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([45, 255, 255]))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((9, 21), dtype=np.uint8))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, np.ones((3, 9), dtype=np.uint8))

    num_labels, _labels, stats, centers = cv2.connectedComponentsWithStats(yellow_mask, connectivity=8)
    button_count = 0
    for label_index in range(1, num_labels):
        _x, _y, component_width, component_height, area = stats[label_index]
        component_center_x = int(width * 0.28) + float(centers[label_index][0])
        component_center_y = int(height * 0.06) + float(centers[label_index][1])
        aspect = component_width / max(1, component_height)
        if (
            1400 <= int(area) <= 7000
            and 80 <= component_width <= 180
            and 24 <= component_height <= 55
            and aspect >= 2.3
            and width * 0.34 <= component_center_x <= width * 0.66
            and height * 0.10 <= component_center_y <= height * 0.18
        ):
            button_count += 1
    return button_count
