from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Any, Protocol

import cv2
import numpy as np

from vision_bot.config import resource_path
from vision_bot.cursor_classifier import (
    CursorClassification,
    CursorSnapshot,
    CursorTemplateClassifier,
    capture_win32_cursor_snapshot,
    cursor_colorfulness,
    perceptual_cursor_hash,
)
from vision_bot.ore_world_detector import (
    OreWorldDetector,
    OreWorldDetectorResult,
    build_ore_detector_probe_points,
)
from vision_bot.regions import crop_region, resolve_region


class MouseLike(Protocol):
    def move_to(self, screen_x: int, screen_y: int) -> None: ...

    def right_click(self, duration: float = 0.05) -> None: ...


class CaptureLike(Protocol):
    def capture_client_region(self) -> np.ndarray: ...

    def client_to_screen_point(self, x: int, y: int) -> tuple[int, int]: ...


@dataclass(frozen=True)
class HoverResult:
    found: bool
    point: tuple[int, int] | None = None
    reason: str = ""
    scanned_points: int = 0


class HoverScanPhase(str, Enum):
    IDLE = "idle"
    BASELINE_SETTLE = "baseline_settle"
    PROBE_SETTLE = "probe_settle"
    FOUND = "found"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class HoverScanStep:
    phase: HoverScanPhase
    point: tuple[int, int] | None = None
    reason: str = ""
    scanned_points: int = 0
    cursor_label: str | None = None
    cursor_similarity: float = 0.0


@dataclass(frozen=True)
class MiningTargetAlignment:
    aligned: bool
    turn_key: str | None
    turn_seconds: float
    horizontal_offset: float


class MiningPhase(str, Enum):
    IDLE = "idle"
    INTERCEPT = "intercept"
    CENTER_TOOLTIP = "center_tooltip"
    FACE_NODE = "face_node"
    WORLD_SCAN = "world_scan"
    GATHER_WAIT = "gather_wait"
    VERIFY = "verify"
    SUSPENDED = "suspended"
    RESUME = "resume"


@dataclass(frozen=True)
class OrePresence:
    bright_points: tuple[tuple[int, int], ...]
    dark_points: tuple[tuple[int, int], ...]
    bright_streak: int
    confirmed_bright: bool
    dark_only: bool
    confirmed_point: tuple[int, int] | None = None
    confirmed_track_id: int | None = None
    target_visible: bool = False
    target_dark_visible: bool = False


@dataclass(frozen=True)
class MiningOutcome:
    success: bool
    reason: str
    node_id: int | None
    node_coord: int
    elapsed_seconds: float
    marker_point: tuple[int, int] | None = None
    resume_coords: tuple[int, ...] = ()
    resume_route_index: int | None = None


@dataclass(frozen=True)
class MiningCandidate:
    node_id: int | None
    coord: int
    ore_type: str
    access_plan: Any | None = None
    marker_intercept_coord: int | None = None
    tooltip_confirmed: bool = False
    centered_tooltip_confirmed: bool = False
    source: str = "database"


@dataclass
class _MarkerTrack:
    track_id: int
    point: tuple[int, int]
    streak: int = 1
    missing_frames: int = 0


class BrightOreTracker:
    """Track minimap marker hypotheses so unrelated bright points cannot verify a mine."""

    def __init__(self, *, max_jump_pixels: float, max_missing_frames: int) -> None:
        self.max_jump_pixels = max(1.0, float(max_jump_pixels))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self._tracks: list[_MarkerTrack] = []
        self._next_track_id = 1

    def update(self, points: tuple[tuple[int, int], ...]) -> tuple[_MarkerTrack, ...]:
        unmatched = set(range(len(points)))
        for track in sorted(self._tracks, key=lambda item: (-item.streak, item.track_id)):
            nearest = min(
                unmatched,
                key=lambda index: _point_distance(track.point, points[index]),
                default=None,
            )
            if (
                nearest is not None
                and _point_distance(track.point, points[nearest]) <= self.max_jump_pixels
            ):
                track.point = points[nearest]
                track.streak += 1
                track.missing_frames = 0
                unmatched.remove(nearest)
            else:
                track.missing_frames += 1

        self._tracks = [
            track
            for track in self._tracks
            if track.missing_frames <= self.max_missing_frames
        ]
        for point_index in sorted(unmatched):
            self._tracks.append(_MarkerTrack(self._next_track_id, points[point_index]))
            self._next_track_id += 1
        return tuple(self._tracks)

    def get(self, track_id: int | None) -> _MarkerTrack | None:
        if track_id is None:
            return None
        return next((track for track in self._tracks if track.track_id == track_id), None)

    def reset(self) -> None:
        self._tracks.clear()


def _point_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _normalize_ore_type(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().split())
    for prefix in ("ooze covered ",):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    for suffix in (" deposit", " vein"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def _ore_types_match(database_ore_type: str, tooltip_ore_type: str) -> bool:
    return _normalize_ore_type(database_ore_type) == _normalize_ore_type(
        tooltip_ore_type
    )


class MinimapTooltipProbePhase(str, Enum):
    IDLE = "idle"
    HOVER = "hover"
    CONFIRMED = "confirmed"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class MinimapTooltipProbeStep:
    phase: MinimapTooltipProbePhase
    action: str
    move_point: tuple[int, int] | None = None
    release_cursor: bool = False
    ore_id: int | None = None
    ore_type: str | None = None
    track_id: int | None = None


class MinimapTooltipProbeController:
    """Bounded hover gate between a CV minimap blip and a database candidate."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        mining_cfg = (config or {}).get("mining", {})
        probe_cfg = mining_cfg.get("minimap_tooltip_probe", {})
        self.enabled = bool(
            mining_cfg.get("route_live_enabled", False)
            and probe_cfg.get("enabled", False)
        )
        self.required = bool(
            mining_cfg.get("require_minimap_tooltip_confirmation", False)
        )
        self.settle_seconds = max(
            0.0, float(probe_cfg.get("settle_seconds", 0.10))
        )
        self.timeout_seconds = max(
            self.settle_seconds,
            float(probe_cfg.get("timeout_seconds", 0.70)),
        )
        self.retry_cooldown_seconds = max(
            0.0, float(probe_cfg.get("retry_cooldown_seconds", 2.0))
        )
        self.confirmation_hold_seconds = max(
            0.0, float(probe_cfg.get("confirmation_hold_seconds", 0.75))
        )
        self.phase = MinimapTooltipProbePhase.IDLE
        self.active_track_id: int | None = None
        self.active_point: tuple[int, int] | None = None
        self.started_at: float | None = None
        self.confirmed_track_id: int | None = None
        self.confirmed_point: tuple[int, int] | None = None
        self.confirmed_ore_id: int | None = None
        self.confirmed_ore_type: str | None = None
        self.confirmed_at: float | None = None
        self.cooldowns: dict[int, float] = {}
        self.last_action = "minimap_tooltip_probe_idle"

    @property
    def active(self) -> bool:
        return self.phase == MinimapTooltipProbePhase.HOVER

    @property
    def confirmed(self) -> bool:
        return (
            self.confirmed_track_id is not None
            and self.confirmed_ore_id is not None
            and self.confirmed_ore_type is not None
        )

    def observe(
        self,
        *,
        track_id: int | None,
        marker_point: tuple[int, int] | None,
        now: float,
        tooltip_ore_id: int | None = None,
        tooltip_ore_type: str | None = None,
        control_available: bool = True,
    ) -> MinimapTooltipProbeStep:
        if not self.enabled:
            self.phase = MinimapTooltipProbePhase.IDLE
            self.last_action = "minimap_tooltip_probe_disabled"
            return self._step(self.last_action)

        self._expire_cooldowns(now)
        if not control_available:
            return self.cancel("control_preempted", now=now)

        if self.active:
            elapsed = max(
                0.0,
                now - (self.started_at if self.started_at is not None else now),
            )
            if (
                elapsed >= self.settle_seconds
                and tooltip_ore_id is not None
                and tooltip_ore_type is not None
            ):
                self.confirmed_track_id = self.active_track_id
                self.confirmed_point = self.active_point
                self.confirmed_ore_id = int(tooltip_ore_id)
                self.confirmed_ore_type = str(tooltip_ore_type)
                self.confirmed_at = now
                self.phase = MinimapTooltipProbePhase.CONFIRMED
                self.started_at = None
                self.last_action = "minimap_tooltip_probe_confirmed"
                return self._step(self.last_action, release_cursor=True)
            if elapsed >= self.timeout_seconds:
                failed_track_id = self.active_track_id
                if failed_track_id is not None:
                    self.cooldowns[failed_track_id] = now + self.retry_cooldown_seconds
                self._clear_active()
                self.phase = MinimapTooltipProbePhase.COOLDOWN
                self.last_action = "minimap_tooltip_probe_timeout"
                return self._step(
                    self.last_action,
                    release_cursor=True,
                    track_id=failed_track_id,
                )
            # Hovering can temporarily obscure the CV blip, so the active track
            # is intentionally retained until the bounded timeout expires.
            self.last_action = "minimap_tooltip_probe_wait"
            return self._step(self.last_action, track_id=self.active_track_id)

        if self.confirmed_track_id is not None:
            missing_within_hold = bool(
                track_id is None
                and self.confirmed_at is not None
                and now - self.confirmed_at <= self.confirmation_hold_seconds
            )
            if track_id == self.confirmed_track_id or missing_within_hold:
                if track_id == self.confirmed_track_id and marker_point is not None:
                    self.confirmed_point = (
                        int(marker_point[0]),
                        int(marker_point[1]),
                    )
                self.phase = MinimapTooltipProbePhase.CONFIRMED
                self.last_action = "minimap_tooltip_probe_confirmed_cached"
                return self._step(self.last_action)
            self._clear_confirmation()

        if track_id is None or marker_point is None:
            self.phase = MinimapTooltipProbePhase.IDLE
            self.last_action = "minimap_tooltip_probe_no_bright_track"
            return self._step(self.last_action)

        if self.cooldowns.get(track_id, 0.0) > now:
            self.phase = MinimapTooltipProbePhase.COOLDOWN
            self.last_action = "minimap_tooltip_probe_cooldown"
            return self._step(self.last_action, track_id=track_id)

        self.phase = MinimapTooltipProbePhase.HOVER
        self.active_track_id = int(track_id)
        self.active_point = (int(marker_point[0]), int(marker_point[1]))
        self.started_at = now
        self.last_action = "minimap_tooltip_probe_hover"
        return self._step(
            self.last_action,
            move_point=self.active_point,
            track_id=self.active_track_id,
        )

    def cancel(self, reason: str, *, now: float) -> MinimapTooltipProbeStep:
        was_active = self.active
        self._clear_active()
        self.phase = MinimapTooltipProbePhase.IDLE
        self.last_action = f"minimap_tooltip_probe_cancelled:{reason}"
        return self._step(self.last_action, release_cursor=was_active)

    def _step(
        self,
        action: str,
        *,
        move_point: tuple[int, int] | None = None,
        release_cursor: bool = False,
        track_id: int | None = None,
    ) -> MinimapTooltipProbeStep:
        return MinimapTooltipProbeStep(
            phase=self.phase,
            action=action,
            move_point=move_point,
            release_cursor=release_cursor,
            ore_id=self.confirmed_ore_id,
            ore_type=self.confirmed_ore_type,
            track_id=(
                track_id
                if track_id is not None
                else self.confirmed_track_id or self.active_track_id
            ),
        )

    def _clear_active(self) -> None:
        self.active_track_id = None
        self.active_point = None
        self.started_at = None

    def _clear_confirmation(self) -> None:
        self.confirmed_track_id = None
        self.confirmed_point = None
        self.confirmed_ore_id = None
        self.confirmed_ore_type = None
        self.confirmed_at = None

    def _expire_cooldowns(self, now: float) -> None:
        self.cooldowns = {
            track_id: deadline
            for track_id, deadline in self.cooldowns.items()
            if deadline > now
        }

    def snapshot(self, *, now: float) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required": self.required,
            "phase": self.phase.value,
            "active": self.active,
            "active_track_id": self.active_track_id,
            "confirmed_track_id": self.confirmed_track_id,
            "confirmed_point": (
                list(self.confirmed_point) if self.confirmed_point is not None else None
            ),
            "ore_id": self.confirmed_ore_id,
            "ore_type": self.confirmed_ore_type,
            "last_action": self.last_action,
            "remaining_seconds": (
                max(0.0, self.timeout_seconds - (now - self.started_at))
                if self.active and self.started_at is not None
                else 0.0
            ),
        }


def minimap_marker_client_point(
    marker_point: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    cfg = config or {}
    x, y, width, height = resolve_region(cfg.get("minimap", {}), frame_shape, cfg)
    marker_x = max(0, min(width - 1, int(marker_point[0])))
    marker_y = max(0, min(height - 1, int(marker_point[1])))
    return x + marker_x, y + marker_y


class MiningCycleController:
    """Pure timing/state authority for one minimap-confirmed mining transaction."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        mining_cfg = (config or {}).get("mining", {})
        self.enabled = bool(mining_cfg.get("route_live_enabled", False))
        self.confirm_frames = max(1, int(mining_cfg.get("bright_confirm_frames", 2)))
        self.clear_frames = max(1, int(mining_cfg.get("verify_clear_frames", 2)))
        self.gather_wait_seconds = max(0.0, float(mining_cfg.get("gather_wait_seconds", 3.0)))
        self.verify_timeout_seconds = max(
            self.gather_wait_seconds,
            float(mining_cfg.get("verify_timeout_seconds", 6.0)),
        )
        self.success_cooldown_seconds = max(0.0, float(mining_cfg.get("cooldown_seconds", 180.0)))
        self.failure_cooldown_seconds = max(
            0.0,
            float(mining_cfg.get("failure_cooldown_seconds", 20.0)),
        )
        self.failure_spatial_cooldown_radius = max(
            0.0,
            float(mining_cfg.get("failure_spatial_cooldown_radius_coord", 0.35)),
        )
        self.centered_marker_radius_pixels = max(
            1.0,
            float(mining_cfg.get("centered_marker_radius_pixels", 6.0)),
        )
        self.centered_marker_occlusion_radius_pixels = max(
            self.centered_marker_radius_pixels,
            float(mining_cfg.get("centered_marker_occlusion_radius_pixels", 18.0)),
        )
        self.anchor_local_world_validation_enabled = bool(
            mining_cfg.get("anchor_local_world_validation_enabled", False)
        )
        self.anchor_local_world_validation_radius_pixels = max(
            self.centered_marker_occlusion_radius_pixels,
            float(
                mining_cfg.get(
                    "anchor_local_world_validation_radius_pixels",
                    24.0,
                )
            ),
        )
        self.anchor_local_world_validation_seconds = max(
            0.0,
            float(
                mining_cfg.get(
                    "anchor_local_world_validation_seconds",
                    3.0,
                )
            ),
        )
        center_probe_cfg = mining_cfg.get("center_tooltip_probe", {})
        self.center_tooltip_settle_seconds = max(
            0.0,
            float(center_probe_cfg.get("settle_seconds", 0.12)),
        )
        self.center_tooltip_timeout_seconds = max(
            self.center_tooltip_settle_seconds,
            float(center_probe_cfg.get("timeout_seconds", 0.90)),
        )
        self.center_tooltip_world_scan_fallback_enabled = bool(
            mining_cfg.get("center_tooltip_world_scan_fallback_enabled", False)
        )
        self.dark_failure_cooldown_seconds = max(
            self.failure_cooldown_seconds,
            float(mining_cfg.get("dark_failure_cooldown_seconds", 3600.0)),
        )
        self.max_total_seconds = max(
            0.0,
            float(mining_cfg.get("max_total_seconds", 35.0)),
        )
        self.max_intercept_seconds = max(
            0.0,
            float(mining_cfg.get("max_intercept_seconds", 20.0)),
        )
        self.intercept_no_progress_seconds = max(
            0.0,
            float(mining_cfg.get("intercept_no_progress_seconds", 4.0)),
        )
        self.intercept_progress_pixels = max(
            0.0,
            float(mining_cfg.get("intercept_progress_pixels", 3.0)),
        )
        self.intercept_progress_coord = max(
            0.0,
            float(mining_cfg.get("intercept_progress_coord", 0.02)),
        )
        self.max_world_scan_seconds = max(
            0.0,
            float(mining_cfg.get("max_world_scan_seconds", 10.0)),
        )
        self.max_face_node_seconds = max(
            0.0,
            float(mining_cfg.get("max_face_node_seconds", 2.5)),
        )
        self.face_node_timeout_starts_world_scan = bool(
            mining_cfg.get("face_node_timeout_starts_world_scan", True)
        )
        self.dark_target_confirm_frames = max(
            1,
            int(mining_cfg.get("dark_target_confirm_frames", 2)),
        )
        self.world_search_distance = max(
            0.0,
            float(mining_cfg.get("world_search_distance_coord", 0.22)),
        )
        self.candidate_handoff_distance = max(
            0.0,
            float(
                mining_cfg.get(
                    "candidate_handoff_distance_coord",
                    self.world_search_distance,
                )
            ),
        )
        self.final_world_scan_distance = max(
            0.01,
            float(
                mining_cfg.get(
                    "final_world_scan_distance_coord",
                    self.world_search_distance,
                )
            ),
        )
        self.candidate_match_distance = max(
            self.world_search_distance,
            float(mining_cfg.get("candidate_match_distance_coord", 1.35)),
        )
        # A native ore tooltip establishes live presence and exact ore type.
        # Its DB identity search must therefore cover the visible minimap
        # envelope, rather than being clipped by the much smaller ordinary
        # player-to-node admission radius. Access ownership remains separately
        # gated by the precomputed route attachment below.
        self.tooltip_confirmed_database_search_distance = max(
            self.candidate_match_distance,
            float(
                mining_cfg.get(
                    "tooltip_confirmed_database_search_distance_coord",
                    2.0,
                )
            ),
        )
        self.require_access_plan = bool(
            mining_cfg.get("require_access_plan_for_route_live", False)
        )
        self.access_attachment_max_distance = max(
            0.0,
            float(mining_cfg.get("access_attachment_max_distance_coord", 0.0)),
        )
        self.access_option_retry_enabled = bool(
            mining_cfg.get("access_option_retry_enabled", True)
        )
        self.max_access_option_retries = max(
            0,
            int(mining_cfg.get("max_access_option_retries", 1)),
        )
        rec_cfg = (config or {}).get("recognition", {})
        route_cfg = (config or {}).get("route", {})
        center_cfg = rec_cfg.get(
            "sensor_circle_center_fraction", {"x": 0.50, "y": 0.48}
        )
        self.marker_center_x_fraction = float(center_cfg.get("x", 0.50))
        self.marker_center_y_fraction = float(center_cfg.get("y", 0.48))
        self.marker_radius_fraction = float(
            rec_cfg.get("sensor_circle_radius_fraction", 0.37)
        )
        self.marker_tracking_radius = max(
            0.01,
            float(route_cfg.get("tracking_radius_coord", 1.15)),
        )
        self.marker_tracking_radius_x = max(
            0.01,
            float(mining_cfg.get("marker_tracking_radius_x_coord", 0.72)),
        )
        self.marker_tracking_radius_y = max(
            0.01,
            float(mining_cfg.get("marker_tracking_radius_y_coord", 1.09)),
        )
        self.direct_marker_intercept_enabled = bool(
            mining_cfg.get("direct_marker_intercept_enabled", True)
        )
        self.marker_database_match_distance = max(
            0.0,
            float(mining_cfg.get("marker_database_match_distance_coord", 0.55)),
        )
        self.require_minimap_tooltip_confirmation = bool(
            mining_cfg.get("require_minimap_tooltip_confirmation", False)
        )
        self.tooltip_confirmed_relaxed_match_enabled = bool(
            mining_cfg.get("tooltip_confirmed_relaxed_match_enabled", True)
        )
        self.tooltip_confirmed_unmatched_enabled = bool(
            mining_cfg.get("tooltip_confirmed_unmatched_enabled", False)
        )
        self.tooltip_confirmed_unmatched_max_distance = max(
            0.0,
            float(
                mining_cfg.get(
                    "tooltip_confirmed_unmatched_max_distance_coord",
                    0.75,
                )
            ),
        )
        self.direct_node_intercept_first_enabled = bool(
            mining_cfg.get("direct_node_intercept_first_enabled", False)
        )
        self.direct_marker_intercept_max_node_delta = max(
            0.0,
            float(mining_cfg.get("direct_marker_intercept_max_node_delta_coord", 1.35)),
        )
        self.direct_marker_world_search_distance = max(
            0.01,
            float(mining_cfg.get("direct_marker_world_search_distance_coord", 0.04)),
        )
        self.direct_marker_world_scan_radius_pixels = max(
            1.0,
            float(mining_cfg.get("direct_marker_world_scan_radius_pixels", 12.0)),
        )
        self.direct_marker_world_scan_candidate_distance = max(
            self.direct_marker_world_search_distance,
            float(
                mining_cfg.get(
                    "direct_marker_world_scan_candidate_distance_coord",
                    0.12,
                )
            ),
        )
        self.lost_marker_world_scan_distance = max(
            self.direct_marker_world_search_distance,
            float(mining_cfg.get("lost_marker_world_scan_distance_coord", 0.45)),
        )
        self.lost_marker_world_scan_radius_pixels = max(
            self.direct_marker_world_scan_radius_pixels,
            float(mining_cfg.get("lost_marker_world_scan_radius_pixels", 36.0)),
        )
        self.intercept_stall_world_scan_distance = max(
            self.final_world_scan_distance,
            float(
                mining_cfg.get(
                    "intercept_stall_world_scan_distance_coord",
                    self.final_world_scan_distance,
                )
            ),
        )
        self.intercept_stall_world_scan_radius_pixels = max(
            self.lost_marker_world_scan_radius_pixels,
            float(
                mining_cfg.get(
                    "intercept_stall_world_scan_radius_pixels",
                    52.0,
                )
            ),
        )
        self.marker_bearing_enabled = bool(
            mining_cfg.get("marker_bearing_enabled", True)
        )
        self.marker_bearing_min_pixels = max(
            0.0, float(mining_cfg.get("marker_bearing_min_pixels", 5.0))
        )
        self.marker_bearing_min_alignment = float(
            mining_cfg.get("marker_bearing_min_alignment", 0.20)
        )
        self.marker_bearing_weight = max(
            0.0, float(mining_cfg.get("marker_bearing_weight", 2.0))
        )
        self.marker_radial_weight = max(
            0.0, float(mining_cfg.get("marker_radial_weight", 1.0))
        )
        self.verify_require_hover_clear = bool(
            mining_cfg.get("verify_require_hover_clear", True)
        )
        self.intercept_marker_missing_limit = max(
            1,
            int(mining_cfg.get("intercept_marker_missing_frames", 3)),
        )
        self.continue_coordinate_approach_after_marker_loss = bool(
            mining_cfg.get(
                "continue_coordinate_approach_after_marker_loss",
                True,
            )
        )
        self.marker_tracker = BrightOreTracker(
            max_jump_pixels=float(mining_cfg.get("marker_track_max_jump_pixels", 24.0)),
            max_missing_frames=max(
                int(mining_cfg.get("marker_track_missing_frames", 1)),
                self.intercept_marker_missing_limit - 1,
            ),
        )
        self.phase = MiningPhase.IDLE
        self.candidate: MiningCandidate | None = None
        self.started_at: float | None = None
        self.phase_started_at: float | None = None
        self.clicked_at: float | None = None
        self.hover_point: tuple[int, int] | None = None
        self.suspended_from: MiningPhase | None = None
        self.suspended_at: float | None = None
        self.bright_streak = 0
        self.clear_streak = 0
        self.intercept_marker_missing_frames = 0
        self.target_marker_id: int | None = None
        self.target_marker_point: tuple[int, int] | None = None
        self.intercept_coords: tuple[int, ...] = ()
        self.intercept_cursor = 0
        self.resume_coords: tuple[int, ...] = ()
        self.resume_route_index: int | None = None
        self.access_option_rank: int | None = None
        self.tried_access_option_ranks: set[int] = set()
        self.access_option_retries = 0
        self.direct_marker_intercept_active = False
        self.direct_node_intercept_active = False
        self.direct_marker_stage_index: int | None = None
        self.database_anchor_arrived = False
        self.live_marker_centering_coord: int | None = None
        self.anchor_local_world_validation_active = False
        self.best_marker_radial_distance: float | None = None
        self.best_intercept_coord_distance: float | None = None
        self.last_intercept_progress_at: float | None = None
        self.last_intercept_target_coord: int | None = None
        self.last_intercept_target_distance: float | None = None
        self.previous_intercept_coord: int | None = None
        self.last_candidate_handoff_reason: str | None = None
        self.last_candidate_segment_distance: float | None = None
        self.dark_target_streak = 0
        self.last_minimap_shape: tuple[int, ...] | None = None
        self.last_presence = OrePresence((), (), 0, False, False)
        self.last_marker_intercept_coord: int | None = None
        self.last_marker_database_distance: float | None = None
        self.last_candidate_rejection_reason: str | None = None
        self.last_access_attachment_distance: float | None = None
        self.last_tooltip_confirmed = False
        self.last_confirmed_ore_type: str | None = None
        self.last_tooltip_relaxed_match = False
        self.last_tooltip_unmatched_candidate = False
        self.last_tooltip_unmatched_distance: float | None = None
        self.last_candidate_search_distance = self.candidate_match_distance
        self.last_candidate_nodes_scanned = 0
        self.last_candidate_same_type_nodes = 0
        self.last_candidate_match_elapsed_ms = 0.0
        self.cooldowns: dict[int, float] = {}
        self.spatial_cooldowns: list[tuple[int, str, float]] = []
        self.center_tooltip_probe_started_at: float | None = None
        self.center_tooltip_confirmed = False
        self.center_tooltip_last_ore_type: str | None = None
        self._pending_outcome: MiningOutcome | None = None

    @property
    def active(self) -> bool:
        return self.phase not in {MiningPhase.IDLE, MiningPhase.RESUME}

    def observe_minimap(
        self,
        bright_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        dark_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    ) -> OrePresence:
        bright = tuple((int(x), int(y)) for x, y in bright_points)
        dark = tuple((int(x), int(y)) for x, y in dark_points)
        tracks = self.marker_tracker.update(bright)
        confirmed_tracks = [
            track
            for track in tracks
            if track.missing_frames == 0 and track.streak >= self.confirm_frames
        ]
        confirmed_track = min(
            confirmed_tracks,
            key=lambda track: (-track.streak, track.track_id),
            default=None,
        )
        target_track = self.marker_tracker.get(self.target_marker_id)
        target_bright_visible = bool(
            target_track is not None and target_track.missing_frames == 0
        )
        if target_bright_visible and target_track is not None:
            self.target_marker_point = target_track.point
        target_dark_point = None
        if (
            not target_bright_visible
            and self.candidate is not None
            and self.target_marker_point is not None
            and dark
        ):
            nearest_dark = min(
                dark,
                key=lambda point: _point_distance(self.target_marker_point, point),
            )
            if (
                _point_distance(self.target_marker_point, nearest_dark)
                <= self.marker_tracker.max_jump_pixels
            ):
                target_dark_point = nearest_dark
                self.target_marker_point = nearest_dark
        target_dark_visible = target_dark_point is not None
        target_visible = target_bright_visible or target_dark_visible
        self.bright_streak = confirmed_track.streak if confirmed_track is not None else 0
        self.last_presence = OrePresence(
            bright_points=bright,
            dark_points=dark,
            bright_streak=self.bright_streak,
            confirmed_bright=confirmed_track is not None,
            dark_only=bool(dark and not bright),
            confirmed_point=confirmed_track.point if confirmed_track is not None else None,
            confirmed_track_id=(
                confirmed_track.track_id if confirmed_track is not None else None
            ),
            target_visible=target_visible,
            target_dark_visible=target_dark_visible,
        )
        return self.last_presence

    def choose_candidate(
        self,
        current_coord: int,
        nodes: list[Any] | tuple[Any, ...],
        *,
        now: float,
        marker_point: tuple[int, int] | None = None,
        minimap_shape: tuple[int, ...] | None = None,
        tooltip_confirmed: bool = False,
        confirmed_ore_type: str | None = None,
    ) -> MiningCandidate | None:
        selection_started_at = time.perf_counter()

        def finish(candidate: MiningCandidate | None) -> MiningCandidate | None:
            self.last_candidate_match_elapsed_ms = max(
                0.0,
                (time.perf_counter() - selection_started_at) * 1000.0,
            )
            return candidate

        self._expire_cooldowns(now)
        if minimap_shape is not None:
            self.last_minimap_shape = tuple(int(value) for value in minimap_shape)
        marker_vector = self._marker_vector(marker_point, minimap_shape)
        marker_intercept_coord = self._marker_intercept_coord(
            current_coord,
            marker_point,
            minimap_shape,
        )
        self.last_marker_intercept_coord = marker_intercept_coord
        self.last_marker_database_distance = None
        self.last_candidate_rejection_reason = None
        self.last_access_attachment_distance = None
        self.last_tooltip_confirmed = bool(
            tooltip_confirmed and confirmed_ore_type
        )
        self.last_confirmed_ore_type = (
            str(confirmed_ore_type) if self.last_tooltip_confirmed else None
        )
        self.last_tooltip_relaxed_match = False
        self.last_tooltip_unmatched_candidate = False
        self.last_tooltip_unmatched_distance = None
        self.last_candidate_nodes_scanned = 0
        self.last_candidate_same_type_nodes = 0
        self.last_candidate_search_distance = (
            self.tooltip_confirmed_database_search_distance
            if self.last_tooltip_confirmed
            else self.candidate_match_distance
        )
        marker_radial_distance = self._marker_radial_distance(
            marker_point,
            minimap_shape,
        )
        centered_tooltip_confirmed = bool(
            self.last_tooltip_confirmed
            and marker_radial_distance is not None
            and marker_radial_distance <= self.centered_marker_radius_pixels
        )
        if self.require_minimap_tooltip_confirmation and not self.last_tooltip_confirmed:
            self.last_candidate_rejection_reason = "minimap_tooltip_unconfirmed"
            return finish(None)

        ranked: list[tuple[int, float, float, int, Any]] = []
        tooltip_identities: list[tuple[int, float, float, int, Any]] = []
        marker_database_match_seen = False
        nearby_node_seen = False
        tooltip_type_match_seen = False
        current_x, current_y = _coord_xy(current_coord)
        for index, node in enumerate(nodes):
            self.last_candidate_nodes_scanned += 1
            coord = int(getattr(node, "coord"))
            distance = _coord_distance(current_coord, coord)
            if distance > self.last_candidate_search_distance:
                continue
            nearby_node_seen = True
            if self.last_tooltip_confirmed:
                if not _ore_types_match(
                    str(getattr(node, "ore_type", "")),
                    str(confirmed_ore_type),
                ):
                    continue
                tooltip_type_match_seen = True
                self.last_candidate_same_type_nodes += 1
            match_class = 0
            score = distance
            if marker_intercept_coord is not None:
                marker_database_distance = _coord_distance(coord, marker_intercept_coord)
                if (
                    self.last_marker_database_distance is None
                    or marker_database_distance < self.last_marker_database_distance
                ):
                    self.last_marker_database_distance = marker_database_distance
                if marker_database_distance > self.marker_database_match_distance:
                    if not (
                        self.last_tooltip_confirmed
                        and self.tooltip_confirmed_relaxed_match_enabled
                    ):
                        continue
                    match_class = 1
                    # Native tooltip identity proves that a live node of this
                    # exact ore type exists. The minimap projection is the signal
                    # being relaxed, so its bearing must not remain a hidden hard
                    # gate here. Choose the nearest compatible database spawn;
                    # strict projected matches above still outrank this class.
                    score = distance
                else:
                    marker_database_match_seen = True
                    score = marker_database_distance
            elif marker_vector is not None:
                marker_x, marker_y, marker_distance, sensor_radius = marker_vector
                node_x, node_y = _coord_xy(coord)
                node_dx = node_x - current_x
                node_dy = node_y - current_y
                node_distance = hypot(node_dx, node_dy)
                if node_distance > 0.001:
                    alignment = (
                        marker_x * node_dx + marker_y * node_dy
                    ) / (marker_distance * node_distance)
                    if alignment < self.marker_bearing_min_alignment:
                        continue
                    expected_distance = min(
                        self.marker_tracking_radius,
                        marker_distance / sensor_radius * self.marker_tracking_radius,
                    )
                    score = (
                        (1.0 - alignment) * self.marker_bearing_weight
                        + abs(node_distance - expected_distance)
                        / self.marker_tracking_radius
                        * self.marker_radial_weight
                    )
            if self.last_tooltip_confirmed:
                # Identity must be chosen before availability. Otherwise a
                # closer DB record without a safe plan could be silently
                # skipped in favor of a different, farther hillside spawn that
                # happens to have an admissible attachment.
                tooltip_identities.append((match_class, score, distance, index, node))
                continue
            access_plan = getattr(node, "access_plan", None)
            if self.require_access_plan and access_plan is None:
                continue
            if self.require_access_plan and self.access_attachment_max_distance > 0.0:
                attachment_distance = _access_attachment_distance(
                    access_plan,
                    current_coord,
                )
                if (
                    attachment_distance is None
                    or attachment_distance > self.access_attachment_max_distance
                ):
                    if (
                        attachment_distance is not None
                        and (
                            self.last_access_attachment_distance is None
                            or attachment_distance < self.last_access_attachment_distance
                        )
                    ):
                        self.last_access_attachment_distance = attachment_distance
                    continue
                self.last_access_attachment_distance = attachment_distance
            if self.cooldowns.get(coord, 0.0) > now:
                continue
            if self.is_spatially_cooled(
                coord,
                str(getattr(node, "ore_type", "Ore")),
                now=now,
            ):
                continue
            ranked.append((match_class, score, distance, index, node))
        if tooltip_identities:
            match_class, _score, _distance, _index, node = min(
                tooltip_identities
            )
            self.last_tooltip_relaxed_match = match_class == 1
            coord = int(getattr(node, "coord"))
            access_plan = getattr(node, "access_plan", None)
            if self.require_access_plan and access_plan is None:
                self.last_candidate_rejection_reason = (
                    "database_node_missing_access_plan"
                )
                return finish(None)
            if self.require_access_plan and self.access_attachment_max_distance > 0.0:
                attachment_distance = _access_attachment_distance(
                    access_plan,
                    current_coord,
                )
                self.last_access_attachment_distance = attachment_distance
                if (
                    attachment_distance is None
                    or attachment_distance > self.access_attachment_max_distance
                ):
                    self.last_candidate_rejection_reason = (
                        "access_attachment_too_far"
                    )
                    return finish(None)
            if self.cooldowns.get(coord, 0.0) > now or self.is_spatially_cooled(
                coord,
                str(getattr(node, "ore_type", "Ore")),
                now=now,
            ):
                self.last_candidate_rejection_reason = "database_node_unavailable"
                return finish(None)
            return finish(MiningCandidate(
                node_id=getattr(node, "node_id", None),
                coord=coord,
                ore_type=str(getattr(node, "ore_type", "Ore")),
                access_plan=access_plan,
                marker_intercept_coord=marker_intercept_coord,
                tooltip_confirmed=True,
                centered_tooltip_confirmed=centered_tooltip_confirmed,
                source="database",
            ))
        if not ranked:
            if (
                self.tooltip_confirmed_unmatched_enabled
                and self.last_tooltip_confirmed
                and marker_intercept_coord is not None
                and not tooltip_type_match_seen
            ):
                projection_distance = _coord_distance(
                    current_coord,
                    marker_intercept_coord,
                )
                self.last_tooltip_unmatched_distance = projection_distance
                projection_available = (
                    self.cooldowns.get(marker_intercept_coord, 0.0) <= now
                    and not self.is_spatially_cooled(
                        marker_intercept_coord,
                        str(confirmed_ore_type),
                        now=now,
                    )
                )
                if (
                    projection_distance
                    <= self.tooltip_confirmed_unmatched_max_distance
                    and projection_available
                ):
                    self.last_tooltip_unmatched_candidate = True
                    synthetic_node_id = -max(1, abs(int(marker_intercept_coord)))
                    return finish(MiningCandidate(
                        node_id=synthetic_node_id,
                        coord=marker_intercept_coord,
                        ore_type=str(confirmed_ore_type),
                        marker_intercept_coord=marker_intercept_coord,
                        tooltip_confirmed=True,
                        centered_tooltip_confirmed=centered_tooltip_confirmed,
                        source="live_tooltip_marker",
                    ))
                if (
                    projection_distance
                    > self.tooltip_confirmed_unmatched_max_distance
                ):
                    self.last_candidate_rejection_reason = (
                        "live_tooltip_projection_too_far"
                    )
                else:
                    self.last_candidate_rejection_reason = (
                        "live_tooltip_projection_on_cooldown"
                    )
                return finish(None)
            if (
                self.last_tooltip_confirmed
                and nearby_node_seen
                and not tooltip_type_match_seen
            ):
                self.last_candidate_rejection_reason = "tooltip_ore_not_in_runtime_database"
            elif (
                self.require_access_plan
                and self.access_attachment_max_distance > 0.0
                and self.last_access_attachment_distance is not None
                and self.last_access_attachment_distance
                > self.access_attachment_max_distance
            ):
                self.last_candidate_rejection_reason = "access_attachment_too_far"
            elif marker_intercept_coord is not None and not marker_database_match_seen:
                self.last_candidate_rejection_reason = "marker_not_in_database"
            else:
                self.last_candidate_rejection_reason = "database_node_unavailable"
            return finish(None)
        match_class, _score, _distance, _index, node = min(ranked)
        self.last_tooltip_relaxed_match = match_class == 1
        return finish(MiningCandidate(
            node_id=getattr(node, "node_id", None),
            coord=int(getattr(node, "coord")),
            ore_type=str(getattr(node, "ore_type", "Ore")),
            access_plan=getattr(node, "access_plan", None),
            marker_intercept_coord=marker_intercept_coord,
            tooltip_confirmed=self.last_tooltip_confirmed,
            centered_tooltip_confirmed=centered_tooltip_confirmed,
            source="database",
        ))

    def is_spatially_cooled(
        self,
        coord: int,
        ore_type: str,
        *,
        now: float,
    ) -> bool:
        """Return whether a recent failure already represents this physical vein."""
        self._expire_cooldowns(now)
        normalized_type = _normalize_ore_type(ore_type)
        return any(
            deadline > now
            and stored_type == normalized_type
            and _coord_distance(coord, center_coord)
            <= self.failure_spatial_cooldown_radius
            for center_coord, stored_type, deadline in self.spatial_cooldowns
        )

    def _marker_intercept_coord(
        self,
        current_coord: int,
        marker_point: tuple[int, int] | None,
        minimap_shape: tuple[int, ...] | None,
    ) -> int | None:
        if (
            not self.direct_marker_intercept_enabled
            or marker_point is None
            or minimap_shape is None
            or len(minimap_shape) < 2
        ):
            return None
        height, width = minimap_shape[:2]
        sensor_radius = min(width, height) * self.marker_radius_fraction
        if sensor_radius <= 0.0:
            return None
        current_x, current_y = _coord_xy(current_coord)
        dx = float(marker_point[0]) - width * self.marker_center_x_fraction
        dy = float(marker_point[1]) - height * self.marker_center_y_fraction
        target_x = max(
            0.0,
            min(100.0, current_x + dx / sensor_radius * self.marker_tracking_radius_x),
        )
        target_y = max(
            0.0,
            min(100.0, current_y + dy / sensor_radius * self.marker_tracking_radius_y),
        )
        from vision_bot.coords import xy_to_coord

        return xy_to_coord(target_x, target_y)

    def _marker_vector(
        self,
        marker_point: tuple[int, int] | None,
        minimap_shape: tuple[int, ...] | None,
    ) -> tuple[float, float, float, float] | None:
        if (
            not self.marker_bearing_enabled
            or marker_point is None
            or minimap_shape is None
            or len(minimap_shape) < 2
        ):
            return None
        height, width = minimap_shape[:2]
        dx = float(marker_point[0]) - width * self.marker_center_x_fraction
        dy = float(marker_point[1]) - height * self.marker_center_y_fraction
        distance = hypot(dx, dy)
        sensor_radius = min(width, height) * self.marker_radius_fraction
        if distance < self.marker_bearing_min_pixels or sensor_radius <= 0:
            return None
        return dx, dy, distance, sensor_radius

    @property
    def current_intercept_coord(self) -> int | None:
        if self.intercept_cursor >= len(self.intercept_coords):
            return None
        return self.intercept_coords[self.intercept_cursor]

    @property
    def face_target_coord(self) -> int | None:
        """Face live ore evidence after arriving at the historical DB anchor."""
        candidate = self.candidate
        if candidate is None:
            return None
        if self.database_anchor_arrived:
            # At the fixed GatherMate interaction anchor a residual minimap
            # projection can be distorted by player-icon occlusion. RUN1 node
            # 134 was already beside a visible vein, but trusting that residual
            # turned the camera away before the world model ran. Hold the
            # arrival bearing; model plus native cursor supplies final aiming.
            return candidate.coord
        if (
            candidate.tooltip_confirmed
            and self.center_tooltip_confirmed
            and self.last_marker_intercept_coord is not None
        ):
            return self.last_marker_intercept_coord
        return candidate.coord

    def begin(
        self,
        candidate: MiningCandidate,
        *,
        now: float,
        current_coord: int | None = None,
        marker_track_id: int | None = None,
        marker_point: tuple[int, int] | None = None,
    ) -> None:
        if self.phase != MiningPhase.IDLE:
            raise RuntimeError(f"Cannot begin mining from {self.phase.value}")
        self.candidate = candidate
        self.started_at = now
        self.phase_started_at = now
        self.clicked_at = None
        self.hover_point = None
        self.clear_streak = 0
        self.intercept_marker_missing_frames = 0
        self.target_marker_id = (
            marker_track_id
            if marker_track_id is not None
            else self.last_presence.confirmed_track_id
        )
        self.target_marker_point = (
            marker_point
            if marker_point is not None
            else self.last_presence.confirmed_point
        )
        self.best_marker_radial_distance = self._target_marker_radial_distance()
        self.last_intercept_progress_at = now
        self.previous_intercept_coord = current_coord
        self.last_candidate_handoff_reason = None
        self.last_candidate_segment_distance = None
        self.dark_target_streak = 0
        self.center_tooltip_probe_started_at = None
        self.center_tooltip_confirmed = bool(candidate.centered_tooltip_confirmed)
        self.center_tooltip_last_ore_type = (
            candidate.ore_type if candidate.centered_tooltip_confirmed else None
        )
        self.database_anchor_arrived = False
        self.live_marker_centering_coord = None
        self.anchor_local_world_validation_active = False
        self.suspended_from = None
        self.suspended_at = None
        option = _primary_access_option(candidate.access_plan)
        direct_marker_intercept = (
            candidate.marker_intercept_coord is not None
            and _coord_distance(candidate.coord, candidate.marker_intercept_coord)
            <= self.direct_marker_intercept_max_node_delta
        )
        direct_node_first = bool(
            self.direct_node_intercept_first_enabled
            and current_coord is not None
        )
        self.access_option_rank = (
            None
            if direct_node_first
            else (int(getattr(option, "rank", 0)) if option is not None else None)
        )
        self.tried_access_option_ranks = (
            set()
            if direct_node_first or self.access_option_rank is None
            else {self.access_option_rank}
        )
        self.access_option_retries = 0
        self.direct_marker_intercept_active = direct_marker_intercept
        self._configure_access_option(option, current_coord=current_coord)
        self.best_intercept_coord_distance = (
            _coord_distance(current_coord, self.current_intercept_coord)
            if current_coord is not None and self.current_intercept_coord is not None
            else None
        )
        self.last_intercept_target_coord = self.current_intercept_coord
        self.last_intercept_target_distance = self.best_intercept_coord_distance
        if (
            candidate.centered_tooltip_confirmed
            and candidate.source != "database"
        ):
            self.intercept_cursor = len(self.intercept_coords)
            self.phase = MiningPhase.FACE_NODE
        else:
            self.phase = MiningPhase.INTERCEPT

    def refresh_direct_marker_intercept(
        self,
        current_coord: int,
        minimap_shape: tuple[int, ...] | None,
    ) -> int | None:
        if (
            self.phase != MiningPhase.INTERCEPT
            or not self.direct_marker_intercept_active
        ):
            return self.current_intercept_coord
        frozen_marker_target = (
            self.live_marker_centering_coord
            if self.database_anchor_arrived
            else None
        )
        if frozen_marker_target is not None and not self.last_presence.target_visible:
            # The residual is intentionally frozen at DB arrival so a player-
            # icon occlusion or short tracking gap cannot erase stage two.
            prefix = self.intercept_coords[: self.intercept_cursor]
            self.intercept_coords = (*prefix, frozen_marker_target)
            self.direct_marker_stage_index = self.intercept_cursor
            return self.current_intercept_coord
        if not self.last_presence.target_visible or self.target_marker_point is None:
            return self.current_intercept_coord
        if self.direct_node_intercept_active:
            return self.current_intercept_coord
        if (
            self.direct_marker_stage_index is not None
            and self.intercept_cursor < self.direct_marker_stage_index
        ):
            return self.current_intercept_coord
        if (
            self.direct_marker_stage_index is not None
            and self.intercept_cursor > self.direct_marker_stage_index
        ):
            return self.current_intercept_coord
        intercept_coord = self._marker_intercept_coord(
            current_coord,
            self.target_marker_point,
            minimap_shape,
        )
        radial_distance = self._target_marker_radial_distance()
        candidate = self.candidate
        if (
            candidate is not None
            and not candidate.tooltip_confirmed
            and radial_distance is not None
            and radial_distance <= self.direct_marker_world_scan_radius_pixels
            and _coord_distance(current_coord, candidate.coord)
            > self.direct_marker_world_scan_candidate_distance
        ):
            intercept_coord = candidate.coord
        self.last_marker_intercept_coord = intercept_coord
        if (
            candidate is not None
            and candidate.source == "database"
            and candidate.tooltip_confirmed
            and not self.database_anchor_arrived
        ):
            # A GatherMate node is the historical visible player coordinate at
            # a successful interaction.  It is the fixed reachable movement
            # anchor.  Keep it fixed until that anchor has actually been
            # reached.  Afterwards the remaining error is visible directly on
            # the minimap, so the live projection becomes the final bounded
            # centering target instead of leaving the character stopped beside
            # (rather than on top of) the ore marker.
            return self.current_intercept_coord
        if (
            candidate is not None
            and candidate.source == "database"
            and candidate.tooltip_confirmed
            and self.database_anchor_arrived
        ):
            # Freeze the residual projection captured at DB-anchor arrival.
            # Reprojecting on every frame makes the target move with the
            # character and continuously resets the stopped-burst state.
            if self.live_marker_centering_coord is None:
                self.live_marker_centering_coord = intercept_coord
            intercept_coord = self.live_marker_centering_coord
        if intercept_coord is not None:
            prefix = self.intercept_coords[: self.intercept_cursor]
            if candidate is not None and candidate.tooltip_confirmed:
                # A native tooltip makes the live minimap marker authoritative.
                # Keep its freshly projected intercept as the final stage.  The
                # generic helper may append the coarse candidate coordinate when
                # the two projections differ by more than the world-scan radius;
                # that turns the dynamic marker stage into a non-final waypoint.
                # Runtime then advances past it and circles back to a stale
                # coordinate instead of continuing to center the visible dot.
                self.intercept_coords = (*prefix, intercept_coord)
            else:
                self.intercept_coords = self._with_final_candidate_coord(
                    (*prefix, intercept_coord)
                )
            self.direct_marker_stage_index = self.intercept_cursor
        return self.current_intercept_coord

    def mark_intercept_reached(self, *, now: float | None = None) -> None:
        if self.phase != MiningPhase.INTERCEPT:
            raise RuntimeError("Mining intercept can only finish from intercept state")
        self.intercept_cursor = len(self.intercept_coords)
        self.phase = MiningPhase.FACE_NODE
        self.phase_started_at = now if now is not None else time.monotonic()

    @property
    def final_intercept_uses_database_anchor(self) -> bool:
        return bool(
            self.phase == MiningPhase.INTERCEPT
            and self.candidate is not None
            and self.candidate.source == "database"
            and self.candidate.tooltip_confirmed
            and self.intercept_cursor >= len(self.intercept_coords) - 1
            and self.current_intercept_coord == self.candidate.coord
        )

    def mark_database_anchor_arrived(self, *, now: float) -> bool:
        if not self.final_intercept_uses_database_anchor:
            raise RuntimeError("Database mining anchor is not the final intercept")
        self.database_anchor_arrived = True
        self.live_marker_centering_coord = self.last_marker_intercept_coord
        self.last_intercept_progress_at = now
        if self._anchor_local_world_validation_is_ready():
            # RUN8 node 145 reached the historical successful-interaction
            # coordinate with the world vein visibly beside the character.
            # A modest 20.5 px minimap residual then projected a new target
            # almost 20 yards into the cliff. Before accepting such a
            # contradictory residual, remain stopped and let the candidate-
            # only world model seek native tooltip/pickaxe authority. Failure
            # to find it falls back to the original one-burst residual path.
            self.anchor_local_world_validation_active = True
            self.intercept_cursor = len(self.intercept_coords)
            self.phase = MiningPhase.WORLD_SCAN
            self.phase_started_at = now
            return True
        return self.mark_intercept_target_reached(now=now)

    def _anchor_local_world_validation_is_ready(self) -> bool:
        candidate = self.candidate
        if (
            not self.anchor_local_world_validation_enabled
            or candidate is None
            or candidate.source != "database"
            or not candidate.tooltip_confirmed
            or not self.direct_marker_intercept_active
        ):
            return False
        distances = [
            value
            for value in (
                self._target_marker_radial_distance(),
                self.best_marker_radial_distance,
            )
            if value is not None
        ]
        if not distances:
            return False
        radial_distance = min(float(value) for value in distances)
        return bool(
            radial_distance > self.centered_marker_occlusion_radius_pixels
            and radial_distance <= self.anchor_local_world_validation_radius_pixels
        )

    def anchor_local_world_validation_timed_out(self, *, now: float) -> bool:
        return bool(
            self.phase == MiningPhase.WORLD_SCAN
            and self.anchor_local_world_validation_active
            and self.phase_started_at is not None
            and self.anchor_local_world_validation_seconds > 0.0
            and now - self.phase_started_at
            >= self.anchor_local_world_validation_seconds
        )

    def resume_residual_after_anchor_world_validation(
        self,
        *,
        now: float,
        current_coord: int | None = None,
    ) -> bool:
        if (
            self.phase != MiningPhase.WORLD_SCAN
            or not self.anchor_local_world_validation_active
        ):
            return False
        self.anchor_local_world_validation_active = False
        self.phase = MiningPhase.INTERCEPT
        self.phase_started_at = now
        self.intercept_cursor = max(0, len(self.intercept_coords) - 1)
        self.last_intercept_progress_at = now
        self.last_intercept_target_coord = self.live_marker_centering_coord
        self.last_intercept_target_distance = (
            _coord_distance(current_coord, self.live_marker_centering_coord)
            if current_coord is not None
            and self.live_marker_centering_coord is not None
            else None
        )
        self.best_intercept_coord_distance = self.last_intercept_target_distance
        return True

    def observe_intercept_marker(
        self,
        *,
        now: float,
        current_coord: int | None = None,
    ) -> MiningOutcome | None:
        if self.phase != MiningPhase.INTERCEPT:
            return None
        if self._candidate_proximity_handoff_ready(current_coord):
            self.mark_intercept_reached(now=now)
            return None
        timeout = self._active_timeout_reason(now)
        if timeout is not None:
            if (
                timeout == "ore_intercept_timeout"
                and self._stalled_intercept_is_ready_for_world_scan(current_coord)
            ):
                self.mark_intercept_reached(now=now)
                return None
            return self._finish(False, timeout, now=now)
        intercept_target = self.current_intercept_coord
        if current_coord is not None and intercept_target is not None:
            intercept_distance = _coord_distance(current_coord, intercept_target)
            if self.last_intercept_target_coord != intercept_target:
                self.last_intercept_target_coord = intercept_target
                self.last_intercept_target_distance = intercept_distance
            elif (
                self.last_intercept_target_distance is not None
                and intercept_distance
                <= self.last_intercept_target_distance - self.intercept_progress_coord
            ):
                # Coordinate OCR and movement are not atomic.  Immediately
                # after advancing a short access waypoint, one sample can still
                # describe momentum from the previous leg and establish an
                # unrealistically good global best.  A later genuine approach
                # to the current target must still refresh the stall timer.
                self.last_intercept_progress_at = now
            self.last_intercept_target_distance = intercept_distance
            if (
                self.best_intercept_coord_distance is None
                or intercept_distance
                <= self.best_intercept_coord_distance - self.intercept_progress_coord
            ):
                self.best_intercept_coord_distance = intercept_distance
                self.last_intercept_progress_at = now
        if self.last_presence.target_visible:
            self.intercept_marker_missing_frames = 0
            radial_distance = self._target_marker_radial_distance()
            if radial_distance is not None:
                if self._direct_marker_is_ready_for_world_scan(
                    radial_distance,
                    current_coord=current_coord,
                ):
                    if self.requires_centered_marker_confirmation:
                        self.begin_center_tooltip_probe(now=now)
                    else:
                        self.mark_intercept_target_reached(now=now)
                    return None
                if (
                    self.best_marker_radial_distance is None
                    or radial_distance
                    <= self.best_marker_radial_distance - self.intercept_progress_pixels
                ):
                    self.best_marker_radial_distance = radial_distance
                    self.last_intercept_progress_at = now
            return self._observe_intercept_stall(now=now, current_coord=current_coord)
        self.intercept_marker_missing_frames += 1
        if self.intercept_marker_missing_frames < self.intercept_marker_missing_limit:
            return self._observe_intercept_stall(now=now, current_coord=current_coord)
        if self._direct_marker_is_ready_for_world_scan(
            self.best_marker_radial_distance,
            current_coord=current_coord,
        ):
            if self.requires_centered_marker_confirmation:
                self.begin_center_tooltip_probe(now=now)
            else:
                self.mark_intercept_target_reached(now=now)
            return None
        candidate = self.candidate
        close_to_candidate = bool(
            current_coord is not None
            and candidate is not None
            and _coord_distance(current_coord, candidate.coord)
            <= self.lost_marker_world_scan_distance
        )
        marker_was_close_without_position = bool(
            current_coord is None
            and candidate is not None
            and self.direct_marker_intercept_active
            and self.best_marker_radial_distance is not None
            and self.best_marker_radial_distance
            <= self.lost_marker_world_scan_radius_pixels
        )
        marker_was_center_occluded = bool(
            candidate is not None
            and candidate.tooltip_confirmed
            and self.direct_marker_intercept_active
            and self.best_marker_radial_distance is not None
            and self.best_marker_radial_distance
            <= self.centered_marker_occlusion_radius_pixels
        )
        marker_was_anchor_local_occluded = bool(
            self.anchor_local_world_validation_enabled
            and candidate is not None
            and candidate.source == "database"
            and candidate.tooltip_confirmed
            and self.direct_marker_intercept_active
            and self.best_marker_radial_distance is not None
            and self.best_marker_radial_distance
            <= self.anchor_local_world_validation_radius_pixels
        )
        pending_frozen_marker_centering = bool(
            candidate is not None
            and candidate.tooltip_confirmed
            and candidate.source == "database"
            and self.database_anchor_arrived
            and self.live_marker_centering_coord is not None
            and not marker_was_center_occluded
        )
        if pending_frozen_marker_centering:
            # V0.8.21 live evidence reached a relaxed-match DB anchor while
            # the last confirmed live marker was still 76 px from center.  A
            # generic "near DB + marker missing" fallback incorrectly skipped
            # the promised second stage and probed an empty minimap center.
            # Preserve the frozen live-marker target instead.  It remains a
            # bounded one-burst transaction; if the dot does not reappear at
            # center, the normal final-marker failure/cooldown still applies.
            return self._observe_intercept_stall(
                now=now,
                current_coord=current_coord,
            )
        if (
            candidate is not None
            and candidate.tooltip_confirmed
            and marker_was_center_occluded
            and (
                candidate.source != "database"
                or self.database_anchor_arrived
            )
        ):
            self.begin_center_tooltip_probe(now=now)
            return None
        if (
            candidate is not None
            and candidate.tooltip_confirmed
            and candidate.source == "database"
            and marker_was_center_occluded
        ):
            # The player icon can cover the live dot during the stopped
            # recheck/final impulse.  Keep approaching the fixed DB anchor;
            # exact-center tooltip validation is allowed only after arrival.
            return self._observe_intercept_stall(
                now=now,
                current_coord=current_coord,
            )
        if marker_was_anchor_local_occluded and not self.database_anchor_arrived:
            # RUN1C node 138 disappeared at 18.36 px while the final bounded
            # DB-anchor burst was being aligned. That is only 0.36 px beyond
            # the old player-occlusion radius and must not be mislabeled as an
            # external gather. Finish the short DB-anchor transaction; on
            # arrival mark_database_anchor_arrived() starts candidate-only
            # world validation for this modest-residual band.
            return self._observe_intercept_stall(
                now=now,
                current_coord=current_coord,
            )
        if (
            candidate is not None
            and candidate.tooltip_confirmed
            and candidate.source == "database"
            and close_to_candidate
        ):
            # Coordinate feedback can jump from several yards outside the node
            # to the historical interaction anchor between two minimap frames.
            # The player icon may therefore cover the dot before the tracker
            # ever records an <= occlusion-radius sample.  Do not mislabel that
            # disappearance as another player's gather.  Finish the bounded
            # stopped/one-burst arrival first, then let an exact-center native
            # tooltip distinguish occlusion from a truly absent vein.
            if self.database_anchor_arrived:
                self.begin_center_tooltip_probe(now=now)
                return None
            return self._observe_intercept_stall(
                now=now,
                current_coord=current_coord,
            )
        if (
            candidate is not None
            and not candidate.tooltip_confirmed
            and (close_to_candidate or marker_was_close_without_position)
        ):
            self.mark_intercept_reached(now=now)
            return None
        if candidate is not None and candidate.tooltip_confirmed:
            # A tooltip-confirmed tracking dot is authoritative visible-world
            # evidence. If it then stays absent while we are still en route,
            # another player most likely gathered it. Do not keep running to a
            # now-empty coordinate; _finish also applies the normal bounded
            # failure and spatial cooldowns.
            return self._finish(False, "ore_taken_by_other_player", now=now)
        if (
            self.continue_coordinate_approach_after_marker_loss
            and current_coord is not None
            and candidate is not None
            and self.current_intercept_coord is not None
        ):
            # Once a bright marker selected a database node, terrain and height
            # can hide the icon before the side approach is complete. Keep the
            # bounded coordinate/access-plan approach; world hover authority or
            # the existing intercept timeout will decide the outcome.
            return self._observe_intercept_stall(now=now, current_coord=current_coord)
        return self._finish(False, "ore_lost_during_intercept", now=now)

    def note_intercept_heading_alignment(self, *, now: float) -> None:
        """Keep the coordinate-stall timer from expiring during an active pivot.

        Precision mining deliberately pivots without pressing W.  A large but
        healthy turn can therefore take longer than the short coordinate
        no-progress window even though heading error is being removed every
        frame.  The overall intercept timeout remains the hard bound; this only
        makes the shorter watchdog measure stalled translation after alignment.
        """
        if self.phase == MiningPhase.INTERCEPT:
            self.last_intercept_progress_at = now

    def _candidate_proximity_handoff_ready(
        self,
        current_coord: int | None,
    ) -> bool:
        """Stop an access leg once it reaches or crosses the confirmed DB spawn.

        Coordinate telemetry is sampled, so a mounted character can move from one
        side of the interaction radius to the other without producing a sample
        inside it.  Checking the travelled segment closes that gap and prevents
        an unfinished side-approach waypoint from carrying the character past a
        tooltip-confirmed candidate that was already reached.
        """
        candidate = self.candidate
        previous_coord = self.previous_intercept_coord
        self.previous_intercept_coord = current_coord
        self.last_candidate_segment_distance = None
        if (
            current_coord is None
            or candidate is None
            or not candidate.tooltip_confirmed
            or candidate.source != "database"
            or (
                self.direct_marker_intercept_active
                and candidate.tooltip_confirmed
            )
            or self.candidate_handoff_distance <= 0.0
        ):
            return False

        current_distance = _coord_distance(current_coord, candidate.coord)
        if current_distance <= self.candidate_handoff_distance:
            self.last_candidate_handoff_reason = "candidate_proximity"
            self.last_candidate_segment_distance = current_distance
            return True
        if previous_coord is None or previous_coord == current_coord:
            return False

        segment_distance = _coord_segment_distance(
            candidate.coord,
            previous_coord,
            current_coord,
        )
        self.last_candidate_segment_distance = segment_distance
        if segment_distance <= self.candidate_handoff_distance:
            self.last_candidate_handoff_reason = "candidate_segment_crossed"
            return True
        return False

    def _observe_intercept_stall(
        self,
        *,
        now: float,
        current_coord: int | None,
    ) -> MiningOutcome | None:
        if (
            self.intercept_no_progress_seconds <= 0.0
            or self.last_intercept_progress_at is None
            or now - self.last_intercept_progress_at
            < self.intercept_no_progress_seconds
        ):
            return None
        if self._stalled_intercept_is_ready_for_world_scan(current_coord):
            self.mark_intercept_reached(now=now)
            return None
        if self._retry_access_option(now=now, current_coord=current_coord):
            return None
        return self._finish(False, "ore_intercept_no_progress", now=now)

    def _stalled_intercept_is_ready_for_world_scan(
        self,
        current_coord: int | None,
    ) -> bool:
        candidate = self.candidate
        if candidate is None or not self.intercept_coords:
            return False
        if self.requires_centered_marker_confirmation:
            return False
        candidate_close = bool(
            current_coord is not None
            and _coord_distance(current_coord, candidate.coord)
            <= self.intercept_stall_world_scan_distance
        )
        return self.intercept_cursor >= len(self.intercept_coords) - 1 and candidate_close

    def mark_intercept_target_reached(self, *, now: float | None = None) -> bool:
        if self.phase != MiningPhase.INTERCEPT:
            raise RuntimeError("Mining intercept target is not active")
        if (
            self.intercept_cursor >= len(self.intercept_coords) - 1
            and self.requires_centered_marker_confirmation
        ):
            if self.target_marker_is_centered:
                self.begin_center_tooltip_probe(now=now)
                return True
            return False
        self.intercept_cursor += 1
        self.best_intercept_coord_distance = None
        self.last_intercept_target_coord = None
        self.last_intercept_target_distance = None
        if now is not None:
            self.last_intercept_progress_at = now
        if self.intercept_cursor >= len(self.intercept_coords):
            self.phase = MiningPhase.FACE_NODE
            self.phase_started_at = now if now is not None else time.monotonic()
            return True
        return False

    @property
    def requires_centered_marker_confirmation(self) -> bool:
        return bool(
            self.candidate is not None
            and self.candidate.tooltip_confirmed
            and self.direct_marker_intercept_active
            and not self.center_tooltip_confirmed
        )

    @property
    def target_marker_is_centered(self) -> bool:
        radial_distance = self._target_marker_radial_distance()
        return bool(
            radial_distance is not None
            and radial_distance <= self.centered_marker_radius_pixels
        )

    def begin_center_tooltip_probe(self, *, now: float | None = None) -> None:
        if self.phase != MiningPhase.INTERCEPT:
            raise RuntimeError("Center tooltip probe can only start during intercept")
        started_at = now if now is not None else time.monotonic()
        self.intercept_cursor = len(self.intercept_coords)
        self.center_tooltip_probe_started_at = started_at
        self.center_tooltip_last_ore_type = None
        self.phase = MiningPhase.CENTER_TOOLTIP
        self.phase_started_at = started_at

    def observe_center_tooltip(
        self,
        *,
        now: float,
        tooltip_ore_type: str | None,
    ) -> MiningOutcome | None:
        if self.phase != MiningPhase.CENTER_TOOLTIP:
            return None
        candidate = self.candidate
        if candidate is None:
            return self._finish(False, "candidate_missing_at_center", now=now)
        started_at = (
            self.center_tooltip_probe_started_at
            if self.center_tooltip_probe_started_at is not None
            else now
        )
        elapsed = max(0.0, now - started_at)
        if elapsed >= self.center_tooltip_settle_seconds and tooltip_ore_type:
            self.center_tooltip_last_ore_type = str(tooltip_ore_type)
            if not _ore_types_match(candidate.ore_type, tooltip_ore_type):
                return self._finish(False, "ore_center_tooltip_type_mismatch", now=now)
            self.center_tooltip_confirmed = True
            self.phase = MiningPhase.FACE_NODE
            self.phase_started_at = now
            return None
        if elapsed >= self.center_tooltip_timeout_seconds:
            if (
                self.center_tooltip_world_scan_fallback_enabled
                and candidate.source == "database"
                and candidate.tooltip_confirmed
            ):
                # RUN3B node 136 reached its historical anchor after a native
                # minimap tooltip, but the player-overlapped center no longer
                # yielded that tooltip. Give the world model plus native world
                # tooltip/pickaxe one bounded chance before cooling the node.
                self.center_tooltip_probe_started_at = None
                self.phase = MiningPhase.WORLD_SCAN
                self.phase_started_at = now
                return None
            return self._finish(False, "ore_center_tooltip_unconfirmed", now=now)
        timeout = self._active_timeout_reason(now)
        if timeout is not None:
            return self._finish(False, timeout, now=now)
        return None

    def observe_final_approach_failure(
        self,
        reason: str,
        *,
        now: float,
        current_coord: int | None = None,
    ) -> MiningOutcome | None:
        """Retry one unused terrain side before cooling a failed final burst."""
        if self.phase != MiningPhase.INTERCEPT:
            raise RuntimeError("Final mining approach is not active")
        if self._retry_access_option(now=now, current_coord=current_coord):
            return None
        return self._finish(False, str(reason), now=now)

    def center_tooltip_marker_point(self) -> tuple[int, int] | None:
        shape = self.last_minimap_shape
        if shape is None or len(shape) < 2:
            return None
        height, width = shape[:2]
        return (
            int(round(width * self.marker_center_x_fraction)),
            int(round(height * self.marker_center_y_fraction)),
        )

    def mark_node_faced(self, *, now: float | None = None) -> None:
        if self.phase != MiningPhase.FACE_NODE:
            raise RuntimeError("Mining node can only be faced after intercept")
        self.phase = MiningPhase.WORLD_SCAN
        self.phase_started_at = now if now is not None else time.monotonic()

    def observe_face_node_timeout(self, *, now: float) -> MiningOutcome | None:
        if self.phase != MiningPhase.FACE_NODE:
            return None
        timeout = self._active_timeout_reason(now)
        if timeout == "ore_face_node_timeout" and self.face_node_timeout_starts_world_scan:
            # Coordinate-derived facing is a coarse setup step. Once the bot is
            # already beside the node, let cursor/tooltip evidence search the
            # visible world instead of abandoning a vein over heading jitter.
            self.mark_node_faced(now=now)
            return None
        if timeout is not None:
            return self._finish(False, timeout, now=now)
        return None

    def observe_dark_target(self, *, now: float) -> MiningOutcome | None:
        if self.phase not in {
            MiningPhase.INTERCEPT,
            MiningPhase.FACE_NODE,
            MiningPhase.WORLD_SCAN,
        }:
            self.dark_target_streak = 0
            return None
        if self.last_presence.target_dark_visible:
            self.dark_target_streak = 0
            return None
        self.dark_target_streak = self.dark_target_streak + 1 if self.last_presence.dark_only else 0
        if self.dark_target_streak < self.dark_target_confirm_frames:
            return None
        return self._finish(False, "ore_marker_dark_below", now=now)

    def observe_world_scan_timeout(
        self,
        *,
        now: float,
        current_coord: int | None = None,
    ) -> MiningOutcome | None:
        if self.phase != MiningPhase.WORLD_SCAN:
            return None
        if self.anchor_local_world_validation_timed_out(now=now):
            self.resume_residual_after_anchor_world_validation(
                now=now,
                current_coord=current_coord,
            )
            return None
        timeout = self._active_timeout_reason(now)
        if timeout == "ore_world_scan_timeout" and self._retry_access_option(
            now=now,
            current_coord=current_coord,
        ):
            self.phase = MiningPhase.INTERCEPT
            return None
        if timeout is not None:
            return self._finish(False, timeout, now=now)
        return None

    def mark_clicked(self, point: tuple[int, int], *, now: float) -> None:
        if self.phase != MiningPhase.WORLD_SCAN:
            raise RuntimeError("Mining click can only be recorded from world scan")
        self.hover_point = point
        self.clicked_at = now
        self.clear_streak = 0
        self.intercept_marker_missing_frames = 0
        self.anchor_local_world_validation_active = False
        self.phase = MiningPhase.GATHER_WAIT

    def observe_after_click(
        self,
        *,
        target_visible: bool,
        hover_mining_ready: bool | None,
        now: float,
    ) -> MiningOutcome | None:
        if self.phase not in {MiningPhase.GATHER_WAIT, MiningPhase.VERIFY}:
            return None
        if self.clicked_at is None:
            raise RuntimeError("Mining verification has no click timestamp")

        elapsed_after_click = max(0.0, now - self.clicked_at)
        if elapsed_after_click < self.gather_wait_seconds:
            return None
        self.phase = MiningPhase.VERIFY
        hover_cleared = hover_mining_ready is False
        clear_confirmed = not target_visible and (
            hover_cleared or not self.verify_require_hover_clear
        )
        self.clear_streak = self.clear_streak + 1 if clear_confirmed else 0
        if self.clear_streak >= self.clear_frames:
            return self._finish(True, "tracked_icon_and_hover_cleared", now=now)
        if elapsed_after_click >= self.verify_timeout_seconds:
            reason = (
                "tracked_bright_minimap_icon_persisted"
                if target_visible
                else "mining_hover_evidence_persisted"
            )
            return self._finish(False, reason, now=now)
        return None

    def fail(self, reason: str, *, now: float) -> MiningOutcome:
        if self.candidate is None:
            raise RuntimeError("No mining candidate is active")
        return self._finish(False, reason, now=now)

    def suspend(self, *, now: float | None = None) -> None:
        if not self.active or self.phase == MiningPhase.SUSPENDED:
            return
        self.suspended_from = self.phase
        self.suspended_at = now if now is not None else time.monotonic()
        self.phase = MiningPhase.SUSPENDED
        self.phase_started_at = None

    def resume_after_interrupt(
        self,
        *,
        target_visible: bool,
        confirmed_bright: bool,
        now: float,
    ) -> MiningOutcome | None:
        if self.phase != MiningPhase.SUSPENDED:
            return None
        if self.suspended_at is not None and self.started_at is not None:
            self.started_at += max(0.0, now - self.suspended_at)
        candidate = self.candidate
        marker_missing = not target_visible and not confirmed_bright
        if marker_missing and not bool(
            candidate is not None and candidate.tooltip_confirmed
        ):
            return self._finish(False, "ore_lost_after_interrupt", now=now)
        if confirmed_bright and self.last_presence.confirmed_track_id is not None:
            # A long combat can expire the old tracker id even though the same
            # dot is visible again. Rebind transaction identity to the newly
            # stable marker before resuming the intercept.
            self.target_marker_id = self.last_presence.confirmed_track_id
            self.target_marker_point = self.last_presence.confirmed_point
        self.clicked_at = None
        self.hover_point = None
        self.clear_streak = 0
        self.intercept_marker_missing_frames = 1 if marker_missing else 0
        self.phase = MiningPhase.INTERCEPT
        self.phase_started_at = now
        self.center_tooltip_probe_started_at = None
        self.center_tooltip_confirmed = False
        self.center_tooltip_last_ore_type = None
        self.anchor_local_world_validation_active = False
        self.last_intercept_progress_at = now
        self.last_intercept_target_coord = self.current_intercept_coord
        self.last_intercept_target_distance = None
        self.best_marker_radial_distance = self._target_marker_radial_distance()
        self.dark_target_streak = 0
        self.previous_intercept_coord = None
        self.last_candidate_handoff_reason = None
        self.last_candidate_segment_distance = None
        self.suspended_from = None
        self.suspended_at = None
        return None

    def consume_outcome(self) -> MiningOutcome | None:
        outcome = self._pending_outcome
        if outcome is None:
            return None
        self._pending_outcome = None
        self.phase = MiningPhase.IDLE
        self.candidate = None
        self.started_at = None
        self.phase_started_at = None
        self.clicked_at = None
        self.hover_point = None
        self.center_tooltip_probe_started_at = None
        self.center_tooltip_confirmed = False
        self.center_tooltip_last_ore_type = None
        self.suspended_from = None
        self.suspended_at = None
        self.clear_streak = 0
        self.intercept_marker_missing_frames = 0
        self.target_marker_id = None
        self.target_marker_point = None
        self.intercept_coords = ()
        self.intercept_cursor = 0
        self.resume_coords = ()
        self.resume_route_index = None
        self.access_option_rank = None
        self.tried_access_option_ranks = set()
        self.access_option_retries = 0
        self.direct_marker_intercept_active = False
        self.direct_node_intercept_active = False
        self.direct_marker_stage_index = None
        self.database_anchor_arrived = False
        self.live_marker_centering_coord = None
        self.anchor_local_world_validation_active = False
        self.best_marker_radial_distance = None
        self.best_intercept_coord_distance = None
        self.last_intercept_progress_at = None
        self.last_intercept_target_coord = None
        self.last_intercept_target_distance = None
        self.previous_intercept_coord = None
        self.last_candidate_handoff_reason = None
        self.last_candidate_segment_distance = None
        self.dark_target_streak = 0
        return outcome

    def snapshot(self, *, now: float) -> dict[str, Any]:
        access_option = (
            _access_option_by_rank(self.candidate.access_plan, self.access_option_rank)
            if self.candidate is not None
            else None
        )
        return {
            "enabled": self.enabled,
            "require_access_plan": self.require_access_plan,
            "access_attachment_max_distance": self.access_attachment_max_distance,
            "access_attachment_distance": self.last_access_attachment_distance,
            "phase": self.phase.value,
            "candidate": (
                {
                    "node_id": self.candidate.node_id,
                    "coord": self.candidate.coord,
                    "ore_type": self.candidate.ore_type,
                    "source": self.candidate.source,
                    "tooltip_confirmed": self.candidate.tooltip_confirmed,
                    "centered_tooltip_confirmed": (
                        self.candidate.centered_tooltip_confirmed
                    ),
                }
                if self.candidate is not None
                else None
            ),
            "bright_points": [list(point) for point in self.last_presence.bright_points],
            "dark_points": [list(point) for point in self.last_presence.dark_points],
            "bright_streak": self.last_presence.bright_streak,
            "confirmed_bright": self.last_presence.confirmed_bright,
            "marker_intercept_coord": self.last_marker_intercept_coord,
            "marker_database_distance": self.last_marker_database_distance,
            "marker_database_match_distance": self.marker_database_match_distance,
            "require_minimap_tooltip_confirmation": self.require_minimap_tooltip_confirmation,
            "minimap_tooltip_confirmed": self.last_tooltip_confirmed,
            "confirmed_ore_type": self.last_confirmed_ore_type,
            "tooltip_confirmed_relaxed_match": self.last_tooltip_relaxed_match,
            "tooltip_confirmed_unmatched_candidate": self.last_tooltip_unmatched_candidate,
            "tooltip_confirmed_unmatched_distance": self.last_tooltip_unmatched_distance,
            "tooltip_confirmed_unmatched_max_distance": self.tooltip_confirmed_unmatched_max_distance,
            "candidate_search_distance": self.last_candidate_search_distance,
            "candidate_nodes_scanned": self.last_candidate_nodes_scanned,
            "candidate_same_type_nodes": self.last_candidate_same_type_nodes,
            "candidate_match_elapsed_ms": self.last_candidate_match_elapsed_ms,
            "candidate_rejection_reason": self.last_candidate_rejection_reason,
            "dark_only": self.last_presence.dark_only,
            "confirmed_point": (
                list(self.last_presence.confirmed_point)
                if self.last_presence.confirmed_point is not None
                else None
            ),
            "target_marker_id": self.target_marker_id,
            "target_marker_point": (
                list(self.target_marker_point)
                if self.target_marker_point is not None
                else None
            ),
            "target_visible": self.last_presence.target_visible,
            "target_dark_visible": self.last_presence.target_dark_visible,
            "intercept_target": self.current_intercept_coord,
            "direct_marker_intercept": self.direct_marker_intercept_active,
            "database_anchor_arrived": self.database_anchor_arrived,
            "live_marker_centering_coord": self.live_marker_centering_coord,
            "anchor_local_world_validation_active": (
                self.anchor_local_world_validation_active
            ),
            "anchor_local_world_validation_radius_pixels": (
                self.anchor_local_world_validation_radius_pixels
            ),
            "final_intercept_uses_database_anchor": (
                self.final_intercept_uses_database_anchor
            ),
            "intercept_remaining": max(
                0,
                len(self.intercept_coords) - self.intercept_cursor,
            ),
            "resume_waypoint_count": len(self.resume_coords),
            "resume_route_index": self.resume_route_index,
            "access_option_rank": self.access_option_rank,
            "access_option_retries": self.access_option_retries,
            "access_approach_distance_yards": (
                getattr(access_option, "approach_distance_yards", None)
                if access_option is not None
                else None
            ),
            "access_node_facing_aligned": bool(
                getattr(access_option, "node_facing_aligned", False)
            ),
            "hover_point": list(self.hover_point) if self.hover_point is not None else None,
            "elapsed_seconds": (
                self._elapsed_seconds(now) if self.started_at is not None else None
            ),
            "phase_elapsed_seconds": (
                max(0.0, now - self.phase_started_at)
                if self.phase_started_at is not None
                else None
            ),
            "best_marker_radial_distance": self.best_marker_radial_distance,
            "centered_marker_radius_pixels": self.centered_marker_radius_pixels,
            "centered_marker_occlusion_radius_pixels": (
                self.centered_marker_occlusion_radius_pixels
            ),
            "target_marker_is_centered": self.target_marker_is_centered,
            "center_tooltip_confirmed": self.center_tooltip_confirmed,
            "center_tooltip_last_ore_type": self.center_tooltip_last_ore_type,
            "center_tooltip_probe_elapsed": (
                max(0.0, now - self.center_tooltip_probe_started_at)
                if self.center_tooltip_probe_started_at is not None
                else None
            ),
            "best_intercept_coord_distance": self.best_intercept_coord_distance,
            "last_intercept_target_coord": self.last_intercept_target_coord,
            "last_intercept_target_distance": self.last_intercept_target_distance,
            "candidate_handoff_distance": self.candidate_handoff_distance,
            "candidate_handoff_reason": self.last_candidate_handoff_reason,
            "candidate_segment_distance": self.last_candidate_segment_distance,
            "last_intercept_progress_age": (
                max(0.0, now - self.last_intercept_progress_at)
                if self.last_intercept_progress_at is not None
                else None
            ),
            "dark_target_streak": self.dark_target_streak,
            "spatial_cooldown_count": len(self.spatial_cooldowns),
            "clear_streak": self.clear_streak,
            "intercept_marker_missing_frames": self.intercept_marker_missing_frames,
        }

    def _configure_access_option(
        self,
        option: Any | None,
        *,
        current_coord: int | None,
    ) -> None:
        candidate = self.candidate
        if candidate is None:
            raise RuntimeError("No mining candidate is active")
        inbound = (
            _start_access_leg_near_current(
                tuple(int(coord) for coord in getattr(option, "inbound_coords", ())),
                current_coord,
            )
            if option is not None
            else ()
        )
        if (
            candidate.tooltip_confirmed
            and current_coord is not None
            and _coord_distance(current_coord, candidate.coord)
            <= self.candidate_handoff_distance
        ):
            # The character is already at the spawn anchor. Do not follow an
            # attachment path away from a live marker; finish by centering the
            # marker itself and validating the minimap-center tooltip.
            inbound = ()
        if option is None:
            self.resume_coords = ()
            self.resume_route_index = None
        else:
            self.resume_coords = tuple(int(coord) for coord in option.resume_coords)
            self.resume_route_index = int(candidate.access_plan.resume_route_index)

        self.direct_marker_stage_index = None
        self.direct_node_intercept_active = False
        self.database_anchor_arrived = False
        self.live_marker_centering_coord = None
        self.anchor_local_world_validation_active = False
        if (
            self.direct_node_intercept_first_enabled
            and self.access_option_retries == 0
            and current_coord is not None
        ):
            self.intercept_coords = (int(candidate.coord),)
            self.intercept_cursor = 0
            self.direct_node_intercept_active = True
            return
        if candidate.source == "database" and candidate.tooltip_confirmed:
            # Reach the historical DB interaction position before allowing
            # the frozen live-marker projection to make one bounded residual
            # correction.  This keeps the noisy marker out of the coarse leg.
            self.intercept_coords = self._with_exact_final_candidate_coord(inbound)
            self.direct_marker_stage_index = len(self.intercept_coords) - 1
        elif self.direct_marker_intercept_active and candidate.marker_intercept_coord is not None:
            marker_coord = int(candidate.marker_intercept_coord)
            if inbound and inbound[-1] == marker_coord:
                self.intercept_coords = inbound
            else:
                self.intercept_coords = (*inbound, marker_coord)
            self.direct_marker_stage_index = len(self.intercept_coords) - 1
        elif inbound:
            self.intercept_coords = inbound
        else:
            self.intercept_coords = (candidate.coord,)
        # A DB candidate has already been fixed above.  Only a synthetic
        # tooltip-only candidate continues to own a dynamic projected target.
        if not (
            (candidate.source == "database" and candidate.tooltip_confirmed)
            or (self.direct_marker_intercept_active and candidate.tooltip_confirmed)
        ):
            self.intercept_coords = self._with_final_candidate_coord(self.intercept_coords)
        self.intercept_cursor = 0

    def _with_final_candidate_coord(self, coords: tuple[int, ...]) -> tuple[int, ...]:
        candidate = self.candidate
        if candidate is None:
            return coords
        final_coord = int(candidate.coord)
        if not coords:
            return (final_coord,)
        if _coord_distance(coords[-1], final_coord) <= self.final_world_scan_distance:
            return coords
        return (*coords, final_coord)

    def _with_exact_final_candidate_coord(
        self,
        coords: tuple[int, ...],
    ) -> tuple[int, ...]:
        candidate = self.candidate
        if candidate is None:
            return coords
        final_coord = int(candidate.coord)
        if not coords:
            return (final_coord,)
        if coords[-1] == final_coord:
            return coords
        return (*coords, final_coord)

    def _retry_access_option(
        self,
        *,
        now: float,
        current_coord: int | None,
    ) -> bool:
        candidate = self.candidate
        if (
            not self.access_option_retry_enabled
            or candidate is None
            or candidate.access_plan is None
            or self.access_option_retries >= self.max_access_option_retries
        ):
            return False
        option = next(
            (
                item
                for item in _ordered_access_options(candidate.access_plan)
                if int(getattr(item, "rank", -1)) not in self.tried_access_option_ranks
            ),
            None,
        )
        if option is None:
            return False
        rank = int(getattr(option, "rank", 0))
        self.access_option_rank = rank
        self.tried_access_option_ranks.add(rank)
        self.access_option_retries += 1
        self._configure_access_option(option, current_coord=current_coord)
        self.phase_started_at = now
        self.best_marker_radial_distance = self._target_marker_radial_distance()
        self.best_intercept_coord_distance = (
            _coord_distance(current_coord, self.current_intercept_coord)
            if current_coord is not None and self.current_intercept_coord is not None
            else None
        )
        self.last_intercept_progress_at = now
        self.last_intercept_target_coord = self.current_intercept_coord
        self.last_intercept_target_distance = self.best_intercept_coord_distance
        self.previous_intercept_coord = current_coord
        self.last_candidate_handoff_reason = None
        self.last_candidate_segment_distance = None
        return True

    def _finish(self, success: bool, reason: str, *, now: float) -> MiningOutcome:
        candidate = self.candidate
        if candidate is None:
            raise RuntimeError("No mining candidate is active")
        elapsed = self._elapsed_seconds(now)
        outcome = MiningOutcome(
            success,
            reason,
            candidate.node_id,
            candidate.coord,
            elapsed,
            self.target_marker_point,
            self.resume_coords,
            self.resume_route_index,
        )
        if success:
            cooldown = self.success_cooldown_seconds
        elif reason == "ore_marker_dark_below":
            cooldown = self.dark_failure_cooldown_seconds
        else:
            cooldown = self.failure_cooldown_seconds
        self.cooldowns[candidate.coord] = now + cooldown
        if not success and self.failure_spatial_cooldown_radius > 0.0:
            normalized_type = _normalize_ore_type(candidate.ore_type)
            cooldown_anchors = {
                int(candidate.coord),
                *(
                    (int(candidate.marker_intercept_coord),)
                    if candidate.marker_intercept_coord is not None
                    else ()
                ),
                *(
                    (int(self.last_marker_intercept_coord),)
                    if self.last_marker_intercept_coord is not None
                    else ()
                ),
            }
            self.spatial_cooldowns.extend(
                (anchor, normalized_type, now + cooldown)
                for anchor in cooldown_anchors
            )
        self._pending_outcome = outcome
        self.phase = MiningPhase.RESUME
        self.phase_started_at = now
        return outcome

    def _expire_cooldowns(self, now: float) -> None:
        self.cooldowns = {
            coord: deadline for coord, deadline in self.cooldowns.items() if deadline > now
        }
        self.spatial_cooldowns = [
            (coord, ore_type, deadline)
            for coord, ore_type, deadline in self.spatial_cooldowns
            if deadline > now
        ]

    def _marker_radial_distance(
        self,
        point: tuple[int, int] | None,
        shape: tuple[int, ...] | None,
    ) -> float | None:
        if point is None or shape is None or len(shape) < 2:
            return None
        height, width = shape[:2]
        dx = float(point[0]) - width * self.marker_center_x_fraction
        dy = float(point[1]) - height * self.marker_center_y_fraction
        return hypot(dx, dy)

    def _target_marker_radial_distance(self) -> float | None:
        return self._marker_radial_distance(
            self.target_marker_point,
            self.last_minimap_shape,
        )

    def _direct_marker_is_ready_for_world_scan(
        self,
        radial_distance: float | None,
        *,
        current_coord: int | None,
    ) -> bool:
        direct_stage_ready = bool(
            self.direct_marker_intercept_active
            and self.direct_marker_stage_index is not None
            and self.intercept_cursor >= self.direct_marker_stage_index
        )
        if not direct_stage_ready or self.candidate is None:
            return False
        if self.candidate.tooltip_confirmed:
            if (
                self.candidate.source == "database"
                and not self.database_anchor_arrived
            ):
                return False
            return bool(
                radial_distance is not None
                and radial_distance <= self.centered_marker_radius_pixels
            )
        candidate_close = bool(
            current_coord is not None
            and _coord_distance(current_coord, self.candidate.coord)
            <= self.direct_marker_world_scan_candidate_distance
        )
        candidate_world_search_close = bool(
            current_coord is not None
            and self.candidate.source == "database"
            and _coord_distance(current_coord, self.candidate.coord)
            <= self.world_search_distance
        )
        return bool(
            candidate_world_search_close
            or (
                radial_distance is not None
                and radial_distance <= self.direct_marker_world_scan_radius_pixels
                and candidate_close
            )
        )

    def _elapsed_seconds(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        elapsed = max(0.0, now - self.started_at)
        if self.phase == MiningPhase.SUSPENDED and self.suspended_at is not None:
            elapsed -= max(0.0, now - self.suspended_at)
        return max(0.0, elapsed)

    def _active_timeout_reason(self, now: float) -> str | None:
        if (
            self.max_total_seconds > 0.0
            and self.started_at is not None
            and self._elapsed_seconds(now) >= self.max_total_seconds
        ):
            return "ore_attempt_timeout"
        if self.phase_started_at is None:
            return None
        phase_elapsed = now - self.phase_started_at
        if self.phase == MiningPhase.INTERCEPT and self.max_intercept_seconds > 0.0:
            if phase_elapsed >= self.max_intercept_seconds:
                return "ore_intercept_timeout"
        if self.phase == MiningPhase.FACE_NODE and self.max_face_node_seconds > 0.0:
            if phase_elapsed >= self.max_face_node_seconds:
                return "ore_face_node_timeout"
        if self.phase == MiningPhase.WORLD_SCAN and self.max_world_scan_seconds > 0.0:
            if phase_elapsed >= self.max_world_scan_seconds:
                return "ore_world_scan_timeout"
        return None


def _primary_access_option(access_plan: Any | None) -> Any | None:
    if access_plan is None:
        return None
    primary_rank = int(getattr(access_plan, "primary_option_rank", 0))
    return next(
        (
            option
            for option in getattr(access_plan, "options", ())
            if int(getattr(option, "rank", -1)) == primary_rank
        ),
        None,
    )


def _ordered_access_options(access_plan: Any | None) -> tuple[Any, ...]:
    if access_plan is None:
        return ()
    return tuple(
        sorted(
            getattr(access_plan, "options", ()),
            key=lambda option: int(getattr(option, "rank", 0)),
        )
    )


def _access_option_by_rank(access_plan: Any | None, rank: int | None) -> Any | None:
    if rank is None:
        return None
    return next(
        (
            option
            for option in _ordered_access_options(access_plan)
            if int(getattr(option, "rank", -1)) == rank
        ),
        None,
    )


def _start_access_leg_near_current(
    coords: tuple[int, ...],
    current_coord: int | None,
) -> tuple[int, ...]:
    if not coords or current_coord is None:
        return coords
    nearest_index = min(
        range(len(coords)),
        key=lambda index: (_coord_distance(current_coord, coords[index]), index),
    )
    nearest_distance = _coord_distance(current_coord, coords[nearest_index])
    if nearest_distance <= 0.06:
        return coords[nearest_index:]
    return coords


class MiningHoverScanner:
    """Moves one hover probe per controller tick and never blocks the live loop."""

    def __init__(
        self,
        capture: CaptureLike,
        mouse: MouseLike,
        config: dict[str, Any],
        *,
        cursor_signature_reader: Any | None = None,
        cursor_snapshot_reader: Any | None = None,
        cursor_classifier: CursorTemplateClassifier | None = None,
        ore_world_detector: Any | None = None,
    ) -> None:
        self.capture = capture
        self.mouse = mouse
        self.config = config
        self.cursor_signature_reader = cursor_signature_reader or read_cursor_signature
        self.cursor_snapshot_reader = (
            cursor_snapshot_reader
            if cursor_snapshot_reader is not None
            else (
                capture_win32_cursor_snapshot
                if cursor_signature_reader is None
                else lambda: None
            )
        )
        self.cursor_classifier = cursor_classifier or _load_cursor_classifier(config)
        self.ore_world_detector = ore_world_detector or OreWorldDetector(config)
        self.last_ore_detector_result = OreWorldDetectorResult()
        self.last_cursor_classification = CursorClassification(
            None,
            0.0,
            None,
            None,
            self.cursor_classifier.has_templates(),
        )
        self.last_probe_evidence: dict[str, Any] = {}
        detector_cfg = config.get("mining", {}).get("ore_world_detector", {})
        self.ore_detector_refresh_interval = max(
            0.0,
            float(detector_cfg.get("refresh_interval_seconds", 0.0)),
        )
        self.ore_detector_refresh_point_dedupe = max(
            0.0,
            float(detector_cfg.get("refresh_point_dedupe_pixels", 20.0)),
        )
        self.next_ore_detector_at = 0.0
        self.ore_detector_runs = 0
        self.ore_detector_detection_runs = 0
        self.ore_detector_probe_points: list[tuple[int, int]] = []
        self.phase = HoverScanPhase.IDLE
        self.points: list[tuple[int, int]] = []
        self.cursor = 0
        self.pending_point: tuple[int, int] | None = None
        self.baseline_signature: int | None = None
        self.ready_at = 0.0
        self.max_points = 0
        self.cursor_candidate_settle_count = 0
        self.cursor_only_confirm_count = 0

    def start(self, frame_shape: tuple[int, ...], *, now: float) -> HoverScanStep:
        mining_cfg = self.config.get("mining", {})
        max_points = max(1, int(mining_cfg.get("scan_max_points", 72)))
        self.max_points = max_points
        self.points = build_scan_points(frame_shape, self.config)[:max_points]
        self.cursor = 0
        self.pending_point = None
        self.baseline_signature = None
        self.next_ore_detector_at = now
        self.ore_detector_runs = 0
        self.ore_detector_detection_runs = 0
        self.ore_detector_probe_points = []
        neutral = _scaled_client_point(
            mining_cfg.get("cursor_baseline_point", {"x": 1280, "y": 170}),
            frame_shape,
            self.config,
        )
        screen_x, screen_y = self.capture.client_to_screen_point(*neutral)
        self.mouse.move_to(screen_x, screen_y)
        self.ready_at = now + max(0.0, float(mining_cfg.get("hover_delay", 0.08)))
        self.phase = HoverScanPhase.BASELINE_SETTLE
        return HoverScanStep(self.phase, neutral, "cursor_baseline", 0)

    def step(self, frame: np.ndarray, *, now: float) -> HoverScanStep:
        if self.phase == HoverScanPhase.IDLE:
            return self.start(frame.shape, now=now)
        if self.phase in {HoverScanPhase.FOUND, HoverScanPhase.EXHAUSTED}:
            return HoverScanStep(self.phase, self.pending_point, self.phase.value, self.cursor)
        if now < self.ready_at:
            return HoverScanStep(self.phase, self.pending_point, "hover_settle", self.cursor)

        if self.phase == HoverScanPhase.BASELINE_SETTLE:
            self.baseline_signature = self.cursor_signature_reader()
            self.last_ore_detector_result = self.ore_world_detector.detect(frame)
            self.ore_detector_runs += 1
            if self.last_ore_detector_result.detections:
                self.ore_detector_detection_runs += 1
            self.next_ore_detector_at = now + self.ore_detector_refresh_interval
            detector_points = build_ore_detector_probe_points(
                self.last_ore_detector_result,
                frame.shape,
                self.config,
            )
            self.ore_detector_probe_points.extend(detector_points)
            priority_points = _merge_probe_points(
                detector_points,
                _merge_probe_points(
                    build_salient_ore_probe_points(frame, self.config),
                    build_front_target_probe_points(frame.shape, self.config),
                    limit=self.max_points,
                ),
                limit=self.max_points,
            )
            if priority_points:
                self.points = _merge_probe_points(
                    priority_points,
                    self.points,
                    limit=self.max_points,
                )
            return self._move_next_probe(now)

        if self.pending_point is None:
            return self._move_next_probe(now)

        if self.current_target_is_mining_ready(frame):
            self.phase = HoverScanPhase.FOUND
            classified = self.last_cursor_classification.label == "mine"
            cursor_only_ready = bool(
                self.last_probe_evidence.get("cursor_only_ready", False)
            )
            tooltip_only_ready = bool(
                self.last_probe_evidence.get("tooltip_only_ready", False)
            )
            return HoverScanStep(
                self.phase,
                self.pending_point,
                (
                    "mine_cursor_confirmed_without_tooltip"
                    if cursor_only_ready
                    else "mining_tooltip_confirmed"
                    if tooltip_only_ready
                    else
                    "tooltip_and_mine_cursor_classified"
                    if classified
                    else "tooltip_and_mining_cursor_confirmed"
                ),
                self.cursor,
                self.last_cursor_classification.label,
                self.last_cursor_classification.similarity,
            )
        if self._should_settle_cursor_candidate(now):
            return HoverScanStep(
                self.phase,
                self.pending_point,
                "cursor_candidate_settle",
                self.cursor,
                self.last_cursor_classification.label,
                self.last_cursor_classification.similarity,
            )
        if self._refresh_ore_detector_if_due(frame, now=now):
            return self._move_next_probe(now)
        return self._move_next_probe(now)

    def current_target_is_mining_ready(self, frame: np.ndarray) -> bool | None:
        if self.pending_point is None:
            return None
        mining_cfg = self.config.get("mining", {})
        tooltip_ready = detect_mining_tooltip(frame, self.config)
        self.last_cursor_classification = self.cursor_classifier.classify(
            self.cursor_snapshot_reader()
        )
        current_signature = self.cursor_signature_reader()
        cursor_changed = (
            self.baseline_signature is not None
            and current_signature is not None
            and current_signature != self.baseline_signature
        )
        pixel_pickaxe = bool(
            mining_cfg.get("cursor_pixel_fallback_enabled", True)
        ) and detect_pickaxe_cursor(frame, self.pending_point, self.config)
        require_tooltip = bool(mining_cfg.get("require_mining_tooltip", True))
        require_cursor = bool(mining_cfg.get("require_cursor_change", True))
        tooltip_ok = tooltip_ready or not require_tooltip
        mine_templates_ready = self.cursor_classifier.has_templates("mine")
        require_classified_when_calibrated = bool(
            mining_cfg.get("require_classified_cursor_when_calibrated", True)
        )
        if mine_templates_ready and require_classified_when_calibrated:
            cursor_ok = self.last_cursor_classification.label == "mine"
        else:
            cursor_ok = cursor_changed or pixel_pickaxe or not require_cursor
        tooltip_only_ready = bool(
            tooltip_ready
            and mining_cfg.get("tooltip_only_click_enabled", True)
            and not cursor_ok
            and not (mine_templates_ready and require_classified_when_calibrated)
        )
        cursor_only_candidate = bool(
            not tooltip_ready
            and (
                self.last_cursor_classification.label == "mine"
                or (pixel_pickaxe and not mining_cfg.get("cursor_color_fallback_enabled", True))
            )
        )
        if cursor_only_candidate:
            self.cursor_only_confirm_count += 1
        else:
            self.cursor_only_confirm_count = 0
        cursor_only_ready = bool(
            mining_cfg.get("cursor_only_click_enabled", False)
            and cursor_only_candidate
            and self.cursor_only_confirm_count
            >= max(1, int(mining_cfg.get("cursor_only_confirm_frames", 3)))
        )
        self.last_probe_evidence = {
            "point": list(self.pending_point),
            "tooltip_ready": bool(tooltip_ready),
            "cursor_changed": bool(cursor_changed),
            "pixel_pickaxe": bool(pixel_pickaxe),
            "cursor_label": self.last_cursor_classification.label,
            "cursor_similarity": self.last_cursor_classification.similarity,
            "mine_templates_ready": bool(mine_templates_ready),
            "tooltip_ok": bool(tooltip_ok),
            "cursor_ok": bool(cursor_ok),
            "tooltip_only_ready": tooltip_only_ready,
            "cursor_only_candidate": cursor_only_candidate,
            "cursor_only_confirm_count": self.cursor_only_confirm_count,
            "cursor_only_ready": cursor_only_ready,
        }
        return bool((tooltip_ok and cursor_ok) or tooltip_only_ready or cursor_only_ready)

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "cursor": self.cursor,
            "point_count": len(self.points),
            "pending_point": (
                list(self.pending_point) if self.pending_point is not None else None
            ),
            "last_probe_evidence": dict(self.last_probe_evidence),
            "ore_world_detector": {
                "reason": self.last_ore_detector_result.reason,
                "elapsed_ms": self.last_ore_detector_result.elapsed_ms,
                "detections": [
                    {
                        "bbox": [
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                        ],
                        "confidence": detection.confidence,
                    }
                    for detection in self.last_ore_detector_result.detections
                ],
                "runs": self.ore_detector_runs,
                "detection_runs": self.ore_detector_detection_runs,
                "refresh_interval_seconds": self.ore_detector_refresh_interval,
                "probe_points": [
                    list(point) for point in self.ore_detector_probe_points
                ],
            },
        }

    def reset(self) -> None:
        self.phase = HoverScanPhase.IDLE
        self.points = []
        self.cursor = 0
        self.pending_point = None
        self.baseline_signature = None
        self.ready_at = 0.0
        self.max_points = 0
        self.cursor_candidate_settle_count = 0
        self.cursor_only_confirm_count = 0
        self.last_cursor_classification = CursorClassification(
            None,
            0.0,
            None,
            None,
            self.cursor_classifier.has_templates(),
        )
        self.last_probe_evidence = {}
        self.last_ore_detector_result = OreWorldDetectorResult()
        self.next_ore_detector_at = 0.0
        self.ore_detector_runs = 0
        self.ore_detector_detection_runs = 0
        self.ore_detector_probe_points = []

    def _refresh_ore_detector_if_due(
        self,
        frame: np.ndarray,
        *,
        now: float,
    ) -> bool:
        if (
            self.ore_detector_refresh_interval <= 0.0
            or now < self.next_ore_detector_at
        ):
            return False
        self.next_ore_detector_at = now + self.ore_detector_refresh_interval
        self.last_ore_detector_result = self.ore_world_detector.detect(frame)
        self.ore_detector_runs += 1
        if not self.last_ore_detector_result.detections:
            return False
        self.ore_detector_detection_runs += 1
        detected_points = build_ore_detector_probe_points(
            self.last_ore_detector_result,
            frame.shape,
            self.config,
        )
        novel_points: list[tuple[int, int]] = []
        for point in detected_points:
            if any(
                _point_distance(point, previous)
                <= self.ore_detector_refresh_point_dedupe
                for previous in self.ore_detector_probe_points
            ):
                continue
            self.ore_detector_probe_points.append(point)
            novel_points.append(point)
        if not novel_points:
            return False

        visited = self.points[: self.cursor]
        remaining = [
            point
            for point in self.points[self.cursor :]
            if all(
                _point_distance(point, detected)
                > self.ore_detector_refresh_point_dedupe
                for detected in novel_points
            )
        ]
        remaining_capacity = max(0, self.max_points - len(visited))
        self.points = visited + (novel_points + remaining)[:remaining_capacity]
        return True

    def _move_next_probe(self, now: float) -> HoverScanStep:
        if self.cursor >= len(self.points):
            self.phase = HoverScanPhase.EXHAUSTED
            self.pending_point = None
            return HoverScanStep(self.phase, None, "hover_not_found", self.cursor)
        point = self.points[self.cursor]
        self.cursor += 1
        self.pending_point = point
        self.cursor_candidate_settle_count = 0
        self.cursor_only_confirm_count = 0
        screen_x, screen_y = self.capture.client_to_screen_point(*point)
        self.mouse.move_to(screen_x, screen_y)
        hover_delay = max(0.0, float(self.config.get("mining", {}).get("hover_delay", 0.08)))
        self.ready_at = now + hover_delay
        self.phase = HoverScanPhase.PROBE_SETTLE
        return HoverScanStep(self.phase, point, "hover_probe", self.cursor)

    def _should_settle_cursor_candidate(self, now: float) -> bool:
        mining_cfg = self.config.get("mining", {})
        if not bool(mining_cfg.get("cursor_candidate_extra_settle_enabled", True)):
            return False
        if not bool(self.last_probe_evidence.get("cursor_only_candidate", False)):
            return False
        max_settle_frames = max(
            0,
            int(mining_cfg.get("cursor_candidate_extra_settle_frames", 2)),
        )
        if self.cursor_candidate_settle_count >= max_settle_frames:
            return False
        self.cursor_candidate_settle_count += 1
        self.ready_at = now + max(
            0.0,
            float(mining_cfg.get("cursor_candidate_extra_settle_seconds", 0.12)),
        )
        return True


class MiningInteractor:
    def __init__(
        self,
        capture: CaptureLike,
        mouse: MouseLike,
        config: dict[str, Any],
        *,
        cursor_snapshot_reader: Any | None = None,
        cursor_classifier: CursorTemplateClassifier | None = None,
        ore_world_detector: Any | None = None,
    ) -> None:
        self.capture = capture
        self.mouse = mouse
        self.config = config
        self.cursor_snapshot_reader = cursor_snapshot_reader or capture_win32_cursor_snapshot
        self.cursor_classifier = cursor_classifier or _load_cursor_classifier(config)
        self.hover_scanner = MiningHoverScanner(
            capture,
            mouse,
            config,
            cursor_snapshot_reader=self.cursor_snapshot_reader,
            cursor_classifier=self.cursor_classifier,
            ore_world_detector=ore_world_detector,
        )

    def warm_up_ore_world_detector(self, frame: np.ndarray) -> OreWorldDetectorResult:
        result = self.hover_scanner.ore_world_detector.warm_up(frame)
        self.hover_scanner.last_ore_detector_result = result
        return result

    def start_hover_scan(self, frame_shape: tuple[int, ...], *, now: float) -> HoverScanStep:
        return self.hover_scanner.start(frame_shape, now=now)

    def poll_hover_scan(self, frame: np.ndarray, *, now: float) -> HoverScanStep:
        return self.hover_scanner.step(frame, now=now)

    def cancel_hover_scan(self) -> None:
        self.hover_scanner.reset()

    def current_cursor_is_mining_ready(self, frame: np.ndarray) -> bool:
        """Accept an already-hovered ore tooltip before moving the user's cursor."""
        if not detect_mining_tooltip(frame, self.config):
            return False
        if not self.cursor_classifier.has_templates("mine"):
            return True
        return (
            self.cursor_classifier.classify(self.cursor_snapshot_reader()).label
            == "mine"
        )

    def right_click(self) -> None:
        click_duration = float(self.config.get("mining", {}).get("right_click_duration", 0.05))
        self.mouse.right_click(duration=click_duration)

    def hover_target_is_mining_ready(self, frame: np.ndarray) -> bool | None:
        result = self.hover_scanner.current_target_is_mining_ready(frame)
        if result is None:
            return detect_mining_tooltip(frame, self.config)
        return result

    def find_hover_target(self, frame: np.ndarray | None = None) -> HoverResult:
        if frame is None:
            frame = self.capture.capture_client_region()
        hover_delay = max(0.01, float(self.config.get("mining", {}).get("hover_delay", 0.08)))
        step = self.start_hover_scan(frame.shape, now=time.monotonic())
        while step.phase not in {HoverScanPhase.FOUND, HoverScanPhase.EXHAUSTED}:
            time.sleep(hover_delay)
            hover_frame = self.capture.capture_client_region()
            step = self.poll_hover_scan(hover_frame, now=time.monotonic())
        return HoverResult(
            found=step.phase == HoverScanPhase.FOUND,
            point=step.point,
            reason=step.reason,
            scanned_points=step.scanned_points,
        )

    def right_click_hover_target(self, frame: np.ndarray | None = None) -> HoverResult:
        result = self.find_hover_target(frame)
        if not result.found:
            return result

        mining_cfg = self.config.get("mining", {})
        post_click_wait = float(mining_cfg.get("right_click_wait_seconds", 1.5))
        self.right_click()
        time.sleep(max(0.0, post_click_wait))
        return result


def read_cursor_signature() -> int | None:
    """Return the active Win32 cursor handle without reading game memory."""
    try:
        import ctypes
        from ctypes import wintypes

        class CursorInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", wintypes.HANDLE),
                ("ptScreenPos", wintypes.POINT),
            ]

        info = CursorInfo()
        info.cbSize = ctypes.sizeof(CursorInfo)
        if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(info)):
            return None
        return int(info.hCursor) if info.hCursor else None
    except Exception:
        return None


def _load_cursor_classifier(config: dict[str, Any]) -> CursorTemplateClassifier:
    classifier_cfg = config.get("cursor_classifier", {})
    if not bool(classifier_cfg.get("enabled", True)):
        return CursorTemplateClassifier()
    path = resource_path(
        str(
            classifier_cfg.get(
                "templates_path",
                "data/cursor_templates/cursor_templates.json",
            )
        )
    )
    return CursorTemplateClassifier.load(
        path,
        min_similarity=float(classifier_cfg.get("min_similarity", 0.90)),
    )


def _coord_distance(a: int, b: int) -> float:
    ax, ay = _coord_xy(a)
    bx, by = _coord_xy(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _coord_segment_distance(point: int, start: int, end: int) -> float:
    point_x, point_y = _coord_xy(point)
    start_x, start_y = _coord_xy(start)
    end_x, end_y = _coord_xy(end)
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 1e-12:
        return hypot(point_x - start_x, point_y - start_y)
    projection = (
        (point_x - start_x) * delta_x + (point_y - start_y) * delta_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest_x = start_x + projection * delta_x
    closest_y = start_y + projection * delta_y
    return hypot(point_x - closest_x, point_y - closest_y)


def _access_attachment_distance(access_plan: Any, current_coord: int) -> float | None:
    if access_plan is None:
        return None
    distances = [
        _coord_distance(current_coord, int(inbound[0]))
        for option in getattr(access_plan, "options", ())
        if (inbound := tuple(getattr(option, "inbound_coords", ())))
    ]
    return min(distances) if distances else None


def _coord_xy(coord: int) -> tuple[float, float]:
    from vision_bot.coords import coord_to_xy

    return coord_to_xy(coord)


def _scaled_client_point(
    point_cfg: Any,
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> tuple[int, int]:
    if not isinstance(point_cfg, dict):
        raise ValueError("cursor_baseline_point must be an object with x/y")
    reference_width = max(1, int(config.get("screen", {}).get("reference_width", frame_shape[1])))
    reference_height = max(1, int(config.get("screen", {}).get("reference_height", frame_shape[0])))
    scale = bool(config.get("screen", {}).get("scale_regions", True))
    x = float(point_cfg.get("x", reference_width / 2.0))
    y = float(point_cfg.get("y", reference_height * 0.12))
    if scale:
        x *= frame_shape[1] / reference_width
        y *= frame_shape[0] / reference_height
    return (
        max(0, min(frame_shape[1] - 1, int(round(x)))),
        max(0, min(frame_shape[0] - 1, int(round(y)))),
    )


def build_scan_points(frame_shape: tuple[int, ...], config: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    frame_height, frame_width = frame_shape[:2]
    default_region = {
        "x": int(frame_width * 0.34),
        "y": int(frame_height * 0.20),
        "width": int(frame_width * 0.36),
        "height": int(frame_height * 0.42),
    }
    if "scan_region" in mining_cfg:
        x, y, width, height = resolve_region(mining_cfg["scan_region"], frame_shape, cfg)
    else:
        x = default_region["x"]
        y = default_region["y"]
        width = default_region["width"]
        height = default_region["height"]
    step = max(8, int(mining_cfg.get("scan_step", 34)))

    center_x = x + width // 2
    center_y = y + height // 2
    points: list[tuple[int, int]] = []
    for row in range(y, y + height + 1, step):
        for col in range(x, x + width + 1, step):
            points.append((col, row))

    return sorted(points, key=lambda point: (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)


def build_front_target_probe_points(
    frame_shape: tuple[int, ...],
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    """Probe the stable near-field where a faced mining node is normally rendered."""
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    if not bool(mining_cfg.get("front_probe_enabled", True)):
        return []

    frame_height, frame_width = frame_shape[:2]
    center_x = frame_width * float(mining_cfg.get("front_probe_center_x_fraction", 0.50))
    y_fractions = tuple(
        float(value)
        for value in mining_cfg.get(
            "front_probe_y_fractions",
            [0.62, 0.68, 0.56, 0.74, 0.50],
        )
    )
    reference_width = max(1, int(cfg.get("screen", {}).get("reference_width", frame_width)))
    scale_x = frame_width / reference_width
    x_offsets = tuple(
        float(value)
        for value in mining_cfg.get(
            "front_probe_x_offsets",
            [0, -32, 32, -64, 64, -96, 96],
        )
    )
    max_points = max(1, int(mining_cfg.get("front_probe_max_points", 35)))

    points: list[tuple[int, int]] = []
    for y_fraction in y_fractions:
        y = max(0, min(frame_height - 1, int(round(frame_height * y_fraction))))
        for offset in x_offsets:
            x = max(
                0,
                min(frame_width - 1, int(round(center_x + offset * scale_x))),
            )
            point = (x, y)
            if point not in points:
                points.append(point)
            if len(points) >= max_points:
                return points
    return points


def mining_target_alignment(
    point: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any] | None = None,
) -> MiningTargetAlignment:
    mining_cfg = (config or {}).get("mining", {})
    width = max(1, int(frame_shape[1]))
    center_x = width * float(mining_cfg.get("world_target_center_x_fraction", 0.50))
    offset = float(point[0]) - center_x
    deadzone = max(
        1.0,
        width * float(mining_cfg.get("world_target_face_deadzone_fraction", 0.055)),
    )
    if abs(offset) <= deadzone:
        return MiningTargetAlignment(True, None, 0.0, offset)
    min_seconds = max(0.0, float(mining_cfg.get("world_target_face_min_seconds", 0.05)))
    max_seconds = max(min_seconds, float(mining_cfg.get("world_target_face_max_seconds", 0.18)))
    seconds_per_width = max(
        0.0,
        float(mining_cfg.get("world_target_face_seconds_per_screen_width", 0.55)),
    )
    duration = max(min_seconds, min(max_seconds, abs(offset) / width * seconds_per_width))
    return MiningTargetAlignment(False, "A" if offset < 0.0 else "D", duration, offset)


def build_salient_ore_probe_points(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    """Prioritize colorful world objects; cursor evidence remains authoritative."""
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    if not bool(mining_cfg.get("salient_probe_enabled", True)):
        return []
    region_cfg = mining_cfg.get(
        "salient_probe_region",
        {"x": 220, "y": 180, "width": 1760, "height": 950},
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, cfg)
    if width <= 0 or height <= 0:
        return []
    roi = frame[y : y + height, x : x + width]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.asarray(mining_cfg.get("salient_probe_hsv_lower", [35, 80, 90]), dtype=np.uint8)
    upper = np.asarray(mining_cfg.get("salient_probe_hsv_upper", [115, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel_size = max(1, int(mining_cfg.get("salient_probe_close_kernel", 5)))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = max(1.0, float(mining_cfg.get("salient_probe_min_area", 24.0)))
    max_area = max(min_area, float(mining_cfg.get("salient_probe_max_area", 24000.0)))
    offsets = tuple(
        int(value)
        for value in mining_cfg.get("salient_probe_offsets", [0, -22, 22])
    )
    candidates: list[tuple[float, int, int, int, int]] = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        center_x = x + bx + bw // 2
        center_y = y + by + bh // 2
        saturation = float(np.mean(hsv[by : by + bh, bx : bx + bw, 1]))
        value = float(np.mean(hsv[by : by + bh, bx : bx + bw, 2]))
        score = area * (0.5 + saturation / 255.0) * (0.5 + value / 255.0)
        candidates.append((score, center_x, center_y, bw, bh))

    points: list[tuple[int, int]] = []
    for _, center_x, center_y, bw, bh in sorted(candidates, reverse=True):
        local_offsets = offsets if max(bw, bh) >= 40 else (0,)
        for dx in local_offsets:
            for dy in local_offsets:
                px = max(x, min(x + width - 1, center_x + dx))
                py = max(y, min(y + height - 1, center_y + dy))
                points.append((px, py))
    return _merge_probe_points(points, [], limit=max(1, int(mining_cfg.get("salient_probe_max_points", 24))))


def _merge_probe_points(
    priority: list[tuple[int, int]],
    fallback: list[tuple[int, int]],
    *,
    limit: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in (*priority, *fallback):
        normalized = (int(point[0]), int(point[1]))
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
        if len(merged) >= max(1, int(limit)):
            break
    return merged


def is_mining_hover_ready(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
    cursor_point: tuple[int, int] | None = None,
) -> bool:
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    if detect_mining_tooltip(frame, cfg):
        return True

    if bool(mining_cfg.get("cursor_pickaxe_detection_enabled", True)) and cursor_point is not None:
        return detect_pickaxe_cursor(frame, cursor_point, cfg)

    return False


def detect_mining_tooltip(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    min_yellow = int(mining_cfg.get("tooltip_min_yellow_pixels", 18))
    min_green = int(mining_cfg.get("tooltip_min_green_pixels", 18))
    min_dark_ratio = float(mining_cfg.get("tooltip_min_dark_ratio", 0.20))
    reject_hostile = bool(mining_cfg.get("reject_hostile_tooltip", True))

    neutral_fallback_enabled = bool(
        mining_cfg.get("tooltip_neutral_fallback_enabled", False)
    )
    neutral_fallback_indices = {
        int(value)
        for value in mining_cfg.get("tooltip_neutral_fallback_region_indices", ())
    }
    min_neutral = int(mining_cfg.get("tooltip_min_neutral_pixels", 180))

    for region_index, tooltip_region in enumerate(_mining_tooltip_regions(mining_cfg)):
        tooltip = crop_region(frame, tooltip_region, cfg)
        if tooltip.size == 0:
            continue
        if reject_hostile and _tooltip_crop_looks_hostile_unit(tooltip, mining_cfg):
            continue
        hsv = cv2.cvtColor(tooltip, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv,
            np.array([18, 70, 90], dtype=np.uint8),
            np.array([42, 255, 255], dtype=np.uint8),
        )
        green = cv2.inRange(
            hsv,
            np.array([45, 50, 70], dtype=np.uint8),
            np.array([95, 255, 255], dtype=np.uint8),
        )
        neutral = cv2.inRange(
            hsv,
            np.array([0, 0, 90], dtype=np.uint8),
            np.array([179, 70, 255], dtype=np.uint8),
        )
        dark = cv2.inRange(
            hsv,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([179, 255, 70], dtype=np.uint8),
        )
        dark_ratio = float(np.count_nonzero(dark)) / float(dark.size)
        yellow_ready = np.count_nonzero(yellow) >= min_yellow
        green_ready = np.count_nonzero(green) >= min_green
        neutral_ready = bool(
            neutral_fallback_enabled
            and region_index in neutral_fallback_indices
            and np.count_nonzero(neutral) >= min_neutral
        )
        if yellow_ready and (green_ready or neutral_ready) and dark_ratio >= min_dark_ratio:
            return True
    return False


def _mining_tooltip_regions(mining_cfg: dict[str, Any]) -> list[Any]:
    default_region = {"x": 2380, "y": 1010, "width": 180, "height": 130}
    configured_regions = mining_cfg.get("tooltip_regions")
    if isinstance(configured_regions, list) and configured_regions:
        return list(configured_regions)
    regions: list[Any] = [mining_cfg.get("tooltip_region", default_region)]
    fallback_regions = mining_cfg.get(
        "tooltip_fallback_regions",
        [{"x": 2100, "y": 1120, "width": 350, "height": 160}],
    )
    if isinstance(fallback_regions, list):
        regions.extend(fallback_regions)
    return regions


def _tooltip_crop_looks_hostile_unit(
    tooltip: np.ndarray,
    mining_cfg: dict[str, Any],
) -> bool:
    if tooltip.size == 0:
        return False
    hsv = cv2.cvtColor(tooltip, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    top = hsv[: max(1, int(height * 0.55)), :, :]
    bottom = hsv[max(0, int(height * 0.70)) :, :, :]
    red_low = cv2.inRange(
        top,
        np.array([0, 80, 80], dtype=np.uint8),
        np.array([14, 255, 255], dtype=np.uint8),
    )
    red_high = cv2.inRange(
        top,
        np.array([170, 80, 80], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    bottom_green = cv2.inRange(
        bottom,
        np.array([45, 70, 70], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    min_red = int(mining_cfg.get("hostile_tooltip_min_red_pixels", 12))
    min_bottom_green = int(
        mining_cfg.get("hostile_tooltip_min_bottom_green_pixels", 55)
    )
    return bool(
        np.count_nonzero(red_low | red_high) >= min_red
        and np.count_nonzero(bottom_green) >= min_bottom_green
    )


def detect_pickaxe_cursor(
    frame: np.ndarray,
    cursor_point: tuple[int, int],
    config: dict[str, Any] | None = None,
) -> bool:
    cfg = config or {}
    mining_cfg = cfg.get("mining", {})
    radius = max(8, int(mining_cfg.get("cursor_detection_radius", 28)))
    x, y = cursor_point
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(frame.shape[1], x + radius + 1)
    y1 = min(frame.shape[0], y + radius + 1)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return False

    template_result = _detect_pickaxe_cursor_template(crop, cfg)
    if template_result is not None:
        return template_result
    if not bool(mining_cfg.get("cursor_color_fallback_enabled", True)):
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright_metal = cv2.inRange(hsv, np.array([0, 0, 155], dtype=np.uint8), np.array([179, 95, 255], dtype=np.uint8))
    amber_handle = cv2.inRange(hsv, np.array([8, 70, 75], dtype=np.uint8), np.array([38, 255, 235], dtype=np.uint8))

    min_metal = int(mining_cfg.get("cursor_min_metal_pixels", 14))
    min_amber = int(mining_cfg.get("cursor_min_amber_pixels", 4))
    return np.count_nonzero(bright_metal) >= min_metal and np.count_nonzero(amber_handle) >= min_amber


_CURSOR_TEMPLATE_IMAGE_CACHE: dict[tuple[str, ...], dict[str, tuple[np.ndarray, ...]]] = {}
_CURSOR_TEMPLATE_CLASSIFIER_CACHE: dict[str, CursorTemplateClassifier] = {}


def _detect_pickaxe_cursor_template(
    crop: np.ndarray,
    config: dict[str, Any],
) -> bool | None:
    mining_cfg = config.get("mining", {})
    if not bool(mining_cfg.get("cursor_template_fallback_enabled", True)):
        return None
    classifier = _load_cursor_asset_classifier(config)
    if classifier.has_templates():
        visual_hash = perceptual_cursor_hash(crop, hash_size=classifier.hash_size)
        classification = classifier.classify(
            CursorSnapshot(
                None,
                crop,
                visual_hash,
                classifier.hash_size,
                cursor_colorfulness(crop),
            )
        )
        if classification.label == "mine":
            return True
        if classification.label == "unable_mine":
            return False
    templates = _load_cursor_template_images(config)
    mine_score = _best_template_score(crop, templates.get("mine", ()))
    unable_score = _best_template_score(crop, templates.get("unable_mine", ()))
    if mine_score is None and unable_score is None:
        return None
    min_score = float(mining_cfg.get("cursor_template_min_score", 0.62))
    margin = float(mining_cfg.get("cursor_template_margin", 0.04))
    if unable_score is not None and unable_score >= min_score and (
        mine_score is None or unable_score >= mine_score - margin
    ):
        return False
    if mine_score is not None and mine_score >= min_score and (
        unable_score is None or mine_score >= unable_score + margin
    ):
        return True
    return False


def _load_cursor_asset_classifier(config: dict[str, Any]) -> CursorTemplateClassifier:
    cursor_cfg = config.get("cursor_classifier", {})
    path = resource_path(
        str(
            cursor_cfg.get(
                "templates_path",
                "data/cursor_templates/cursor_templates.json",
            )
        )
    )
    cache_key = str(path)
    if cache_key not in _CURSOR_TEMPLATE_CLASSIFIER_CACHE:
        _CURSOR_TEMPLATE_CLASSIFIER_CACHE[cache_key] = CursorTemplateClassifier.load(
            path,
            min_similarity=float(cursor_cfg.get("min_similarity", 0.90)),
        )
    return _CURSOR_TEMPLATE_CLASSIFIER_CACHE[cache_key]


def _load_cursor_template_images(config: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    cursor_cfg = config.get("cursor_classifier", {})
    root = resource_path(
        str(
            cursor_cfg.get(
                "asset_template_dir",
                "data/cursor_templates/client_assets",
            )
        )
    )
    labels = {
        "mine": ("Mine.png", "Mine2.png", "Mine4.png"),
        "unable_mine": ("UnableMine.png", "UnableMine2.png", "UnableMine4.png"),
    }
    cache_key = tuple(str(root / filename) for names in labels.values() for filename in names)
    if cache_key in _CURSOR_TEMPLATE_IMAGE_CACHE:
        return _CURSOR_TEMPLATE_IMAGE_CACHE[cache_key]
    loaded: dict[str, tuple[np.ndarray, ...]] = {}
    for label, filenames in labels.items():
        images: list[np.ndarray] = []
        for filename in filenames:
            path = root / filename
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                continue
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            elif image.shape[2] == 3:
                alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
                image = np.concatenate([image, alpha], axis=2)
            images.append(image)
        loaded[label] = tuple(images)
    _CURSOR_TEMPLATE_IMAGE_CACHE[cache_key] = loaded
    return loaded


def _best_template_score(crop: np.ndarray, templates: tuple[np.ndarray, ...]) -> float | None:
    if crop.size == 0 or not templates:
        return None
    crop_bgr = crop[:, :, :3] if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    best: float | None = None
    for template in templates:
        template_bgr = template[:, :, :3]
        alpha = template[:, :, 3] if template.shape[2] >= 4 else None
        template_mask = alpha if alpha is not None and np.any(alpha > 0) else None
        for scale in (1.0, 0.75, 0.50):
            width = max(4, int(round(template_bgr.shape[1] * scale)))
            height = max(4, int(round(template_bgr.shape[0] * scale)))
            if width > crop_bgr.shape[1] or height > crop_bgr.shape[0]:
                continue
            resized = cv2.resize(template_bgr, (width, height), interpolation=cv2.INTER_AREA)
            mask = None
            if template_mask is not None:
                mask = cv2.resize(template_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                if np.count_nonzero(mask) < 8:
                    mask = None
            method = cv2.TM_CCORR_NORMED if mask is not None else cv2.TM_CCOEFF_NORMED
            result = cv2.matchTemplate(crop_bgr, resized, method, mask=mask)
            if result.size == 0:
                continue
            finite_scores = result[np.isfinite(result)]
            if finite_scores.size == 0:
                continue
            score = float(np.max(finite_scores))
            if np.isfinite(score):
                best = score if best is None else max(best, score)
    return best
