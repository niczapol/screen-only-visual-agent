from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Any

from vision_bot.coords import coord_to_xy


class MiningFinalApproachAction(str, Enum):
    COARSE = "coarse"
    STOP_RECHECK = "stop_recheck"
    ALIGN = "align"
    WAIT_FEEDBACK = "wait_feedback"
    ARRIVED = "arrived"
    FAILED = "failed"


@dataclass(frozen=True)
class MiningFinalApproachDecision:
    action: MiningFinalApproachAction
    distance_yards: float
    burst_seconds: float | None = None
    reason: str | None = None


def coordinate_distance_yards(
    first: int,
    second: int,
    *,
    map_width_yards: float,
    map_height_yards: float,
) -> float:
    """Return physical distance for percentage-space map coordinates."""
    first_x, first_y = coord_to_xy(first)
    second_x, second_y = coord_to_xy(second)
    return hypot(
        (second_x - first_x) * float(map_width_yards) / 100.0,
        (second_y - first_y) * float(map_height_yards) / 100.0,
    )


class MiningFinalApproachController:
    """One stopped recheck followed by at most one timed forward impulse.

    This controller owns only the final database-anchor arrival.  Coarse travel
    remains continuous.  Once the character enters the short final zone it must
    stop, wait for a later accepted coordinate sample, align without moving,
    then issue one calculated W pulse.  The pulse is never repeated: visible
    coordinate feedback either accepts arrival or fails the bounded mining
    transaction so the existing spatial cooldown can prevent a loop.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        mining_cfg = (config or {}).get("mining", {})
        cfg = mining_cfg.get("final_database_approach", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self.enabled = bool(cfg.get("enabled", True))
        self.map_width_yards = max(1.0, float(cfg.get("map_width_yards", 6900.0)))
        self.map_height_yards = max(1.0, float(cfg.get("map_height_yards", 4600.0)))
        self.trigger_distance_yards = max(
            0.1, float(cfg.get("trigger_distance_yards", 3.0))
        )
        self.arrival_tolerance_yards = max(
            0.0,
            min(
                self.trigger_distance_yards,
                float(cfg.get("arrival_tolerance_yards", 0.9)),
            ),
        )
        # The one-shot burst is intentionally non-repeatable. A slightly wider
        # post-burst handoff may therefore enter the tooltip-gated world scan
        # without declaring a reachable, visible vein inaccessible. It does
        # not weaken the initial arrival test or authorize a click by itself.
        self.post_burst_world_scan_tolerance_yards = max(
            self.arrival_tolerance_yards,
            min(
                self.trigger_distance_yards,
                float(
                    cfg.get(
                        "post_burst_world_scan_tolerance_yards",
                        self.arrival_tolerance_yards,
                    )
                ),
            ),
        )
        self.mounted_speed_yards_per_second = max(
            0.1, float(cfg.get("mounted_speed_yards_per_second", 16.8))
        )
        self.foot_speed_yards_per_second = max(
            0.1, float(cfg.get("foot_speed_yards_per_second", 7.0))
        )
        self.min_burst_seconds = max(
            0.01, float(cfg.get("min_burst_seconds", 0.05))
        )
        legacy_max_burst_seconds = max(
            self.min_burst_seconds,
            float(cfg.get("max_burst_seconds", 0.24)),
        )
        self.mounted_max_burst_seconds = max(
            self.min_burst_seconds,
            float(cfg.get("mounted_max_burst_seconds", legacy_max_burst_seconds)),
        )
        self.foot_max_burst_seconds = max(
            self.min_burst_seconds,
            float(cfg.get("foot_max_burst_seconds", legacy_max_burst_seconds)),
        )
        # Keep the legacy aggregate available to diagnostics/tests while the
        # actual cap is selected from the observed mounted state.
        self.max_burst_seconds = max(
            self.mounted_max_burst_seconds,
            self.foot_max_burst_seconds,
        )
        self.burst_stop_short_yards = max(
            0.0,
            min(
                self.arrival_tolerance_yards,
                float(cfg.get("burst_stop_short_yards", 0.0)),
            ),
        )
        self.precision_coord_interval_seconds = max(
            0.05,
            float(cfg.get("precision_coord_interval_seconds", 0.15)),
        )
        self.feedback_settle_seconds = max(
            0.0, float(cfg.get("feedback_settle_seconds", 0.35))
        )
        self.feedback_timeout_seconds = max(
            self.feedback_settle_seconds,
            float(cfg.get("feedback_timeout_seconds", 1.75)),
        )
        self.reset()

    def reset(self) -> None:
        self.target_coord: int | None = None
        self.phase = "idle"
        self.recheck_started_at: float | None = None
        self.burst_sent_at: float | None = None
        self.burst_origin_coord: int | None = None
        self.burst_seconds: float | None = None
        self.last_distance_yards: float | None = None
        self.last_reason: str | None = None

    def distance_yards(self, first: int, second: int) -> float:
        return coordinate_distance_yards(
            first,
            second,
            map_width_yards=self.map_width_yards,
            map_height_yards=self.map_height_yards,
        )

    @property
    def precision_coord_feedback_active(self) -> bool:
        """Request faster OCR only while stopped for final feedback."""
        return self.phase in {"wait_recheck", "burst_sent"}

    def observe(
        self,
        current_coord: int,
        target_coord: int,
        *,
        now: float,
        coord_fresh: bool,
    ) -> MiningFinalApproachDecision:
        target_coord = int(target_coord)
        if self.target_coord is not None and self.target_coord != target_coord:
            self.reset()
        self.target_coord = target_coord
        distance = self.distance_yards(current_coord, target_coord)
        self.last_distance_yards = distance

        if not self.enabled:
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.COARSE,
                distance,
                reason="final_database_approach_disabled",
            )
        if self.phase == "burst_sent":
            return self._observe_burst_feedback(
                current_coord,
                now=now,
                coord_fresh=coord_fresh,
            )
        if distance <= self.arrival_tolerance_yards:
            self.phase = "arrived"
            self.last_reason = "database_anchor_within_tolerance"
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.ARRIVED,
                distance,
                reason=self.last_reason,
            )
        if self.phase == "arrived":
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.FAILED,
                distance,
                reason="database_anchor_drift_after_arrival",
            )
        if distance > self.trigger_distance_yards:
            self.phase = "idle"
            self.recheck_started_at = None
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.COARSE,
                distance,
            )
        if self.phase == "idle":
            self.phase = "wait_recheck"
            self.recheck_started_at = now
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.STOP_RECHECK,
                distance,
                reason="entered_final_database_zone",
            )
        if self.phase == "wait_recheck":
            if not coord_fresh or (
                self.recheck_started_at is not None
                and now <= self.recheck_started_at
            ):
                return MiningFinalApproachDecision(
                    MiningFinalApproachAction.STOP_RECHECK,
                    distance,
                    reason="awaiting_fresh_stopped_coordinate",
                )
            self.phase = "align"
        return MiningFinalApproachDecision(
            MiningFinalApproachAction.ALIGN,
            distance,
            reason="fresh_coordinate_rechecked",
        )

    def commit_burst(
        self,
        current_coord: int,
        target_coord: int,
        *,
        now: float,
        mounted: bool,
    ) -> MiningFinalApproachDecision:
        if self.phase != "align" or self.target_coord != int(target_coord):
            raise RuntimeError("Final mining burst is not aligned and armed")
        distance = self.distance_yards(current_coord, target_coord)
        speed = (
            self.mounted_speed_yards_per_second
            if mounted
            else self.foot_speed_yards_per_second
        )
        max_burst_seconds = (
            self.mounted_max_burst_seconds
            if mounted
            else self.foot_max_burst_seconds
        )
        travel_yards = max(0.0, distance - self.burst_stop_short_yards)
        duration = max(
            self.min_burst_seconds,
            min(max_burst_seconds, travel_yards / speed),
        )
        self.phase = "burst_sent"
        self.burst_sent_at = now
        self.burst_origin_coord = int(current_coord)
        self.burst_seconds = duration
        self.last_distance_yards = distance
        self.last_reason = "single_timed_forward_burst"
        return MiningFinalApproachDecision(
            MiningFinalApproachAction.WAIT_FEEDBACK,
            distance,
            burst_seconds=duration,
            reason=self.last_reason,
        )

    def _observe_burst_feedback(
        self,
        current_coord: int,
        *,
        now: float,
        coord_fresh: bool,
    ) -> MiningFinalApproachDecision:
        if self.target_coord is None or self.burst_sent_at is None:
            raise RuntimeError("Final mining burst has no target or timestamp")
        distance = self.distance_yards(current_coord, self.target_coord)
        self.last_distance_yards = distance
        elapsed = max(0.0, now - self.burst_sent_at)
        if coord_fresh and elapsed >= self.feedback_settle_seconds:
            if distance <= self.arrival_tolerance_yards:
                self.phase = "arrived"
                self.last_reason = "database_anchor_reached_after_single_burst"
                return MiningFinalApproachDecision(
                    MiningFinalApproachAction.ARRIVED,
                    distance,
                    burst_seconds=self.burst_seconds,
                    reason=self.last_reason,
                )
            if distance <= self.post_burst_world_scan_tolerance_yards:
                self.phase = "arrived"
                self.last_reason = "database_anchor_close_after_single_burst"
                return MiningFinalApproachDecision(
                    MiningFinalApproachAction.ARRIVED,
                    distance,
                    burst_seconds=self.burst_seconds,
                    reason=self.last_reason,
                )
            self.phase = "failed"
            self.last_reason = "ore_final_database_burst_missed"
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.FAILED,
                distance,
                burst_seconds=self.burst_seconds,
                reason=self.last_reason,
            )
        if elapsed >= self.feedback_timeout_seconds:
            self.phase = "failed"
            self.last_reason = "ore_final_database_burst_feedback_timeout"
            return MiningFinalApproachDecision(
                MiningFinalApproachAction.FAILED,
                distance,
                burst_seconds=self.burst_seconds,
                reason=self.last_reason,
            )
        return MiningFinalApproachDecision(
            MiningFinalApproachAction.WAIT_FEEDBACK,
            distance,
            burst_seconds=self.burst_seconds,
            reason="awaiting_post_burst_coordinate",
        )

    def snapshot(self, *, now: float) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "target_coord": self.target_coord,
            "distance_yards": self.last_distance_yards,
            "trigger_distance_yards": self.trigger_distance_yards,
            "arrival_tolerance_yards": self.arrival_tolerance_yards,
            "post_burst_world_scan_tolerance_yards": (
                self.post_burst_world_scan_tolerance_yards
            ),
            "burst_stop_short_yards": self.burst_stop_short_yards,
            "mounted_max_burst_seconds": self.mounted_max_burst_seconds,
            "foot_max_burst_seconds": self.foot_max_burst_seconds,
            "precision_coord_interval_seconds": self.precision_coord_interval_seconds,
            "burst_origin_coord": self.burst_origin_coord,
            "burst_seconds": self.burst_seconds,
            "burst_age_seconds": (
                max(0.0, now - self.burst_sent_at)
                if self.burst_sent_at is not None
                else None
            ),
            "reason": self.last_reason,
        }
