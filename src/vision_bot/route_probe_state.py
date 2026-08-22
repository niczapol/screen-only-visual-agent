from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_bot.threat_detection import ThreatDetection


@dataclass(frozen=True)
class RouteLiveSummary:
    output_dir: Path
    target_coord: int
    target_node_id: int | None
    target_zone_id: int | None
    target_ore_type: str | None
    start_coord: int | None
    final_coord: int | None
    start_distance: float | None
    final_distance: float | None
    steps: int
    reached: bool
    stuck_events: int
    completed_targets: int
    target_history: list[dict[str, object]]
    completed_route_nodes: int
    route_node_history: list[dict[str, object]]
    skipped_targets: int
    skipped_target_history: list[dict[str, object]]
    blocked_target_coords: list[int]
    detected_zone_name: str | None
    route_zone_ids: list[int]
    armor_critical_observed: bool
    mounted_observed: bool
    mount_casts: int
    mounted_escape_episodes: int
    mounted_escape_escalations: int
    mounted_escape_last_reason: str | None
    hard_stuck_escapes: int
    hard_stuck_escape_history: list[dict[str, object]]
    mining_attempts: int
    mining_successes: int
    mining_failures: int
    mining_outcomes: list[dict[str, object]]
    post_combat_loot_attempts: int
    post_combat_loot_outcomes: list[dict[str, object]]
    image_writes_submitted: int
    image_writes_written: int
    image_writes_dropped: int
    image_write_failures: int
    termination_reason: str


@dataclass
class ThreatAvoidanceCommitment:
    commit_duration: float
    committed_turn_key: str | None = None
    committed_until: float = 0.0
    last_committed_turn_key: str | None = None

    def choose_turn_key(
        self,
        threat: ThreatDetection | None,
        *,
        now: float,
        fallback_turn_key: str,
    ) -> str | None:
        if threat is None:
            self.committed_turn_key = None
            self.committed_until = 0.0
            return None

        if self.committed_turn_key in {"A", "D"} and now < self.committed_until:
            return self.committed_turn_key

        turn_key = self._initial_turn_key(threat, fallback_turn_key)
        self.committed_turn_key = turn_key
        self.last_committed_turn_key = turn_key
        self.committed_until = now + max(0.0, self.commit_duration)
        return turn_key

    def _initial_turn_key(self, threat: ThreatDetection, fallback_turn_key: str) -> str:
        if threat.reason == "hostile_target_frame":
            return _first_valid_turn_key(self.last_committed_turn_key, fallback_turn_key, "D")
        return _first_valid_turn_key(
            threat.turn_key,
            self.last_committed_turn_key,
            fallback_turn_key,
            "D",
        )


@dataclass
class LocalAvoidanceCommitment:
    retry_interval: float
    clear_commit_seconds: float = 0.65
    max_turn_pulses_per_side: int = 3
    committed_turn_key: str | None = None
    committed_turn_pulses: int = 0
    next_retry_at: float = 0.0
    last_blocked_at: float | None = None
    bypass_active: bool = False
    side_switch_pending: bool = False
    jump_only_pending: bool = False

    def next_turn_key(
        self,
        *,
        blocked: bool,
        now: float,
        preferred_turn_key: str | None,
        fallback_turn_key: str,
        distance_progress: float | None,
        progress_threshold: float,
    ) -> str | None:
        if not blocked:
            if self.should_continue_forward(now):
                return None
            self.clear()
            return None

        self.last_blocked_at = now
        self.bypass_active = True
        if self.committed_turn_key not in {"A", "D"}:
            self.committed_turn_key = _first_valid_turn_key(
                preferred_turn_key,
                fallback_turn_key,
                "D",
            )
        if distance_progress is not None and distance_progress > max(0.0, progress_threshold):
            self.committed_turn_pulses = 0
            self.jump_only_pending = True
        if now < self.next_retry_at:
            return None

        if (
            self.committed_turn_pulses >= max(1, self.max_turn_pulses_per_side)
            and (distance_progress is None or distance_progress <= max(0.0, progress_threshold))
        ):
            self.committed_turn_key = "A" if self.committed_turn_key == "D" else "D"
            self.committed_turn_pulses = 0
            self.side_switch_pending = True
            self.jump_only_pending = False
        elif self.committed_turn_pulses == 0:
            # The first response to a fresh blockage is a forward jump. Only a
            # later failed pulse is allowed to add a turn.
            self.jump_only_pending = True
        self.next_retry_at = now + max(0.0, self.retry_interval)
        self.committed_turn_pulses += 1
        return self.committed_turn_key

    def consume_jump_only(self) -> bool:
        pending = self.jump_only_pending
        self.jump_only_pending = False
        return pending

    def consume_side_switch(self) -> bool:
        pending = self.side_switch_pending
        self.side_switch_pending = False
        return pending

    def should_continue_forward(self, now: float) -> bool:
        return bool(
            self.bypass_active
            and self.last_blocked_at is not None
            and now - self.last_blocked_at < max(0.0, self.clear_commit_seconds)
        )

    def clear(self) -> None:
        self.committed_turn_key = None
        self.committed_turn_pulses = 0
        self.next_retry_at = 0.0
        self.last_blocked_at = None
        self.bypass_active = False
        self.side_switch_pending = False
        self.jump_only_pending = False


@dataclass
class RouteTargetProgressWatchdog:
    timeout_seconds: float
    min_improvement: float
    target_coord: int | None = None
    reference_distance: float | None = None
    last_improvement_at: float | None = None

    def observe(
        self,
        *,
        target_coord: int,
        distance: float | None,
        now: float,
        active: bool,
    ) -> bool:
        if not active or distance is None:
            self.reset()
            return False
        if self.target_coord != target_coord or self.reference_distance is None:
            self.target_coord = target_coord
            self.reference_distance = distance
            self.last_improvement_at = now
            return False
        if distance <= self.reference_distance - max(0.0, self.min_improvement):
            self.reference_distance = distance
            self.last_improvement_at = now
            return False
        if self.last_improvement_at is None:
            self.last_improvement_at = now
            return False
        if now - self.last_improvement_at < max(0.0, self.timeout_seconds):
            return False
        self.reference_distance = distance
        self.last_improvement_at = now
        return True

    def reset(self) -> None:
        self.target_coord = None
        self.reference_distance = None
        self.last_improvement_at = None


@dataclass
class SpiritSearchLegProgress:
    reached_distance: float
    pass_distance: float
    pass_margin: float
    max_seconds: float
    target_coord: int | None = None
    started_at: float | None = None
    best_distance: float | None = None

    def observe(
        self,
        *,
        target_coord: int,
        distance: float,
        now: float,
    ) -> tuple[bool, str]:
        if self.target_coord != target_coord or self.started_at is None:
            self.target_coord = target_coord
            self.started_at = now
            self.best_distance = distance
        else:
            self.best_distance = (
                distance
                if self.best_distance is None
                else min(self.best_distance, distance)
            )

        if distance <= max(0.0, self.reached_distance):
            return True, "reached"
        if (
            self.best_distance is not None
            and self.best_distance <= max(self.reached_distance, self.pass_distance)
            and distance >= self.best_distance + max(0.0, self.pass_margin)
        ):
            return True, "passed_closest_approach"
        if now - self.started_at >= max(0.05, self.max_seconds):
            return True, "leg_timeout"
        return False, "tracking"

    def reset(self) -> None:
        self.target_coord = None
        self.started_at = None
        self.best_distance = None


def _first_valid_turn_key(*values: str | None) -> str:
    return next((value for value in values if value in {"A", "D"}), "D")
