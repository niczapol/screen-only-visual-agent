from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vision_bot.coords import coord_to_xy


@dataclass(frozen=True)
class MountedEscapeDecision:
    active: bool
    reason: str
    combat_episode: int
    escalated: bool
    active_seconds: float
    progress_age_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "combat_episode": self.combat_episode,
            "escalated": self.escalated,
            "active_seconds": round(self.active_seconds, 3),
            "progress_age_seconds": (
                round(self.progress_age_seconds, 3)
                if self.progress_age_seconds is not None
                else None
            ),
        }


@dataclass
class MountedEscapeState:
    enabled: bool = False
    min_health: float = 0.55
    max_active_seconds: float = 12.0
    progress_timeout_seconds: float = 1.0
    min_coord_delta: float = 0.015
    active: bool = False
    escalated: bool = False
    combat_episode_active: bool = False
    combat_episodes: int = 0
    escape_episodes: int = 0
    escalations: int = 0
    entered_at: float | None = None
    last_progress_at: float | None = None
    last_progress_coord: int | None = None
    escalation_reason: str | None = None
    last_escalation_reason: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MountedEscapeState:
        combat_cfg = config.get("safety", {}).get("combat", {})
        return cls(
            enabled=bool(
                combat_cfg.get("mounted_escape_enabled", False)
            ),
            min_health=max(
                0.0,
                min(
                    1.0,
                    float(
                        combat_cfg.get("mounted_escape_min_health", 0.55)
                    ),
                ),
            ),
            max_active_seconds=max(
                0.1,
                float(combat_cfg.get("mounted_escape_max_active_seconds", 12.0)),
            ),
            progress_timeout_seconds=max(
                0.1,
                float(combat_cfg.get("mounted_escape_progress_timeout_seconds", 1.0)),
            ),
            min_coord_delta=max(
                0.0,
                float(combat_cfg.get("mounted_escape_min_coord_delta", 0.015)),
            ),
        )

    def observe(
        self,
        *,
        now: float,
        combat_engaged: bool,
        mounted: bool,
        player_health_fraction: float | None,
        current_coord: int | None,
        mining_conflict: bool,
        heal_casting: bool,
        threat_level: str = "normal",
        route_hazard: bool = False,
    ) -> MountedEscapeDecision:
        if not combat_engaged:
            self._clear_combat_episode()
            return self._decision(now, "combat_clear")

        if not self.combat_episode_active:
            self.combat_episode_active = True
            self.combat_episodes += 1

        if not self.enabled:
            return self._decision(now, "disabled")
        if self.escalated:
            return self._decision(now, self.escalation_reason or "combat_escalated")
        if route_hazard:
            return self._escalate(now, "route_hazard")
        if _threat_rank(threat_level) >= _threat_rank("elevated"):
            return self._escalate(now, f"threat_{str(threat_level).lower()}")
        if heal_casting:
            return self._escalate(now, "heal_casting")
        if mining_conflict:
            return self._escalate(now, "mining_conflict")
        if not mounted:
            return self._escalate(now, "dismounted")
        if player_health_fraction is None:
            self.active = False
            return self._decision(now, "health_unknown")
        if player_health_fraction < self.min_health:
            return self._escalate(now, "low_health")

        if not self.active:
            self.active = True
            self.escape_episodes += 1
            self.entered_at = now
            self.last_progress_at = now
            self.last_progress_coord = current_coord
        else:
            self._observe_progress(current_coord, now)

        if (
            self.entered_at is not None
            and now - self.entered_at >= self.max_active_seconds
        ):
            return self._escalate(now, "escape_duration_limit")

        progress_age = self._progress_age(now)
        if progress_age is not None and progress_age >= self.progress_timeout_seconds:
            return self._escalate(now, "no_coordinate_progress")
        return self._decision(now, "mounted_escape")

    def _observe_progress(self, current_coord: int | None, now: float) -> None:
        if current_coord is None:
            return
        if self.last_progress_coord is None:
            self.last_progress_coord = current_coord
            self.last_progress_at = now
            return
        if _coord_distance(self.last_progress_coord, current_coord) < self.min_coord_delta:
            return
        self.last_progress_coord = current_coord
        self.last_progress_at = now

    def _escalate(self, now: float, reason: str) -> MountedEscapeDecision:
        if not self.escalated:
            self.escalations += 1
        self.active = False
        self.escalated = True
        self.escalation_reason = reason
        self.last_escalation_reason = reason
        return self._decision(now, reason)

    def _clear_combat_episode(self) -> None:
        self.active = False
        self.escalated = False
        self.combat_episode_active = False
        self.entered_at = None
        self.last_progress_at = None
        self.last_progress_coord = None
        self.escalation_reason = None

    def _progress_age(self, now: float) -> float | None:
        if self.last_progress_at is None:
            return None
        return max(0.0, now - self.last_progress_at)

    def _decision(self, now: float, reason: str) -> MountedEscapeDecision:
        active_seconds = (
            max(0.0, now - self.entered_at)
            if self.active and self.entered_at is not None
            else 0.0
        )
        return MountedEscapeDecision(
            active=self.active,
            reason=reason,
            combat_episode=self.combat_episodes,
            escalated=self.escalated,
            active_seconds=active_seconds,
            progress_age_seconds=self._progress_age(now) if self.active else None,
        )


def _coord_distance(first: int, second: int) -> float:
    first_x, first_y = coord_to_xy(first)
    second_x, second_y = coord_to_xy(second)
    return math.hypot(second_x - first_x, second_y - first_y)


def _threat_rank(level: str | None) -> int:
    return {
        "none": 0,
        "normal": 1,
        "elevated": 2,
        "high": 3,
        "critical": 4,
    }.get(str(level or "none").lower(), 0)
