from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from vision_bot.coords import coord_to_xy


class CorpseRunPhase(str, Enum):
    IDLE = "idle"
    FOLLOW_ROUTE = "follow_route"
    WAIT_FOR_RECLAIM = "wait_for_reclaim"


@dataclass(frozen=True)
class CorpseRunDecision:
    action: str
    active: bool
    target_coord: int | None = None
    request_reclaim: bool = False


@dataclass(frozen=True)
class CorpseRunOutcome:
    success: bool
    reason: str
    death_coord: int | None
    elapsed_seconds: float


class CorpseRunController:
    """Offline-gated state controller for a future ghost route to the corpse."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        corpse_cfg = (config or {}).get("corpse_run", {})
        self.enabled = bool(corpse_cfg.get("enabled", False))
        self.waypoint_reached_distance = max(
            0.01,
            float(corpse_cfg.get("waypoint_reached_distance_coord", 0.22)),
        )
        self.corpse_reached_distance = max(
            0.01,
            float(corpse_cfg.get("corpse_reached_distance_coord", 0.35)),
        )
        self.max_total_seconds = max(
            1.0,
            float(corpse_cfg.get("max_total_seconds", 240.0)),
        )
        self.phase = CorpseRunPhase.IDLE
        self.death_coord: int | None = None
        self.route: tuple[int, ...] = ()
        self.route_index = 0
        self.started_at = 0.0
        self._outcome: CorpseRunOutcome | None = None

    @property
    def active(self) -> bool:
        return self.phase != CorpseRunPhase.IDLE

    def record_death_position(self, coord: int | None) -> None:
        if coord is not None:
            coord_to_xy(coord)
        self.death_coord = coord

    def begin(self, route_coords: Iterable[int], *, now: float) -> bool:
        """Accept a path produced by the existing terrain/navmesh planner."""

        if not self.enabled or self.active or self.death_coord is None:
            return False
        route = tuple(int(coord) for coord in route_coords)
        for coord in route:
            coord_to_xy(coord)
        if not route or route[-1] != self.death_coord:
            route = (*route, self.death_coord)
        self.route = route
        self.route_index = 0
        self.started_at = now
        self._outcome = None
        self.phase = CorpseRunPhase.FOLLOW_ROUTE
        return True

    def observe(
        self,
        current_coord: int | None,
        *,
        now: float,
        reclaim_available: bool,
        alive: bool = False,
    ) -> CorpseRunDecision:
        if self.phase == CorpseRunPhase.IDLE:
            return CorpseRunDecision("corpse_run_idle", False)
        if alive:
            self._finish(now=now, success=True, reason="alive_confirmed")
            return CorpseRunDecision("corpse_run_complete", False)
        if now - self.started_at >= self.max_total_seconds:
            self._finish(now=now, success=False, reason="timeout")
            return CorpseRunDecision("corpse_run_timeout", False)

        if self.phase == CorpseRunPhase.FOLLOW_ROUTE:
            if current_coord is None:
                return CorpseRunDecision("corpse_run_wait_coordinate", True)
            while self.route_index < len(self.route):
                target = self.route[self.route_index]
                reached_distance = (
                    self.corpse_reached_distance
                    if self.route_index == len(self.route) - 1
                    else self.waypoint_reached_distance
                )
                if _distance(current_coord, target) > reached_distance:
                    return CorpseRunDecision(
                        "corpse_run_follow_route",
                        True,
                        target_coord=target,
                    )
                self.route_index += 1
            self.phase = CorpseRunPhase.WAIT_FOR_RECLAIM

        if reclaim_available:
            return CorpseRunDecision(
                "corpse_run_request_reclaim",
                True,
                target_coord=self.death_coord,
                request_reclaim=True,
            )
        return CorpseRunDecision(
            "corpse_run_wait_reclaim",
            True,
            target_coord=self.death_coord,
        )

    def fail(self, *, now: float, reason: str) -> CorpseRunOutcome | None:
        if not self.active:
            return None
        self._finish(now=now, success=False, reason=reason)
        return self._outcome

    def consume_outcome(self) -> CorpseRunOutcome | None:
        outcome = self._outcome
        self._outcome = None
        return outcome

    def _finish(self, *, now: float, success: bool, reason: str) -> None:
        self._outcome = CorpseRunOutcome(
            success=success,
            reason=reason,
            death_coord=self.death_coord,
            elapsed_seconds=max(0.0, now - self.started_at),
        )
        self.phase = CorpseRunPhase.IDLE
        self.route = ()
        self.route_index = 0


def _distance(first: int, second: int) -> float:
    first_x, first_y = coord_to_xy(first)
    second_x, second_y = coord_to_xy(second)
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5
