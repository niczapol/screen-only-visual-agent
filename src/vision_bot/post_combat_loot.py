from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PostCombatLootPhase(str, Enum):
    IDLE = "idle"
    ACQUIRE = "acquire"
    RETARGET_SETTLE = "retarget_settle"
    VERIFY = "verify"
    MARKER_CLEAR = "marker_clear"
    RETURN_TARGET_SETTLE = "return_target_settle"


@dataclass(frozen=True)
class PostCombatLootDecision:
    action: str
    active: bool
    input_key: str | None = None


@dataclass(frozen=True)
class PostCombatLootOutcome:
    attempted: bool
    reason: str
    elapsed_seconds: float
    interactions: int = 0
    confirmed_loots: int = 0


class PostCombatLootController:
    """Loot the current or previous dead target and require visible confirmation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        loot_cfg = (config or {}).get("post_combat_loot", {})
        self.enabled = bool(loot_cfg.get("enabled", False))
        self.interact_key = str(loot_cfg.get("interact_key", "G")).strip() or "G"
        self.target_last_enabled = bool(loot_cfg.get("target_last_enabled", True))
        self.target_last_key = (
            str(loot_cfg.get("target_last_key", "F9")).strip() or "F9"
        )
        self.acquisition_timeout_seconds = max(
            0.0,
            float(loot_cfg.get("acquisition_timeout_seconds", 1.5)),
        )
        self.retarget_settle_seconds = max(
            0.0,
            float(loot_cfg.get("retarget_settle_seconds", 0.35)),
        )
        self.confirmation_timeout_seconds = max(
            0.0,
            float(loot_cfg.get("confirmation_timeout_seconds", 1.5)),
        )
        self.marker_clear_timeout_seconds = max(
            0.0,
            float(loot_cfg.get("marker_clear_timeout_seconds", 1.0)),
        )
        self.max_interactions = max(
            1,
            int(loot_cfg.get("max_interactions", 2)),
        )
        self.in_combat_enabled = bool(loot_cfg.get("in_combat_enabled", True))
        self.return_target_settle_seconds = max(
            0.0,
            float(loot_cfg.get("return_target_settle_seconds", 0.35)),
        )
        self.phase = PostCombatLootPhase.IDLE
        self.started_at = 0.0
        self.deadline = 0.0
        self._outcome: PostCombatLootOutcome | None = None
        self.interactions = 0
        self.confirmed_loots = 0
        self.retarget_requests = 0
        self.armed_in_combat = False

    @property
    def active(self) -> bool:
        return self.phase != PostCombatLootPhase.IDLE

    def arm(self, *, now: float, in_combat: bool = False) -> bool:
        if (
            not self.enabled
            or self.active
            or (in_combat and not self.in_combat_enabled)
        ):
            return False
        self.phase = PostCombatLootPhase.ACQUIRE
        self.started_at = now
        self.deadline = now + self.acquisition_timeout_seconds
        self._outcome = None
        self.interactions = 0
        self.confirmed_loots = 0
        self.retarget_requests = 0
        self.armed_in_combat = bool(in_combat)
        return True

    def observe(
        self,
        *,
        now: float,
        dead_hostile_target_visible: bool,
        loot_opened_visible: bool = False,
        target_is_attacker_visible: bool = False,
        control_available: bool = True,
    ) -> PostCombatLootDecision:
        if not control_available:
            return PostCombatLootDecision(
                "post_combat_loot_wait_control",
                self.active,
            )
        if self.phase == PostCombatLootPhase.IDLE:
            return PostCombatLootDecision("post_combat_loot_idle", False)

        if self.phase == PostCombatLootPhase.ACQUIRE:
            if dead_hostile_target_visible:
                return self._request_interact(now=now)
            if self.target_last_enabled and self.retarget_requests == 0:
                return self._request_target_last(now=now)
            if now >= self.deadline:
                self._finish(now=now, reason="dead_target_not_visible")
                return PostCombatLootDecision("post_combat_loot_no_target", False)
            return PostCombatLootDecision("post_combat_loot_wait_target", True)

        if self.phase == PostCombatLootPhase.RETARGET_SETTLE:
            if dead_hostile_target_visible:
                return self._request_interact(now=now)
            if now >= self.deadline:
                # Target-last is the acquisition action. Some 3.3.5 clients keep
                # UnitCanAttack false for a selected corpse, so the external dead
                # target marker can lag or remain absent even though F9 succeeded.
                # Always follow a settled retarget with Interact; confirmation is
                # still fail-closed on the explicit LOOT_OPENED marker.
                return self._request_interact(now=now)
            return PostCombatLootDecision("post_combat_loot_wait_retarget", True)

        if self.phase == PostCombatLootPhase.VERIFY:
            if loot_opened_visible:
                self.confirmed_loots += 1
                if (
                    not self.armed_in_combat
                    and (
                        self.interactions >= self.max_interactions
                        or not self.target_last_enabled
                    )
                ):
                    self._finish(now=now, reason="loot_confirmed")
                    return PostCombatLootDecision("post_combat_loot_complete", False)
                self.phase = PostCombatLootPhase.MARKER_CLEAR
                self.deadline = now + self.marker_clear_timeout_seconds
                return PostCombatLootDecision("post_combat_loot_confirmed", True)
            if now >= self.deadline:
                if (
                    self.target_last_enabled
                    and self.interactions < self.max_interactions
                    and not self.armed_in_combat
                ):
                    return self._request_target_last(now=now)
                self._finish(now=now, reason="loot_not_confirmed")
                return PostCombatLootDecision("post_combat_loot_unconfirmed", False)
            return PostCombatLootDecision("post_combat_loot_verify", True)

        if self.phase == PostCombatLootPhase.MARKER_CLEAR:
            if not loot_opened_visible:
                if self.armed_in_combat:
                    return self._request_return_target(now=now)
                return self._request_target_last(now=now)
            if now >= self.deadline:
                self._finish(now=now, reason="loot_confirmed_marker_stuck")
                return PostCombatLootDecision("post_combat_loot_complete", False)
            return PostCombatLootDecision("post_combat_loot_wait_marker_clear", True)

        if target_is_attacker_visible:
            self._finish(now=now, reason="combat_loot_confirmed_target_restored")
            return PostCombatLootDecision("combat_loot_target_restored", False)
        if now >= self.deadline:
            self._finish(now=now, reason="combat_loot_confirmed_target_unverified")
            return PostCombatLootDecision("combat_loot_target_return_timeout", False)
        return PostCombatLootDecision("combat_loot_wait_target_return", True)

    def cancel(self, *, now: float, reason: str) -> None:
        if self.active:
            self._finish(now=now, reason=reason)

    def consume_outcome(self) -> PostCombatLootOutcome | None:
        outcome = self._outcome
        self._outcome = None
        return outcome

    def _request_target_last(self, *, now: float) -> PostCombatLootDecision:
        self.phase = PostCombatLootPhase.RETARGET_SETTLE
        self.deadline = now + self.retarget_settle_seconds
        self.retarget_requests += 1
        return PostCombatLootDecision(
            "post_combat_loot_target_last",
            True,
            self.target_last_key,
        )

    def _request_interact(self, *, now: float) -> PostCombatLootDecision:
        self.phase = PostCombatLootPhase.VERIFY
        self.deadline = now + self.confirmation_timeout_seconds
        self.interactions += 1
        return PostCombatLootDecision(
            "post_combat_loot_interact",
            True,
            self.interact_key,
        )

    def _request_return_target(self, *, now: float) -> PostCombatLootDecision:
        self.phase = PostCombatLootPhase.RETURN_TARGET_SETTLE
        self.deadline = now + self.return_target_settle_seconds
        return PostCombatLootDecision(
            "combat_loot_return_target",
            True,
            self.target_last_key,
        )

    def _finish(self, *, now: float, reason: str) -> None:
        self._outcome = PostCombatLootOutcome(
            attempted=self.interactions > 0,
            reason=reason,
            elapsed_seconds=max(0.0, now - self.started_at),
            interactions=self.interactions,
            confirmed_loots=self.confirmed_loots,
        )
        self.phase = PostCombatLootPhase.IDLE
        self.armed_in_combat = False
