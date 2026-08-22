from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MountTravelState:
    enabled: bool = False
    key: str = "1"
    cast_seconds: float = 2.0
    retry_seconds: float = 4.0
    max_attempts_before_fallback: int = 2
    post_combat_settle_seconds: float = 3.0
    attempts: int = 0
    cast_until: float = 0.0
    retry_at: float = 0.0
    fallback_unmounted: bool = False
    settle_until: float = 0.0

    def observe(
        self,
        *,
        now: float,
        mounted: bool,
        combat_active: bool,
        post_combat_loot_active: bool = False,
        control_available: bool = True,
    ) -> str | None:
        if not self.enabled:
            return None
        if combat_active or post_combat_loot_active:
            self.cast_until = 0.0
            self.retry_at = 0.0
            self.attempts = 0
            self.fallback_unmounted = False
            self.settle_until = max(
                self.settle_until,
                now + max(0.0, self.post_combat_settle_seconds),
            )
            return None
        if mounted:
            self.cast_until = 0.0
            self.retry_at = 0.0
            self.attempts = 0
            self.fallback_unmounted = False
            self.settle_until = 0.0
            return None
        if now < self.settle_until:
            return "mount_combat_clear_wait"
        if not control_available:
            return None
        if self.fallback_unmounted:
            return None
        if now < self.cast_until:
            return "mount_cast_wait"
        if now < self.retry_at:
            return "mount_retry_wait"
        if self.attempts >= max(1, self.max_attempts_before_fallback):
            self.fallback_unmounted = True
            return None
        self.attempts += 1
        self.cast_until = now + max(0.0, self.cast_seconds)
        self.retry_at = now + max(self.cast_seconds, self.retry_seconds)
        return "mount_cast"

    def reset_after_route_event(self) -> None:
        self.attempts = 0
        self.cast_until = 0.0
        self.retry_at = 0.0
        self.fallback_unmounted = False

    def defer_mount(self, *, now: float, seconds: float) -> None:
        """Keep a deliberate on-foot escape from being cancelled by remounting."""

        self.attempts = 0
        self.cast_until = 0.0
        self.retry_at = 0.0
        self.fallback_unmounted = False
        self.settle_until = max(self.settle_until, now + max(0.0, seconds))
