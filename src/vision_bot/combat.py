from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from vision_bot.regions import resolve_region
from vision_bot.screen_objects import BoundingBox
from vision_bot.threat_detection import (
    _avoidance_cfg,
    _detect_central_red_nameplate,
    _detect_hostile_target_frame,
)


@dataclass(frozen=True)
class CombatState:
    active: bool = False
    combat_marker_visible: bool = False
    outgoing_hit_marker_visible: bool = False
    facing_error_marker_visible: bool = False
    out_of_range_marker_visible: bool = False
    target_is_attacker: bool = False
    target_present: bool = False
    hostile_target_hint: bool = False
    nameplate_visible: bool = False
    outgoing_damage_visible: bool = False
    facing_error_visible: bool = False
    face_turn_key: str | None = None
    source: str = "none"
    bbox: BoundingBox | None = None
    outgoing_damage_bbox: BoundingBox | None = None
    player_health_fraction: float | None = None
    attacker_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "combat_marker_visible": self.combat_marker_visible,
            "outgoing_hit_marker_visible": self.outgoing_hit_marker_visible,
            "facing_error_marker_visible": self.facing_error_marker_visible,
            "out_of_range_marker_visible": self.out_of_range_marker_visible,
            "target_is_attacker": self.target_is_attacker,
            "target_present": self.target_present,
            "hostile_target_hint": self.hostile_target_hint,
            "nameplate_visible": self.nameplate_visible,
            "outgoing_damage_visible": self.outgoing_damage_visible,
            "facing_error_visible": self.facing_error_visible,
            "face_turn_key": self.face_turn_key,
            "source": self.source,
            "bbox": _bbox_to_dict(self.bbox),
            "outgoing_damage_bbox": _bbox_to_dict(self.outgoing_damage_bbox),
            "player_health_fraction": self.player_health_fraction,
            "attacker_count": self.attacker_count,
        }

    def has_latching_evidence(self) -> bool:
        return (
            self.active
            or self.combat_marker_visible
            or self.outgoing_hit_marker_visible
            or self.facing_error_marker_visible
            or self.out_of_range_marker_visible
            or self.target_present
            or self.hostile_target_hint
            or self.nameplate_visible
            or self.outgoing_damage_visible
            or self.facing_error_visible
            or self.attacker_count > 0
        )


@dataclass
class CombatFallbackState:
    attack_keys: tuple[str, ...]
    attack_interval: float
    clear_frames: int
    clear_seconds: float = 3.0
    marker_clear_seconds: float = 0.75
    damage_timeout: float = 2.4
    face_search_cooldown: float = 1.0
    target_lock_no_hit_seconds: float = 8.0
    stale_soft_target_seconds: float = 6.0
    health_drop_latch_threshold: float = 0.02
    engaged: bool = False
    missing_frames: int = 0
    next_attack_index: int = 0
    last_attack_at: float | None = None
    last_attack_key: str | None = None
    last_e_attack_at: float | None = None
    unconfirmed_e_started_at: float | None = None
    last_outgoing_damage_at: float | None = None
    last_e_damage_confirmed_at: float | None = None
    target_facing_locked: bool = False
    target_facing_locked_at: float | None = None
    last_locked_hit_at: float | None = None
    last_latching_evidence_at: float | None = None
    last_combat_marker_at: float | None = None
    authoritative_marker_seen: bool = False
    attack_started_at: float | None = None
    last_attack_by_key: dict[str, float] = field(default_factory=dict)
    last_player_health_fraction: float | None = None
    last_health_drop_at: float | None = None
    last_hard_combat_evidence_at: float | None = None
    last_observation_latching_evidence: bool = False
    last_face_search_at: float | None = None
    face_search_index: int = 0
    face_search_turn_key: str = "D"
    face_search_held_key: str | None = None
    face_search_keyboard_held: bool = False
    last_range_approach_at: float | None = None
    last_combat_key_at: float | None = None
    last_tapped_keys: list[str] = field(default_factory=list)
    cleared_after_engaged: bool = False
    cleared_after_stale_soft_target: bool = False
    cleared_after_confirmed_outgoing_damage: bool = False
    last_target_cycle_at: float | None = None
    last_target_interact_at: float | None = None

    def observe(self, combat: CombatState, now: float | None = None) -> bool:
        self.cleared_after_engaged = False
        self.cleared_after_stale_soft_target = False
        self.cleared_after_confirmed_outgoing_damage = False
        health_drop_evidence = self._observe_health_drop(combat, now)
        has_latching_evidence = combat.has_latching_evidence() or health_drop_evidence
        hard_combat_evidence = bool(
            combat.active
            or combat.combat_marker_visible
            or combat.outgoing_hit_marker_visible
            or combat.facing_error_marker_visible
            or combat.out_of_range_marker_visible
            or combat.target_is_attacker
            or combat.outgoing_damage_visible
            or combat.attacker_count > 0
            or health_drop_evidence
        )
        if hard_combat_evidence and now is not None:
            self.last_hard_combat_evidence_at = now
        self.last_observation_latching_evidence = has_latching_evidence
        if combat.combat_marker_visible:
            self.authoritative_marker_seen = True
            if now is not None:
                self.last_combat_marker_at = now
        if combat.active or health_drop_evidence:
            self.engaged = True
            self.missing_frames = 0
            if now is not None:
                self.last_latching_evidence_at = now
            return True

        if not self.engaged:
            self.missing_frames = 0
            return False

        if self.authoritative_marker_seen and now is not None:
            marker_age = (
                0.0
                if self.last_combat_marker_at is None
                else now - self.last_combat_marker_at
            )
            if marker_age < max(0.0, self.marker_clear_seconds):
                return True
            self._clear_engagement()
            return False

        if has_latching_evidence:
            soft_target_evidence = bool(
                combat.target_present
                or combat.hostile_target_hint
                or combat.nameplate_visible
            )
            if (
                soft_target_evidence
                and not hard_combat_evidence
                and now is not None
                and self.last_hard_combat_evidence_at is not None
                and now - self.last_hard_combat_evidence_at
                >= max(0.0, self.stale_soft_target_seconds)
            ):
                self._clear_engagement(stale_soft_target=True)
                return False
            self.missing_frames = 0
            if now is not None:
                self.last_latching_evidence_at = now
            return True

        if now is not None:
            if self.last_latching_evidence_at is None:
                self.last_latching_evidence_at = now
                return True
            if now - self.last_latching_evidence_at < max(0.0, self.clear_seconds):
                return True
            self._clear_engagement()
            return False

        self.missing_frames += 1
        if self.missing_frames >= max(1, self.clear_frames):
            self._clear_engagement()
            return False
        return True

    def _clear_engagement(self, *, stale_soft_target: bool = False) -> None:
        self.cleared_after_confirmed_outgoing_damage = bool(
            self.last_outgoing_damage_at is not None
            or self.last_e_damage_confirmed_at is not None
        )
        self.cleared_after_stale_soft_target = bool(stale_soft_target)
        self.engaged = False
        self.missing_frames = 0
        self.cleared_after_engaged = True
        self.next_attack_index = 0
        self.last_attack_at = None
        self.last_attack_key = None
        self.last_e_attack_at = None
        self.unconfirmed_e_started_at = None
        self.last_outgoing_damage_at = None
        self.last_e_damage_confirmed_at = None
        self.target_facing_locked = False
        self.target_facing_locked_at = None
        self.last_locked_hit_at = None
        self.attack_started_at = None
        self.last_combat_marker_at = None
        self.last_hard_combat_evidence_at = None
        self.authoritative_marker_seen = False
        self.last_attack_by_key.clear()
        self.last_face_search_at = None
        self.face_search_index = 0
        self.last_range_approach_at = None
        self.last_combat_key_at = None
        self.last_tapped_keys.clear()
        self.last_target_cycle_at = None
        self.last_target_interact_at = None

    def can_cycle_target(self, now: float, interval: float) -> bool:
        if self.last_target_cycle_at is not None and now - self.last_target_cycle_at < max(0.0, interval):
            return False
        self.last_target_cycle_at = now
        return True

    def can_interact_target(self, now: float, interval: float) -> bool:
        if (
            self.last_target_interact_at is not None
            and now - self.last_target_interact_at < max(0.0, interval)
        ):
            return False
        self.last_target_interact_at = now
        return True

    def _observe_health_drop(self, combat: CombatState, now: float | None) -> bool:
        current_health = combat.player_health_fraction
        previous_health = self.last_player_health_fraction
        if current_health is not None:
            self.last_player_health_fraction = current_health
        if previous_health is None or current_health is None:
            return False
        if previous_health - current_health < max(0.0, self.health_drop_latch_threshold):
            return False
        if now is not None:
            self.last_health_drop_at = now
        return True

    def consume_clear_event(self) -> bool:
        if not self.cleared_after_engaged:
            return False
        self.cleared_after_engaged = False
        return True

    def consume_stale_soft_target_clear_event(self) -> bool:
        if not self.cleared_after_stale_soft_target:
            return False
        self.cleared_after_stale_soft_target = False
        return True

    def consume_confirmed_outgoing_clear_event(self) -> bool:
        if not self.cleared_after_confirmed_outgoing_damage:
            return False
        self.cleared_after_confirmed_outgoing_damage = False
        return True

    def next_attack_key(self, now: float) -> str | None:
        return self.consume_next_attack_key(now)

    def next_attack_keys(self, now: float) -> list[str]:
        key = self.consume_next_attack_key(now)
        return [key] if key is not None else []

    def peek_next_attack_key(self, now: float) -> str | None:
        if not self.attack_keys:
            return None

        interval = max(0.0, self.attack_interval)
        if self.last_attack_at is not None and now - self.last_attack_at < interval:
            return None

        return self.attack_keys[self.next_attack_index % len(self.attack_keys)]

    def consume_next_attack_key(self, now: float) -> str | None:
        key = self.peek_next_attack_key(now)
        if key is None:
            return None
        if self.attack_started_at is None:
            self.attack_started_at = now
        self.next_attack_index += 1
        self.last_attack_at = now
        self.last_attack_key = key
        self.last_attack_by_key[key] = now
        if key == "E":
            self.last_e_attack_at = now
            if self.unconfirmed_e_started_at is None:
                self.unconfirmed_e_started_at = now
        return key

    def can_tap_combat_key(self, now: float, min_interval: float) -> bool:
        if self.last_combat_key_at is None:
            return True
        return now - self.last_combat_key_at >= max(0.0, min_interval)

    def clear_tapped_keys(self) -> None:
        self.last_tapped_keys.clear()

    def record_tapped_keys(self, keys: list[str], *, now: float | None = None) -> None:
        self.last_tapped_keys = list(keys)
        if keys and now is not None:
            self.last_combat_key_at = now

    def observe_outgoing_damage(
        self,
        now: float,
        visible: bool,
        *,
        facing_error_visible: bool = False,
    ) -> None:
        if facing_error_visible:
            self.release_target_facing_lock()
        if visible:
            self.last_outgoing_damage_at = now
            if self.target_facing_locked:
                self.last_locked_hit_at = now
            if self.engaged:
                self.last_latching_evidence_at = now
            if (
                not facing_error_visible
                and self.last_e_attack_at is not None
                and self.unconfirmed_e_started_at is not None
                and now >= self.unconfirmed_e_started_at
            ):
                self.last_e_damage_confirmed_at = now
                self.unconfirmed_e_started_at = None
                self.face_search_index = 0
                self.target_facing_locked = True
                self.target_facing_locked_at = now
                self.last_locked_hit_at = now

    def has_target_facing_lock(self, now: float) -> bool:
        if not self.target_facing_locked:
            return False
        reference = self.last_locked_hit_at or self.target_facing_locked_at
        timeout = max(0.0, self.target_lock_no_hit_seconds)
        if timeout > 0.0 and reference is not None and now - reference > timeout:
            self.release_target_facing_lock()
            return False
        return True

    def release_target_facing_lock(self) -> None:
        self.target_facing_locked = False
        self.target_facing_locked_at = None
        self.last_locked_hit_at = None

    def has_recent_outgoing_damage(self, now: float) -> bool:
        return self.last_outgoing_damage_at is not None and now - self.last_outgoing_damage_at <= max(
            0.0,
            self.damage_timeout,
        )

    def next_face_search_turn(
        self,
        now: float,
        *,
        turn_180_duration: float,
        turn_90_duration: float,
        force: bool = False,
    ) -> tuple[str, float, str] | None:
        if not force and self.has_target_facing_lock(now):
            return None
        if not force:
            e_wait_started_at = self.unconfirmed_e_started_at or self.last_e_attack_at
            if e_wait_started_at is None:
                return None
            if (
                self.unconfirmed_e_started_at is None
                and self.last_e_damage_confirmed_at is not None
                and self.last_e_damage_confirmed_at >= e_wait_started_at
            ):
                return None
            if now - e_wait_started_at < max(0.0, self.damage_timeout):
                return None
        if self.last_face_search_at is not None and now - self.last_face_search_at < max(0.0, self.face_search_cooldown):
            return None

        turn_key = self.face_search_turn_key if self.face_search_turn_key in {"A", "D"} else "D"
        self.face_search_index += 1
        self.last_face_search_at = now
        return turn_key, max(0.0, turn_90_duration), "combat_face_search_sweep"

    def can_approach_range(self, now: float, cooldown: float) -> bool:
        if self.last_range_approach_at is not None and now - self.last_range_approach_at < max(0.0, cooldown):
            return False
        self.last_range_approach_at = now
        return True

    def start_continuous_face_search(
        self,
        turn_key: str,
        *,
        keyboard_held: bool = True,
    ) -> str:
        normalized = turn_key if turn_key in {"A", "D"} else "D"
        self.face_search_held_key = normalized
        self.face_search_keyboard_held = bool(keyboard_held)
        return normalized

    def stop_continuous_face_search(self) -> str | None:
        held_key = self.face_search_held_key
        self.face_search_held_key = None
        self.face_search_keyboard_held = False
        return held_key


@dataclass
class PeriodicCombatKeyState:
    keys: tuple[str, ...] = ()
    interval: float = 15.0
    last_tap_by_key: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_tap_by_key.clear()

    def next_keys(self, now: float) -> list[str]:
        if not self.keys:
            return []
        interval = max(0.0, self.interval)
        due_keys: list[str] = []
        for key in self.keys:
            last_tap = self.last_tap_by_key.get(key)
            if last_tap is None or now - last_tap >= interval:
                due_keys.append(key)
                self.last_tap_by_key[key] = now
        return due_keys

    def peek_next_key(self, now: float) -> str | None:
        if not self.keys:
            return None
        interval = max(0.0, self.interval)
        for key in self.keys:
            last_tap = self.last_tap_by_key.get(key)
            if last_tap is None or now - last_tap >= interval:
                return key
        return None

    def consume_next_key(self, now: float) -> str | None:
        key = self.peek_next_key(now)
        if key is None:
            return None
        self.last_tap_by_key[key] = now
        return key


@dataclass
class CombatHealState:
    enabled: bool = True
    threshold: float = 0.50
    cooldown_seconds: float = 8.0
    cast_seconds: float = 2.0
    last_heal_at: float | None = None
    casting_until: float | None = None

    def is_casting(self, now: float) -> bool:
        return self.casting_until is not None and now < self.casting_until

    def should_start(self, combat: CombatState, now: float) -> bool:
        if not self.enabled or self.is_casting(now):
            return False
        health = combat.player_health_fraction
        if health is None or health > max(0.0, self.threshold):
            return False
        if self.last_heal_at is not None and now - self.last_heal_at < max(0.0, self.cooldown_seconds):
            return False
        return True

    def start(self, now: float) -> None:
        self.last_heal_at = now
        self.casting_until = now + max(0.0, self.cast_seconds)

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "cast_seconds": self.cast_seconds,
            "last_heal_at": self.last_heal_at,
            "casting_until": self.casting_until,
            "casting": self.is_casting(now) if now is not None else False,
        }


@dataclass(frozen=True)
class CombatThreatAssessment:
    level: str
    attacker_count: int
    player_health_fraction: float | None
    hp_loss_per_second: float
    time_to_death_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "attacker_count": self.attacker_count,
            "player_health_fraction": self.player_health_fraction,
            "hp_loss_per_second": round(self.hp_loss_per_second, 6),
            "time_to_death_seconds": (
                round(self.time_to_death_seconds, 3)
                if self.time_to_death_seconds is not None
                else None
            ),
        }


@dataclass
class CombatThreatEstimator:
    window_seconds: float = 5.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)

    def observe(
        self,
        combat: CombatState,
        *,
        now: float,
        engaged: bool,
    ) -> CombatThreatAssessment:
        health = combat.player_health_fraction
        if not engaged:
            self.samples.clear()
        elif health is not None:
            self.samples.append((now, float(health)))
            cutoff = now - max(0.5, self.window_seconds)
            while len(self.samples) > 1 and self.samples[0][0] < cutoff:
                self.samples.popleft()

        loss = 0.0
        elapsed = 0.0
        if len(self.samples) >= 2:
            elapsed = max(0.0, self.samples[-1][0] - self.samples[0][0])
            for previous, current in zip(self.samples, list(self.samples)[1:]):
                loss += max(0.0, previous[1] - current[1])
        loss_per_second = loss / elapsed if elapsed > 0 else 0.0
        time_to_death = (
            max(0.0, float(health)) / loss_per_second
            if health is not None and loss_per_second > 0.002
            else None
        )
        attackers = max(0, int(combat.attacker_count))
        if not engaged:
            level = "none"
        elif (health is not None and health <= 0.20) or (
            time_to_death is not None and time_to_death <= 4.0
        ):
            level = "critical"
        elif attackers >= 3 or (health is not None and health <= 0.35) or (
            time_to_death is not None and time_to_death <= 8.0
        ):
            level = "high"
        elif attackers >= 2 or (health is not None and health <= 0.55) or (
            time_to_death is not None and time_to_death <= 15.0
        ):
            level = "elevated"
        else:
            level = "normal"
        return CombatThreatAssessment(
            level=level,
            attacker_count=attackers,
            player_health_fraction=health,
            hp_loss_per_second=loss_per_second,
            time_to_death_seconds=time_to_death,
        )


def detect_combat_state(frame: np.ndarray, config: dict[str, Any] | None = None) -> CombatState:
    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    player_health_fraction = detect_player_health_fraction(frame, cfg)
    legacy_damage_enabled = bool(
        combat_cfg.get("legacy_outgoing_damage_confirmation_enabled", True)
    )
    legacy_facing_enabled = bool(
        combat_cfg.get("legacy_facing_error_confirmation_enabled", True)
    )
    raw_outgoing_damage_bbox = (
        detect_outgoing_damage_numbers(frame, cfg)
        if legacy_damage_enabled
        else None
    )
    legacy_facing_error_visible = (
        detect_combat_facing_error_text(frame, cfg)
        if legacy_facing_enabled
        else False
    )
    combat_marker_visible = detect_combat_marker(frame, cfg)
    outgoing_hit_marker_visible = detect_outgoing_hit_marker(frame, cfg)
    facing_error_marker_visible = detect_facing_error_marker(frame, cfg)
    out_of_range_marker_visible = detect_out_of_range_marker(frame, cfg)
    target_is_attacker = detect_attacker_target_marker(frame, cfg)
    attacker_count = detect_attacker_count_marker(frame, cfg)
    facing_error_visible = facing_error_marker_visible or (
        legacy_facing_error_visible
        and legacy_facing_enabled
    )
    if not bool(combat_cfg.get("enabled", True)):
        return CombatState(
            player_health_fraction=player_health_fraction,
            attacker_count=attacker_count,
        )

    avoid_cfg = _avoidance_cfg(cfg)
    target_hint = _detect_hostile_target_frame(frame, cfg, avoid_cfg, "D")
    confirmed_target_bbox = _detect_confirmed_target_frame(frame, cfg, combat_cfg)
    needs_nameplate = bool(
        combat_cfg.get("engage_on_nameplate", False)
        or combat_cfg.get("nameplate_facing_enabled", False)
        or legacy_damage_enabled
    )
    nameplate = (
        _detect_central_red_nameplate(frame, cfg, avoid_cfg)
        if needs_nameplate
        else None
    )
    if (
        not combat_marker_visible
        and confirmed_target_bbox is None
        and nameplate is None
        and target_hint is None
    ):
        return CombatState(
            outgoing_hit_marker_visible=outgoing_hit_marker_visible,
            facing_error_marker_visible=facing_error_marker_visible,
            out_of_range_marker_visible=out_of_range_marker_visible,
            target_is_attacker=target_is_attacker,
            outgoing_damage_visible=outgoing_hit_marker_visible,
            facing_error_visible=facing_error_visible,
            player_health_fraction=player_health_fraction,
            attacker_count=attacker_count,
        )

    face_turn_key = _face_turn_key(nameplate.bbox, frame.shape, combat_cfg) if nameplate is not None else None
    bbox = nameplate.bbox if nameplate is not None else confirmed_target_bbox
    if bbox is None and target_hint is not None:
        bbox = target_hint.bbox
    legacy_outgoing_damage_bbox = (
        raw_outgoing_damage_bbox
        if raw_outgoing_damage_bbox is not None
        and nameplate is not None
        and _damage_near_combat_target(raw_outgoing_damage_bbox, nameplate.bbox, frame.shape, combat_cfg)
        else None
    )
    outgoing_damage_visible = outgoing_hit_marker_visible or (
        legacy_outgoing_damage_bbox is not None
        and legacy_damage_enabled
    )
    target_present = confirmed_target_bbox is not None
    active = (
        combat_marker_visible
        or (target_present and bool(combat_cfg.get("target_frame_starts_combat", False)))
        or (nameplate is not None and bool(combat_cfg.get("engage_on_nameplate", False)))
    )
    sources = []
    if combat_marker_visible:
        sources.append("addon_combat_marker")
    if outgoing_hit_marker_visible:
        sources.append("addon_outgoing_hit_marker")
    if facing_error_marker_visible:
        sources.append("addon_facing_error_marker")
    if out_of_range_marker_visible:
        sources.append("addon_out_of_range_marker")
    if target_is_attacker:
        sources.append("addon_attacker_target_marker")
    if attacker_count > 0:
        sources.append(f"addon_attacker_count_{attacker_count}")
    if confirmed_target_bbox is not None:
        sources.append("top_left_target_frame_bars")
    elif target_hint is not None:
        sources.append("top_left_hostile_target_ui_hint")
    if nameplate is not None:
        sources.append(nameplate.source)

    return CombatState(
        active=active,
        combat_marker_visible=combat_marker_visible,
        outgoing_hit_marker_visible=outgoing_hit_marker_visible,
        facing_error_marker_visible=facing_error_marker_visible,
        out_of_range_marker_visible=out_of_range_marker_visible,
        target_is_attacker=target_is_attacker,
        target_present=target_present,
        hostile_target_hint=target_hint is not None,
        nameplate_visible=nameplate is not None,
        outgoing_damage_visible=outgoing_damage_visible,
        facing_error_visible=facing_error_visible,
        face_turn_key=face_turn_key,
        source="+".join(sources),
        bbox=bbox,
        outgoing_damage_bbox=legacy_outgoing_damage_bbox,
        player_health_fraction=player_health_fraction,
        attacker_count=attacker_count,
    )


def detect_combat_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    if frame.size == 0:
        return False

    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    if not bool(combat_cfg.get("combat_marker_enabled", False)):
        return False

    return _detect_fixed_hsv_marker(
        frame,
        cfg,
        region_key="combat_marker_region",
        default_region={"x": 1238, "y": 0, "width": 84, "height": 48},
        lower_key="combat_marker_hsv_lower",
        default_lower=[145, 220, 220],
        upper_key="combat_marker_hsv_upper",
        default_upper=[155, 255, 255],
        min_pixels_key="combat_marker_min_pixels",
        default_min_pixels=800,
    )


def detect_outgoing_hit_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    return _detect_combat_outcome_marker(
        frame,
        config,
        enabled_key="outcome_marker_enabled",
        region_key="outgoing_hit_marker_region",
        default_region={"x": 1108, "y": 0, "width": 112, "height": 56},
        lower_key="outgoing_hit_marker_hsv_lower",
        default_lower=[55, 220, 220],
        upper_key="outgoing_hit_marker_hsv_upper",
        default_upper=[65, 255, 255],
        min_pixels_key="outcome_marker_min_pixels",
    )


def detect_attacker_target_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    return _detect_combat_outcome_marker(
        frame,
        config,
        enabled_key="attacker_target_marker_enabled",
        region_key="attacker_target_marker_region",
        default_region={"x": 996, "y": 0, "width": 112, "height": 56},
        lower_key="attacker_target_marker_hsv_lower",
        default_lower=[115, 220, 220],
        upper_key="attacker_target_marker_hsv_upper",
        default_upper=[125, 255, 255],
        min_pixels_key="attacker_target_marker_min_pixels",
    )


def detect_attacker_count_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> int:
    """Read the addon's capped visible unique-attacker marker (0, 1, 2, 3+)."""
    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    if not bool(combat_cfg.get("attacker_count_marker_enabled", False)):
        return 0
    common = {
        "region_key": "attacker_count_marker_region",
        "default_region": {"x": 760, "y": 0, "width": 130, "height": 56},
        "min_pixels_key": "attacker_count_marker_min_pixels",
        "default_min_pixels": 800,
    }
    if _detect_fixed_hsv_marker(
        frame,
        cfg,
        lower_key="attacker_count_three_hsv_lower",
        default_lower=[0, 0, 225],
        upper_key="attacker_count_three_hsv_upper",
        default_upper=[179, 45, 255],
        **common,
    ):
        return 3
    if _detect_fixed_hsv_marker(
        frame,
        cfg,
        lower_key="attacker_count_two_hsv_lower",
        default_lower=[10, 220, 220],
        upper_key="attacker_count_two_hsv_upper",
        default_upper=[22, 255, 255],
        **common,
    ):
        return 2
    if _detect_fixed_hsv_marker(
        frame,
        cfg,
        lower_key="attacker_count_one_hsv_lower",
        default_lower=[0, 220, 220],
        upper_key="attacker_count_one_hsv_upper",
        default_upper=[6, 255, 255],
        **common,
    ):
        return 1
    return 0


def detect_facing_error_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    return _detect_combat_outcome_marker(
        frame,
        config,
        enabled_key="outcome_marker_enabled",
        region_key="facing_error_marker_region",
        default_region={"x": 1338, "y": 0, "width": 112, "height": 56},
        lower_key="facing_error_marker_hsv_lower",
        default_lower=[85, 220, 220],
        upper_key="facing_error_marker_hsv_upper",
        default_upper=[95, 255, 255],
        min_pixels_key="outcome_marker_min_pixels",
    )


def detect_out_of_range_marker(frame: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    return _detect_combat_outcome_marker(
        frame,
        config,
        enabled_key="outcome_marker_enabled",
        region_key="out_of_range_marker_region",
        default_region={"x": 1452, "y": 0, "width": 112, "height": 56},
        lower_key="out_of_range_marker_hsv_lower",
        default_lower=[8, 220, 220],
        upper_key="out_of_range_marker_hsv_upper",
        default_upper=[20, 255, 255],
        min_pixels_key="outcome_marker_min_pixels",
    )


def _detect_combat_outcome_marker(
    frame: np.ndarray,
    config: dict[str, Any] | None,
    *,
    enabled_key: str,
    region_key: str,
    default_region: dict[str, int],
    lower_key: str,
    default_lower: list[int],
    upper_key: str,
    default_upper: list[int],
    min_pixels_key: str,
) -> bool:
    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    if not bool(combat_cfg.get(enabled_key, False)):
        return False
    return _detect_fixed_hsv_marker(
        frame,
        cfg,
        region_key=region_key,
        default_region=default_region,
        lower_key=lower_key,
        default_lower=default_lower,
        upper_key=upper_key,
        default_upper=default_upper,
        min_pixels_key=min_pixels_key,
        default_min_pixels=800,
    )


def _detect_fixed_hsv_marker(
    frame: np.ndarray,
    config: dict[str, Any],
    *,
    region_key: str,
    default_region: dict[str, int],
    lower_key: str,
    default_lower: list[int],
    upper_key: str,
    default_upper: list[int],
    min_pixels_key: str,
    default_min_pixels: int,
) -> bool:
    if frame.size == 0:
        return False
    combat_cfg = _combat_cfg(config)
    region_cfg = combat_cfg.get(region_key, default_region)
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    marker_mask = cv2.inRange(
        hsv,
        np.array(combat_cfg.get(lower_key, default_lower), dtype=np.uint8),
        np.array(combat_cfg.get(upper_key, default_upper), dtype=np.uint8),
    )
    return int(cv2.countNonZero(marker_mask)) >= int(
        combat_cfg.get(min_pixels_key, default_min_pixels)
    )


def detect_player_health_fraction(frame: np.ndarray, config: dict[str, Any] | None = None) -> float | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    region_cfg = combat_cfg.get("player_health_region", {"x": 155, "y": 70, "width": 190, "height": 25})
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    health_mask = cv2.inRange(
        hsv,
        np.array(combat_cfg.get("player_health_hsv_lower", [35, 60, 50]), dtype=np.uint8),
        np.array(combat_cfg.get("player_health_hsv_upper", [95, 255, 255]), dtype=np.uint8),
    )
    min_pixels = max(1, int(combat_cfg.get("player_health_column_min_pixels", 4)))
    filled_columns = np.count_nonzero(np.count_nonzero(health_mask, axis=0) >= min_pixels)
    if filled_columns <= 0:
        return None
    return max(0.0, min(1.0, filled_columns / float(roi_width)))


def detect_outgoing_damage_numbers(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> BoundingBox | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    region_cfg = combat_cfg.get("outgoing_damage_region", {"x": 520, "y": 180, "width": 1320, "height": 650})
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array(combat_cfg.get("outgoing_damage_yellow_hsv_lower", [15, 45, 145]), dtype=np.uint8),
        np.array(combat_cfg.get("outgoing_damage_yellow_hsv_upper", [45, 255, 255]), dtype=np.uint8),
    )
    white_mask = cv2.inRange(
        hsv,
        np.array(combat_cfg.get("outgoing_damage_white_hsv_lower", [0, 0, 190]), dtype=np.uint8),
        np.array(combat_cfg.get("outgoing_damage_white_hsv_upper", [179, 85, 255]), dtype=np.uint8),
    )
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    close_kernel = int(combat_cfg.get("outgoing_damage_close_kernel", 5))
    if close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), dtype=np.uint8))
    boxes = _component_boxes(
        mask,
        offset=(roi_x, roi_y),
        min_area=int(combat_cfg.get("outgoing_damage_min_area", 120)),
        max_area=int(combat_cfg.get("outgoing_damage_max_area", 9000)),
        min_width=int(combat_cfg.get("outgoing_damage_min_width", 8)),
        max_width=int(combat_cfg.get("outgoing_damage_max_width", 260)),
        min_height=int(combat_cfg.get("outgoing_damage_min_height", 10)),
        max_height=int(combat_cfg.get("outgoing_damage_max_height", 95)),
        min_fill_ratio=float(combat_cfg.get("outgoing_damage_min_fill_ratio", 0.18)),
    )
    boxes = [box for box in boxes if not _looks_like_nameplate_label(box, frame.shape, combat_cfg)]
    if not boxes:
        return None

    return max(boxes, key=lambda item: item.area())


def detect_combat_facing_error_text(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    if frame.size == 0:
        return False

    cfg = config or {}
    combat_cfg = _combat_cfg(cfg)
    if not bool(combat_cfg.get("facing_error_detector_enabled", True)):
        return False

    region_cfg = combat_cfg.get("facing_error_region", {"x": 720, "y": 120, "width": 1120, "height": 190})
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return False

    red_mask = _facing_error_mask(roi, combat_cfg)
    min_pixels = int(combat_cfg.get("facing_error_min_pixels", 80))
    if int(cv2.countNonZero(red_mask)) < min_pixels:
        return False

    num_labels, _labels, stats, _centers = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    component_count = 0
    min_x: int | None = None
    min_y: int | None = None
    max_x = 0
    max_y = 0
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if int(area) < int(combat_cfg.get("facing_error_component_min_area", 4)):
            continue
        if int(width) < int(combat_cfg.get("facing_error_component_min_width", 2)):
            continue
        if int(height) < int(combat_cfg.get("facing_error_component_min_height", 4)):
            continue
        component_count += 1
        min_x = int(x) if min_x is None else min(min_x, int(x))
        min_y = int(y) if min_y is None else min(min_y, int(y))
        max_x = max(max_x, int(x) + int(width))
        max_y = max(max_y, int(y) + int(height))

    if component_count < int(combat_cfg.get("facing_error_min_components", 3)):
        return False

    row_counts = np.count_nonzero(red_mask, axis=1)
    band_height = max(1, int(combat_cfg.get("facing_error_row_band_height", 5)))
    row_band_score = 0
    for row_index in range(len(row_counts)):
        row_band_score = max(
            row_band_score,
            int(np.sum(row_counts[row_index : row_index + band_height])),
        )
    if row_band_score < int(combat_cfg.get("facing_error_min_row_band_pixels", 120)):
        return False

    column_counts = np.count_nonzero(red_mask, axis=0)
    active_columns = np.where(column_counts > 0)[0]
    if active_columns.size == 0:
        return False
    span_width = int(active_columns[-1] - active_columns[0])
    return span_width >= int(combat_cfg.get("facing_error_min_span_width", 80))


def _facing_error_mask(roi: np.ndarray, combat_cfg: dict[str, Any]) -> np.ndarray:
    blue, green, red = cv2.split(roi)
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    mask = (
        (red_i >= int(combat_cfg.get("facing_error_min_red", 80)))
        & (green_i <= int(combat_cfg.get("facing_error_max_green", 30)))
        & (blue_i <= int(combat_cfg.get("facing_error_max_blue", 30)))
        & ((red_i - green_i) >= int(combat_cfg.get("facing_error_min_red_green_delta", 60)))
        & ((red_i - blue_i) >= int(combat_cfg.get("facing_error_min_red_blue_delta", 60)))
    )
    return (mask.astype(np.uint8)) * 255


def normalize_attack_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ("E",)
    keys = tuple(str(item).strip().upper() for item in value if str(item).strip())
    return keys or ("E",)


def normalize_periodic_combat_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    keys = tuple(str(item).strip().upper() for item in value if str(item).strip())
    return keys


def _detect_confirmed_target_frame(
    frame: np.ndarray,
    config: dict[str, Any],
    combat_cfg: dict[str, Any],
) -> BoundingBox | None:
    if not bool(combat_cfg.get("target_frame_detector_enabled", True)):
        return None

    region_cfg = combat_cfg.get(
        "confirmed_target_frame_region",
        {"x": 300, "y": 15, "width": 390, "height": 120},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array(combat_cfg.get("target_health_hsv_lower", [35, 80, 55]), dtype=np.uint8),
        np.array(combat_cfg.get("target_health_hsv_upper", [95, 255, 255]), dtype=np.uint8),
    )
    red_mask = _combat_red_mask(hsv, combat_cfg)

    health_boxes = _component_boxes(
        green_mask,
        offset=(roi_x, roi_y),
        min_area=int(combat_cfg.get("target_health_min_area", 700)),
        max_area=int(combat_cfg.get("target_health_max_area", 6000)),
        min_width=int(combat_cfg.get("target_health_min_width", 90)),
        max_width=int(combat_cfg.get("target_health_max_width", 260)),
        min_height=int(combat_cfg.get("target_health_min_height", 7)),
        max_height=int(combat_cfg.get("target_health_max_height", 22)),
        min_fill_ratio=float(combat_cfg.get("target_health_min_fill_ratio", 0.75)),
    )
    red_health_boxes = _component_boxes(
        red_mask,
        offset=(roi_x, roi_y),
        min_area=int(combat_cfg.get("target_health_min_area", 700)),
        max_area=int(combat_cfg.get("target_health_max_area", 6000)),
        min_width=int(combat_cfg.get("target_health_min_width", 90)),
        max_width=int(combat_cfg.get("target_health_max_width", 260)),
        min_height=int(combat_cfg.get("target_health_min_height", 7)),
        max_height=int(combat_cfg.get("target_health_max_height", 22)),
        min_fill_ratio=float(combat_cfg.get("target_red_health_min_fill_ratio", 0.60)),
    )
    health_boxes.extend(red_health_boxes)
    hostile_title_boxes = _component_boxes(
        red_mask,
        offset=(roi_x, roi_y),
        min_area=int(combat_cfg.get("target_hostile_title_min_area", 500)),
        max_area=int(combat_cfg.get("target_hostile_title_max_area", 6000)),
        min_width=int(combat_cfg.get("target_hostile_title_min_width", 80)),
        max_width=int(combat_cfg.get("target_hostile_title_max_width", 260)),
        min_height=int(combat_cfg.get("target_hostile_title_min_height", 8)),
        max_height=int(combat_cfg.get("target_hostile_title_max_height", 28)),
        min_fill_ratio=float(combat_cfg.get("target_hostile_title_min_fill_ratio", 0.35)),
    )
    if red_health_boxes and bool(combat_cfg.get("target_red_health_bar_confirms_target", True)):
        return max(red_health_boxes, key=lambda item: item.area())
    if not health_boxes or not hostile_title_boxes:
        return None

    health_box = max(health_boxes, key=lambda item: item.area())
    title_box = max(hostile_title_boxes, key=lambda item: item.area())
    x1 = min(health_box.x, title_box.x)
    y1 = min(health_box.y, title_box.y)
    x2 = max(health_box.x2, title_box.x2)
    y2 = max(health_box.y2, title_box.y2)
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def _combat_red_mask(hsv: np.ndarray, combat_cfg: dict[str, Any]) -> np.ndarray:
    ranges = combat_cfg.get(
        "target_hostile_hsv_ranges",
        [
            [[0, 90, 80], [10, 255, 255]],
            [[170, 90, 80], [179, 255, 255]],
        ],
    )
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
        )
    return mask


def _component_boxes(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    min_area: int,
    max_area: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    min_fill_ratio: float,
) -> list[BoundingBox]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    offset_x, offset_y = offset
    boxes: list[BoundingBox] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        fill_ratio = area / float(width * height)
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and fill_ratio >= min_fill_ratio
        ):
            continue
        boxes.append(BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)))
    return boxes


def _face_turn_key(
    bbox: BoundingBox,
    frame_shape: tuple[int, ...],
    combat_cfg: dict[str, Any],
) -> str | None:
    frame_width = frame_shape[1]
    reference_x = frame_width * float(combat_cfg.get("face_reference_x_fraction", 0.5))
    deadzone_px = float(combat_cfg.get("face_center_deadzone_px", 35))
    bbox_center_x = bbox.x + bbox.width / 2.0
    if bbox_center_x < reference_x - deadzone_px:
        return "A"
    if bbox_center_x > reference_x + deadzone_px:
        return "D"
    return None


def _looks_like_nameplate_label(
    bbox: BoundingBox,
    frame_shape: tuple[int, ...],
    combat_cfg: dict[str, Any],
) -> bool:
    del frame_shape
    aspect = bbox.width / float(max(1, bbox.height))
    return (
        aspect >= float(combat_cfg.get("outgoing_damage_label_min_aspect", 5.0))
        and bbox.height <= int(combat_cfg.get("outgoing_damage_label_max_height", 34))
    )


def _damage_near_combat_target(
    damage_bbox: BoundingBox,
    target_bbox: BoundingBox,
    frame_shape: tuple[int, ...],
    combat_cfg: dict[str, Any],
) -> bool:
    frame_height, frame_width = frame_shape[:2]
    damage_center_x = damage_bbox.x + damage_bbox.width / 2.0
    damage_center_y = damage_bbox.y + damage_bbox.height / 2.0
    target_center_x = target_bbox.x + target_bbox.width / 2.0
    target_center_y = target_bbox.y + target_bbox.height / 2.0
    max_dx = frame_width * float(combat_cfg.get("outgoing_damage_target_max_dx_fraction", 0.20))
    max_dy = frame_height * float(combat_cfg.get("outgoing_damage_target_max_dy_fraction", 0.22))
    return abs(damage_center_x - target_center_x) <= max_dx and abs(damage_center_y - target_center_y) <= max_dy


def _combat_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("safety", {}).get("combat", {})


def _bbox_to_dict(bbox: BoundingBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }
