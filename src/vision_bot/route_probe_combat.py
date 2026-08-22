from __future__ import annotations

import time
from typing import Any

from vision_bot.combat import (
    CombatFallbackState,
    CombatHealState,
    CombatState,
    PeriodicCombatKeyState,
)
from vision_bot.movement import InputController, MouseSteeringController


def _maybe_invert_turn_key(
    turn_key: str | None,
    *,
    invert_turn_direction: bool,
) -> str | None:
    if turn_key not in {"A", "D"}:
        return None
    if not invert_turn_direction:
        return turn_key
    return "A" if turn_key == "D" else "D"


def _combat_face_search_turn_key(value: Any) -> str:
    turn_key = str(value or "D").strip().upper()
    return turn_key if turn_key in {"A", "D"} else "D"


def _apply_combat_fallback(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    combat: CombatState,
    *,
    now: float,
    face_turn_duration: float,
    face_search_turn_180_duration: float,
    face_search_turn_90_duration: float,
    attack_tap_duration: float,
    combat_key_min_interval: float = 0.12,
    range_approach_duration: float = 0.35,
    range_approach_cooldown: float = 0.55,
    periodic_combat_keys: PeriodicCombatKeyState | None = None,
    aligned_periodic_combat_keys: PeriodicCombatKeyState | None = None,
    continuous_face_search: bool = False,
    attacker_target_selection: bool = False,
    attacker_target_cycle_key: str = "F8",
    attacker_target_cycle_interval: float = 0.35,
    nameplate_facing: bool = False,
    nameplate_face_turn_duration: float = 0.08,
    target_interact_facing: bool = False,
    target_interact_key: str = "G",
    target_interact_cooldown: float = 0.45,
    target_interact_probe_delay: float = 0.35,
    invert_turn_direction: bool = False,
    mouse_steering: MouseSteeringController | None = None,
) -> tuple[str, str | None]:
    combat_fallback.observe_outgoing_damage(
        now,
        combat.outgoing_damage_visible,
        facing_error_visible=combat.facing_error_visible,
    )
    recent_hit = combat_fallback.has_recent_outgoing_damage(now)
    priority_periodic_keys = _tap_due_priority_periodic_combat_key(
        input_controller,
        combat_fallback,
        periodic_combat_keys,
        now=now,
        attack_tap_duration=attack_tap_duration,
        combat_key_min_interval=combat_key_min_interval,
    )
    if priority_periodic_keys:
        return "combat_priority_periodic", None
    if attacker_target_selection and not combat.target_is_attacker:
        combat_fallback.release_target_facing_lock()
        _release_continuous_face_search(input_controller, combat_fallback, mouse_steering)
        if combat_fallback.can_cycle_target(now, attacker_target_cycle_interval):
            input_controller.tap_key(attacker_target_cycle_key, duration=attack_tap_duration)
            combat_fallback.record_tapped_keys([attacker_target_cycle_key], now=now)
            return "combat_select_attacker", None
        combat_fallback.record_tapped_keys([])
        return "combat_select_attacker_wait", None

    if target_interact_facing and combat.target_is_attacker:
        interaction_action: str | None = None
        if combat.out_of_range_marker_visible:
            interaction_action = "combat_interact_approach_target"
        elif combat.facing_error_marker_visible:
            interaction_action = "combat_interact_face_target"
        elif (
            not recent_hit
            and combat_fallback.unconfirmed_e_started_at is not None
            and now - combat_fallback.unconfirmed_e_started_at
            >= max(0.0, target_interact_probe_delay)
        ):
            interaction_action = "combat_interact_probe_target"
        if (
            interaction_action is not None
            and target_interact_key
            and combat_fallback.can_interact_target(now, target_interact_cooldown)
        ):
            _release_continuous_face_search(input_controller, combat_fallback, mouse_steering)
            input_controller.tap_key(target_interact_key, duration=attack_tap_duration)
            combat_fallback.record_tapped_keys([target_interact_key], now=now)
            return interaction_action, None

    if nameplate_facing and combat.target_is_attacker:
        _release_continuous_face_search(input_controller, combat_fallback, mouse_steering)
        if combat.nameplate_visible and combat.face_turn_key is not None:
            turn_key = _maybe_invert_turn_key(
                combat.face_turn_key,
                invert_turn_direction=invert_turn_direction,
            ) or combat.face_turn_key
            _turn_character(
                input_controller,
                mouse_steering,
                turn_key,
                duration=max(0.0, nameplate_face_turn_duration),
            )
            combat_fallback.record_tapped_keys([])
            return "combat_face_target_nameplate", turn_key
        if combat.nameplate_visible:
            attack_keys = _tap_due_combat_keys(
                input_controller,
                combat_fallback,
                now=now,
                attack_tap_duration=attack_tap_duration,
                combat_key_min_interval=combat_key_min_interval,
                periodic_combat_keys=periodic_combat_keys,
                aligned_periodic_combat_keys=aligned_periodic_combat_keys,
            )
            return ("combat_attack" if attack_keys else "combat_hold"), None
        if combat.facing_error_visible:
            face_search_turn = combat_fallback.next_face_search_turn(
                now,
                turn_180_duration=face_search_turn_180_duration,
                turn_90_duration=face_search_turn_90_duration,
                force=True,
            )
            if face_search_turn is not None:
                turn_key, _duration, _action = face_search_turn
                turn_key = _maybe_invert_turn_key(
                    turn_key,
                    invert_turn_direction=invert_turn_direction,
                ) or turn_key
                _turn_character(
                    input_controller,
                    mouse_steering,
                    turn_key,
                    duration=max(0.0, nameplate_face_turn_duration),
                )
                combat_fallback.record_tapped_keys([])
                return "combat_face_target_probe", turn_key
        attack_keys = _tap_due_combat_keys(
            input_controller,
            combat_fallback,
            now=now,
            attack_tap_duration=attack_tap_duration,
            combat_key_min_interval=combat_key_min_interval,
            periodic_combat_keys=periodic_combat_keys,
            aligned_periodic_combat_keys=aligned_periodic_combat_keys,
        )
        return ("combat_attack" if attack_keys else "combat_hold"), None

    if continuous_face_search:
        return _apply_continuous_combat_facing(
            input_controller,
            combat_fallback,
            combat,
            now=now,
            recent_hit=recent_hit,
            face_turn_duration=face_turn_duration,
            face_search_turn_180_duration=face_search_turn_180_duration,
            face_search_turn_90_duration=face_search_turn_90_duration,
            attack_tap_duration=attack_tap_duration,
            combat_key_min_interval=combat_key_min_interval,
            range_approach_duration=range_approach_duration,
            range_approach_cooldown=range_approach_cooldown,
            periodic_combat_keys=periodic_combat_keys,
            aligned_periodic_combat_keys=aligned_periodic_combat_keys,
            invert_turn_direction=invert_turn_direction,
            mouse_steering=mouse_steering,
        )
    if combat.facing_error_visible and not recent_hit:
        face_search_turn = combat_fallback.next_face_search_turn(
            now,
            turn_180_duration=face_search_turn_180_duration,
            turn_90_duration=face_search_turn_90_duration,
            force=True,
        )
        if face_search_turn is not None:
            local_turn_key, turn_duration, face_search_action = face_search_turn
            local_turn_key = _maybe_invert_turn_key(
                local_turn_key,
                invert_turn_direction=invert_turn_direction,
            )
            _turn_character(
                input_controller,
                mouse_steering,
                local_turn_key,
                duration=turn_duration,
            )
            combat_fallback.record_tapped_keys([])
            return face_search_action, local_turn_key

    local_turn_key = _maybe_invert_turn_key(
        combat.face_turn_key,
        invert_turn_direction=invert_turn_direction,
    )
    if combat.out_of_range_marker_visible:
        if combat_fallback.can_approach_range(now, range_approach_cooldown):
            input_controller.tap_key("W", duration=max(0.0, range_approach_duration))
            combat_fallback.record_tapped_keys([])
            return "combat_close_range", None

    if not recent_hit:
        face_search_turn = combat_fallback.next_face_search_turn(
            now,
            turn_180_duration=face_search_turn_180_duration,
            turn_90_duration=face_search_turn_90_duration,
        )
        if face_search_turn is not None:
            local_turn_key, turn_duration, face_search_action = face_search_turn
            local_turn_key = _maybe_invert_turn_key(
                local_turn_key,
                invert_turn_direction=invert_turn_direction,
            )
            _turn_character(
                input_controller,
                mouse_steering,
                local_turn_key,
                duration=turn_duration,
            )
            combat_fallback.record_tapped_keys([])
            return face_search_action, local_turn_key

    attack_keys = _tap_due_combat_keys(
        input_controller,
        combat_fallback,
        now=now,
        attack_tap_duration=attack_tap_duration,
        combat_key_min_interval=combat_key_min_interval,
        periodic_combat_keys=periodic_combat_keys,
        aligned_periodic_combat_keys=aligned_periodic_combat_keys,
    )
    if attack_keys:
        return "combat_attack", None
    return "combat_hold", None


def _apply_continuous_combat_facing(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    combat: CombatState,
    *,
    now: float,
    recent_hit: bool,
    face_turn_duration: float,
    face_search_turn_180_duration: float,
    face_search_turn_90_duration: float,
    attack_tap_duration: float,
    combat_key_min_interval: float,
    range_approach_duration: float,
    range_approach_cooldown: float,
    periodic_combat_keys: PeriodicCombatKeyState | None,
    aligned_periodic_combat_keys: PeriodicCombatKeyState | None,
    invert_turn_direction: bool,
    mouse_steering: MouseSteeringController | None,
) -> tuple[str, str | None]:
    if recent_hit:
        _release_continuous_face_search(input_controller, combat_fallback, mouse_steering)

    local_turn_key = _maybe_invert_turn_key(
        combat.face_turn_key,
        invert_turn_direction=invert_turn_direction,
    )
    if not recent_hit and combat.out_of_range_marker_visible:
        _release_continuous_face_search(input_controller, combat_fallback, mouse_steering)
        if combat_fallback.can_approach_range(now, range_approach_cooldown):
            input_controller.tap_key("W", duration=max(0.0, range_approach_duration))
            combat_fallback.record_tapped_keys([])
            return "combat_close_range", None

    should_force_search = combat.facing_error_visible and not recent_hit
    if not recent_hit and combat_fallback.face_search_held_key is None:
        face_search_turn = combat_fallback.next_face_search_turn(
            now,
            turn_180_duration=face_search_turn_180_duration,
            turn_90_duration=face_search_turn_90_duration,
            force=should_force_search,
        )
        if face_search_turn is not None:
            turn_key, _turn_duration, _action = face_search_turn
            turn_key = _maybe_invert_turn_key(
                turn_key,
                invert_turn_direction=invert_turn_direction,
            ) or "D"
            keyboard_held = mouse_steering is None or not mouse_steering.enabled
            held_turn_key = combat_fallback.start_continuous_face_search(
                turn_key,
                keyboard_held=keyboard_held,
            )
            if keyboard_held:
                input_controller.press_key(held_turn_key)
            elif mouse_steering is not None:
                mouse_steering.start_continuous_turn(held_turn_key, context="combat")

    attack_keys = _tap_due_combat_keys(
        input_controller,
        combat_fallback,
        now=now,
        attack_tap_duration=attack_tap_duration,
        combat_key_min_interval=combat_key_min_interval,
        periodic_combat_keys=periodic_combat_keys,
        aligned_periodic_combat_keys=aligned_periodic_combat_keys,
    )
    held_turn_key = combat_fallback.face_search_held_key
    if held_turn_key is not None:
        if mouse_steering is not None and mouse_steering.enabled:
            mouse_steering.start_continuous_turn(held_turn_key, context="combat")
        return (
            "combat_face_search_attack" if attack_keys else "combat_face_search_hold",
            held_turn_key,
        )
    if attack_keys:
        return "combat_attack", None
    return "combat_hold", None


def _release_continuous_face_search(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    mouse_steering: MouseSteeringController | None = None,
) -> None:
    keyboard_held = combat_fallback.face_search_keyboard_held
    held_key = combat_fallback.stop_continuous_face_search()
    if mouse_steering is not None:
        mouse_steering.stop_continuous_turn()
    if held_key is not None and keyboard_held:
        input_controller.release_key(held_key)


def _turn_character(
    input_controller: InputController,
    mouse_steering: MouseSteeringController | None,
    turn_key: str | None,
    *,
    duration: float,
) -> None:
    if mouse_steering is not None and mouse_steering.turn(
        turn_key,
        duration=duration,
        context="combat",
    ):
        return
    if turn_key in {"A", "D"}:
        input_controller.tap_key(turn_key, duration=max(0.0, duration))


def _tap_due_combat_keys(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    *,
    now: float,
    attack_tap_duration: float,
    combat_key_min_interval: float = 0.12,
    periodic_combat_keys: PeriodicCombatKeyState | None = None,
    aligned_periodic_combat_keys: PeriodicCombatKeyState | None = None,
) -> list[str]:
    if not combat_fallback.can_tap_combat_key(now, combat_key_min_interval):
        combat_fallback.record_tapped_keys([])
        return []

    attack_key = combat_fallback.peek_next_attack_key(now)
    periodic_key = periodic_combat_keys.peek_next_key(now) if periodic_combat_keys is not None else None
    aligned_periodic_key = (
        aligned_periodic_combat_keys.peek_next_key(now)
        if aligned_periodic_combat_keys is not None
        and combat_fallback.has_target_facing_lock(now)
        else None
    )
    key = periodic_key or aligned_periodic_key or attack_key
    if key is None:
        combat_fallback.record_tapped_keys([])
        return []

    if periodic_key is not None and periodic_combat_keys is not None:
        key = periodic_combat_keys.consume_next_key(now)
    elif (
        aligned_periodic_key is not None
        and aligned_periodic_combat_keys is not None
    ):
        key = aligned_periodic_combat_keys.consume_next_key(now)
    elif attack_key is not None:
        key = combat_fallback.consume_next_attack_key(now)
    if key is None:
        combat_fallback.record_tapped_keys([])
        return []

    input_controller.tap_key(key, duration=attack_tap_duration)
    combat_fallback.record_tapped_keys([key], now=now)
    return [key]


def _tap_due_priority_periodic_combat_key(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    state: PeriodicCombatKeyState | None,
    *,
    now: float,
    attack_tap_duration: float,
    combat_key_min_interval: float = 0.12,
) -> list[str]:
    """Tap combat-wide periodic keys before targeting, facing and E/Q."""
    if state is None or state.peek_next_key(now) is None:
        return []
    if not combat_fallback.can_tap_combat_key(now, combat_key_min_interval):
        return []
    key = state.consume_next_key(now)
    if key is None:
        return []
    input_controller.tap_key(key, duration=attack_tap_duration)
    combat_fallback.record_tapped_keys([key], now=now)
    return [key]


def _tap_due_low_health_combat_key(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    combat: CombatState,
    state: PeriodicCombatKeyState,
    *,
    now: float,
    health_threshold: float,
    attack_tap_duration: float,
    combat_key_min_interval: float = 0.12,
) -> list[str]:
    """Tap the emergency key first when visible player HP crosses the threshold."""
    health = combat.player_health_fraction
    if health is None or health > max(0.0, min(1.0, health_threshold)):
        state.reset()
        return []
    return _tap_due_priority_periodic_combat_key(
        input_controller,
        combat_fallback,
        state,
        now=now,
        attack_tap_duration=attack_tap_duration,
        combat_key_min_interval=combat_key_min_interval,
    )


def _apply_combat_low_health_heal(
    input_controller: InputController,
    heal_state: CombatHealState,
    combat: CombatState,
    *,
    now: float,
    heal_key: str,
    heal_tap_duration: float,
) -> str | None:
    if not heal_state.should_start(combat, now):
        return None
    if heal_key:
        input_controller.tap_key(heal_key, duration=max(0.0, heal_tap_duration))
    heal_state.start(now)
    return "combat_low_health_heal"


def _apply_emergency_heal_followup(
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    heal_state: CombatHealState,
    trigger_keys: list[str],
    *,
    now: float,
    heal_key: str,
    heal_tap_duration: float,
    delay_seconds: float,
) -> list[str]:
    """Serialize one emergency heal immediately after a fired F key."""
    key = str(heal_key).strip().upper()
    if not trigger_keys or not key:
        return []
    delay = max(0.0, float(delay_seconds))
    if delay > 0.0:
        time.sleep(delay)
    input_controller.tap_key(key, duration=max(0.0, heal_tap_duration))
    followup_now = now + delay
    heal_state.start(followup_now)
    combined = [*trigger_keys, key]
    combat_fallback.record_tapped_keys(combined, now=followup_now)
    return [key]
