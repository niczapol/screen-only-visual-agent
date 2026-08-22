from vision_bot.coords import xy_to_coord
from vision_bot.mounted_escape import MountedEscapeState


def _observe(
    state: MountedEscapeState,
    *,
    now: float,
    coord: int | None,
    mounted: bool = True,
    health: float | None = 0.80,
    combat: bool = True,
    mining: bool = False,
    healing: bool = False,
    threat: str = "normal",
):
    return state.observe(
        now=now,
        combat_engaged=combat,
        mounted=mounted,
        player_health_fraction=health,
        current_coord=coord,
        mining_conflict=mining,
        heal_casting=healing,
        threat_level=threat,
    )


def test_mounted_escape_continues_through_large_health_drop_above_floor():
    state = MountedEscapeState(
        enabled=True,
        min_health=0.55,
        progress_timeout_seconds=1.0,
        min_coord_delta=0.015,
    )
    start = xy_to_coord(45.00, 60.00)
    moved = xy_to_coord(45.03, 60.00)

    assert _observe(state, now=10.0, coord=start, health=0.95).active
    decision = _observe(state, now=10.5, coord=moved, health=0.60)

    assert decision.active
    assert decision.reason == "mounted_escape"
    assert not decision.escalated


def test_mounted_escape_escalates_after_coordinate_progress_timeout_and_latches():
    state = MountedEscapeState(
        enabled=True,
        progress_timeout_seconds=1.0,
        min_coord_delta=0.015,
    )
    coord = xy_to_coord(45.00, 60.00)

    assert _observe(state, now=1.0, coord=coord).active
    assert _observe(state, now=1.9, coord=coord).active
    stalled = _observe(state, now=2.0, coord=coord)

    assert not stalled.active
    assert stalled.escalated
    assert stalled.reason == "no_coordinate_progress"
    assert state.escalations == 1

    moved = xy_to_coord(45.10, 60.00)
    latched = _observe(state, now=2.2, coord=moved)
    assert not latched.active
    assert latched.reason == "no_coordinate_progress"
    assert state.escalations == 1


def test_mounted_escape_progress_resets_timeout():
    state = MountedEscapeState(
        enabled=True,
        progress_timeout_seconds=1.0,
        min_coord_delta=0.015,
    )
    start = xy_to_coord(45.00, 60.00)
    moved = xy_to_coord(45.02, 60.00)

    assert _observe(state, now=1.0, coord=start).active
    assert _observe(state, now=1.8, coord=moved).active
    decision = _observe(state, now=2.6, coord=moved)

    assert decision.active
    assert decision.progress_age_seconds == 0.8


def test_mounted_escape_escalates_for_dismount_low_health_mining_or_heal():
    coord = xy_to_coord(45.00, 60.00)
    cases = (
        ({"mounted": False}, "dismounted"),
        ({"health": 0.54}, "low_health"),
        ({"mining": True}, "mining_conflict"),
        ({"healing": True}, "heal_casting"),
    )

    for overrides, expected_reason in cases:
        state = MountedEscapeState(enabled=True, min_health=0.55)
        decision = _observe(state, now=1.0, coord=coord, **overrides)
        assert not decision.active
        assert decision.escalated
        assert decision.reason == expected_reason


def test_mounted_escape_escalates_when_visible_threat_becomes_elevated():
    state = MountedEscapeState(enabled=True, min_health=0.55)
    coord = xy_to_coord(45.00, 60.00)

    assert _observe(state, now=1.0, coord=coord, threat="normal").active
    decision = _observe(
        state,
        now=1.5,
        coord=xy_to_coord(45.03, 60.00),
        threat="elevated",
    )

    assert not decision.active
    assert decision.escalated
    assert decision.reason == "threat_elevated"


def test_mounted_escape_escalates_at_route_hazard_boundary():
    state = MountedEscapeState(enabled=True, min_health=0.55)

    decision = state.observe(
        now=1.0,
        combat_engaged=True,
        mounted=True,
        player_health_fraction=0.90,
        current_coord=xy_to_coord(63.53, 49.70),
        mining_conflict=False,
        heal_casting=False,
        route_hazard=True,
    )

    assert not decision.active
    assert decision.escalated
    assert decision.reason == "route_hazard"


def test_mounted_escape_escalates_after_duration_limit_despite_progress():
    state = MountedEscapeState(
        enabled=True,
        max_active_seconds=2.0,
        progress_timeout_seconds=1.0,
    )

    assert _observe(
        state,
        now=1.0,
        coord=xy_to_coord(45.00, 60.00),
    ).active
    assert _observe(
        state,
        now=2.0,
        coord=xy_to_coord(45.03, 60.00),
    ).active
    decision = _observe(
        state,
        now=3.0,
        coord=xy_to_coord(45.06, 60.00),
    )

    assert not decision.active
    assert decision.escalated
    assert decision.reason == "escape_duration_limit"


def test_mounted_escape_waits_for_health_without_latching_combat_escalation():
    state = MountedEscapeState(enabled=True)
    coord = xy_to_coord(45.00, 60.00)

    unknown = _observe(state, now=1.0, coord=coord, health=None)
    assert not unknown.active
    assert not unknown.escalated
    assert unknown.reason == "health_unknown"

    ready = _observe(state, now=1.2, coord=coord, health=0.80)
    assert ready.active
    assert ready.reason == "mounted_escape"


def test_mounted_escape_resets_only_after_combat_clear():
    state = MountedEscapeState(enabled=True)
    coord = xy_to_coord(45.00, 60.00)

    assert _observe(state, now=1.0, coord=coord, mounted=False).escalated
    cleared = _observe(state, now=2.0, coord=coord, combat=False)
    assert not cleared.active
    assert not cleared.escalated

    restarted = _observe(state, now=3.0, coord=coord)
    assert restarted.active
    assert state.combat_episodes == 2
    assert state.escape_episodes == 1
    assert state.last_escalation_reason == "dismounted"


def test_mounted_escape_reads_new_configuration():
    state = MountedEscapeState.from_config(
        {
            "safety": {
                "combat": {
                    "mounted_escape_enabled": True,
                    "mounted_escape_min_health": 0.50,
                    "mounted_escape_max_active_seconds": 9.0,
                    "mounted_escape_progress_timeout_seconds": 0.8,
                    "mounted_escape_min_coord_delta": 0.02,
                }
            }
        }
    )

    assert state.enabled
    assert state.min_health == 0.50
    assert state.max_active_seconds == 9.0
    assert state.progress_timeout_seconds == 0.8
    assert state.min_coord_delta == 0.02
