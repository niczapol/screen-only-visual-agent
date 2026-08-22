from vision_bot.mounting import MountTravelState


def test_mount_state_casts_waits_retries_then_falls_back():
    state = MountTravelState(
        enabled=True,
        cast_seconds=2.0,
        retry_seconds=4.0,
        max_attempts_before_fallback=2,
        post_combat_settle_seconds=0.0,
    )

    assert state.observe(now=10.0, mounted=False, combat_active=False) == "mount_cast"
    assert state.observe(now=11.0, mounted=False, combat_active=False) == "mount_cast_wait"
    assert state.observe(now=12.5, mounted=False, combat_active=False) == "mount_retry_wait"
    assert state.observe(now=14.0, mounted=False, combat_active=False) == "mount_cast"
    assert state.observe(now=18.0, mounted=False, combat_active=False) is None
    assert state.fallback_unmounted


def test_mount_state_resets_after_combat_and_confirmed_mount():
    state = MountTravelState(enabled=True, post_combat_settle_seconds=3.0)
    assert state.observe(now=1.0, mounted=False, combat_active=False) == "mount_cast"

    assert state.observe(now=1.1, mounted=False, combat_active=True) is None
    assert state.observe(now=2.0, mounted=False, combat_active=False) == "mount_combat_clear_wait"
    assert state.observe(now=4.0, mounted=False, combat_active=False) == "mount_combat_clear_wait"
    assert state.observe(now=4.1, mounted=False, combat_active=False) == "mount_cast"
    assert state.observe(now=4.2, mounted=True, combat_active=False) is None
    assert state.attempts == 0


def test_mount_state_waits_three_seconds_after_loot_and_does_not_spend_attempt() -> None:
    state = MountTravelState(enabled=True, post_combat_settle_seconds=3.0)

    assert state.observe(
        now=10.0,
        mounted=False,
        combat_active=False,
        post_combat_loot_active=True,
        control_available=False,
    ) is None
    assert state.attempts == 0
    assert state.observe(now=12.9, mounted=False, combat_active=False) == "mount_combat_clear_wait"
    assert state.observe(now=13.0, mounted=False, combat_active=False) == "mount_cast"


def test_mount_state_does_not_advance_without_input_ownership() -> None:
    state = MountTravelState(enabled=True, post_combat_settle_seconds=0.0)

    assert state.observe(
        now=10.0,
        mounted=False,
        combat_active=False,
        control_available=False,
    ) is None
    assert state.attempts == 0
    assert state.observe(now=10.1, mounted=False, combat_active=False) == "mount_cast"
