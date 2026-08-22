from scripts.run_luna_overnight_live_tests import RouteLiveWatchdog, _termination_requires_review


def _row(
    *,
    coord: int = 5000500000,
    progress: float = 10.0,
    completed: int = 1,
    armor: bool = False,
    skipped: int = 0,
):
    return {
        "coord": coord,
        "coord_feedback_age": 0.0,
        "route_following": {"progress": progress},
        "completed_targets": completed,
        "skipped_targets": skipped,
        "armor_critical": armor,
        "combat": {"active": False},
        "game_state": {"death_or_blocking_modal": False},
        "mining": {"phase": "idle"},
        "post_combat_loot": {"active": False},
    }


def test_watchdog_stops_after_persistent_critical_armor():
    watchdog = RouteLiveWatchdog()

    assert watchdog.observe(_row(armor=True), now=0.0) is None
    assert watchdog.observe(_row(armor=True), now=1.0) is None
    decision = watchdog.observe(_row(armor=True), now=2.0)

    assert decision is not None
    assert decision.reason == "armor_critical"


def test_watchdog_detects_stationary_route_without_progress():
    watchdog = RouteLiveWatchdog(route_window_seconds=10.0)

    assert watchdog.observe(_row(), now=0.0) is None
    assert watchdog.observe(_row(), now=5.0) is None
    decision = watchdog.observe(_row(), now=10.0)

    assert decision is not None
    assert decision.reason == "stationary_without_route_progress"


def test_watchdog_does_not_call_combat_stationary_navigation_stuck():
    watchdog = RouteLiveWatchdog(route_window_seconds=10.0)
    row = _row()
    row["combat"] = {"active": True}

    assert watchdog.observe(row, now=0.0) is None
    assert watchdog.observe(row, now=10.0) is None
    assert watchdog.observe(row, now=20.0) is None


def test_watchdog_pauses_mining_timeout_while_combat_suspends_transaction():
    watchdog = RouteLiveWatchdog(mining_timeout_seconds=5.0)
    active = _row()
    active["mining"] = {"phase": "intercept"}
    suspended = _row()
    suspended["combat"] = {"active": True}
    suspended["mining"] = {"phase": "suspended"}

    assert watchdog.observe(active, now=0.0) is None
    assert watchdog.observe(suspended, now=3.0) is None
    assert watchdog.observe(suspended, now=20.0) is None
    assert watchdog.observe(active, now=21.0) is None


def test_watchdog_stops_skip_storm():
    watchdog = RouteLiveWatchdog()

    assert watchdog.observe(_row(skipped=1), now=0.0) is None
    decision = watchdog.observe(_row(skipped=11), now=30.0)

    assert decision is not None
    assert decision.reason == "route_skip_storm"


def test_overnight_supervisor_stops_after_death_recovery_failure():
    assert _termination_requires_review({"termination_reason": "death_recovery_failed"})
    assert not _termination_requires_review({"termination_reason": "max_wall_time"})
