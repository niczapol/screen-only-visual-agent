from vision_bot.coords import xy_to_coord
from vision_bot.corpse_run import CorpseRunController, CorpseRunPhase


def _config(enabled=True):
    return {
        "corpse_run": {
            "enabled": enabled,
            "waypoint_reached_distance_coord": 0.10,
            "corpse_reached_distance_coord": 0.20,
            "max_total_seconds": 30.0,
        }
    }


def test_corpse_run_is_disabled_by_default_and_cannot_begin():
    controller = CorpseRunController({})
    controller.record_death_position(xy_to_coord(20.0, 20.0))

    assert not controller.begin([], now=0.0)
    assert not controller.active


def test_corpse_run_follows_waypoints_then_requests_reclaim():
    controller = CorpseRunController(_config())
    waypoint = xy_to_coord(10.0, 10.0)
    death = xy_to_coord(11.0, 10.0)
    controller.record_death_position(death)
    assert controller.begin([waypoint], now=2.0)

    moving = controller.observe(
        xy_to_coord(9.0, 10.0),
        now=3.0,
        reclaim_available=False,
    )
    next_leg = controller.observe(
        waypoint,
        now=4.0,
        reclaim_available=False,
    )
    reclaim = controller.observe(
        death,
        now=5.0,
        reclaim_available=True,
    )

    assert moving.target_coord == waypoint
    assert next_leg.target_coord == death
    assert reclaim.request_reclaim
    assert controller.phase == CorpseRunPhase.WAIT_FOR_RECLAIM


def test_corpse_run_completes_only_after_alive_confirmation():
    controller = CorpseRunController(_config())
    death = xy_to_coord(11.0, 10.0)
    controller.record_death_position(death)
    controller.begin([], now=2.0)
    controller.observe(death, now=3.0, reclaim_available=True)

    complete = controller.observe(death, now=4.0, reclaim_available=False, alive=True)

    assert not complete.active
    outcome = controller.consume_outcome()
    assert outcome is not None and outcome.success


def test_corpse_run_times_out_without_movement():
    controller = CorpseRunController(_config())
    death = xy_to_coord(11.0, 10.0)
    controller.record_death_position(death)
    controller.begin([], now=2.0)

    timeout = controller.observe(None, now=32.0, reclaim_available=False)

    assert not timeout.active
    assert controller.consume_outcome().reason == "timeout"
