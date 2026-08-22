from vision_bot.post_combat_loot import (
    PostCombatLootController,
    PostCombatLootPhase,
)


def _controller(enabled=True):
    return PostCombatLootController(
        {
            "post_combat_loot": {
                "enabled": enabled,
                "interact_key": "G",
                "target_last_key": "F9",
                "acquisition_timeout_seconds": 1.5,
                "retarget_settle_seconds": 0.35,
                "confirmation_timeout_seconds": 1.5,
                "marker_clear_timeout_seconds": 1.0,
                "max_interactions": 2,
            }
        }
    )


def test_disabled_loot_controller_cannot_arm():
    controller = _controller(enabled=False)

    assert not controller.arm(now=1.0)
    assert not controller.active


def test_loot_controller_requires_visible_confirmation():
    controller = _controller()
    assert controller.arm(now=10.0)

    interact = controller.observe(now=10.2, dead_hostile_target_visible=True)
    verifying = controller.observe(now=10.5, dead_hostile_target_visible=True)
    confirmed = controller.observe(
        now=10.6,
        dead_hostile_target_visible=True,
        loot_opened_visible=True,
    )

    assert interact.input_key == "G"
    assert verifying.active and verifying.input_key is None
    assert confirmed.active
    assert controller.phase == PostCombatLootPhase.MARKER_CLEAR


def test_loot_controller_targets_previous_corpse_and_loots_two_targets():
    controller = _controller()
    controller.arm(now=10.0)

    retarget = controller.observe(now=10.0, dead_hostile_target_visible=False)
    first_interact = controller.observe(now=10.1, dead_hostile_target_visible=True)
    first_confirm = controller.observe(
        now=10.2,
        dead_hostile_target_visible=True,
        loot_opened_visible=True,
    )
    second_retarget = controller.observe(
        now=10.9,
        dead_hostile_target_visible=True,
        loot_opened_visible=False,
    )
    second_interact = controller.observe(now=11.0, dead_hostile_target_visible=True)
    complete = controller.observe(
        now=11.1,
        dead_hostile_target_visible=True,
        loot_opened_visible=True,
    )

    assert retarget.input_key == "F9"
    assert first_interact.input_key == "G"
    assert first_confirm.active
    assert second_retarget.input_key == "F9"
    assert second_interact.input_key == "G"
    assert not complete.active
    outcome = controller.consume_outcome()
    assert outcome is not None
    assert outcome.attempted
    assert outcome.reason == "loot_confirmed"
    assert outcome.interactions == 2
    assert outcome.confirmed_loots == 2


def test_loot_controller_interacts_after_target_last_without_dead_marker():
    controller = _controller()
    controller.arm(now=3.0)

    retarget = controller.observe(now=3.0, dead_hostile_target_visible=False)
    interact = controller.observe(now=3.4, dead_hostile_target_visible=False)
    second_retarget = controller.observe(
        now=4.91,
        dead_hostile_target_visible=False,
    )
    second_interact = controller.observe(
        now=5.3,
        dead_hostile_target_visible=False,
    )
    decision = controller.observe(now=6.81, dead_hostile_target_visible=False)

    assert retarget.input_key == "F9"
    assert interact.input_key == "G"
    assert second_retarget.input_key == "F9"
    assert second_interact.input_key == "G"
    assert not decision.active
    assert controller.phase == PostCombatLootPhase.IDLE
    outcome = controller.consume_outcome()
    assert outcome is not None
    assert outcome.attempted
    assert outcome.reason == "loot_not_confirmed"
    assert outcome.interactions == 2


def test_loot_controller_does_not_advance_while_control_is_preempted():
    controller = _controller()
    controller.arm(now=3.0, in_combat=True)

    waiting = controller.observe(
        now=3.0,
        dead_hostile_target_visible=False,
        control_available=False,
    )

    assert waiting.action == "post_combat_loot_wait_control"
    assert waiting.active
    assert waiting.input_key is None
    assert controller.phase == PostCombatLootPhase.ACQUIRE
    assert controller.retarget_requests == 0

    retarget = controller.observe(
        now=5.1,
        dead_hostile_target_visible=False,
        control_available=True,
    )
    assert retarget.input_key == "F9"


def test_loot_controller_cancels_when_combat_reenters():
    controller = _controller()
    controller.arm(now=2.0)

    controller.cancel(now=2.2, reason="combat_reentered")

    assert not controller.active
    assert controller.consume_outcome().reason == "combat_reentered"


def test_loot_controller_reports_unconfirmed_interaction():
    controller = _controller()
    controller.arm(now=1.0)
    controller.observe(now=1.0, dead_hostile_target_visible=True)
    retarget = controller.observe(now=2.6, dead_hostile_target_visible=True)
    controller.observe(now=2.7, dead_hostile_target_visible=True)
    complete = controller.observe(now=4.3, dead_hostile_target_visible=True)

    assert retarget.input_key == "F9"
    assert not complete.active
    outcome = controller.consume_outcome()
    assert outcome is not None
    assert outcome.reason == "loot_not_confirmed"
    assert outcome.interactions == 2
    assert outcome.confirmed_loots == 0


def test_in_combat_loot_returns_to_confirmed_attacker_after_looting():
    controller = _controller()
    assert controller.arm(now=10.0, in_combat=True)

    retarget_corpse = controller.observe(
        now=10.0,
        dead_hostile_target_visible=False,
    )
    interact = controller.observe(
        now=10.1,
        dead_hostile_target_visible=True,
    )
    controller.observe(
        now=10.2,
        dead_hostile_target_visible=True,
        loot_opened_visible=True,
    )
    return_target = controller.observe(
        now=10.9,
        dead_hostile_target_visible=True,
        loot_opened_visible=False,
    )
    complete = controller.observe(
        now=11.0,
        dead_hostile_target_visible=False,
        target_is_attacker_visible=True,
    )

    assert retarget_corpse.input_key == "F9"
    assert interact.input_key == "G"
    assert return_target.input_key == "F9"
    assert not complete.active
    outcome = controller.consume_outcome()
    assert outcome is not None
    assert outcome.reason == "combat_loot_confirmed_target_restored"
    assert outcome.confirmed_loots == 1
