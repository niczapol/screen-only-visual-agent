from vision_bot.control_arbitration import (
    OperationalControlOwner,
    OperationalControlSignals,
    select_operational_control_owner,
)


def test_route_owns_control_without_interruptions() -> None:
    assert (
        select_operational_control_owner(OperationalControlSignals())
        == OperationalControlOwner.ROUTE
    )


def test_operational_control_priority_is_explicit() -> None:
    signals = OperationalControlSignals(
        combat_heal_casting=True,
        combat_blocks_route=True,
        mining_active=True,
        mount_requested=True,
    )
    assert select_operational_control_owner(signals) == OperationalControlOwner.COMBAT_HEAL

    signals = OperationalControlSignals(
        combat_blocks_route=True,
        post_combat_loot_active=True,
        mining_active=True,
        mount_requested=True,
    )
    assert (
        select_operational_control_owner(signals)
        == OperationalControlOwner.POST_COMBAT_LOOT
    )

    signals = OperationalControlSignals(
        post_combat_loot_active=True,
        mining_active=True,
        mount_requested=True,
    )
    assert (
        select_operational_control_owner(signals)
        == OperationalControlOwner.POST_COMBAT_LOOT
    )

    signals = OperationalControlSignals(mining_active=True, mount_requested=True)
    assert select_operational_control_owner(signals) == OperationalControlOwner.MINING

    signals = OperationalControlSignals(mount_requested=True)
    assert select_operational_control_owner(signals) == OperationalControlOwner.MOUNT
