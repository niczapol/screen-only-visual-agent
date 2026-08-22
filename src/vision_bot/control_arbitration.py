from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationalControlOwner(str, Enum):
    """Subsystem allowed to issue movement or combat input for this tick."""

    COMBAT_HEAL = "combat_heal"
    COMBAT = "combat"
    POST_COMBAT_LOOT = "post_combat_loot"
    MINING = "mining"
    MOUNT = "mount"
    ROUTE = "route"


@dataclass(frozen=True)
class OperationalControlSignals:
    combat_heal_casting: bool = False
    combat_blocks_route: bool = False
    post_combat_loot_active: bool = False
    mining_active: bool = False
    mount_requested: bool = False


def select_operational_control_owner(
    signals: OperationalControlSignals,
) -> OperationalControlOwner:
    """Resolve mutually exclusive operational control with explicit priority."""

    if signals.combat_heal_casting:
        return OperationalControlOwner.COMBAT_HEAL
    if signals.post_combat_loot_active:
        return OperationalControlOwner.POST_COMBAT_LOOT
    if signals.combat_blocks_route:
        return OperationalControlOwner.COMBAT
    if signals.mining_active:
        return OperationalControlOwner.MINING
    if signals.mount_requested:
        return OperationalControlOwner.MOUNT
    return OperationalControlOwner.ROUTE
