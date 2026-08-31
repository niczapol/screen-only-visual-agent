from __future__ import annotations

from vision_bot.core.commands import Command, CommandGroup, CommandKind, ControlIntent
from vision_bot.core.evidence import EvidenceState
from vision_bot.core.world import CombatView, WorldSnapshot
from vision_bot.engine.supervisor import ControlDecision
from vision_bot.post_combat_loot import PostCombatLootController


class LootController:
    """v0.9 intent adapter around the accepted pure loot state machine."""

    def __init__(self, project_config: dict) -> None:
        self.project_config = project_config
        self.controller = PostCombatLootController(project_config)
        self.tap_seconds = 0.06

    @property
    def active(self) -> bool:
        return self.controller.active

    def reset(self) -> None:
        self.controller = PostCombatLootController(self.project_config)

    def plan(
        self,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> ControlIntent | None:
        combat = snapshot.combat.present_value(now=now, max_age_seconds=0.75)
        loot_pending = _observed_bool(snapshot, now=now)
        if not self.controller.active and loot_pending:
            self.controller.arm(
                now=now,
                in_combat=bool(isinstance(combat, CombatView) and combat.active),
            )
        if not self.controller.active:
            return None
        view = combat if isinstance(combat, CombatView) else CombatView(active=False)
        loot_decision = self.controller.observe(
            now=now,
            dead_hostile_target_visible=view.dead_hostile_target,
            loot_opened_visible=view.loot_opened,
            target_is_attacker_visible=view.target_is_attacker,
            control_available=True,
        )
        if loot_decision.input_key is None:
            return None
        return ControlIntent(
            CommandGroup.LOOT,
            (
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.LOOT,
                    key=loot_decision.input_key,
                    duration=self.tap_seconds,
                    reason=loot_decision.action,
                    exclusive=True,
                    deadline=now + self.tap_seconds + 0.30,
                ),
            ),
            loot_decision.action,
        )


def _observed_bool(snapshot: WorldSnapshot, *, now: float) -> bool:
    evidence = snapshot.loot_pending
    if not evidence.is_fresh(now=now, max_age_seconds=0.75):
        return False
    if evidence.state is EvidenceState.PRESENT:
        return bool(evidence.value)
    return False
