from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vision_bot.core.commands import CommandGroup, CommandKind
from vision_bot.core.evidence import EvidenceState
from vision_bot.core.world import WorldSnapshot
from vision_bot.engine.kernel import KernelStep
from vision_bot.engine.supervisor import BotState


@dataclass(frozen=True)
class ReplayInvariantViolation:
    frame_id: int
    invariant: str
    detail: str


def check_replay_invariants(
    snapshots: Iterable[WorldSnapshot],
    steps: Iterable[KernelStep],
) -> tuple[ReplayInvariantViolation, ...]:
    snapshot_items = tuple(snapshots)
    step_items = tuple(steps)
    violations: list[ReplayInvariantViolation] = []
    if len(snapshot_items) != len(step_items):
        violations.append(
            ReplayInvariantViolation(
                frame_id=-1,
                invariant="step_alignment",
                detail=f"snapshots={len(snapshot_items)} steps={len(step_items)}",
            )
        )
    previous_frame_id = -1
    for snapshot, step in zip(snapshot_items, step_items):
        if snapshot.frame_id <= previous_frame_id:
            violations.append(
                ReplayInvariantViolation(
                    snapshot.frame_id,
                    "monotonic_frames",
                    "snapshot frame id did not increase",
                )
            )
        previous_frame_id = snapshot.frame_id
        if step.frame_id != snapshot.frame_id:
            violations.append(
                ReplayInvariantViolation(
                    snapshot.frame_id,
                    "step_alignment",
                    f"step frame={step.frame_id}",
                )
            )
        intent = step.intent
        if intent is not None and intent.owner is not step.decision.owner:
            violations.append(
                ReplayInvariantViolation(
                    snapshot.frame_id,
                    "single_operational_owner",
                    f"decision={step.decision.owner} intent={intent.owner}",
                )
            )
        if step.decision.state in {
            BotState.WAIT_FOR_CLIENT,
            BotState.PAUSED,
            BotState.RESURRECTION_SICKNESS,
            BotState.FAULT,
        } and intent is not None:
            violations.append(
                ReplayInvariantViolation(
                    snapshot.frame_id,
                    "fail_closed_state_has_no_intent",
                    step.decision.state.value,
                )
            )
        if (
            snapshot.input_ready.state is EvidenceState.UNKNOWN
            and intent is not None
        ):
            violations.append(
                ReplayInvariantViolation(
                    snapshot.frame_id,
                    "unknown_input_readiness_has_no_intent",
                    intent.reason,
                )
            )
        if intent is not None and intent.owner is CommandGroup.MINING:
            mining = snapshot.mining.value
            for command in intent.commands:
                if command.kind is CommandKind.CLICK and not (
                    mining is not None and mining.interaction_authority
                ):
                    violations.append(
                        ReplayInvariantViolation(
                            snapshot.frame_id,
                            "mining_click_requires_authority",
                            command.reason,
                        )
                    )
    return tuple(violations)
