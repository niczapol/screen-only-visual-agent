from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from vision_bot.core.commands import Command, ControlIntent
from vision_bot.core.world import WorldSnapshot
from vision_bot.engine.kernel import DeterministicKernel, KernelStep


@dataclass(frozen=True)
class ReplayResult:
    steps: tuple[KernelStep, ...]
    digest: str
    state_counts: dict[str, int]


@dataclass(frozen=True)
class ShadowDifference:
    frame_id: int
    field: str
    baseline: object
    candidate: object


class ReplayRunner:
    def __init__(self, kernel: DeterministicKernel) -> None:
        self.kernel = kernel

    def run(self, snapshots: Iterable[WorldSnapshot]) -> ReplayResult:
        self.kernel.reset()
        steps = tuple(self.kernel.step(snapshot) for snapshot in snapshots)
        payload = [_step_signature(step) for step in steps]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        state_counts: dict[str, int] = {}
        for step in steps:
            state = step.decision.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        return ReplayResult(
            steps=steps,
            digest=hashlib.sha256(encoded).hexdigest(),
            state_counts=state_counts,
        )


def compare_steps(
    baseline: Iterable[KernelStep],
    candidate: Iterable[KernelStep],
) -> tuple[ShadowDifference, ...]:
    baseline_steps = tuple(baseline)
    candidate_steps = tuple(candidate)
    differences: list[ShadowDifference] = []
    if len(baseline_steps) != len(candidate_steps):
        differences.append(
            ShadowDifference(
                frame_id=-1,
                field="step_count",
                baseline=len(baseline_steps),
                candidate=len(candidate_steps),
            )
        )
    for baseline_step, candidate_step in zip(baseline_steps, candidate_steps):
        frame_id = baseline_step.frame_id
        if frame_id != candidate_step.frame_id:
            differences.append(
                ShadowDifference(
                    frame_id=frame_id,
                    field="frame_id",
                    baseline=frame_id,
                    candidate=candidate_step.frame_id,
                )
            )
        baseline_signature = _step_signature(baseline_step)
        candidate_signature = _step_signature(candidate_step)
        for field in ("state", "owner", "reason", "intent"):
            if baseline_signature[field] != candidate_signature[field]:
                differences.append(
                    ShadowDifference(
                        frame_id=frame_id,
                        field=field,
                        baseline=baseline_signature[field],
                        candidate=candidate_signature[field],
                    )
                )
    return tuple(differences)


def _step_signature(step: KernelStep) -> dict[str, object]:
    return {
        "frame_id": step.frame_id,
        "state": step.decision.state.value,
        "owner": step.decision.owner.value if step.decision.owner is not None else None,
        "reason": step.decision.reason,
        "intent": _intent_signature(step.intent),
    }


def _intent_signature(intent: ControlIntent | None) -> object:
    if intent is None:
        return None
    return {
        "owner": intent.owner.value,
        "reason": intent.reason,
        "commands": [_command_signature(command) for command in intent.commands],
    }


def _command_signature(command: Command) -> dict[str, object]:
    return {
        "kind": command.kind.value,
        "group": command.group.value,
        "reason": command.reason,
        "not_before": command.not_before,
        "deadline": command.deadline,
        "key": command.key,
        "duration": command.duration,
        "point": command.point,
        "point_space": command.point_space.value,
        "delta_x": command.delta_x,
        "mouse_button": (
            command.mouse_button.value if command.mouse_button is not None else None
        ),
        "exclusive": command.exclusive,
    }
