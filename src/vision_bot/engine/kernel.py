from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from vision_bot.core.commands import CommandGroup, ControlIntent
from vision_bot.core.world import WorldSnapshot
from vision_bot.engine.supervisor import ControlDecision, Supervisor, SupervisorMemory


class DomainController(Protocol):
    @property
    def active(self) -> bool: ...

    def reset(self) -> None: ...

    def plan(
        self,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> ControlIntent | None: ...


@dataclass(frozen=True)
class KernelStep:
    frame_id: int
    decision: ControlDecision
    intent: ControlIntent | None


class DeterministicKernel:
    """Pure owner selection and domain-controller dispatch."""

    def __init__(
        self,
        supervisor: Supervisor,
        controllers: Mapping[CommandGroup, DomainController] | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.controllers = dict(controllers or {})
        self.memory = SupervisorMemory()
        self.last_frame_id = -1

    def reset(self) -> None:
        self.memory = SupervisorMemory()
        self.last_frame_id = -1
        for controller in self.controllers.values():
            reset = getattr(controller, "reset", None)
            if callable(reset):
                reset()

    def step(self, snapshot: WorldSnapshot, *, now: float | None = None) -> KernelStep:
        if snapshot.frame_id <= self.last_frame_id:
            raise ValueError("kernel snapshot frame ids must increase monotonically")
        decision_at = snapshot.captured_at if now is None else float(now)
        active_groups = frozenset(
            group
            for group, controller in self.controllers.items()
            if bool(getattr(controller, "active", False))
        )
        decision = self.supervisor.reduce(
            self.memory,
            snapshot,
            now=decision_at,
            active_groups=active_groups,
        )
        self.memory = decision.memory
        self.last_frame_id = snapshot.frame_id
        intent: ControlIntent | None = None
        if decision.owner is not None:
            controller = self.controllers.get(decision.owner)
            if controller is not None:
                intent = controller.plan(snapshot, decision, now=decision_at)
                if intent is not None and intent.owner is not decision.owner:
                    raise ValueError(
                        "domain controller returned an intent for a non-owning group"
                    )
        return KernelStep(
            frame_id=snapshot.frame_id,
            decision=decision,
            intent=intent,
        )
