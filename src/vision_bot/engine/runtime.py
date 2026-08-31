from __future__ import annotations

from dataclasses import dataclass

from vision_bot.core.commands import CommandGroup
from vision_bot.core.evidence import EvidenceState
from vision_bot.core.world import WorldSnapshot
from vision_bot.engine.input_executor import InputEvent, InputExecutor
from vision_bot.engine.kernel import DeterministicKernel, KernelStep


@dataclass(frozen=True)
class RuntimeStep:
    kernel: KernelStep
    input_events: tuple[InputEvent, ...]
    live_input_enabled: bool
    pending_input_actions: int = 0
    held_keys: tuple[str, ...] = ()
    held_buttons: tuple[str, ...] = ()
    actuation_delay_seconds: float = 0.0
    permitted_groups: tuple[str, ...] = ()


class V09Runtime:
    """Bridge the pure kernel to one optional, explicitly enabled executor."""

    def __init__(
        self,
        kernel: DeterministicKernel,
        *,
        executor: InputExecutor | None = None,
        live_input_enabled: bool = False,
    ) -> None:
        if live_input_enabled and executor is None:
            raise ValueError("live input requires an InputExecutor")
        self.kernel = kernel
        self.executor = executor
        self.live_input_enabled = bool(live_input_enabled)

    def reset(self, *, now: float = 0.0) -> None:
        self.kernel.reset()
        if self.executor is not None:
            self.executor.preempt_all(now=now, reason="runtime_reset")
            self.executor.set_input_permitted(False, now=now)

    def step(
        self,
        snapshot: WorldSnapshot,
        *,
        now: float | None = None,
        actuation_now: float | None = None,
        permitted_groups: frozenset[CommandGroup] | None = None,
    ) -> RuntimeStep:
        current_time = snapshot.captured_at if now is None else float(now)
        execution_time = current_time if actuation_now is None else float(actuation_now)
        kernel_step = self.kernel.step(snapshot, now=current_time)
        if self.executor is None:
            return RuntimeStep(kernel_step, (), False)

        input_ready = bool(
            snapshot.input_ready.state is EvidenceState.PRESENT
            and snapshot.input_ready.value
            and snapshot.input_ready.is_fresh(now=current_time, max_age_seconds=0.50)
        )
        scope = (
            frozenset(CommandGroup)
            if permitted_groups is None
            else frozenset(permitted_groups)
        )
        owner_permitted = (
            kernel_step.decision.owner is not None
            and kernel_step.decision.owner in scope
        )
        permitted = self.live_input_enabled and input_ready and owner_permitted
        self.executor.set_input_permitted(permitted, now=execution_time)
        for group in kernel_step.decision.preempt_groups:
            self.executor.cancel_group(
                group,
                now=execution_time,
                reason=f"supervisor_owner_{kernel_step.decision.owner}",
            )
        if permitted and kernel_step.intent is not None:
            self.executor.submit(
                kernel_step.intent,
                now=execution_time,
                reference_time=current_time,
            )
        events = (
            self.executor.drain_events()
            if self.executor.background_running
            else self.executor.run_due(now=execution_time)
        )
        return RuntimeStep(
            kernel_step,
            events,
            permitted,
            pending_input_actions=self.executor.pending_count,
            held_keys=tuple(sorted(self.executor.held_keys)),
            held_buttons=tuple(sorted(button.value for button in self.executor.held_buttons)),
            actuation_delay_seconds=max(0.0, execution_time - snapshot.captured_at),
            permitted_groups=tuple(sorted(group.value for group in scope)),
        )
