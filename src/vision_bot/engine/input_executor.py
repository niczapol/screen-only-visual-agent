from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)


class InputBackend(Protocol):
    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def move_cursor(self, x: int, y: int) -> None: ...

    def mouse_down(self, button: MouseButton) -> None: ...

    def mouse_up(self, button: MouseButton) -> None: ...

    def move_mouse_relative(self, delta_x: int, delta_y: int) -> None: ...


@dataclass(frozen=True)
class InputEvent:
    happened_at: float
    group: CommandGroup
    operation: str
    value: str | int | tuple[int, int] | None
    reason: str
    executed: bool


@dataclass(order=True)
class _ScheduledAction:
    due_at: float
    sequence: int
    group: CommandGroup = field(compare=False)
    generation: int = field(compare=False)
    operation: str = field(compare=False)
    value: str | int | tuple[int, int] | None = field(compare=False)
    reason: str = field(compare=False)
    deadline: float | None = field(compare=False, default=None)
    lane: str | None = field(compare=False, default=None)


class InputExecutor:
    """Single, non-blocking authority for all keyboard and mouse input."""

    def __init__(
        self,
        backend: InputBackend,
        *,
        drag_step_seconds: float = 0.012,
        cursor_settle_seconds: float = 0.12,
        client_to_screen: Callable[[int, int], tuple[int, int]] | None = None,
    ) -> None:
        self.backend = backend
        self.drag_step_seconds = max(0.004, float(drag_step_seconds))
        self.cursor_settle_seconds = max(0.0, float(cursor_settle_seconds))
        self.client_to_screen = client_to_screen or (lambda x, y: (x, y))
        self._lock = threading.RLock()
        self._queue: list[_ScheduledAction] = []
        self._sequence = 0
        self._generations = {group: 0 for group in CommandGroup}
        self._held_keys: dict[str, CommandGroup] = {}
        self._held_buttons: dict[MouseButton, CommandGroup] = {}
        self._held_button_lanes: dict[MouseButton, str | None] = {}
        self._input_permitted = False
        self.events: list[InputEvent] = []
        self._pump_stop: threading.Event | None = None
        self._pump_thread: threading.Thread | None = None

    @property
    def held_keys(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held_keys)

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        with self._lock:
            return frozenset(self._held_buttons)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def background_running(self) -> bool:
        with self._lock:
            return bool(self._pump_thread is not None and self._pump_thread.is_alive())

    def start_background(self) -> None:
        with self._lock:
            if self._pump_thread is not None and self._pump_thread.is_alive():
                return
            stop = threading.Event()
            thread = threading.Thread(
                target=self._background_loop,
                args=(stop,),
                name="v09-input-executor",
                daemon=True,
            )
            self._pump_stop = stop
            self._pump_thread = thread
            thread.start()

    def stop_background(self, *, now: float, reason: str) -> None:
        with self._lock:
            stop = self._pump_stop
            thread = self._pump_thread
            if stop is not None:
                stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            self._pump_stop = None
            self._pump_thread = None
            self._input_permitted = False
            self._preempt_all_locked(now=float(now), reason=reason)

    def _background_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            with self._lock:
                self._run_due_locked(now=time.monotonic())
            stop.wait(self.drag_step_seconds)

    def set_input_permitted(self, permitted: bool, *, now: float) -> None:
        with self._lock:
            permitted = bool(permitted)
            if self._input_permitted and not permitted:
                self._preempt_all_locked(now=now, reason="input_permission_revoked")
            self._input_permitted = permitted

    def submit(
        self,
        intent: ControlIntent,
        *,
        now: float,
        reference_time: float | None = None,
    ) -> None:
        with self._lock:
            # Steering is a latest-observation control lane.  A new controller
            # decision must replace the unexecuted tail of its previous drag;
            # it must never queue behind stale yaw from older frames.
            self._cancel_drag_lane(
                intent.owner,
                now=float(now),
                reason="drag_superseded_by_latest_intent",
            )
            offset = float(now) - (
                float(now) if reference_time is None else float(reference_time)
            )
            cursor_ready_at: float | None = None
            for command in intent.commands:
                earliest_due_at = (
                    cursor_ready_at
                    if command.kind is CommandKind.CLICK
                    else None
                )
                due_at = self._schedule_command(
                    command,
                    now=float(now),
                    time_offset=offset,
                    earliest_due_at=earliest_due_at,
                )
                if command.kind is CommandKind.MOVE_CURSOR:
                    cursor_ready_at = due_at + self.cursor_settle_seconds

    def cancel_group(self, group: CommandGroup, *, now: float, reason: str) -> None:
        with self._lock:
            self._cancel_group_locked(group, now=now, reason=reason)

    def _cancel_group_locked(self, group: CommandGroup, *, now: float, reason: str) -> None:
        self._generations[group] += 1
        for key, owner in tuple(self._held_keys.items()):
            if owner is group:
                self.backend.key_up(key)
                self._held_keys.pop(key, None)
                self._record(now, group, "key_up", key, reason, executed=True)
        for button, owner in tuple(self._held_buttons.items()):
            if owner is group:
                self.backend.mouse_up(button)
                self._held_buttons.pop(button, None)
                self._held_button_lanes.pop(button, None)
                self._record(now, group, "mouse_up", button.value, reason, executed=True)

    def preempt_all(self, *, now: float, reason: str) -> None:
        with self._lock:
            self._preempt_all_locked(now=now, reason=reason)

    def _preempt_all_locked(self, *, now: float, reason: str) -> None:
        for group in CommandGroup:
            self._generations[group] += 1
        for key, owner in tuple(self._held_keys.items()):
            self.backend.key_up(key)
            self._record(now, owner, "key_up", key, reason, executed=True)
        self._held_keys.clear()
        for button, owner in tuple(self._held_buttons.items()):
            self.backend.mouse_up(button)
            self._record(now, owner, "mouse_up", button.value, reason, executed=True)
        self._held_buttons.clear()
        self._held_button_lanes.clear()

    def run_due(self, *, now: float) -> tuple[InputEvent, ...]:
        with self._lock:
            self._run_due_locked(now=float(now))
            return self._drain_events_locked()

    def drain_events(self) -> tuple[InputEvent, ...]:
        with self._lock:
            return self._drain_events_locked()

    def _run_due_locked(self, *, now: float) -> None:
        while self._queue and self._queue[0].due_at <= now:
            action = heapq.heappop(self._queue)
            if action.generation != self._generations[action.group]:
                self._record(
                    now,
                    action.group,
                    action.operation,
                    action.value,
                    "cancelled_generation",
                    executed=False,
                )
                continue
            if action.deadline is not None and now > action.deadline:
                self._record(
                    now,
                    action.group,
                    action.operation,
                    action.value,
                    "expired_deadline",
                    executed=False,
                )
                continue
            self._execute(action, now=now)

    def _drain_events_locked(self) -> tuple[InputEvent, ...]:
        events = tuple(self.events)
        self.events.clear()
        return events

    def _cancel_drag_lane(
        self,
        group: CommandGroup,
        *,
        now: float,
        reason: str,
    ) -> None:
        if self._queue:
            retained = [
                action
                for action in self._queue
                if not (action.group is group and action.lane == "drag")
            ]
            if len(retained) != len(self._queue):
                self._queue = retained
                heapq.heapify(self._queue)
        for button, owner in tuple(self._held_buttons.items()):
            if owner is group and self._held_button_lanes.get(button) == "drag":
                self.backend.mouse_up(button)
                self._held_buttons.pop(button, None)
                self._held_button_lanes.pop(button, None)
                self._record(
                    now,
                    group,
                    "mouse_up",
                    button.value,
                    reason,
                    executed=True,
                )

    def _schedule_command(
        self,
        command: Command,
        *,
        now: float,
        time_offset: float = 0.0,
        earliest_due_at: float | None = None,
    ) -> float:
        due_at = max(
            now,
            float(command.not_before) + time_offset,
            now if earliest_due_at is None else float(earliest_due_at),
        )
        deadline = (
            None
            if command.deadline is None
            else float(command.deadline) + time_offset
        )
        if deadline is not None and due_at > deadline:
            self._record(
                now,
                command.group,
                command.kind.value,
                command.key,
                "expired_before_submit",
                executed=False,
            )
            return due_at
        if command.exclusive:
            self._push(
                due_at,
                command,
                operation="release_all",
                value=None,
                deadline=deadline,
            )
        if command.kind is CommandKind.HOLD_KEY:
            self._push(
                due_at,
                command,
                operation="key_down",
                value=command.key,
                deadline=deadline,
            )
        elif command.kind is CommandKind.RELEASE_KEY:
            self._push(
                due_at,
                command,
                operation="key_up",
                value=command.key,
                deadline=deadline,
            )
        elif command.kind is CommandKind.TAP_KEY:
            self._push(
                due_at,
                command,
                operation="key_down",
                value=command.key,
                deadline=deadline,
            )
            self._push(
                due_at + max(0.01, command.duration),
                command,
                operation="key_up",
                value=command.key,
                deadline=deadline,
            )
        elif command.kind is CommandKind.MOVE_CURSOR:
            self._push(
                due_at,
                command,
                operation=f"move_cursor_{command.point_space.value}",
                value=command.point,
                deadline=deadline,
            )
        elif command.kind is CommandKind.CLICK:
            button = command.mouse_button
            assert button is not None
            self._push(
                due_at,
                command,
                operation="mouse_down",
                value=button.value,
                deadline=deadline,
            )
            self._push(
                due_at + max(0.01, command.duration),
                command,
                operation="mouse_up",
                value=button.value,
                deadline=deadline,
            )
        elif command.kind is CommandKind.DRAG_RELATIVE:
            button = command.mouse_button
            assert button is not None
            duration = max(self.drag_step_seconds, command.duration)
            steps = max(1, int(round(duration / self.drag_step_seconds)))
            self._push(
                due_at,
                command,
                operation="mouse_down",
                value=button.value,
                lane="drag",
                deadline=deadline,
            )
            sent = 0
            for index in range(steps):
                target = round(command.delta_x * (index + 1) / steps)
                delta = target - sent
                sent = target
                if delta:
                    self._push(
                        due_at + duration * (index + 1) / steps,
                        command,
                        operation="mouse_move_relative",
                        value=delta,
                        lane="drag",
                        deadline=deadline,
                    )
            self._push(
                due_at + duration,
                command,
                operation="mouse_up",
                value=button.value,
                lane="drag",
                deadline=deadline,
            )
        elif command.kind is CommandKind.RELEASE_ALL:
            self._push(
                due_at,
                command,
                operation="release_all",
                value=None,
                deadline=deadline,
            )
        else:
            raise ValueError(f"unsupported command kind: {command.kind}")
        return due_at

    def _push(
        self,
        due_at: float,
        command: Command,
        *,
        operation: str,
        value: str | int | tuple[int, int] | None,
        lane: str | None = None,
        deadline: float | None = None,
    ) -> None:
        self._sequence += 1
        heapq.heappush(
            self._queue,
            _ScheduledAction(
                due_at=due_at,
                sequence=self._sequence,
                group=command.group,
                generation=self._generations[command.group],
                operation=operation,
                value=value,
                reason=command.reason,
                deadline=deadline,
                lane=lane,
            ),
        )

    def _execute(self, action: _ScheduledAction, *, now: float) -> None:
        if action.operation == "release_all":
            self._release_all_resources(now=now, reason=action.reason)
            self._record(now, action.group, action.operation, None, action.reason, True)
            return
        if not self._input_permitted:
            self._record(
                now,
                action.group,
                action.operation,
                action.value,
                "input_not_permitted",
                executed=False,
            )
            return
        if action.operation == "key_down":
            key = str(action.value).upper()
            current_owner = self._held_keys.get(key)
            if current_owner is None:
                self.backend.key_down(key)
                self._held_keys[key] = action.group
            self._record(now, action.group, action.operation, key, action.reason, True)
        elif action.operation == "key_up":
            key = str(action.value).upper()
            if key in self._held_keys:
                self.backend.key_up(key)
                self._held_keys.pop(key, None)
            self._record(now, action.group, action.operation, key, action.reason, True)
        elif action.operation in {"move_cursor_client", "move_cursor_screen"}:
            point = action.value
            assert isinstance(point, tuple)
            screen_point = (
                self.client_to_screen(int(point[0]), int(point[1]))
                if action.operation == "move_cursor_client"
                else (int(point[0]), int(point[1]))
            )
            self.backend.move_cursor(*screen_point)
            self._record(now, action.group, action.operation, point, action.reason, True)
        elif action.operation == "mouse_down":
            button = MouseButton(str(action.value))
            if button not in self._held_buttons:
                self.backend.mouse_down(button)
                self._held_buttons[button] = action.group
                self._held_button_lanes[button] = action.lane
            self._record(now, action.group, action.operation, button.value, action.reason, True)
        elif action.operation == "mouse_up":
            button = MouseButton(str(action.value))
            if button in self._held_buttons:
                self.backend.mouse_up(button)
                self._held_buttons.pop(button, None)
                self._held_button_lanes.pop(button, None)
            self._record(now, action.group, action.operation, button.value, action.reason, True)
        elif action.operation == "mouse_move_relative":
            delta_x = int(action.value)
            self.backend.move_mouse_relative(delta_x, 0)
            self._record(now, action.group, action.operation, delta_x, action.reason, True)
        else:
            raise ValueError(f"unsupported scheduled operation: {action.operation}")

    def _release_all_resources(self, *, now: float, reason: str) -> None:
        for key, owner in tuple(self._held_keys.items()):
            self.backend.key_up(key)
            self._record(now, owner, "key_up", key, reason, True)
        self._held_keys.clear()
        for button, owner in tuple(self._held_buttons.items()):
            self.backend.mouse_up(button)
            self._record(now, owner, "mouse_up", button.value, reason, True)
        self._held_buttons.clear()
        self._held_button_lanes.clear()

    def _record(
        self,
        happened_at: float,
        group: CommandGroup,
        operation: str,
        value: str | int | tuple[int, int] | None,
        reason: str,
        executed: bool,
    ) -> None:
        self.events.append(
            InputEvent(
                happened_at=float(happened_at),
                group=group,
                operation=operation,
                value=value,
                reason=reason,
                executed=bool(executed),
            )
        )
