from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandGroup(str, Enum):
    SYSTEM = "system"
    RECOVERY = "recovery"
    COMBAT = "combat"
    LOOT = "loot"
    MINING = "mining"
    MOUNT = "mount"
    TRAVEL = "travel"


class CommandKind(str, Enum):
    HOLD_KEY = "hold_key"
    RELEASE_KEY = "release_key"
    TAP_KEY = "tap_key"
    MOVE_CURSOR = "move_cursor"
    CLICK = "click"
    DRAG_RELATIVE = "drag_relative"
    RELEASE_ALL = "release_all"


class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class CoordinateSpace(str, Enum):
    CLIENT = "client"
    SCREEN = "screen"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    group: CommandGroup
    reason: str
    not_before: float = 0.0
    deadline: float | None = None
    key: str | None = None
    duration: float = 0.0
    point: tuple[int, int] | None = None
    point_space: CoordinateSpace = CoordinateSpace.CLIENT
    delta_x: int = 0
    mouse_button: MouseButton | None = None
    exclusive: bool = False

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("command reason must not be empty")
        if self.deadline is not None and self.deadline < self.not_before:
            raise ValueError("command deadline precedes not_before")
        if self.duration < 0.0:
            raise ValueError("command duration must be non-negative")
        if self.kind in {
            CommandKind.HOLD_KEY,
            CommandKind.RELEASE_KEY,
            CommandKind.TAP_KEY,
        } and not self.key:
            raise ValueError(f"{self.kind.value} requires a key")
        if self.kind is CommandKind.MOVE_CURSOR and self.point is None:
            raise ValueError("move_cursor requires a point")
        if self.kind in {CommandKind.CLICK, CommandKind.DRAG_RELATIVE} and self.mouse_button is None:
            raise ValueError(f"{self.kind.value} requires a mouse button")


@dataclass(frozen=True)
class ControlIntent:
    owner: CommandGroup
    commands: tuple[Command, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("intent reason must not be empty")
        if any(command.group is not self.owner for command in self.commands):
            raise ValueError("every command must belong to the intent owner")
