from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint


class LifeState(str, Enum):
    ALIVE = "alive"
    DEAD = "dead"
    GHOST = "ghost"
    RESURRECTION_SICKNESS = "resurrection_sickness"


class ModalState(str, Enum):
    CLEAR = "clear"
    DISMISSABLE = "dismissable"
    BLOCKING = "blocking"


class MiningSignal(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    APPROACH = "approach"
    VISUAL_SEARCH = "visual_search"
    INTERACTION_READY = "interaction_ready"
    VERIFYING = "verifying"
    RESUME = "resume"


class RecoverySignal(str, Enum):
    NONE = "none"
    RELEASE_SPIRIT = "release_spirit"
    GHOST_SEARCH = "ghost_search"
    HEALER_TARGETED = "healer_targeted"
    HEALER_DIALOG = "healer_dialog"
    RETURN_TO_LIFE = "return_to_life"
    ACCEPT_RESURRECTION = "accept_resurrection"
    SICKNESS = "sickness"


@dataclass(frozen=True)
class PlayerPose:
    zone_id: int
    position: MapPoint
    heading_degrees: float


@dataclass(frozen=True)
class CombatView:
    active: bool
    hp_fraction: float | None = None
    target_confirmed: bool = False
    outgoing_hit: bool = False
    wrong_facing: bool = False
    out_of_range: bool = False
    attacker_count: int | None = None
    outcome_sequence: int | None = None
    dead_hostile_target: bool = False
    loot_opened: bool = False
    target_is_attacker: bool = False


@dataclass(frozen=True)
class MiningPerception:
    signal: MiningSignal = MiningSignal.IDLE
    candidate_id: int | None = None
    candidate_position: MapPoint | None = None
    minimap_offset_px: tuple[float, float] | None = None
    minimap_centered: bool = False
    bright_minimap_points: tuple[tuple[int, int], ...] = ()
    dark_minimap_points: tuple[tuple[int, int], ...] = ()
    tooltip_ore_type: str | None = None
    world_candidate_points: tuple[tuple[int, int], ...] = ()
    interaction_authority: bool = False
    interaction_point: tuple[int, int] | None = None
    interaction_outcome: str | None = None


@dataclass(frozen=True)
class RecoveryPerception:
    signal: RecoverySignal = RecoverySignal.NONE
    click_point: tuple[int, int] | None = None
    visual_bearing_point: tuple[int, int] | None = None


@dataclass(frozen=True)
class WorldSnapshot:
    """Complete immutable controller input for one captured frame."""

    frame_id: int
    captured_at: float
    input_ready: Evidence[bool]
    pose: Evidence[PlayerPose]
    life: Evidence[LifeState]
    modal: Evidence[ModalState]
    combat: Evidence[CombatView]
    mounted: Evidence[bool]
    mining: Evidence[MiningPerception]
    loot_pending: Evidence[bool]
    forbidden_subzone: Evidence[bool]
    recovery: Evidence[RecoveryPerception]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("snapshot frame_id must be non-negative")
        evidence_items = (
            self.input_ready,
            self.pose,
            self.life,
            self.modal,
            self.combat,
            self.mounted,
            self.mining,
            self.loot_pending,
            self.forbidden_subzone,
            self.recovery,
        )
        if any(item.frame_id > self.frame_id for item in evidence_items):
            raise ValueError("snapshot cannot contain evidence from a future frame")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "captured_at", float(self.captured_at))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
