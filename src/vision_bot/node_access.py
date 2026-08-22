from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vision_bot.route_planner import NodeAccessOption, NodeAccessPlan


class NodeAccessState(str, Enum):
    IDLE = "idle"
    INBOUND = "inbound"
    ORE_CHECK = "ore_check"
    BACKTRACK = "backtrack"
    RESUME = "resume"
    EXHAUSTED = "exhausted"
    COMPLETE = "complete"


@dataclass
class NodeAccessController:
    """Advance through a precomputed node-access plan without pathfinding."""

    state: NodeAccessState = NodeAccessState.IDLE
    plan: NodeAccessPlan | None = None
    _ordered_options: tuple[NodeAccessOption, ...] = field(default=(), init=False, repr=False)
    _option_index: int = field(default=0, init=False, repr=False)
    _leg_coords: tuple[int, ...] = field(default=(), init=False, repr=False)
    _leg_cursor: int = field(default=0, init=False, repr=False)
    _reached_inbound: int = field(default=0, init=False, repr=False)

    def start(self, plan: NodeAccessPlan) -> None:
        primary = [option for option in plan.options if option.rank == plan.primary_option_rank]
        if len(primary) != 1:
            raise ValueError("Node access plan must contain exactly one primary option")
        fallbacks = sorted(
            (option for option in plan.options if option.rank != plan.primary_option_rank),
            key=lambda option: option.rank,
        )
        for option in (primary[0], *fallbacks):
            if option.return_coords != tuple(reversed(option.inbound_coords)):
                raise ValueError("Node access return path must reverse its inbound path")
        self.plan = plan
        self._ordered_options = (primary[0], *fallbacks)
        self._option_index = 0
        self._activate_inbound()

    @property
    def current_option(self) -> NodeAccessOption | None:
        if not self._ordered_options or self._option_index >= len(self._ordered_options):
            return None
        return self._ordered_options[self._option_index]

    @property
    def current_target_coord(self) -> int | None:
        if self._leg_cursor >= len(self._leg_coords):
            return None
        return self._leg_coords[self._leg_cursor]

    @property
    def attempted_option_count(self) -> int:
        if self.state == NodeAccessState.IDLE:
            return 0
        return min(self._option_index + 1, len(self._ordered_options))

    @property
    def resume_route_index(self) -> int | None:
        return self.plan.resume_route_index if self.plan is not None else None

    def mark_target_reached(self) -> None:
        if self.current_target_coord is None:
            raise RuntimeError(f"No target is active in state {self.state.value}")
        self._leg_cursor += 1

        if self.state == NodeAccessState.INBOUND:
            self._reached_inbound = self._leg_cursor
            if self._leg_cursor >= len(self._leg_coords):
                self.state = NodeAccessState.ORE_CHECK
                self._clear_leg()
            return

        if self.state == NodeAccessState.BACKTRACK:
            if self._leg_cursor >= len(self._leg_coords):
                self._activate_next_option()
            return

        if self.state == NodeAccessState.RESUME:
            if self._leg_cursor >= len(self._leg_coords):
                self.state = NodeAccessState.COMPLETE
                self._clear_leg()
            return

        raise RuntimeError(f"Targets are not valid in state {self.state.value}")

    def mark_access_blocked(self) -> None:
        if self.state != NodeAccessState.INBOUND:
            raise RuntimeError("Access can only be marked blocked during inbound movement")

        option = self._require_current_option()
        if self._reached_inbound == 0:
            self._activate_next_option()
            return

        # Return only over the part of the inbound breadcrumb actually reached.
        self.state = NodeAccessState.BACKTRACK
        self._leg_coords = option.return_coords[-self._reached_inbound :]
        self._leg_cursor = 0

    def resume_after_ore_check(self) -> None:
        """Resume after mined or absent ore; the caller owns cooldown semantics."""
        if self.state != NodeAccessState.ORE_CHECK:
            raise RuntimeError("Ore check can only finish after reaching an approach point")
        option = self._require_current_option()
        self.state = NodeAccessState.RESUME
        self._leg_coords = option.resume_coords
        self._leg_cursor = 0

    def reset(self) -> None:
        self.state = NodeAccessState.IDLE
        self.plan = None
        self._ordered_options = ()
        self._option_index = 0
        self._clear_leg()
        self._reached_inbound = 0

    def _activate_inbound(self) -> None:
        option = self._require_current_option()
        self.state = NodeAccessState.INBOUND
        self._leg_coords = option.inbound_coords
        self._leg_cursor = 0
        self._reached_inbound = 0

    def _activate_next_option(self) -> None:
        self._option_index += 1
        if self._option_index >= len(self._ordered_options):
            self.state = NodeAccessState.EXHAUSTED
            self._clear_leg()
            self._reached_inbound = 0
            return
        self._activate_inbound()

    def _require_current_option(self) -> NodeAccessOption:
        option = self.current_option
        if option is None:
            raise RuntimeError("No node access option is active")
        return option

    def _clear_leg(self) -> None:
        self._leg_coords = ()
        self._leg_cursor = 0
