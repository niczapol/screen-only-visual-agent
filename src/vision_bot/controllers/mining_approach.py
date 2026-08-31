from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.core.geometry import MapPoint, WorldPoint, ZoneGeometry
from vision_bot.core.world import MiningPerception, MiningSignal, PlayerPose, WorldSnapshot
from vision_bot.engine.supervisor import ControlDecision
from vision_bot.heading_navigation import signed_heading_error_degrees
from vision_bot.route_planner import MiningNode, NodeAccessOption
from vision_bot.v09_config import MiningApproachConfig, NavigationConfig


class MiningApproachPhase(str, Enum):
    IDLE = "idle"
    IDENTITY_PROBE = "identity_probe"
    ACCESS_APPROACH = "access_approach"
    CONTINUOUS_APPROACH = "continuous_approach"
    VISUAL_SEARCH = "visual_search"
    VERIFY = "verify"
    RETURN_TO_ROUTE = "return_to_route"
    RESUME = "resume"
    FAILED = "failed"


@dataclass(frozen=True)
class MiningTargetEstimate:
    position: MapPoint
    source: str
    disagreement_yards: float | None
    marker_centered: bool


@dataclass(frozen=True)
class MiningApproachState:
    phase: MiningApproachPhase = MiningApproachPhase.IDLE
    target_id: int | None = None
    database_position: MapPoint | None = None
    estimate: MiningTargetEstimate | None = None
    target_ore_type: str | None = None
    pending_marker_point: tuple[int, int] | None = None
    marker_streak: int = 0
    last_marker_frame_id: int | None = None
    access_path: tuple[MapPoint, ...] = ()
    access_index: int = 0
    return_path: tuple[MapPoint, ...] = ()
    return_index: int = 0
    started_at: float | None = None
    phase_entered_at: float | None = None
    last_search_action_at: float | None = None
    search_step: int = 0
    clicked_at: float | None = None
    out_of_range_recoveries: int = 0
    last_interaction_point: tuple[int, int] | None = None
    last_turn_sign: int = 0
    last_turn_at: float | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class MiningApproachDecision:
    state: MiningApproachState
    intent: ControlIntent | None
    distance_yards: float | None
    reason: str


class MiningApproachController:
    """Continuous fused-coordinate approach with visual interaction authority.

    Normal operation never walks forward in repeated microsteps.  It keeps W
    held while fresh pose/target evidence closes the distance, releases inside
    the capture envelope (or on a centered live marker), and then switches to
    screen-space acquisition.  One short W action exists only as an explicit
    post-click `out_of_range` recovery.
    """

    def __init__(
        self,
        geometry: ZoneGeometry,
        config: MiningApproachConfig,
        navigation: NavigationConfig,
        nodes: tuple[MiningNode, ...] | list[MiningNode] = (),
    ) -> None:
        self.geometry = geometry
        self.config = config
        self.navigation = navigation
        self.nodes = tuple(nodes)
        self.state = MiningApproachState()

    @property
    def active(self) -> bool:
        return self.state.phase not in {
            MiningApproachPhase.IDLE,
            MiningApproachPhase.RESUME,
            MiningApproachPhase.FAILED,
        }

    def reset(self) -> None:
        self.state = MiningApproachState()

    def filter_perception(
        self,
        perception: MiningPerception,
        *,
        now: float,
    ) -> MiningPerception:
        """Suppress only the failed marker during its bounded retry cooldown."""
        state = self.state
        if (
            state.phase is not MiningApproachPhase.FAILED
            or state.phase_entered_at is None
            or now - state.phase_entered_at >= self.config.failure_suppression_seconds
        ):
            return perception
        candidates = perception.bright_minimap_points or perception.dark_minimap_points
        same_marker = (
            state.pending_marker_point is None
            or _track_marker(
                state.pending_marker_point,
                candidates,
                self.config.marker_track_max_jump_pixels,
            )
            is not None
        )
        if not same_marker:
            return perception
        return replace(
            perception,
            signal=MiningSignal.IDLE,
            bright_minimap_points=(),
            dark_minimap_points=(),
        )

    def plan(
        self,
        snapshot: WorldSnapshot,
        decision: ControlDecision,
        *,
        now: float,
    ) -> ControlIntent | None:
        result = self.reduce(self.state, snapshot, now=now)
        self.state = result.state
        return result.intent

    def reduce(
        self,
        state: MiningApproachState,
        snapshot: WorldSnapshot,
        *,
        now: float,
    ) -> MiningApproachDecision:
        perception = snapshot.mining.present_value(
            now=now,
            max_age_seconds=self.config.target_max_age_seconds,
        )
        pose = snapshot.pose.present_value(
            now=now,
            max_age_seconds=self.config.target_max_age_seconds,
        )

        if state.phase is MiningApproachPhase.RESUME:
            state = MiningApproachState()
        elif state.phase is MiningApproachPhase.FAILED:
            raw_marker_visible = bool(
                perception is not None
                and (
                    perception.bright_minimap_points
                    or perception.dark_minimap_points
                )
            )
            same_marker = bool(
                raw_marker_visible
                and (
                    state.pending_marker_point is None
                    or _track_marker(
                        state.pending_marker_point,
                        (
                            perception.bright_minimap_points
                            or perception.dark_minimap_points
                        ),
                        self.config.marker_track_max_jump_pixels,
                    )
                    is not None
                )
            )
            cooldown_active = bool(
                state.phase_entered_at is not None
                and now - state.phase_entered_at
                < self.config.failure_suppression_seconds
            )
            if same_marker and cooldown_active:
                return MiningApproachDecision(
                    state,
                    None,
                    None,
                    "failed_target_waiting_for_marker_clear",
                )
            state = MiningApproachState()

        if state.phase is MiningApproachPhase.IDLE:
            if (
                perception is not None
                and perception.candidate_position is None
                and perception.bright_minimap_points
            ):
                marker = _closest_to_minimap_center(
                    perception.bright_minimap_points,
                    snapshot,
                )
                state = MiningApproachState(
                    phase=MiningApproachPhase.IDENTITY_PROBE,
                    pending_marker_point=marker,
                    marker_streak=1,
                    last_marker_frame_id=snapshot.frame_id,
                    started_at=now,
                    phase_entered_at=now,
                )
            else:
                state = self._begin_if_authorized(state, perception, pose, now=now)
            if state.phase is MiningApproachPhase.IDLE:
                return MiningApproachDecision(state, None, None, "no_admitted_target")

        if state.phase is MiningApproachPhase.IDENTITY_PROBE:
            return self._reduce_identity_probe(
                state,
                perception,
                pose,
                snapshot=snapshot,
                now=now,
            )

        state = self._refresh_target(state, perception, pose, snapshot=snapshot)
        if (
            state.phase
            in {
                MiningApproachPhase.IDENTITY_PROBE,
                MiningApproachPhase.ACCESS_APPROACH,
                MiningApproachPhase.CONTINUOUS_APPROACH,
            }
            and state.started_at is not None
            and now - state.started_at > self.config.max_approach_seconds
        ):
            failed = replace(
                state,
                phase=MiningApproachPhase.FAILED,
                phase_entered_at=now,
                outcome="approach_timeout",
            )
            return MiningApproachDecision(
                failed,
                _release_forward(now, "mining_approach_timeout"),
                self._distance(pose, state.estimate),
                "approach_timeout",
            )

        if state.phase is MiningApproachPhase.ACCESS_APPROACH:
            access = self._reduce_path(
                state,
                pose,
                path=state.access_path,
                index=state.access_index,
                returning=False,
                now=now,
            )
            if access.state.phase is MiningApproachPhase.CONTINUOUS_APPROACH:
                return self._reduce_continuous_approach(
                    access.state,
                    perception,
                    pose,
                    now=now,
                )
            return access
        if state.phase is MiningApproachPhase.CONTINUOUS_APPROACH:
            return self._reduce_continuous_approach(state, perception, pose, now=now)
        if state.phase is MiningApproachPhase.VISUAL_SEARCH:
            return self._reduce_visual_search(state, perception, now=now, snapshot=snapshot)
        if state.phase is MiningApproachPhase.VERIFY:
            return self._reduce_verify(state, perception, now=now, snapshot=snapshot)
        if state.phase is MiningApproachPhase.RETURN_TO_ROUTE:
            return self._reduce_path(
                state,
                pose,
                path=state.return_path,
                index=state.return_index,
                returning=True,
                now=now,
            )
        return MiningApproachDecision(state, None, self._distance(pose, state.estimate), state.phase.value)

    def _begin_if_authorized(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        pose: PlayerPose | None,
        *,
        now: float,
    ) -> MiningApproachState:
        if (
            perception is None
            or perception.signal is MiningSignal.IDLE
            or perception.candidate_position is None
            or pose is None
        ):
            return state

        candidate = perception
        option: NodeAccessOption | None = None
        if self.config.require_access_plan:
            node = next(
                (
                    item
                    for item in self.nodes
                    if item.node_id is not None
                    and item.node_id == perception.candidate_id
                ),
                None,
            )
            if node is None:
                return state
            option = _primary_access_option(node)
            if option is None:
                return state
            if option.inbound_coords:
                attachment = MapPoint.from_coord(option.inbound_coords[0])
                if (
                    self.geometry.distance(pose.position, attachment)
                    > self.config.access_attachment_radius_yards
                ):
                    return state
            candidate = replace(
                perception,
                candidate_position=MapPoint.from_coord(node.coord),
            )

        estimate = self._estimate_target(candidate, pose)
        if estimate is None:
            return state
        access_path = tuple(
            MapPoint.from_coord(coord)
            for coord in (option.inbound_coords if option is not None else ())
        )
        if (
            not access_path
            and self.geometry.distance(pose.position, estimate.position)
            > self.config.acquisition_radius_yards
        ):
            return state
        return MiningApproachState(
            phase=(
                MiningApproachPhase.ACCESS_APPROACH
                if access_path
                else MiningApproachPhase.CONTINUOUS_APPROACH
            ),
            target_id=candidate.candidate_id,
            database_position=candidate.candidate_position,
            estimate=estimate,
            target_ore_type=candidate.tooltip_ore_type,
            access_path=access_path,
            return_path=_return_path(option),
            started_at=now,
            phase_entered_at=now,
        )

    def _reduce_identity_probe(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        pose: PlayerPose | None,
        *,
        snapshot: WorldSnapshot,
        now: float,
    ) -> MiningApproachDecision:
        if state.phase_entered_at is None:
            state = replace(state, phase_entered_at=now)
        if now - float(state.phase_entered_at) > self.config.identity_timeout_seconds:
            return MiningApproachDecision(
                replace(
                    state,
                    phase=MiningApproachPhase.FAILED,
                    phase_entered_at=now,
                    outcome="minimap_identity_timeout",
                ),
                None,
                None,
                "minimap_identity_timeout",
            )
        if perception is None or not perception.bright_minimap_points:
            return MiningApproachDecision(state, None, None, "identity_marker_temporarily_missing")
        marker = _track_marker(
            state.pending_marker_point,
            perception.bright_minimap_points,
            self.config.marker_track_max_jump_pixels,
        )
        if marker is None:
            return MiningApproachDecision(state, None, None, "identity_marker_track_lost")
        streak = state.marker_streak + (
            1 if snapshot.frame_id != state.last_marker_frame_id else 0
        )
        state = replace(
            state,
            pending_marker_point=marker,
            marker_streak=streak,
            last_marker_frame_id=snapshot.frame_id,
        )
        if (
            streak >= self.config.identity_confirm_frames
            and perception.tooltip_ore_type
            and pose is not None
        ):
            admitted = self._admit_database_target(
                state,
                marker,
                perception.tooltip_ore_type,
                pose,
                snapshot=snapshot,
                now=now,
            )
            if admitted is not None:
                if admitted.phase is MiningApproachPhase.ACCESS_APPROACH:
                    access = self._reduce_path(
                        admitted,
                        pose,
                        path=admitted.access_path,
                        index=admitted.access_index,
                        returning=False,
                        now=now,
                    )
                    if access.state.phase is not MiningApproachPhase.CONTINUOUS_APPROACH:
                        return access
                    admitted = access.state
                return self._reduce_continuous_approach(
                    admitted,
                    perception,
                    pose,
                    now=now,
                )
        if streak < self.config.identity_confirm_frames:
            return MiningApproachDecision(state, None, None, "identity_marker_confirming")
        client_point = _minimap_client_point(marker, snapshot)
        if client_point is None:
            return MiningApproachDecision(state, None, None, "minimap_geometry_missing")
        return MiningApproachDecision(
            state,
            ControlIntent(
                CommandGroup.MINING,
                (
                    Command(
                        kind=CommandKind.MOVE_CURSOR,
                        group=CommandGroup.MINING,
                        point=client_point,
                        reason="mining_native_minimap_identity_probe",
                        deadline=now + 0.25,
                    ),
                ),
                "mining_native_minimap_identity_probe",
            ),
            None,
            "identity_tooltip_probe",
        )

    def _admit_database_target(
        self,
        state: MiningApproachState,
        marker: tuple[int, int],
        ore_type: str,
        pose: PlayerPose,
        *,
        snapshot: WorldSnapshot,
        now: float,
    ) -> MiningApproachState | None:
        center = _metadata_pair(snapshot, "minimap_center")
        if center is None:
            return None
        offset = marker[0] - center[0], marker[1] - center[1]
        live_position = self._live_marker_position(pose, offset)
        ranked: list[tuple[float, MiningNode, NodeAccessOption | None]] = []
        for node in self.nodes:
            if not _ore_types_match(node.ore_type, ore_type):
                continue
            node_position = MapPoint.from_coord(node.coord)
            marker_distance = self.geometry.distance(live_position, node_position)
            if marker_distance > self.config.database_match_radius_yards:
                continue
            option = _primary_access_option(node)
            if self.config.require_access_plan and option is None:
                continue
            if option is not None and option.inbound_coords:
                attachment = MapPoint.from_coord(option.inbound_coords[0])
                if (
                    self.geometry.distance(pose.position, attachment)
                    > self.config.access_attachment_radius_yards
                ):
                    continue
            ranked.append((marker_distance, node, option))
        if not ranked:
            return None
        _distance, node, option = min(ranked, key=lambda item: (item[0], item[1].node_id or 0))
        database_position = MapPoint.from_coord(node.coord)
        enriched = MiningPerception(
            signal=MiningSignal.APPROACH,
            candidate_id=node.node_id,
            candidate_position=database_position,
            minimap_offset_px=offset,
            minimap_centered=math.hypot(*offset) <= self.config.centered_radius_pixels,
            tooltip_ore_type=ore_type,
        )
        estimate = self._estimate_target(enriched, pose)
        access_path = tuple(
            MapPoint.from_coord(coord)
            for coord in (option.inbound_coords if option is not None else ())
        )
        return_path = _return_path(option)
        return MiningApproachState(
            phase=(
                MiningApproachPhase.ACCESS_APPROACH
                if access_path
                else MiningApproachPhase.CONTINUOUS_APPROACH
            ),
            target_id=node.node_id,
            database_position=database_position,
            estimate=estimate,
            target_ore_type=ore_type,
            pending_marker_point=marker,
            marker_streak=state.marker_streak,
            last_marker_frame_id=state.last_marker_frame_id,
            access_path=access_path,
            return_path=return_path,
            started_at=state.started_at or now,
            phase_entered_at=now,
        )

    def _live_marker_position(
        self,
        pose: PlayerPose,
        offset: tuple[float, float],
    ) -> MapPoint:
        player_world = self.geometry.to_world(pose.position)
        return self.geometry.to_map(
            WorldPoint(
                x_yards=(
                    player_world.x_yards
                    + offset[0]
                    / self.config.minimap_sensor_radius_pixels
                    * self.config.minimap_radius_x_yards
                ),
                y_yards=(
                    player_world.y_yards
                    + offset[1]
                    / self.config.minimap_sensor_radius_pixels
                    * self.config.minimap_radius_y_yards
                ),
            )
        )

    def _reduce_path(
        self,
        state: MiningApproachState,
        pose: PlayerPose | None,
        *,
        path: tuple[MapPoint, ...],
        index: int,
        returning: bool,
        now: float,
    ) -> MiningApproachDecision:
        if pose is None:
            return MiningApproachDecision(
                state,
                _release_forward(now, "mining_access_pose_unknown"),
                None,
                "access_pose_unknown",
            )
        cursor = index
        while cursor < len(path) and self.geometry.distance(pose.position, path[cursor]) <= self.config.capture_radius_yards:
            cursor += 1
        if cursor >= len(path):
            next_phase = (
                MiningApproachPhase.RESUME
                if returning
                else MiningApproachPhase.CONTINUOUS_APPROACH
            )
            next_state = replace(
                state,
                phase=next_phase,
                phase_entered_at=now,
                access_index=cursor if not returning else state.access_index,
                return_index=cursor if returning else state.return_index,
            )
            return MiningApproachDecision(
                next_state,
                _release_forward(
                    now,
                    "mining_route_resume_reached" if returning else "mining_access_path_reached",
                ),
                0.0,
                "route_resume_reached" if returning else "access_path_reached",
            )
        target = path[cursor]
        desired = self.geometry.heading_degrees(pose.position, target)
        distance = self.geometry.distance(pose.position, target)
        if desired is None:
            return MiningApproachDecision(state, None, distance, "access_target_reached")
        error = signed_heading_error_degrees(pose.heading_degrees, desired)
        next_state = replace(
            state,
            access_index=cursor if not returning else state.access_index,
            return_index=cursor if returning else state.return_index,
        )
        next_state, intent = self._continuous_motion_intent(
            next_state,
            error,
            now=now,
            inside_braking_radius=distance <= self.config.braking_radius_yards,
        )
        return MiningApproachDecision(
            next_state,
            intent,
            distance,
            "continuous_return_path" if returning else "continuous_access_path",
        )

    def _refresh_target(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        pose: PlayerPose | None,
        *,
        snapshot: WorldSnapshot,
    ) -> MiningApproachState:
        if perception is None:
            return state
        if (
            state.target_id is not None
            and perception.candidate_id is not None
            and perception.candidate_id != state.target_id
        ):
            return state
        database_position = perception.candidate_position or state.database_position
        if database_position is None:
            return state
        marker_candidates = (
            perception.bright_minimap_points or perception.dark_minimap_points
        )
        marker = _track_marker(
            state.pending_marker_point,
            marker_candidates,
            self.config.marker_track_max_jump_pixels,
        )
        center = _metadata_pair(snapshot, "minimap_center")
        offset = perception.minimap_offset_px
        centered = perception.minimap_centered
        if marker is not None and center is not None:
            offset = marker[0] - center[0], marker[1] - center[1]
            centered = math.hypot(*offset) <= self.config.centered_radius_pixels
        refreshed = replace(
            perception,
            candidate_id=state.target_id,
            candidate_position=database_position,
            minimap_offset_px=offset,
            minimap_centered=centered,
        )
        estimate = self._estimate_target(refreshed, pose)
        return replace(
            state,
            database_position=database_position,
            estimate=estimate or state.estimate,
            pending_marker_point=marker or state.pending_marker_point,
        )

    def _estimate_target(
        self,
        perception: MiningPerception,
        pose: PlayerPose | None,
    ) -> MiningTargetEstimate | None:
        database_position = perception.candidate_position
        centered = perception.minimap_centered
        offset = perception.minimap_offset_px
        if offset is not None:
            centered = centered or math.hypot(*offset) <= self.config.centered_radius_pixels
        if database_position is None:
            return None
        if pose is None or offset is None:
            return MiningTargetEstimate(
                position=database_position,
                source="database",
                disagreement_yards=None,
                marker_centered=centered,
            )

        player_world = self.geometry.to_world(pose.position)
        live_world = WorldPoint(
            x_yards=(
                player_world.x_yards
                + offset[0]
                / self.config.minimap_sensor_radius_pixels
                * self.config.minimap_radius_x_yards
            ),
            y_yards=(
                player_world.y_yards
                + offset[1]
                / self.config.minimap_sensor_radius_pixels
                * self.config.minimap_radius_y_yards
            ),
        )
        live_position = self.geometry.to_map(live_world)
        disagreement = self.geometry.distance(database_position, live_position)
        if disagreement > self.config.max_fusion_disagreement_yards:
            return MiningTargetEstimate(
                position=database_position,
                source="database_live_projection_rejected",
                disagreement_yards=disagreement,
                marker_centered=centered,
            )
        database_world = self.geometry.to_world(database_position)
        live_weight = self.config.live_marker_weight
        fused_world = WorldPoint(
            x_yards=database_world.x_yards * (1.0 - live_weight) + live_world.x_yards * live_weight,
            y_yards=database_world.y_yards * (1.0 - live_weight) + live_world.y_yards * live_weight,
        )
        return MiningTargetEstimate(
            position=self.geometry.to_map(fused_world),
            source="database_live_marker_fused",
            disagreement_yards=disagreement,
            marker_centered=centered,
        )

    def _reduce_continuous_approach(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        pose: PlayerPose | None,
        *,
        now: float,
    ) -> MiningApproachDecision:
        distance = self._distance(pose, state.estimate)
        if pose is None or state.estimate is None or distance is None:
            return MiningApproachDecision(
                state,
                _release_forward(now, "mining_pose_or_target_unknown"),
                None,
                "pose_or_target_unknown",
            )
        centered = bool(
            state.estimate.marker_centered
            or (perception is not None and perception.minimap_centered)
        )
        if centered or distance <= self.config.capture_radius_yards:
            search_state = replace(
                state,
                phase=MiningApproachPhase.VISUAL_SEARCH,
                phase_entered_at=now,
                last_search_action_at=None,
                search_step=0,
            )
            return MiningApproachDecision(
                search_state,
                _release_forward(now, "mining_capture_envelope_reached"),
                distance,
                "marker_centered" if centered else "capture_radius_reached",
            )

        desired_heading = self.geometry.heading_degrees(
            pose.position,
            state.estimate.position,
        )
        if desired_heading is None:
            return MiningApproachDecision(
                state,
                _release_forward(now, "mining_target_reached"),
                distance,
                "target_reached",
            )
        heading_error = signed_heading_error_degrees(
            pose.heading_degrees,
            desired_heading,
        )
        state, intent = self._continuous_motion_intent(
            state,
            heading_error,
            now=now,
            inside_braking_radius=distance <= self.config.braking_radius_yards,
        )
        return MiningApproachDecision(
            state,
            intent,
            distance,
            (
                "continuous_capture_approach"
                if distance <= self.config.braking_radius_yards
                else "continuous_coarse_approach"
            ),
        )

    def _continuous_motion_intent(
        self,
        state: MiningApproachState,
        heading_error: float,
        *,
        now: float,
        inside_braking_radius: bool,
    ) -> tuple[MiningApproachState, ControlIntent]:
        magnitude = abs(heading_error)
        pivot = magnitude >= self.navigation.pivot_degrees
        turn_sign = 1 if heading_error > 0.0 else -1
        reversal_suppressed = bool(
            magnitude < 50.0
            and state.last_turn_sign not in {0, turn_sign}
            and state.last_turn_at is not None
            and now - state.last_turn_at < self.navigation.turn_reversal_guard_seconds
        )
        commands: list[Command] = [
            Command(
                kind=(CommandKind.RELEASE_KEY if pivot else CommandKind.HOLD_KEY),
                group=CommandGroup.MINING,
                key="W",
                reason="mining_pivot" if pivot else "mining_continuous_forward",
                deadline=now + 0.30,
            )
        ]
        if magnitude > self.navigation.turn_engage_degrees and not reversal_suppressed:
            max_duration = (
                self.navigation.pivot_drag_max_seconds
                if pivot
                else self.navigation.moving_drag_max_seconds
            )
            pixels = int(round(-heading_error * self.navigation.yaw_pixels_per_degree))
            max_pixels = max(2, int(round(self.navigation.yaw_pixels_per_second * max_duration)))
            pixels = max(-max_pixels, min(max_pixels, pixels))
            if pixels:
                commands.append(
                    Command(
                        kind=CommandKind.DRAG_RELATIVE,
                        group=CommandGroup.MINING,
                        mouse_button=MouseButton.RIGHT,
                        delta_x=pixels,
                        duration=max(
                            0.04,
                            min(max_duration, abs(pixels) / self.navigation.yaw_pixels_per_second),
                        ),
                        reason="mining_fused_target_yaw",
                        deadline=now + 0.50,
                    )
                )
                state = replace(
                    state,
                    last_turn_sign=turn_sign,
                    last_turn_at=now,
                )
        return state, ControlIntent(
            owner=CommandGroup.MINING,
            commands=tuple(commands),
            reason=(
                "mining_yaw_reversal_hysteresis"
                if reversal_suppressed
                else (
                    "mining_continuous_near_capture"
                    if inside_braking_radius
                    else "mining_continuous_approach"
                )
            ),
        )

    def _reduce_visual_search(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        *,
        now: float,
        snapshot: WorldSnapshot,
    ) -> MiningApproachDecision:
        if state.phase_entered_at is None:
            state = replace(state, phase_entered_at=now)
        if now - float(state.phase_entered_at) > self.config.visual_search_seconds:
            failed = replace(
                state,
                phase=MiningApproachPhase.FAILED,
                phase_entered_at=now,
                outcome="visual_search_timeout",
            )
            return MiningApproachDecision(failed, None, None, "visual_search_timeout")
        if (
            perception is not None
            and perception.interaction_authority
            and perception.interaction_point is not None
        ):
            point = perception.interaction_point
            verify_state = replace(
                state,
                phase=MiningApproachPhase.VERIFY,
                phase_entered_at=now,
                clicked_at=now,
                last_interaction_point=point,
            )
            return MiningApproachDecision(
                verify_state,
                ControlIntent(
                    CommandGroup.MINING,
                    (
                        Command(
                            kind=CommandKind.MOVE_CURSOR,
                            group=CommandGroup.MINING,
                            point=point,
                            reason="mining_authorized_point",
                            deadline=now + 0.25,
                        ),
                        Command(
                            kind=CommandKind.CLICK,
                            group=CommandGroup.MINING,
                            mouse_button=MouseButton.RIGHT,
                            duration=0.05,
                            reason="mining_native_authority_click",
                            deadline=now + 0.25,
                            exclusive=True,
                        ),
                    ),
                    "mining_authorized_interaction",
                ),
                None,
                "interaction_authority",
            )
        candidates = perception.world_candidate_points if perception is not None else ()
        if candidates:
            point = _closest_to_screen_center(candidates, snapshot)
            return MiningApproachDecision(
                replace(state, last_interaction_point=point),
                ControlIntent(
                    CommandGroup.MINING,
                    (
                        Command(
                            kind=CommandKind.MOVE_CURSOR,
                            group=CommandGroup.MINING,
                            point=point,
                            reason="mining_probe_world_candidate",
                            deadline=now + 0.25,
                        ),
                    ),
                    "mining_probe_world_candidate",
                ),
                None,
                "world_candidate_probe",
            )
        if (
            state.last_search_action_at is not None
            and now - state.last_search_action_at < 0.35
        ):
            return MiningApproachDecision(state, None, None, "visual_search_observing")
        search_step = state.search_step
        if search_step == 0:
            command = Command(
                kind=CommandKind.TAP_KEY,
                group=CommandGroup.MINING,
                key="F7",
                duration=0.04,
                reason="mining_camera_reset",
                exclusive=True,
                deadline=now + 0.25,
            )
        else:
            sweep_pattern = (-28, -28, -28, 56, 56, -28)
            delta = sweep_pattern[(search_step - 1) % len(sweep_pattern)]
            command = Command(
                kind=CommandKind.DRAG_RELATIVE,
                group=CommandGroup.MINING,
                mouse_button=MouseButton.RIGHT,
                delta_x=delta,
                duration=max(0.06, abs(delta) / 260.0),
                reason="mining_bounded_world_sweep",
                deadline=now + 0.35,
            )
        next_state = replace(
            state,
            last_search_action_at=now,
            search_step=search_step + 1,
        )
        return MiningApproachDecision(
            next_state,
            ControlIntent(CommandGroup.MINING, (command,), command.reason),
            None,
            command.reason,
        )

    def _reduce_verify(
        self,
        state: MiningApproachState,
        perception: MiningPerception | None,
        *,
        now: float,
        snapshot: WorldSnapshot,
    ) -> MiningApproachDecision:
        outcome = perception.interaction_outcome if perception is not None else None
        normalized = str(outcome).strip().lower() if outcome else ""
        if normalized in {"success", "gathered", "loot_opened"}:
            complete = replace(
                state,
                phase=(
                    MiningApproachPhase.RETURN_TO_ROUTE
                    if state.return_path
                    else MiningApproachPhase.RESUME
                ),
                phase_entered_at=now,
                outcome="success",
            )
            return MiningApproachDecision(
                complete,
                _release_forward(now, "gather_verified"),
                None,
                "gather_verified",
            )
        if normalized == "out_of_range":
            if state.out_of_range_recoveries >= self.config.max_out_of_range_recoveries:
                failed = replace(
                    state,
                    phase=MiningApproachPhase.FAILED,
                    phase_entered_at=now,
                    outcome="out_of_range_exhausted",
                )
                return MiningApproachDecision(failed, None, None, "out_of_range_exhausted")
            commands: list[Command] = []
            point = (
                perception.interaction_point
                if perception is not None and perception.interaction_point is not None
                else state.last_interaction_point
            )
            if point is not None:
                screen_width = _screen_size(snapshot)[0]
                horizontal_error = point[0] - screen_width * 0.5
                if abs(horizontal_error) > screen_width * 0.04:
                    commands.append(
                        Command(
                            kind=CommandKind.DRAG_RELATIVE,
                            group=CommandGroup.MINING,
                            mouse_button=MouseButton.RIGHT,
                            delta_x=int(round(horizontal_error * 0.30)),
                            duration=0.10,
                            reason="mining_out_of_range_face",
                            deadline=now + 0.30,
                        )
                    )
            commands.append(
                Command(
                    kind=CommandKind.TAP_KEY,
                    group=CommandGroup.MINING,
                    key="W",
                    duration=self.config.out_of_range_recovery_seconds,
                    reason="mining_confirmed_out_of_range_recovery",
                    exclusive=True,
                    deadline=now + self.config.out_of_range_recovery_seconds + 0.30,
                )
            )
            retry = replace(
                state,
                phase=MiningApproachPhase.VISUAL_SEARCH,
                phase_entered_at=now,
                last_search_action_at=now,
                out_of_range_recoveries=state.out_of_range_recoveries + 1,
            )
            return MiningApproachDecision(
                retry,
                ControlIntent(
                    CommandGroup.MINING,
                    tuple(commands),
                    "mining_confirmed_out_of_range_recovery",
                ),
                None,
                "out_of_range_recovery",
            )
        if normalized in {"taken", "failed", "interrupted", "no_target"}:
            failed = replace(
                state,
                phase=MiningApproachPhase.FAILED,
                phase_entered_at=now,
                outcome=normalized,
            )
            return MiningApproachDecision(failed, None, None, f"interaction_{normalized}")
        if state.clicked_at is not None and now - state.clicked_at > self.config.verify_seconds:
            failed = replace(
                state,
                phase=MiningApproachPhase.FAILED,
                phase_entered_at=now,
                outcome="verify_timeout",
            )
            return MiningApproachDecision(failed, None, None, "verify_timeout")
        return MiningApproachDecision(state, None, None, "awaiting_interaction_outcome")

    def _distance(
        self,
        pose: PlayerPose | None,
        estimate: MiningTargetEstimate | None,
    ) -> float | None:
        if pose is None or estimate is None:
            return None
        return self.geometry.distance(pose.position, estimate.position)


def _release_forward(now: float, reason: str) -> ControlIntent:
    return ControlIntent(
        CommandGroup.MINING,
        (
            Command(
                kind=CommandKind.RELEASE_KEY,
                group=CommandGroup.MINING,
                key="W",
                reason=reason,
                deadline=now + 0.25,
            ),
        ),
        reason,
    )


def _screen_size(snapshot: WorldSnapshot) -> tuple[int, int]:
    value = snapshot.metadata.get("screen_size")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 2560, 1440


def _closest_to_screen_center(
    candidates: tuple[tuple[int, int], ...],
    snapshot: WorldSnapshot,
) -> tuple[int, int]:
    width, height = _screen_size(snapshot)
    center = width * 0.5, height * 0.5
    return min(
        candidates,
        key=lambda point: math.hypot(point[0] - center[0], point[1] - center[1]),
    )


def _closest_to_minimap_center(
    candidates: tuple[tuple[int, int], ...],
    snapshot: WorldSnapshot,
) -> tuple[int, int]:
    center = _metadata_pair(snapshot, "minimap_center") or (146.5, 133.4)
    return min(
        candidates,
        key=lambda point: math.hypot(point[0] - center[0], point[1] - center[1]),
    )


def _track_marker(
    previous: tuple[int, int] | None,
    candidates: tuple[tuple[int, int], ...],
    max_jump_pixels: float,
) -> tuple[int, int] | None:
    if not candidates:
        return None
    if previous is None:
        return candidates[0]
    nearest = min(
        candidates,
        key=lambda point: math.hypot(point[0] - previous[0], point[1] - previous[1]),
    )
    if math.hypot(nearest[0] - previous[0], nearest[1] - previous[1]) > max_jump_pixels:
        return None
    return nearest


def _metadata_pair(
    snapshot: WorldSnapshot,
    key: str,
) -> tuple[float, float] | None:
    value = snapshot.metadata.get(key)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def _minimap_client_point(
    marker: tuple[int, int],
    snapshot: WorldSnapshot,
) -> tuple[int, int] | None:
    origin = _metadata_pair(snapshot, "minimap_origin")
    shape = _metadata_pair(snapshot, "minimap_shape")
    if origin is None or shape is None:
        return None
    height, width = shape
    x = max(0, min(int(width) - 1, int(marker[0])))
    y = max(0, min(int(height) - 1, int(marker[1])))
    return int(round(origin[0])) + x, int(round(origin[1])) + y


def _ore_types_match(database_type: str, tooltip_type: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(
            str(value)
            .casefold()
            .replace("deposit", "")
            .replace("vein", "")
            .replace("ooze covered", "")
            .split()
        )

    database = normalize(database_type)
    tooltip = normalize(tooltip_type)
    return database == tooltip or database in tooltip or tooltip in database


def _primary_access_option(node: MiningNode) -> NodeAccessOption | None:
    plan = node.access_plan
    if plan is None:
        return None
    for option in plan.options:
        if option.rank == plan.primary_option_rank:
            return option
    return plan.options[0] if plan.options else None


def _return_path(option: NodeAccessOption | None) -> tuple[MapPoint, ...]:
    if option is None:
        return ()
    coords = (*option.return_coords, *option.resume_coords)
    points: list[MapPoint] = []
    for coord in coords:
        point = MapPoint.from_coord(coord)
        if not points or point != points[-1]:
            points.append(point)
    return tuple(points)
