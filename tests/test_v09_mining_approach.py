from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vision_bot.controllers.mining_approach import (
    MiningApproachController,
    MiningApproachPhase,
    MiningApproachState,
)
from vision_bot.core.commands import CommandKind
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    MiningSignal,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    WorldSnapshot,
)
from vision_bot.v09_config import V09Config
from vision_bot.route_planner import MiningNode, NodeAccessOption, NodeAccessPlan


ROOT = Path(__file__).parents[1]
CONFIG = V09Config.from_mapping(
    yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
)


def _evidence(value, frame_id: int, now: float):
    return Evidence.present(
        value,
        observed_at=now,
        frame_id=frame_id,
        source="test",
    )


def _snapshot(
    perception: MiningPerception,
    *,
    position: MapPoint = MapPoint(50.0, 50.0),
    heading: float = 270.0,
    frame_id: int = 1,
    now: float = 1.0,
) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=_evidence(True, frame_id, now),
        pose=_evidence(PlayerPose(162, position, heading), frame_id, now),
        life=_evidence(LifeState.ALIVE, frame_id, now),
        modal=_evidence(ModalState.CLEAR, frame_id, now),
        combat=_evidence(CombatView(active=False), frame_id, now),
        mounted=_evidence(True, frame_id, now),
        mining=_evidence(perception, frame_id, now),
        loot_pending=_evidence(False, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(RecoveryPerception(), frame_id, now),
        metadata={
            "screen_size": (2560, 1440),
            "minimap_origin": (2267, 0),
            "minimap_shape": (278, 293),
            "minimap_center": (146.5, 133.44),
        },
    )


def _controller(nodes=(), *, strict: bool = False) -> MiningApproachController:
    mining_config = (
        CONFIG.mining
        if strict
        else replace(CONFIG.mining, require_access_plan=False)
    )
    return MiningApproachController(
        CONFIG.zone,
        mining_config,
        CONFIG.navigation,
        nodes,
    )


def _access_node() -> MiningNode:
    option = NodeAccessOption(
        rank=0,
        candidate_index=0,
        primary=True,
        approach_coord=MapPoint(50.20, 50.0).to_coord(),
        sector=0,
        bearing_degrees=0.0,
        objective=1.0,
        inbound_coords=(
            MapPoint(50.0, 50.0).to_coord(),
            MapPoint(50.10, 50.0).to_coord(),
        ),
        return_coords=(
            MapPoint(50.10, 50.0).to_coord(),
            MapPoint(50.0, 50.0).to_coord(),
        ),
        resume_coords=(MapPoint(49.95, 50.0).to_coord(),),
    )
    return MiningNode(
        zone_id=162,
        coord=MapPoint(50.20, 50.0).to_coord(),
        ore_type="Small Thorium",
        node_id=77,
        access_plan=NodeAccessPlan(
            attachment_route_index=1,
            resume_route_index=2,
            primary_option_rank=0,
            options=(option,),
        ),
    )


def test_normal_mining_approach_holds_forward_continuously_without_microsteps() -> None:
    controller = _controller()
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=12,
        candidate_position=MapPoint(50.30, 50.0),
    )

    result = controller.reduce(
        MiningApproachState(),
        _snapshot(perception, heading=270.0),
        now=1.0,
    )

    assert result.state.phase is MiningApproachPhase.CONTINUOUS_APPROACH
    assert result.intent is not None
    assert any(
        command.kind is CommandKind.HOLD_KEY and command.key == "W"
        for command in result.intent.commands
    )
    assert not any(
        command.kind is CommandKind.TAP_KEY and command.key == "W"
        for command in result.intent.commands
    )


def test_mining_approach_suppresses_rapid_yaw_reversal_without_stopping_forward() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.CONTINUOUS_APPROACH,
        target_id=12,
        database_position=MapPoint(50.30, 50.0),
        started_at=1.0,
        phase_entered_at=1.0,
        last_turn_sign=1,
        last_turn_at=1.0,
    )
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=12,
        candidate_position=MapPoint(50.30, 50.0),
    )

    result = controller.reduce(
        state,
        _snapshot(perception, heading=310.0, now=1.1),
        now=1.1,
    )

    assert result.intent is not None
    assert result.intent.reason == "mining_yaw_reversal_hysteresis"
    assert any(
        command.kind is CommandKind.HOLD_KEY and command.key == "W"
        for command in result.intent.commands
    )
    assert not any(
        command.kind is CommandKind.DRAG_RELATIVE
        for command in result.intent.commands
    )


def test_raw_marker_requires_stable_native_tooltip_and_safe_access_plan() -> None:
    controller = _controller((_access_node(),), strict=True)
    marker = (158, 133)
    first_perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=(marker,),
    )
    first = controller.reduce(
        MiningApproachState(),
        _snapshot(first_perception, now=1.0),
        now=1.0,
    )

    assert first.state.phase is MiningApproachPhase.IDENTITY_PROBE
    assert first.intent is None

    identified = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=(marker,),
        tooltip_ore_type="Small Thorium",
    )
    admitted = controller.reduce(
        first.state,
        _snapshot(identified, frame_id=2, now=1.1),
        now=1.1,
    )

    assert admitted.state.target_id == 77
    assert admitted.state.phase is MiningApproachPhase.ACCESS_APPROACH
    assert admitted.state.access_path
    assert admitted.state.return_path
    assert admitted.intent is not None
    assert any(
        command.kind is CommandKind.HOLD_KEY and command.key == "W"
        for command in admitted.intent.commands
    )
    assert not any(
        command.kind is CommandKind.RELEASE_KEY and command.key == "W"
        for command in admitted.intent.commands
    )


def test_identity_probe_moves_cursor_but_never_clicks_without_tooltip() -> None:
    controller = _controller((_access_node(),), strict=True)
    marker = (158, 133)
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=(marker,),
    )
    first = controller.reduce(
        MiningApproachState(),
        _snapshot(perception, now=1.0),
        now=1.0,
    )
    second = controller.reduce(
        first.state,
        _snapshot(perception, frame_id=2, now=1.1),
        now=1.1,
    )

    assert second.intent is not None
    assert second.intent.commands[0].kind is CommandKind.MOVE_CURSOR
    assert second.intent.commands[0].point == (2267 + 158, 133)
    assert not any(command.kind is CommandKind.CLICK for command in second.intent.commands)


def test_live_minimap_vector_is_fused_with_database_anchor() -> None:
    controller = _controller()
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=12,
        candidate_position=MapPoint(50.30, 50.0),
        minimap_offset_px=(10.0, 0.0),
    )

    result = controller.reduce(
        MiningApproachState(),
        _snapshot(perception),
        now=1.0,
    )

    assert result.state.estimate is not None
    assert result.state.estimate.source == "database_live_marker_fused"
    assert 50.20 < result.state.estimate.position.x_percent < 50.30


def test_strict_runtime_rejects_prepopulated_coordinate_without_planned_node() -> None:
    controller = _controller(strict=True)
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=999,
        candidate_position=MapPoint(50.10, 50.0),
    )

    result = controller.reduce(
        MiningApproachState(),
        _snapshot(perception),
        now=1.0,
    )

    assert result.state.phase is MiningApproachPhase.IDLE
    assert result.intent is None


def test_unplanned_candidate_cannot_take_control_outside_acquisition_radius() -> None:
    controller = _controller()
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=12,
        candidate_position=MapPoint(51.0, 50.0),
    )

    result = controller.reduce(
        MiningApproachState(),
        _snapshot(perception),
        now=1.0,
    )

    assert result.state.phase is MiningApproachPhase.IDLE
    assert result.intent is None


def test_centered_marker_enters_visual_search_without_distant_microsteps() -> None:
    controller = _controller()
    perception = MiningPerception(
        signal=MiningSignal.APPROACH,
        candidate_id=12,
        candidate_position=MapPoint(50.30, 50.0),
        minimap_offset_px=(2.0, 1.0),
        minimap_centered=True,
    )

    result = controller.reduce(
        MiningApproachState(),
        _snapshot(perception),
        now=1.0,
    )

    assert result.state.phase is MiningApproachPhase.VISUAL_SEARCH
    assert result.reason == "marker_centered"
    assert result.intent is not None
    assert result.intent.commands[0].kind is CommandKind.RELEASE_KEY
    assert result.intent.commands[0].key == "W"


def test_admitted_target_refreshes_from_each_tracked_live_marker_frame() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.CONTINUOUS_APPROACH,
        target_id=12,
        database_position=MapPoint(50.30, 50.0),
        pending_marker_point=(170, 133),
        started_at=1.0,
        phase_entered_at=1.0,
    )
    # The same visible marker moves under the player icon.  No synthetic
    # candidate_position is needed on subsequent frames: identity remains
    # latched while the current minimap vector drives the handoff.
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=((149, 134),),
    )

    result = controller.reduce(
        state,
        _snapshot(perception, frame_id=2, now=1.1),
        now=1.1,
    )

    assert result.state.pending_marker_point == (149, 134)
    assert result.state.estimate is not None
    assert result.state.estimate.marker_centered
    assert result.state.estimate.source == "database_live_marker_fused"
    assert result.state.phase is MiningApproachPhase.VISUAL_SEARCH
    assert result.reason == "marker_centered"


def test_visual_candidate_moves_cursor_but_cannot_authorize_click() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.VISUAL_SEARCH,
        target_id=12,
        database_position=MapPoint(50.0, 50.0),
        started_at=1.0,
        phase_entered_at=1.0,
    )
    perception = MiningPerception(
        signal=MiningSignal.VISUAL_SEARCH,
        candidate_id=12,
        candidate_position=MapPoint(50.0, 50.0),
        world_candidate_points=((400, 400), (1275, 710)),
        interaction_authority=False,
    )

    result = controller.reduce(state, _snapshot(perception, now=1.1), now=1.1)

    assert result.intent is not None
    assert result.intent.commands[0].kind is CommandKind.MOVE_CURSOR
    assert result.intent.commands[0].point == (1275, 710)
    assert not any(command.kind is CommandKind.CLICK for command in result.intent.commands)


def test_native_authority_allows_exactly_one_interaction_click() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.VISUAL_SEARCH,
        target_id=12,
        database_position=MapPoint(50.0, 50.0),
        started_at=1.0,
        phase_entered_at=1.0,
    )
    perception = MiningPerception(
        signal=MiningSignal.INTERACTION_READY,
        candidate_id=12,
        candidate_position=MapPoint(50.0, 50.0),
        interaction_authority=True,
        interaction_point=(1275, 710),
    )

    result = controller.reduce(state, _snapshot(perception, now=1.1), now=1.1)

    assert result.state.phase is MiningApproachPhase.VERIFY
    assert result.intent is not None
    clicks = [command for command in result.intent.commands if command.kind is CommandKind.CLICK]
    assert len(clicks) == 1
    assert clicks[0].reason == "mining_native_authority_click"


def test_micro_forward_action_exists_only_after_confirmed_out_of_range() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.VERIFY,
        target_id=12,
        database_position=MapPoint(50.0, 50.0),
        started_at=1.0,
        phase_entered_at=1.0,
        clicked_at=1.0,
        last_interaction_point=(1400, 700),
    )
    perception = MiningPerception(
        signal=MiningSignal.VERIFYING,
        candidate_id=12,
        candidate_position=MapPoint(50.0, 50.0),
        interaction_outcome="out_of_range",
        interaction_point=(1400, 700),
    )

    result = controller.reduce(state, _snapshot(perception, now=1.2), now=1.2)

    assert result.state.phase is MiningApproachPhase.VISUAL_SEARCH
    assert result.state.out_of_range_recoveries == 1
    assert result.intent is not None
    recoveries = [
        command
        for command in result.intent.commands
        if command.kind is CommandKind.TAP_KEY and command.key == "W"
    ]
    assert len(recoveries) == 1
    assert recoveries[0].reason == "mining_confirmed_out_of_range_recovery"


def test_verified_success_releases_mining_ownership() -> None:
    controller = _controller()
    controller.state = MiningApproachState(
        phase=MiningApproachPhase.VERIFY,
        target_id=12,
        database_position=MapPoint(50.0, 50.0),
        started_at=1.0,
        phase_entered_at=1.0,
        clicked_at=1.0,
    )
    perception = MiningPerception(
        signal=MiningSignal.VERIFYING,
        candidate_id=12,
        candidate_position=MapPoint(50.0, 50.0),
        interaction_outcome="success",
    )

    result = controller.reduce(
        controller.state,
        _snapshot(perception, now=2.0),
        now=2.0,
    )
    controller.state = result.state

    assert result.state.phase is MiningApproachPhase.RESUME
    assert not controller.active
    assert result.intent is not None
    assert result.intent.commands[0].kind is CommandKind.RELEASE_KEY


def test_resume_terminal_rearms_controller_for_the_next_visible_candidate() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.RESUME,
        target_id=12,
        outcome="success",
    )
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        candidate_id=13,
        candidate_position=MapPoint(50.30, 50.0),
    )

    result = controller.reduce(state, _snapshot(perception, now=3.0), now=3.0)

    assert result.state.target_id == 13
    assert result.state.phase is MiningApproachPhase.CONTINUOUS_APPROACH


def test_failed_target_does_not_immediately_loop_while_marker_remains() -> None:
    controller = _controller()
    state = MiningApproachState(
        phase=MiningApproachPhase.FAILED,
        target_id=12,
        phase_entered_at=1.0,
        outcome="visual_search_timeout",
    )
    perception = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=((150, 130),),
    )

    result = controller.reduce(state, _snapshot(perception, now=3.0), now=3.0)

    assert result.state.phase is MiningApproachPhase.FAILED
    assert result.intent is None
    assert result.reason == "failed_target_waiting_for_marker_clear"


def test_failed_marker_is_filtered_during_cooldown_but_new_marker_is_not() -> None:
    controller = _controller()
    controller.state = MiningApproachState(
        phase=MiningApproachPhase.FAILED,
        pending_marker_point=(150, 130),
        phase_entered_at=1.0,
        outcome="visual_search_timeout",
    )
    same = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=((152, 131),),
    )
    new = MiningPerception(
        signal=MiningSignal.CANDIDATE,
        bright_minimap_points=((220, 200),),
    )

    assert controller.filter_perception(same, now=2.0).signal is MiningSignal.IDLE
    assert controller.filter_perception(new, now=2.0).signal is MiningSignal.CANDIDATE
    assert controller.filter_perception(same, now=31.1).signal is MiningSignal.CANDIDATE
