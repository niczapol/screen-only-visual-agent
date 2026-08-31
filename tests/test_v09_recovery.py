from __future__ import annotations

from pathlib import Path

import yaml

from vision_bot.controllers.recovery import (
    RecoveryController,
    RecoveryControllerState,
    RecoveryPhase,
)
from vision_bot.core.commands import CommandKind, MouseButton
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    RecoverySignal,
    WorldSnapshot,
)
from vision_bot.v09_config import V09Config


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
    life: LifeState,
    recovery: RecoveryPerception,
    *,
    frame_id: int,
    now: float,
    position: MapPoint | None = None,
    heading: float = 180.0,
) -> WorldSnapshot:
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=_evidence(True, frame_id, now),
        pose=_evidence(
            PlayerPose(162, position or MapPoint(50.0, 50.0), heading),
            frame_id,
            now,
        ),
        life=_evidence(life, frame_id, now),
        modal=_evidence(ModalState.CLEAR, frame_id, now),
        combat=_evidence(CombatView(active=False), frame_id, now),
        mounted=_evidence(False, frame_id, now),
        mining=_evidence(MiningPerception(), frame_id, now),
        loot_pending=_evidence(False, frame_id, now),
        forbidden_subzone=_evidence(False, frame_id, now),
        recovery=_evidence(recovery, frame_id, now),
    )


def _controller() -> RecoveryController:
    return RecoveryController(CONFIG.recovery, CONFIG.zone, CONFIG.navigation)


def test_release_spirit_requires_visible_click_point() -> None:
    controller = _controller()
    hidden = controller.reduce(
        RecoveryControllerState(),
        _snapshot(
            LifeState.DEAD,
            RecoveryPerception(signal=RecoverySignal.RELEASE_SPIRIT),
            frame_id=1,
            now=1.0,
        ),
        now=1.0,
    )
    assert hidden.intent is None
    assert hidden.action == "release_button_not_visible"

    visible = controller.reduce(
        hidden.state,
        _snapshot(
            LifeState.DEAD,
            RecoveryPerception(
                signal=RecoverySignal.RELEASE_SPIRIT,
                click_point=(1200, 260),
            ),
            frame_id=2,
            now=1.1,
        ),
        now=1.1,
    )
    assert visible.state.phase is RecoveryPhase.WAIT_FOR_GHOST
    assert visible.intent is not None
    assert [command.kind for command in visible.intent.commands] == [
        CommandKind.MOVE_CURSOR,
        CommandKind.CLICK,
    ]
    assert visible.intent.commands[1].mouse_button is MouseButton.LEFT


def test_ghost_search_uses_C_as_primary_target_action() -> None:
    controller = _controller()

    result = controller.reduce(
        RecoveryControllerState(),
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.GHOST_SEARCH),
            frame_id=1,
            now=2.0,
        ),
        now=2.0,
    )

    assert result.state.phase is RecoveryPhase.SEEK_HEALER
    assert result.intent is not None
    assert result.intent.commands[0].key == "C"


def test_targeted_healer_uses_G_only_inside_anchor_interact_radius() -> None:
    controller = _controller()
    state = RecoveryControllerState(
        phase=RecoveryPhase.SEEK_HEALER,
        started_at=1.0,
        phase_entered_at=1.0,
    )

    result = controller.reduce(
        state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.HEALER_TARGETED),
            frame_id=2,
            now=2.0,
            position=CONFIG.recovery.healer_anchors[0],
        ),
        now=2.0,
    )

    assert result.state.phase is RecoveryPhase.INTERACT_HEALER
    assert result.intent is not None
    assert result.intent.commands[0].key == "G"


def test_targeted_healer_turns_to_authorized_anchor_before_bounded_W() -> None:
    controller = _controller()
    anchor = CONFIG.recovery.healer_anchors[3]
    position = MapPoint(anchor.x_percent + 0.30, anchor.y_percent - 0.40)
    desired = CONFIG.zone.heading_degrees(position, anchor)
    assert desired is not None
    state = RecoveryControllerState(
        phase=RecoveryPhase.INTERACT_HEALER,
        started_at=1.0,
        phase_entered_at=1.0,
    )

    turning = controller.reduce(
        state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.HEALER_TARGETED),
            frame_id=3,
            now=2.0,
            position=position,
            heading=(desired + 180.0) % 360.0,
        ),
        now=2.0,
    )
    assert turning.intent is not None
    assert any(
        command.kind is CommandKind.DRAG_RELATIVE
        for command in turning.intent.commands
    )

    approached = controller.reduce(
        turning.state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.HEALER_TARGETED),
            frame_id=4,
            now=2.1,
            position=position,
            heading=desired,
        ),
        now=2.1,
    )
    assert approached.intent is not None
    assert approached.intent.commands[0].key == "W"
    assert approached.intent.commands[0].duration == CONFIG.recovery.approach_seconds


def test_failed_recovery_is_terminal_while_ghost_signal_remains_visible() -> None:
    controller = _controller()
    failed = RecoveryControllerState(
        phase=RecoveryPhase.FAILED,
        started_at=1.0,
        phase_entered_at=2.0,
        outcome="test_failure",
    )

    result = controller.reduce(
        failed,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.HEALER_TARGETED),
            frame_id=5,
            now=3.0,
        ),
        now=3.0,
    )

    assert result.state.phase is RecoveryPhase.FAILED
    assert result.intent is None


def test_failed_release_resumes_when_client_visibly_auto_releases_to_ghost() -> None:
    controller = _controller()
    failed = RecoveryControllerState(
        phase=RecoveryPhase.FAILED,
        started_at=1.0,
        phase_entered_at=9.0,
        outcome="release_attempts_exhausted",
    )

    result = controller.reduce(
        failed,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(signal=RecoverySignal.GHOST_SEARCH),
            frame_id=6,
            now=10.0,
        ),
        now=10.0,
    )

    assert result.state.phase is RecoveryPhase.SEEK_HEALER
    assert result.intent is not None
    assert result.intent.commands[0].key == "C"


def test_release_retries_are_spaced_by_typed_interval() -> None:
    controller = _controller()
    state = RecoveryControllerState(
        phase=RecoveryPhase.WAIT_FOR_GHOST,
        started_at=1.0,
        phase_entered_at=2.0,
        last_action_at=2.0,
        release_attempts=1,
    )
    snapshot = _snapshot(
        LifeState.DEAD,
        RecoveryPerception(
            signal=RecoverySignal.RELEASE_SPIRIT,
            click_point=(1161, 290),
        ),
        frame_id=7,
        now=2.5,
    )

    early = controller.reduce(state, snapshot, now=2.5)
    due = controller.reduce(
        state,
        _snapshot(
            LifeState.DEAD,
            RecoveryPerception(
                signal=RecoverySignal.RELEASE_SPIRIT,
                click_point=(1161, 290),
            ),
            frame_id=8,
            now=3.6,
        ),
        now=3.6,
    )

    assert early.intent is None
    assert early.action == "awaiting_ghost_transition"
    assert due.intent is not None
    assert due.state.release_attempts == 2


def test_return_to_life_and_accept_require_their_own_visible_points() -> None:
    controller = _controller()
    state = RecoveryControllerState(
        phase=RecoveryPhase.WAIT_FOR_RETURN,
        started_at=1.0,
        phase_entered_at=1.0,
    )
    returned = controller.reduce(
        state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(
                signal=RecoverySignal.RETURN_TO_LIFE,
                click_point=(1300, 800),
            ),
            frame_id=2,
            now=2.0,
        ),
        now=2.0,
    )
    assert returned.state.phase is RecoveryPhase.RETURN_TO_LIFE
    assert returned.intent is not None

    accepted = controller.reduce(
        returned.state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(
                signal=RecoverySignal.ACCEPT_RESURRECTION,
                click_point=(1280, 850),
            ),
            frame_id=3,
            now=2.5,
        ),
        now=2.5,
    )
    assert accepted.state.phase is RecoveryPhase.ACCEPT_RESURRECTION
    assert accepted.intent is not None


def test_each_visible_recovery_phase_gets_its_own_bounded_attempt_budget() -> None:
    controller = _controller()
    state = RecoveryControllerState(
        phase=RecoveryPhase.RETURN_TO_LIFE,
        started_at=1.0,
        phase_entered_at=1.0,
        interact_attempts=controller.config.max_interact_attempts,
    )

    accepted = controller.reduce(
        state,
        _snapshot(
            LifeState.GHOST,
            RecoveryPerception(
                signal=RecoverySignal.ACCEPT_RESURRECTION,
                click_point=(1280, 850),
            ),
            frame_id=3,
            now=2.0,
        ),
        now=2.0,
    )

    assert accepted.state.phase is RecoveryPhase.ACCEPT_RESURRECTION
    assert accepted.state.interact_attempts == 1
    assert accepted.intent is not None


def test_sickness_completes_input_recovery_and_leaves_wait_to_supervisor() -> None:
    controller = _controller()
    state = RecoveryControllerState(
        phase=RecoveryPhase.ACCEPT_RESURRECTION,
        started_at=1.0,
        phase_entered_at=2.0,
    )

    result = controller.reduce(
        state,
        _snapshot(
            LifeState.RESURRECTION_SICKNESS,
            RecoveryPerception(signal=RecoverySignal.SICKNESS),
            frame_id=4,
            now=3.0,
        ),
        now=3.0,
    )

    assert result.state.phase is RecoveryPhase.COMPLETE
    assert result.intent is None
    assert result.action == "sickness_supervisor_wait"
