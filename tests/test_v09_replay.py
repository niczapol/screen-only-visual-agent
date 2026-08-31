from __future__ import annotations

from vision_bot.core.commands import (
    Command,
    CommandGroup,
    CommandKind,
    ControlIntent,
    MouseButton,
)
from vision_bot.core.evidence import Evidence
from vision_bot.core.geometry import MapPoint
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    WorldSnapshot,
)
from vision_bot.engine.kernel import DeterministicKernel
from vision_bot.engine.supervisor import Supervisor
from vision_bot.replay import (
    EpisodeManifest,
    ReplayRunner,
    check_replay_invariants,
    compare_steps,
    snapshot_from_dict,
    snapshot_to_dict,
    read_episode,
    write_episode,
)


def _evidence(value, frame_id: int, observed_at: float):
    return Evidence.present(
        value,
        observed_at=observed_at,
        frame_id=frame_id,
        source="replay_test",
    )


def _snapshot(frame_id: int, *, combat: bool = False) -> WorldSnapshot:
    observed_at = float(frame_id)
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=observed_at,
        input_ready=_evidence(True, frame_id, observed_at),
        pose=_evidence(
            PlayerPose(162, MapPoint(50.0 + frame_id / 100.0, 40.0), 90.0),
            frame_id,
            observed_at,
        ),
        life=_evidence(LifeState.ALIVE, frame_id, observed_at),
        modal=_evidence(ModalState.CLEAR, frame_id, observed_at),
        combat=_evidence(
            CombatView(active=combat, attacker_count=2 if combat else 0),
            frame_id,
            observed_at,
        ),
        mounted=_evidence(True, frame_id, observed_at),
        mining=_evidence(MiningPerception(), frame_id, observed_at),
        loot_pending=_evidence(False, frame_id, observed_at),
        forbidden_subzone=_evidence(False, frame_id, observed_at),
        recovery=_evidence(RecoveryPerception(), frame_id, observed_at),
        metadata={"episode": "test"},
    )


class TravelController:
    def plan(self, snapshot, decision, *, now):
        return ControlIntent(
            CommandGroup.TRAVEL,
            (
                Command(
                    kind=CommandKind.HOLD_KEY,
                    group=CommandGroup.TRAVEL,
                    key="W",
                    reason="route_forward",
                ),
            ),
            "route_forward",
        )


class UnsafeMiningController:
    def plan(self, snapshot, decision, *, now):
        return ControlIntent(
            CommandGroup.MINING,
            (
                Command(
                    kind=CommandKind.CLICK,
                    group=CommandGroup.MINING,
                    mouse_button=MouseButton.RIGHT,
                    reason="unsafe_click",
                ),
            ),
            "unsafe_click",
        )


def test_snapshot_codec_round_trip_preserves_controller_input() -> None:
    snapshot = _snapshot(7, combat=True)

    restored = snapshot_from_dict(snapshot_to_dict(snapshot))

    assert restored == snapshot


def test_replay_is_deterministic() -> None:
    snapshots = [_snapshot(1), _snapshot(2), _snapshot(3, combat=True)]
    first = ReplayRunner(
        DeterministicKernel(
            Supervisor(),
            {CommandGroup.TRAVEL: TravelController()},
        )
    ).run(snapshots)
    second = ReplayRunner(
        DeterministicKernel(
            Supervisor(),
            {CommandGroup.TRAVEL: TravelController()},
        )
    ).run(snapshots)

    assert first.digest == second.digest
    assert first.state_counts == {"travel": 2, "combat": 1}
    assert not compare_steps(first.steps, second.steps)
    assert not check_replay_invariants(snapshots, first.steps)


def test_shadow_comparison_reports_control_change() -> None:
    snapshots = [_snapshot(1)]
    baseline = ReplayRunner(
        DeterministicKernel(Supervisor(), {CommandGroup.TRAVEL: TravelController()})
    ).run(snapshots)
    candidate = ReplayRunner(DeterministicKernel(Supervisor())).run(snapshots)

    differences = compare_steps(baseline.steps, candidate.steps)

    assert any(item.field == "intent" for item in differences)


def test_episode_jsonl_round_trip(tmp_path) -> None:
    snapshots = (_snapshot(1), _snapshot(2, combat=True))
    path = tmp_path / "episode.jsonl"

    write_episode(
        path,
        EpisodeManifest(name="two_frame_gate", config_fingerprint="abc"),
        snapshots,
    )
    restored = read_episode(path)

    assert restored.manifest.name == "two_frame_gate"
    assert restored.manifest.config_fingerprint == "abc"
    assert restored.snapshots == snapshots
