from __future__ import annotations

from dataclasses import replace

import pytest

from vision_bot.cli import (
    _require_portfolio_live_opt_in,
    _require_safe_live_start,
    _shutdown_guard_is_quiet,
    build_parser,
    main,
)
from vision_bot.core.evidence import Evidence
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    ModalState,
    RecoveryPerception,
    WorldSnapshot,
)


def test_no_arguments_never_select_live_mode() -> None:
    assert main([]) == 2


def test_live_parser_requires_preflight_manifest_and_separate_enable_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "live",
            "--client-root",
            "client",
            "--preflight-manifest",
            "preflight.json",
        ]
    )

    assert args.command == "live"
    assert not args.enable_live_input


def test_legacy_runtime_is_only_available_through_explicit_subcommand() -> None:
    args = build_parser().parse_args(["legacy-gui"])

    assert args.command == "legacy-gui"
    assert not args.enable_legacy_input


def test_public_snapshot_requires_an_independent_local_live_opt_in() -> None:
    with pytest.raises(SystemExit, match="portfolio snapshot"):
        _require_portfolio_live_opt_in({"portfolio": {"allow_live_input": False}})

    _require_portfolio_live_opt_in({"portfolio": {"allow_live_input": True}})


def test_recognized_death_is_a_recovery_only_valid_live_start() -> None:
    now = 1.0
    frame_id = 0

    def present(value):
        return Evidence.present(
            value,
            observed_at=now,
            frame_id=frame_id,
            source="test",
        )

    snapshot = WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=present(True),
        pose=Evidence.unknown(
            observed_at=now,
            frame_id=frame_id,
            source="test",
            reason="death_pose_not_required",
        ),
        life=present(LifeState.DEAD),
        modal=present(ModalState.CLEAR),
        combat=present(CombatView(active=False)),
        mounted=present(False),
        mining=present(MiningPerception()),
        loot_pending=present(False),
        forbidden_subzone=present(False),
        recovery=present(RecoveryPerception()),
    )

    _require_safe_live_start(snapshot, object())


def test_shutdown_guard_requires_stable_alive_clear_state() -> None:
    now = 1.0
    frame_id = 1

    def present(value):
        return Evidence.present(
            value,
            observed_at=now,
            frame_id=frame_id,
            source="test",
        )

    base = WorldSnapshot(
        frame_id=frame_id,
        captured_at=now,
        input_ready=present(True),
        pose=Evidence.unknown(
            observed_at=now,
            frame_id=frame_id,
            source="test",
            reason="not_required",
        ),
        life=present(LifeState.ALIVE),
        modal=present(ModalState.CLEAR),
        combat=present(CombatView(active=False)),
        mounted=present(False),
        mining=present(MiningPerception()),
        loot_pending=present(False),
        forbidden_subzone=present(False),
        recovery=present(RecoveryPerception()),
    )

    assert _shutdown_guard_is_quiet(base, recovery_active=False)
    assert not _shutdown_guard_is_quiet(
        replace(base, combat=present(CombatView(active=True))),
        recovery_active=False,
    )
    assert not _shutdown_guard_is_quiet(base, recovery_active=True)
    assert _shutdown_guard_is_quiet(
        replace(base, life=present(LifeState.RESURRECTION_SICKNESS)),
        recovery_active=True,
    )
