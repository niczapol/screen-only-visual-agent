from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Sequence

from vision_bot.addon_install import install_addon
from vision_bot.capture import ScreenCapture
from vision_bot.config import load_config
from vision_bot.core.commands import CommandGroup
from vision_bot.core.evidence import EvidenceState
from vision_bot.core.world import LifeState, ModalState
from vision_bot.engine.factory import V09KernelBundle, build_v09_kernel
from vision_bot.engine.input_executor import InputExecutor
from vision_bot.engine.orchestrator import RuntimeEventLog, V09FrameOrchestrator
from vision_bot.engine.preflight import (
    build_preflight_report,
    validate_preflight_manifest,
    write_preflight_report,
)
from vision_bot.engine.runtime import V09Runtime
from vision_bot.engine.windows_input_backend import WindowsInputBackend
from vision_bot.movement import InputController, MouseController
from vision_bot.perception.snapshot_builder import SnapshotBuilder
from vision_bot.replay.episode import read_episode
from vision_bot.replay.invariants import check_replay_invariants
from vision_bot.replay.runner import ReplayRunner


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHUTDOWN_GUARD_GROUPS = frozenset(
    {CommandGroup.RECOVERY, CommandGroup.COMBAT, CommandGroup.LOOT}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screen-vision-agent",
        description="Deterministic screen-only v0.9 runtime",
    )
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser("preflight", help="validate static live prerequisites")
    preflight.add_argument("--client-root", required=True)
    preflight.add_argument("--output", default="artifacts/v09/preflight.json")

    install = subparsers.add_parser("install-addon", help="install the repository addon")
    install.add_argument("--client-root", required=True)

    replay = subparsers.add_parser("replay", help="run a recorded v0.9 episode")
    replay.add_argument("episode")

    for name in ("shadow", "live"):
        command = subparsers.add_parser(name)
        command.add_argument("--client-root", required=True)
        command.add_argument("--max-frames", type=int, default=0)
        command.add_argument("--log", default=f"artifacts/v09/{name}.jsonl")
    live = subparsers.choices["live"]
    live.add_argument("--enable-live-input", action="store_true")
    live.add_argument("--preflight-manifest", required=True)
    live.add_argument("--purpose", choices=("validation", "production"), default="validation")

    legacy = subparsers.add_parser("legacy-gui", help="isolated v0.8 GUI entrypoint")
    legacy.add_argument("--enable-legacy-input", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print("\nNo mode selected; live input remains disabled.", file=sys.stderr)
        return 2
    config = load_config(args.config)

    if args.command == "install-addon":
        result = install_addon(REPOSITORY_ROOT, args.client_root)
        print(
            json.dumps(
                {
                    "destination": str(result.destination),
                    "files": [str(path) for path in result.copied_files],
                    "fingerprint": result.fingerprint,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "preflight":
        report = build_preflight_report(
            config,
            repository_root=REPOSITORY_ROOT,
            client_root=args.client_root,
        )
        write_preflight_report(args.output, report)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ready_for_validation else 1
    if args.command == "legacy-gui":
        if not args.enable_legacy_input:
            raise SystemExit("legacy GUI requires --enable-legacy-input")
        _require_portfolio_live_opt_in(config)
        from vision_bot.runner import MiningRouter

        MiningRouter(config).run()
        return 0

    bundle = build_v09_kernel(config)
    if args.command == "replay":
        episode = read_episode(args.episode)
        result = ReplayRunner(bundle.kernel).run(episode.snapshots)
        violations = check_replay_invariants(episode.snapshots, result.steps)
        if violations:
            detail = "; ".join(
                f"frame={item.frame_id}:{item.invariant}:{item.detail}"
                for item in violations[:10]
            )
            raise RuntimeError(f"replay invariant violation: {detail}")
        print(
            json.dumps(
                {
                    "episode": episode.manifest.name,
                    "frames": len(episode.snapshots),
                    "digest": result.digest,
                    "states": result.state_counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "live":
        if not args.enable_live_input:
            raise SystemExit("live mode requires --enable-live-input")
        _require_portfolio_live_opt_in(config)
        report = build_preflight_report(
            config,
            repository_root=REPOSITORY_ROOT,
            client_root=args.client_root,
            bundle=bundle,
        )
        validate_preflight_manifest(
            args.preflight_manifest,
            report,
            purpose=args.purpose,
        )
        return _run_screen_mode(
            config,
            bundle,
            live=True,
            max_frames=args.max_frames,
            log_path=args.log,
            config_fingerprint=report.fingerprint,
        )
    report = build_preflight_report(
        config,
        repository_root=REPOSITORY_ROOT,
        client_root=args.client_root,
        bundle=bundle,
    )
    if not report.ready_for_shadow:
        raise RuntimeError("shadow start blocked: static preflight is not ready")
    return _run_screen_mode(
        config,
        bundle,
        live=False,
        max_frames=args.max_frames,
        log_path=args.log,
        config_fingerprint=report.fingerprint,
    )


def _require_portfolio_live_opt_in(config: dict) -> None:
    """Keep the public snapshot inert unless a local config opts in explicitly."""

    if bool(config.get("portfolio", {}).get("allow_live_input", False)):
        return
    raise SystemExit(
        "live input is disabled in this portfolio snapshot; use an ignored "
        "local config and set portfolio.allow_live_input=true only for an "
        "authorized environment"
    )


def _run_screen_mode(
    config: dict,
    bundle: V09KernelBundle,
    *,
    live: bool,
    max_frames: int,
    log_path: str | Path,
    config_fingerprint: str,
) -> int:
    capture = ScreenCapture(config)
    if capture.find_window() is None:
        raise RuntimeError("game window not found; the runtime never launches the client")
    if live and not capture.activate_window():
        raise RuntimeError(
            "live start blocked: game window could not be confirmed foreground"
        )
    snapshot_builder = SnapshotBuilder(config, zone_id=bundle.config.zone.zone_id)

    if live:
        gate_frame = capture.capture_client_region()
        gate_minimap = capture.crop_minimap(gate_frame, config)
        gate_now = time.monotonic()
        gate_snapshot = snapshot_builder.observe(
            gate_frame,
            frame_id=0,
            captured_at=gate_now,
            input_ready=True,
            minimap=gate_minimap,
            recognize_ore=True,
        )
        _require_safe_live_start(gate_snapshot, bundle)

    executor = None
    if live:
        keyboard = InputController(
            target_hwnd=capture.window_handle,
            backend=str(config.get("input", {}).get("backend", "sendinput")),
        )
        mouse = MouseController(target_hwnd=capture.window_handle)
        executor = InputExecutor(
            WindowsInputBackend(keyboard, mouse),
            cursor_settle_seconds=float(
                config.get("input", {}).get("cursor_settle_seconds", 0.12)
            ),
            client_to_screen=capture.client_to_screen_point,
        )
    runtime = V09Runtime(
        bundle.kernel,
        executor=executor,
        live_input_enabled=live,
    )
    event_log = RuntimeEventLog(
        log_path,
        name=f"v09-{'live' if live else 'shadow'}-{int(time.time())}",
        source="v09-live" if live else "v09-shadow",
        config_fingerprint=config_fingerprint,
    )
    orchestrator = V09FrameOrchestrator(
        config,
        bundle,
        runtime,
        snapshot_builder=snapshot_builder,
        event_log=event_log,
    )
    model_state = orchestrator.mining_sensor.detector.load()
    if live and model_state.reason != "model_ready":
        orchestrator.close()
        raise RuntimeError(
            "live start blocked: world-ore model runtime is unavailable "
            f"({model_state.reason})"
        )
    if executor is not None:
        executor.start_background()
    frame_id = 1
    period = 1.0 / bundle.config.performance.control_hz
    try:
        while max_frames <= 0 or frame_id <= max_frames:
            started = time.monotonic()
            frame = capture.capture_client_region()
            minimap = capture.crop_minimap(frame, config)
            outcome = orchestrator.process_frame(
                frame,
                frame_id=frame_id,
                captured_at=started,
                input_ready=True,
                cursor_client_point=_cursor_client_point(capture),
                minimap=minimap,
            )
            if outcome.runtime.kernel.decision.state.value in {"fault", "paused", "wait_for_client"}:
                # The executor has already released/cancelled input through the
                # supervisor decision.  Emit transitions for operator triage.
                print(
                    json.dumps(
                        {
                            "frame": frame_id,
                            "state": outcome.runtime.kernel.decision.state.value,
                            "reason": outcome.runtime.kernel.decision.reason,
                        }
                    )
                )
            frame_id += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        if live and max_frames > 0:
            frame_id = _run_live_shutdown_guard(
                config,
                bundle,
                capture=capture,
                orchestrator=orchestrator,
                executor=executor,
                frame_id=frame_id,
                period=period,
            )
    except KeyboardInterrupt:
        return 130
    finally:
        runtime.reset(now=time.monotonic())
        if executor is not None:
            executor.stop_background(
                now=time.monotonic(),
                reason="screen_mode_exit",
            )
        orchestrator.close()
    return 0


def _run_live_shutdown_guard(
    config: dict,
    bundle: V09KernelBundle,
    *,
    capture: ScreenCapture,
    orchestrator: V09FrameOrchestrator,
    executor: InputExecutor,
    frame_id: int,
    period: float,
) -> int:
    """Drain a bounded live run without resuming travel from its stop point."""

    quiet_required = bundle.config.performance.shutdown_guard_quiet_seconds
    maximum = bundle.config.performance.shutdown_guard_max_seconds
    started_at = time.monotonic()
    quiet_since: float | None = None
    executor.set_input_permitted(False, now=started_at)
    print(
        json.dumps(
            {
                "state": "shutdown_guard",
                "reason": "bounded_frame_budget_complete",
                "quiet_seconds_required": quiet_required,
                "maximum_seconds": maximum,
            }
        )
    )
    while time.monotonic() - started_at < maximum:
        frame_started = time.monotonic()
        frame = capture.capture_client_region()
        minimap = capture.crop_minimap(frame, config)
        outcome = orchestrator.process_frame(
            frame,
            frame_id=frame_id,
            captured_at=frame_started,
            input_ready=True,
            cursor_client_point=_cursor_client_point(capture),
            minimap=minimap,
            permitted_groups=SHUTDOWN_GUARD_GROUPS,
        )
        if _shutdown_guard_is_quiet(
            outcome.snapshot,
            recovery_active=orchestrator.recovery_controller.active,
        ):
            if quiet_since is None:
                quiet_since = frame_started
            elif frame_started - quiet_since >= quiet_required:
                print(
                    json.dumps(
                        {
                            "frame": frame_id,
                            "state": "shutdown_guard_complete",
                            "quiet_seconds": frame_started - quiet_since,
                        }
                    )
                )
                return frame_id + 1
        else:
            quiet_since = None
        frame_id += 1
        remaining = period - (time.monotonic() - frame_started)
        if remaining > 0.0:
            time.sleep(remaining)
    raise RuntimeError(
        "live shutdown guard timed out before a stable safe state; "
        "input was released during final cleanup"
    )


def _shutdown_guard_is_quiet(snapshot, *, recovery_active: bool) -> bool:
    life = snapshot.life.value
    combat = snapshot.combat.value
    return bool(
        life in {LifeState.ALIVE, LifeState.RESURRECTION_SICKNESS}
        and snapshot.modal.value is ModalState.CLEAR
        and combat is not None
        and not combat.active
        # Sickness is an authoritative completed-resurrection state.  The
        # supervisor intentionally suppresses controllers there, so a stale
        # in-memory recovery phase must not hold the shutdown guard for ten
        # minutes after the safe transition has already completed.
        and (life is LifeState.RESURRECTION_SICKNESS or not recovery_active)
    )


def _require_safe_live_start(snapshot, bundle: V09KernelBundle) -> None:
    if snapshot.input_ready.state is not EvidenceState.PRESENT or not snapshot.input_ready.value:
        raise RuntimeError("live start blocked: visible addon protocol v2 is absent or stale")
    life = snapshot.life.value
    if life in {LifeState.DEAD, LifeState.GHOST}:
        # A recognized death state is a valid recovery-only live start.  The
        # supervisor gives RECOVERY exclusive ownership before modal/combat,
        # route or mounting can issue any command.
        return
    if life is not LifeState.ALIVE:
        raise RuntimeError(f"live start blocked: life state is {life}")
    if snapshot.modal.value is not ModalState.CLEAR:
        raise RuntimeError("live start blocked: a modal is visible")
    if snapshot.combat.value is not None and snapshot.combat.value.active:
        raise RuntimeError("live start blocked: combat is already active")
    if snapshot.forbidden_subzone.value:
        raise RuntimeError("live start blocked: forbidden subzone marker is visible")
    if bool(snapshot.metadata.get("armor_critical")):
        raise RuntimeError("live start blocked: durability must be repaired")
    pose = snapshot.pose.value
    if pose is None or pose.zone_id != bundle.config.zone.zone_id:
        raise RuntimeError("live start blocked: current zone/position is unknown")
    projection = bundle.route.project(pose.position)
    if projection.cross_track_yards > bundle.config.navigation.corridor_radius_yards:
        raise RuntimeError(
            "live start blocked: character is outside the configured rail corridor "
            f"({projection.cross_track_yards:.1f} yd)"
        )


def _cursor_client_point(capture: ScreenCapture) -> tuple[int, int] | None:
    class Point(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    point = Point()
    try:
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return None
    except (AttributeError, OSError):
        return None
    return capture.screen_to_client_point(point.x, point.y)


if __name__ == "__main__":
    raise SystemExit(main())
