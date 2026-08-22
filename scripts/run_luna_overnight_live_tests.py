from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from vision_bot.coords import coord_to_xy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
CLIENT_LAUNCHER = PROJECT_ROOT / "scripts" / "launch_ascension_medium.ps1"

PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "desolace": {
        "locations": [102],
        "database_path": "data/routes/generated/desolace_shadowprey_to_kodo_north_topographic_cycle.json",
        "permanent_exclusions_path": "data/permanent_exclusions_desolace.json",
    },
    "tanaris": {
        "locations": [162],
        "database_path": "data/routes/generated/tanaris_terrain_coverage_cycle_v4.json",
        "permanent_exclusions_path": "data/permanent_exclusions.json",
    },
}

TERMINATIONS_REQUIRING_REVIEW = frozenset(
    {
        "death_recovery_failed",
        "forbidden_underground_subzone",
        "safe_drain_limit",
        "external_stop_safe_drain_timeout",
        "route_entry_no_target",
        "route_entry_blocked",
        "target_blocked_no_target",
    }
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    duration_seconds: float
    combat_loot: bool


@dataclass(frozen=True)
class WatchdogDecision:
    reason: str
    details: dict[str, Any]


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.remainder = ""

    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            text = handle.read()
            self.offset = handle.tell()
        if not text:
            return []
        chunks = (self.remainder + text).split("\n")
        self.remainder = chunks.pop()
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                value = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows


class RouteLiveWatchdog:
    def __init__(
        self,
        *,
        route_window_seconds: float = 90.0,
        coordinate_loss_seconds: float = 20.0,
        combat_timeout_seconds: float = 240.0,
        recovery_timeout_seconds: float = 300.0,
        mining_timeout_seconds: float = 60.0,
    ) -> None:
        self.route_window_seconds = route_window_seconds
        self.coordinate_loss_seconds = coordinate_loss_seconds
        self.combat_timeout_seconds = combat_timeout_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.mining_timeout_seconds = mining_timeout_seconds
        self.armor_critical_frames = 0
        self.coordinate_loss_started: float | None = None
        self.combat_started: float | None = None
        self.recovery_started: float | None = None
        self.mining_started: float | None = None
        self.route_samples: deque[tuple[float, float, float, float, int]] = deque()
        self.skip_samples: deque[tuple[float, int]] = deque()

    @staticmethod
    def _active_states(row: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
        combat = bool((row.get("combat") or {}).get("active", False))
        recovery = bool((row.get("game_state") or {}).get("death_or_blocking_modal", False))
        mining_phase = str((row.get("mining") or {}).get("phase", "idle"))
        mining = mining_phase not in {
            "",
            "idle",
            "disabled",
            "complete",
            "failed",
            "suspended",
        }
        loot = bool((row.get("post_combat_loot") or {}).get("active", False))
        return combat, recovery, mining, loot

    @staticmethod
    def _track_duration(
        active: bool,
        started: float | None,
        now: float,
    ) -> float | None:
        if not active:
            return None
        return now if started is None else started

    def observe(self, row: dict[str, Any], *, now: float) -> WatchdogDecision | None:
        self.armor_critical_frames = (
            self.armor_critical_frames + 1 if bool(row.get("armor_critical")) else 0
        )
        if self.armor_critical_frames >= 3:
            return WatchdogDecision("armor_critical", {"frames": self.armor_critical_frames})

        combat, recovery, mining, loot = self._active_states(row)
        self.combat_started = self._track_duration(combat, self.combat_started, now)
        self.recovery_started = self._track_duration(recovery, self.recovery_started, now)
        self.mining_started = self._track_duration(mining, self.mining_started, now)
        if self.combat_started is not None and now - self.combat_started >= self.combat_timeout_seconds:
            return WatchdogDecision(
                "combat_timeout",
                {"seconds": round(now - self.combat_started, 1)},
            )
        if self.recovery_started is not None and now - self.recovery_started >= self.recovery_timeout_seconds:
            return WatchdogDecision(
                "death_recovery_timeout",
                {"seconds": round(now - self.recovery_started, 1)},
            )
        if self.mining_started is not None and now - self.mining_started >= self.mining_timeout_seconds:
            return WatchdogDecision(
                "mining_transaction_timeout",
                {"seconds": round(now - self.mining_started, 1)},
            )

        busy = combat or recovery or mining or loot
        coord = row.get("coord")
        feedback_age = row.get("coord_feedback_age")
        coordinate_missing = coord is None or (
            isinstance(feedback_age, (int, float)) and float(feedback_age) > 15.0
        )
        if not busy and coordinate_missing:
            if self.coordinate_loss_started is None:
                self.coordinate_loss_started = now
            elif now - self.coordinate_loss_started >= self.coordinate_loss_seconds:
                return WatchdogDecision(
                    "coordinate_feedback_lost",
                    {"seconds": round(now - self.coordinate_loss_started, 1)},
                )
        else:
            self.coordinate_loss_started = None

        skipped = int(row.get("skipped_targets", 0) or 0)
        self.skip_samples.append((now, skipped))
        while self.skip_samples and now - self.skip_samples[0][0] > 60.0:
            self.skip_samples.popleft()
        if (
            len(self.skip_samples) >= 2
            and self.skip_samples[-1][1] - self.skip_samples[0][1] >= 10
        ):
            return WatchdogDecision(
                "route_skip_storm",
                {"skipped_in_window": self.skip_samples[-1][1] - self.skip_samples[0][1]},
            )

        route_progress = (row.get("route_following") or {}).get("progress")
        completed = int(row.get("completed_targets", 0) or 0)
        if busy or not isinstance(coord, int) or not isinstance(route_progress, (int, float)):
            self.route_samples.clear()
            return None
        x, y = coord_to_xy(coord)
        self.route_samples.append((now, x, y, float(route_progress), completed))
        while self.route_samples and now - self.route_samples[0][0] > self.route_window_seconds:
            self.route_samples.popleft()
        if len(self.route_samples) < 3:
            return None
        first = self.route_samples[0]
        last = self.route_samples[-1]
        if last[0] - first[0] < self.route_window_seconds * 0.90:
            return None
        progress_delta = last[3] - first[3]
        if last[4] != first[4] or progress_delta >= 0.50:
            return None
        path_length = sum(
            math.hypot(right[1] - left[1], right[2] - left[2])
            for left, right in zip(self.route_samples, list(self.route_samples)[1:])
        )
        net_displacement = math.hypot(last[1] - first[1], last[2] - first[2])
        details = {
            "window_seconds": round(last[0] - first[0], 1),
            "route_progress_delta": round(progress_delta, 3),
            "path_length": round(path_length, 3),
            "net_displacement": round(net_displacement, 3),
        }
        if path_length < 0.35:
            return WatchdogDecision("stationary_without_route_progress", details)
        if path_length >= 2.0 and net_displacement < 0.50:
            return WatchdogDecision("route_looping_without_progress", details)
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_profile(config: dict[str, Any], profile: str) -> None:
    if profile == "current":
        return
    route_cfg = config.setdefault("route", {})
    route_cfg.update(PROFILE_OVERRIDES[profile])
    route_cfg["follow_route_loop"] = True


def _parse_durations(value: str) -> list[float]:
    durations = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(durations) < 3 or any(item <= 0.0 for item in durations):
        raise argparse.ArgumentTypeError("Provide at least three positive comma-separated durations")
    return durations


def _run_specs(durations: Iterable[float]) -> list[RunSpec]:
    values = list(durations)
    return [
        RunSpec("01_core_route_combat_mining", values[0], False),
        RunSpec("02_route_mining_combat_loot", values[1], True),
        RunSpec("03_endurance_all_enabled", values[2], True),
        *[
            RunSpec(f"{index + 1:02d}_extended_all_enabled", duration, True)
            for index, duration in enumerate(values[3:], start=3)
        ],
    ]


def _preflight_command(config_path: Path, *, combat_loot: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vision_bot.runner",
        "--config",
        str(config_path),
        "--route-live-preflight",
        "--route-live-enable-mining",
    ]
    if combat_loot:
        command.append("--route-live-enable-combat-loot")
    return command


def _live_command(
    config_path: Path,
    output_dir: Path,
    stop_file: Path,
    *,
    combat_loot: bool,
    frame_interval: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vision_bot.runner",
        "--config",
        str(config_path),
        "--route-live-test",
        "20000",
        "--route-live-output-dir",
        str(output_dir),
        "--route-live-follow-loop",
        "--route-live-enable-mining",
        "--route-live-cycle-targets",
        "9999",
        "--route-live-frame-interval",
        str(frame_interval),
        "--route-live-report-interval",
        "10",
        "--route-live-stop-file",
        str(stop_file),
    ]
    if combat_loot:
        command.append("--route-live-enable-combat-loot")
    return command


def _remaining_resurrection_wait(config: dict[str, Any]) -> float:
    value = (
        config.get("safety", {})
        .get("death_recovery", {})
        .get("resurrection_sickness_state_path", "data/runtime/resurrection_sickness.json")
    )
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        remaining = float(data.get("wait_until_epoch", 0.0)) - time.time()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    return max(0.0, remaining)


def _request_stop(stop_file: Path, decision: WatchdogDecision) -> None:
    _write_json(
        stop_file,
        {
            "requested_at_epoch": time.time(),
            "reason": decision.reason,
            "details": decision.details,
        },
    )


def _parse_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _termination_requires_review(summary: dict[str, Any] | None) -> bool:
    termination = str((summary or {}).get("termination_reason", ""))
    return termination in TERMINATIONS_REQUIRING_REVIEW


def _monitor_run(
    process: subprocess.Popen[Any],
    run_dir: Path,
    stop_file: Path,
    *,
    duration_seconds: float,
    min_free_gb: float,
    safe_drain_seconds: float,
) -> tuple[WatchdogDecision | None, bool]:
    metadata_path = run_dir / "metadata.jsonl"
    tail = JsonlTail(metadata_path)
    watchdog = RouteLiveWatchdog()
    started = time.time()
    last_metadata_at: float | None = None
    decision: WatchdogDecision | None = None
    forced = False
    while process.poll() is None:
        now = time.time()
        rows = tail.read_new()
        if rows:
            last_metadata_at = now
        for row in rows:
            decision = watchdog.observe(row, now=now)
            if decision is not None:
                break
        if decision is None and now - started >= duration_seconds:
            decision = WatchdogDecision(
                "max_wall_time",
                {"seconds": round(now - started, 1)},
            )
        if decision is None:
            free_gb = shutil.disk_usage(run_dir).free / (1024**3)
            if free_gb < min_free_gb:
                decision = WatchdogDecision(
                    "low_disk_space",
                    {"free_gb": round(free_gb, 2), "minimum_gb": min_free_gb},
                )
        if decision is None and last_metadata_at is None and now - started >= 60.0:
            decision = WatchdogDecision(
                "metadata_never_started",
                {"seconds": round(now - started, 1)},
            )
        if decision is None and last_metadata_at is not None and now - last_metadata_at >= 30.0:
            decision = WatchdogDecision(
                "metadata_stalled",
                {"seconds": round(now - last_metadata_at, 1)},
            )
        if decision is not None:
            _request_stop(stop_file, decision)
            deadline = time.time() + safe_drain_seconds + 30.0
            while process.poll() is None and time.time() < deadline:
                time.sleep(1.0)
            if process.poll() is None:
                forced = True
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    process.wait(timeout=15.0)
                except (AttributeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
            break
        time.sleep(1.0)
    return decision, forced


def _run_preflight(
    config_path: Path,
    output_path: Path,
    *,
    combat_loot: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        _preflight_command(config_path, combat_loot=combat_loot),
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Preflight failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not value.get("ready_offline"):
        raise RuntimeError(f"Preflight is not ready: {value.get('issues')}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three supervised route-live recordings with an automatic safety watchdog."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("current", "desolace", "tanaris"), default="current")
    parser.add_argument(
        "--durations",
        type=_parse_durations,
        default=_parse_durations("1200,1800,2700"),
        help="Wall-time seconds for at least three runs (default: 20m,30m,45m)",
    )
    parser.add_argument("--frame-interval", type=float, default=3.0)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--settle-seconds", type=float, default=30.0)
    parser.add_argument("--client-wait-seconds", type=float, default=20.0)
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--no-launch-client", action="store_true")
    parser.add_argument("--confirm-manual-gates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if not args.dry_run and not args.confirm_manual_gates:
        raise SystemExit(
            "Refusing live input. Re-run with --confirm-manual-gates only after the character is "
            "logged into the matching zone, Find Minerals is active, GatherMate pins are off, "
            "and C/G/E/X/1/F7 bindings are verified."
        )

    session_dir = args.session_root.resolve() / f"live_overnight_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=False)
    config = _load_yaml(args.config.resolve())
    _apply_profile(config, args.profile)
    route_live_cfg = config.setdefault("training_capture", {}).setdefault("route_live", {})
    route_live_cfg["frame_interval_seconds"] = max(0.0, float(args.frame_interval))
    route_live_cfg.setdefault("external_stop_safe_drain_seconds", 120.0)
    config_path = session_dir / "effective_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    specs = _run_specs(args.durations)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_epoch": time.time(),
        "profile": args.profile,
        "config": str(config_path),
        "dry_run": bool(args.dry_run),
        "runs": [],
    }
    _write_json(session_dir / "session_manifest.json", manifest)

    if not args.no_launch_client and not args.dry_run:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(CLIENT_LAUNCHER),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        time.sleep(max(0.0, float(args.client_wait_seconds)))

    for spec in specs:
        run_dir = session_dir / spec.name
        run_dir.mkdir(parents=True, exist_ok=False)
        stop_file = run_dir / "STOP_REQUESTED.json"
        preflight = _run_preflight(
            config_path,
            run_dir / "preflight.json",
            combat_loot=spec.combat_loot,
            env=env,
        )
        command = _live_command(
            config_path,
            run_dir,
            stop_file,
            combat_loot=spec.combat_loot,
            frame_interval=max(0.0, float(args.frame_interval)),
        )
        run_record: dict[str, Any] = {
            "name": spec.name,
            "duration_seconds": spec.duration_seconds,
            "combat_loot": spec.combat_loot,
            "preflight": preflight,
            "command": command,
            "output_dir": str(run_dir),
        }
        manifest["runs"].append(run_record)
        _write_json(session_dir / "session_manifest.json", manifest)
        if args.dry_run:
            run_record["status"] = "dry_run_prepared"
            _write_json(session_dir / "session_manifest.json", manifest)
            continue

        remaining_wait = _remaining_resurrection_wait(config)
        if remaining_wait > 0.0:
            run_record["resurrection_sickness_wait_seconds"] = round(remaining_wait, 1)
            _write_json(session_dir / "session_manifest.json", manifest)
            time.sleep(remaining_wait + 2.0)

        runner_log = (run_dir / "runner.log").open("w", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        decision, forced = _monitor_run(
            process,
            run_dir,
            stop_file,
            duration_seconds=spec.duration_seconds,
            min_free_gb=max(1.0, float(args.min_free_gb)),
            safe_drain_seconds=float(route_live_cfg["external_stop_safe_drain_seconds"]),
        )
        return_code = process.wait()
        runner_log.close()
        summary = _parse_summary(run_dir / "summary.json")
        run_record.update(
            {
                "status": "completed" if return_code == 0 else "runner_failed",
                "return_code": return_code,
                "watchdog": (
                    {"reason": decision.reason, "details": decision.details}
                    if decision is not None
                    else None
                ),
                "forced_process_stop": forced,
                "summary": summary,
            }
        )
        _write_json(session_dir / "session_manifest.json", manifest)

        safe_time_limit = decision is not None and decision.reason == "max_wall_time"
        if return_code != 0 or forced or (decision is not None and not safe_time_limit):
            manifest["status"] = "stopped_for_review"
            manifest["stop_reason"] = (
                decision.reason if decision is not None else f"runner_exit_{return_code}"
            )
            _write_json(session_dir / "session_manifest.json", manifest)
            return 2
        termination = str((summary or {}).get("termination_reason", ""))
        if _termination_requires_review(summary):
            manifest["status"] = "stopped_for_review"
            manifest["stop_reason"] = termination
            _write_json(session_dir / "session_manifest.json", manifest)
            return 2
        time.sleep(max(0.0, float(args.settle_seconds)))

    manifest["status"] = "completed"
    manifest["completed_at_epoch"] = time.time()
    _write_json(session_dir / "session_manifest.json", manifest)
    print(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
