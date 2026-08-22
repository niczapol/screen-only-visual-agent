"""Remove reproducible project artifacts while preserving source evidence.

The default mode is a read-only dry run. Only explicitly allowlisted generated
directories and known broken/downloaded files can be removed with ``--apply``.
The optional ``--reviewed-live-media`` scope also removes videos and explicitly
superseded sessions after their findings have been reviewed. Metadata, labels,
models, maps, active datasets, and raw frames in retained sessions are preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupPlan:
    directories: tuple[Path, ...]
    files: tuple[Path, ...]
    file_count: int
    byte_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--apply", action="store_true", help="Perform the allowlisted removal.")
    parser.add_argument(
        "--reviewed-live-media",
        action="store_true",
        help="Also remove reviewed live videos and explicitly superseded live/debug sessions.",
    )
    return parser.parse_args()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_cleanup_plan(root: Path, *, include_reviewed_live_media: bool = False) -> CleanupPlan:
    root = root.resolve()
    data_root = root / "data"
    live_roots = sorted(
        path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("live")
    ) if data_root.is_dir() else []

    directories: set[Path] = set()
    for live_root in live_roots:
        directories.update(path for path in live_root.rglob("reports") if path.is_dir())

    explicit_directories = (
        data_root / "vision_dataset" / "overlays",
        root / "build",
    )
    directories.update(path for path in explicit_directories if path.is_dir())
    directories.update(path for path in root.glob(".pytest*") if path.is_dir())
    artifacts_root = root / "artifacts"
    debug_root = root / "debug_output"
    if artifacts_root.is_dir():
        directories.update(path for path in artifacts_root.glob("pytest-*") if path.is_dir())
        directories.update(path for path in artifacts_root.glob("overnight_dry_run*") if path.is_dir())
    if debug_root.is_dir():
        directories.update(path for path in debug_root.glob("pytest_*") if path.is_dir())

    if include_reviewed_live_media:
        reviewed_directories = (
            root / "artifacts" / "live",
            root / "debug_output" / "turn_scan_dataset_test_1837",
            root / "debug_output" / "window_video_smoke",
            data_root / "live_overnight_20260804_033122",
            data_root / "live_overnight_20260804_034239",
            data_root / "live_overnight_20260804_034311",
            data_root / "live_overnight_20260804_034429",
            data_root / "live_desolace_autonomous_entry_v03_20260801_run3",
            data_root / "live_route_probe_combat_modal_ignored_20260731_1651",
            data_root / "live_route_probe_barrens_run3d_combat_20260731_032724",
            data_root / "live_desolace_death_recovery_v7_20260801_0332",
            data_root / "live_tanaris_dense_mining_v03_20260802_172629",
            data_root / "live_navigation_actions_guard_check",
            data_root / "live_desolace_v03_route_20260801_223239",
            data_root / "live_desolace_v03_closed_loop_focus_20260801",
            data_root / "live_reconnect_20260801_iteration2",
            data_root / "live_hp_probe_20260730_205003",
            data_root / "live_death_recovery_smoke_20260730_2000",
            data_root / "live_navigation_manual_input_check",
        )
        directories.update(path for path in reviewed_directories if path.is_dir())

    files: set[Path] = set()
    explicit_files = (
        data_root
        / "live_desolace_v03_attacker_servo_20260801_235728"
        / "window_capture.mp4",
        root / "debug_output" / "server_data_research" / "ac_data_v20" / "Data.zip",
    )
    files.update(path for path in explicit_files if path.is_file())
    if include_reviewed_live_media:
        # Superseded root-level login/gate screenshots from completed v0.8
        # iterations. Their behavioral findings are in iteration reports and
        # compact evidence packs; retaining these unrelated full-screen PNGs
        # only duplicates reviewed media.
        reviewed_root_media = (
            "ascension_launcher_after_retry.png",
            "ascension_launcher_current.png",
            "live_v0812_after_forbidden.png",
            "live_v0812_start_state.png",
            "live_v0813_recovery_current.png",
            "live_v0814_login_after_retry.png",
            "live_v0814_login_retry2.png",
            "live_v0814_login_retry3.png",
            "live_v0814_pre_run5.png",
            "live_v0816_after_login.png",
            "live_v0816_after_login2.png",
            "live_v0816_current_screen.png",
            "live_v0816_login_result.png",
            "live_v0816_world_ready.png",
            "live_v0817_after_okay.png",
            "live_v0817_after_okay_dpi.png",
            "live_v0817_after_okay_scaled.png",
            "live_v0817_login_result.png",
            "live_v0817_login_result2.png",
            "live_v0817_run8_start_gate.png",
            "live_v0817_world_ready2.png",
            "v0815_after_login.png",
            "v0815_dpi_cursor.png",
            "v0815_login_result.png",
            "v0815_login_retry.png",
            "v0815_pre_live.png",
            "v0815_scaled_cursor.png",
            "v0815_world_gate.png",
            "v0815_world_ready.png",
            "v0819_login_after_okay.png",
            "v0819_login_after_okay_dpi.png",
            "v0819_login_result.png",
            "v0819_run5_current.png",
            "v0819_run5_stalled.png",
            "v0819_world_ready.png",
            "v0821_after_login.png",
            "v0821_before_run3_real.png",
            "v0821_gate_final_check.png",
            "v0821_run1_final_check.png",
            "v0821_run2_diag_final.png",
            "v0821_run2b_final.png",
            "v0821_start_gate.png",
        )
        files.update(
            data_root / name
            for name in reviewed_root_media
            if (data_root / name).is_file()
        )
    for live_root in live_roots:
        for path in live_root.rglob("*report*.png"):
            if path.is_file() and "reports" not in path.relative_to(live_root).parts:
                files.add(path)
        if include_reviewed_live_media:
            files.update(
                path
                for path in live_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".avi", ".mkv", ".mp4"}
            )

    safe_directories = tuple(sorted((_validate(path, root) for path in directories), key=str))
    validated_files = tuple(_validate(path, root) for path in files)
    safe_files = tuple(
        sorted(
            (
                path
                for path in validated_files
                if not any(path.is_relative_to(directory) for directory in safe_directories)
            ),
            key=str,
        )
    )

    file_count = len(safe_files)
    byte_count = sum(path.stat().st_size for path in safe_files)
    for directory in safe_directories:
        for path in directory.rglob("*"):
            if path.is_file():
                file_count += 1
                byte_count += path.stat().st_size
    return CleanupPlan(safe_directories, safe_files, file_count, byte_count)


def apply_cleanup_plan(plan: CleanupPlan, root: Path) -> None:
    root = root.resolve()
    for path in plan.files:
        _validate(path, root).unlink(missing_ok=True)
    for path in sorted(plan.directories, key=lambda item: len(item.parts), reverse=True):
        safe_path = _validate(path, root)
        if safe_path.is_dir():
            shutil.rmtree(safe_path)


def plan_payload(plan: CleanupPlan, root: Path, applied: bool) -> dict[str, object]:
    root = root.resolve()
    return {
        "applied": applied,
        "file_count": plan.file_count,
        "byte_count": plan.byte_count,
        "size_gib": round(plan.byte_count / (1024**3), 3),
        "directories": [str(path.relative_to(root)) for path in plan.directories],
        "files": [str(path.relative_to(root)) for path in plan.files],
    }


def _validate(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not is_within(resolved, root):
        raise ValueError(f"Refusing unsafe cleanup target: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    plan = build_cleanup_plan(root, include_reviewed_live_media=bool(args.reviewed_live_media))
    if args.apply:
        apply_cleanup_plan(plan, root)
    print(json.dumps(plan_payload(plan, root, bool(args.apply)), indent=2))


if __name__ == "__main__":
    main()
