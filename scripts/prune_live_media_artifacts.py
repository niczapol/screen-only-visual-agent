"""Prune obsolete live-run media while preserving datasets and key evidence.

The bot produces high-volume PNG/MP4 evidence under ``data/live_*`` during live
tests. Most of those frames are temporary diagnostics once the useful samples
have been promoted into durable datasets or the failure has been written down.

This script deletes only allowlisted media suffixes from non-retained live-run
directories. It intentionally leaves JSON/JSONL/CSV/MD telemetry in place and
does not touch model datasets, runtime assets, routes, maps, source files, or
human demonstrations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


MEDIA_SUFFIXES = frozenset(
    {
        ".avi",
        ".bmp",
        ".jpeg",
        ".jpg",
        ".mkv",
        ".mp4",
        ".png",
        ".webm",
    }
)


# Keep full visual evidence for the current behavior target and the latest
# reviewed bot runs. Human demos are not live_* directories, so they are
# preserved by construction; they remain documented here for clarity.
DEFAULT_RETAIN_LIVE_NAMES = frozenset(
    {
        "live_v080_direct_node_mining_20260808_152018",
        "live_v080_rmb_mining_final_approach_20260808_144454",
        "live_v080_tanaris_rmb_endurance_20260808_140011",
        "live_v072_acceptance_20260808_115600_correct",
        "live_overnight_20260808_105157",
    }
)

DEFAULT_RETAIN_NON_LIVE_NAMES = frozenset(
    {
        "human_demo_tanaris_20260802_113953",
        "human_demo_tanaris_20260802_114252",
    }
)


@dataclass(frozen=True)
class PlannedFile:
    path: Path
    root_name: str
    byte_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--apply", action="store_true", help="Delete the planned files.")
    parser.add_argument(
        "--keep-live",
        action="append",
        default=[],
        help="Additional data/live_* directory name to keep with all media.",
    )
    parser.add_argument(
        "--drop-default-keeps",
        action="store_true",
        help="Ignore the built-in retained live-run list.",
    )
    return parser.parse_args()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not is_within(resolved, root):
        raise ValueError(f"Refusing unsafe prune target: {resolved}")
    return resolved


def iter_live_roots(root: Path) -> list[Path]:
    data_root = root / "data"
    debug_root = root / "debug_output"
    roots: list[Path] = []
    for base in (data_root, debug_root):
        if not base.is_dir():
            continue
        roots.extend(
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.lower().startswith("live")
        )
    return sorted(roots, key=lambda path: str(path).lower())


def build_plan(root: Path, keep_live_names: set[str]) -> list[PlannedFile]:
    root = root.resolve()
    planned: list[PlannedFile] = []
    for live_root in iter_live_roots(root):
        validate_path(live_root, root)
        if live_root.name in keep_live_names:
            continue
        if live_root.name in DEFAULT_RETAIN_NON_LIVE_NAMES:
            continue
        for path in live_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            safe_path = validate_path(path, root)
            planned.append(
                PlannedFile(
                    path=safe_path,
                    root_name=str(live_root.relative_to(root)),
                    byte_count=safe_path.stat().st_size,
                )
            )
    return planned


def remove_empty_dirs(root: Path) -> int:
    root = root.resolve()
    removed = 0
    for live_root in sorted(iter_live_roots(root), key=lambda item: len(item.parts), reverse=True):
        for directory in sorted(
            (path for path in live_root.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            safe_dir = validate_path(directory, root)
            try:
                safe_dir.rmdir()
            except OSError:
                continue
            removed += 1
    return removed


def apply_plan(plan: list[PlannedFile], root: Path) -> int:
    for item in plan:
        validate_path(item.path, root).unlink(missing_ok=True)
    return remove_empty_dirs(root)


def payload(plan: list[PlannedFile], root: Path, applied: bool, empty_dirs_removed: int) -> dict[str, object]:
    root = root.resolve()
    by_root: dict[str, dict[str, object]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for item in plan:
        by_root[item.root_name]["files"] = int(by_root[item.root_name]["files"]) + 1
        by_root[item.root_name]["bytes"] = int(by_root[item.root_name]["bytes"]) + item.byte_count

    top_roots = sorted(
        (
            {
                "root": root_name,
                "files": values["files"],
                "size_gib": round(int(values["bytes"]) / (1024**3), 3),
            }
            for root_name, values in by_root.items()
        ),
        key=lambda row: float(row["size_gib"]),
        reverse=True,
    )
    return {
        "applied": applied,
        "file_count": len(plan),
        "byte_count": sum(item.byte_count for item in plan),
        "size_gib": round(sum(item.byte_count for item in plan) / (1024**3), 3),
        "empty_dirs_removed": empty_dirs_removed,
        "retained_live_roots": sorted(
            name
            for name in DEFAULT_RETAIN_LIVE_NAMES
            if (root / "data" / name).exists() or (root / "debug_output" / name).exists()
        ),
        "retained_non_live_roots": sorted(
            name for name in DEFAULT_RETAIN_NON_LIVE_NAMES if (root / "data" / name).exists()
        ),
        "top_pruned_roots": top_roots[:50],
        "sample_files": [str(item.path.relative_to(root)) for item in plan[:25]],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    keep_live_names = set(args.keep_live)
    if not args.drop_default_keeps:
        keep_live_names.update(DEFAULT_RETAIN_LIVE_NAMES)
    plan = build_plan(root, keep_live_names)
    empty_dirs_removed = apply_plan(plan, root) if args.apply else 0
    print(json.dumps(payload(plan, root, bool(args.apply), empty_dirs_removed), indent=2))


if __name__ == "__main__":
    main()
