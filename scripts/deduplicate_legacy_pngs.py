"""Replace exact legacy-dataset PNG copies with NTFS hardlinks.

The command is read-only unless ``--apply`` is supplied. It scans only the
allowlisted legacy dataset roots, groups files by size and SHA-256, and never
links files whose bytes differ. Existing paths and manifests remain valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RELATIVE_ROOTS = (
    Path("data/vision_dataset_walkability_live"),
    Path("data/vision_dataset"),
    Path("data/vision_dataset_walkability"),
    Path("data/local_navigation_dataset"),
)


@dataclass(frozen=True)
class DuplicateGroup:
    canonical: Path
    duplicates: tuple[Path, ...]
    file_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def file_digest(path: Path) -> bytes:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").digest()


def build_duplicate_groups(root: Path, relative_roots: Iterable[Path] = RELATIVE_ROOTS) -> tuple[DuplicateGroup, ...]:
    root = root.resolve()
    candidates: list[Path] = []
    priority: dict[Path, int] = {}
    for rank, relative in enumerate(relative_roots):
        dataset_root = (root / relative).resolve()
        if not _is_within(dataset_root, root) or not dataset_root.is_dir():
            continue
        for path in dataset_root.rglob("*.png"):
            if path.is_file():
                resolved = path.resolve()
                candidates.append(resolved)
                priority[resolved] = rank

    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in candidates:
        by_size[path.stat().st_size].append(path)

    by_content: dict[tuple[int, bytes], list[Path]] = defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            by_content[(size, file_digest(path))].append(path)

    groups: list[DuplicateGroup] = []
    for (size, _digest), paths in by_content.items():
        if len(paths) < 2:
            continue
        ordered = sorted(paths, key=lambda path: (priority[path], str(path).lower()))
        canonical = ordered[0]
        duplicates = tuple(path for path in ordered[1:] if not os.path.samefile(canonical, path))
        if duplicates:
            groups.append(DuplicateGroup(canonical, duplicates, size))
    return tuple(sorted(groups, key=lambda group: str(group.canonical).lower()))


def apply_duplicate_groups(groups: Iterable[DuplicateGroup], root: Path) -> None:
    root = root.resolve()
    for group in groups:
        canonical = _validate(group.canonical, root)
        if canonical.stat().st_size != group.file_size:
            raise RuntimeError(f"Canonical file changed during deduplication: {canonical}")
        canonical_digest = file_digest(canonical)
        for duplicate in group.duplicates:
            duplicate = _validate(duplicate, root)
            if os.path.samefile(canonical, duplicate):
                continue
            if duplicate.stat().st_size != group.file_size or file_digest(duplicate) != canonical_digest:
                raise RuntimeError(f"Duplicate file changed during deduplication: {duplicate}")
            temporary = duplicate.with_name(duplicate.name + ".dedupe-link")
            temporary.unlink(missing_ok=True)
            try:
                os.link(canonical, temporary)
                os.replace(temporary, duplicate)
            finally:
                temporary.unlink(missing_ok=True)


def plan_payload(groups: Iterable[DuplicateGroup], root: Path, applied: bool) -> dict[str, object]:
    root = root.resolve()
    group_list = list(groups)
    duplicate_count = sum(len(group.duplicates) for group in group_list)
    byte_count = sum(group.file_size * len(group.duplicates) for group in group_list)
    return {
        "applied": applied,
        "groups": len(group_list),
        "duplicate_paths": duplicate_count,
        "physical_bytes_reclaimable": byte_count,
        "physical_gib_reclaimable": round(byte_count / (1024**3), 3),
        "sample_groups": [
            {
                "canonical": str(group.canonical.relative_to(root)),
                "duplicate_count": len(group.duplicates),
                "file_size": group.file_size,
            }
            for group in group_list[:25]
        ],
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not _is_within(resolved, root):
        raise ValueError(f"Refusing unsafe deduplication target: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    groups = build_duplicate_groups(root)
    if args.apply:
        apply_duplicate_groups(groups, root)
    print(json.dumps(plan_payload(groups, root, bool(args.apply)), indent=2))


if __name__ == "__main__":
    main()
