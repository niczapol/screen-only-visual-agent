import os
from pathlib import Path

from scripts.deduplicate_legacy_pngs import apply_duplicate_groups, build_duplicate_groups


def test_exact_legacy_png_duplicates_become_hardlinks(tmp_path: Path):
    canonical = tmp_path / "data" / "vision_dataset_walkability_live" / "images" / "bootstrap" / "a.png"
    duplicate = tmp_path / "data" / "vision_dataset" / "images" / "raw" / "a.png"
    second_duplicate = tmp_path / "data" / "local_navigation_dataset" / "images" / "raw" / "000_a.png"
    different = tmp_path / "data" / "vision_dataset_walkability" / "images" / "raw" / "b.png"
    for path in (canonical, duplicate, second_duplicate, different):
        path.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"exact-png-content")
    duplicate.write_bytes(b"exact-png-content")
    second_duplicate.write_bytes(b"exact-png-content")
    different.write_bytes(b"different-content")

    groups = build_duplicate_groups(tmp_path)
    assert sum(len(group.duplicates) for group in groups) == 2

    apply_duplicate_groups(groups, tmp_path)

    assert os.path.samefile(canonical, duplicate)
    assert os.path.samefile(canonical, second_duplicate)
    assert not os.path.samefile(canonical, different)
    assert different.read_bytes() == b"different-content"


def test_deduplication_is_idempotent(tmp_path: Path):
    first = tmp_path / "data" / "vision_dataset_walkability_live" / "a.png"
    second = tmp_path / "data" / "vision_dataset" / "a.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"same")
    second.write_bytes(b"same")

    apply_duplicate_groups(build_duplicate_groups(tmp_path), tmp_path)

    assert build_duplicate_groups(tmp_path) == ()
