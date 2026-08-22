from __future__ import annotations

import pytest

from pathlib import Path

from scripts.extract_wow_client_data import _extract_targets, _route_tiles, build_world_targets, parse_args


def test_extract_client_data_defaults_to_kalimdor() -> None:
    assert parse_args([]).map_directory == "Kalimdor"


def test_build_world_targets_supports_other_map_directories() -> None:
    targets = build_world_targets(" Azeroth ", [(31, 42), (32, 43)])

    assert targets == [
        "World\\Maps\\Azeroth\\Azeroth.wdt",
        "World\\Maps\\Azeroth\\Azeroth_31_42.adt",
        "World\\Maps\\Azeroth\\Azeroth_32_43.adt",
    ]


def test_route_tiles_use_wow_map_axis_orientation() -> None:
    route = {"route_loop": [{"x": 38.36, "y": 59.71}]}
    area = {
        "left": 4233.0,
        "right": -262.0,
        "top": 452.0,
        "bottom": -2545.0,
    }

    assert _route_tiles(route, area, padding=0) == [(27, 34)]


@pytest.mark.parametrize("value", ["", "   ", "World/Maps", "World\\Maps"])
def test_build_world_targets_rejects_invalid_map_directory(value: str) -> None:
    with pytest.raises(ValueError, match="one non-empty MPQ path component"):
        build_world_targets(value, [])


def test_extract_targets_directly_probes_paths_missing_from_listfile(monkeypatch) -> None:
    target = "World\\Maps\\Kalimdor\\Kalimdor_45_40.adt"

    class FakeArchive:
        files = [b"World\\Maps\\Kalimdor\\Kalimdor.wdt"]

        def __init__(self, _path: str) -> None:
            pass

        def read_file(self, path: str) -> bytes:
            if path == target:
                return b"terrain"
            raise KeyError(path)

    monkeypatch.setattr("scripts.extract_wow_client_data.mpyq.MPQArchive", FakeArchive)

    assert _extract_targets([Path("base.MPQ")], [target]) == {target: b"terrain"}
