from __future__ import annotations

import json

import pytest

from scripts.extract_gathermate_saved_nodes import (
    main,
    normalize_zone,
    parse_saved_mining_nodes,
)


SAVED_VARIABLES = """
GatherMate2DB = { [\"profiles\"] = {} }
GatherMate2MineDB = {
    [162] = {
        [5000250000] = 203,
        [4000350000] = 205,
    },
    [163] = {
        [1000200000] = 201,
    },
}
"""


def test_parse_saved_mining_nodes_ignores_other_tables() -> None:
    parsed = parse_saved_mining_nodes(SAVED_VARIABLES)

    assert parsed == {
        162: [(5000250000, 203), (4000350000, 205)],
        163: [(1000200000, 201)],
    }


def test_normalize_zone_sorts_and_deduplicates() -> None:
    records = normalize_zone(
        [(5000250000, 203), (4000350000, 205), (5000250000, 203)],
        zone_id=162,
    )

    assert [(item["x"], item["y"]) for item in records] == [
        (40.0, 35.0),
        (50.0, 25.0),
    ]
    assert [item["ore_type"] for item in records] == ["Gold", "Iron"]


def test_main_writes_normalized_source_and_extraction_seed(tmp_path) -> None:
    source = tmp_path / "GatherMate2.lua"
    source.write_text(SAVED_VARIABLES, encoding="utf-8")
    output = tmp_path / "nodes.json"
    seed = tmp_path / "seed.json"

    assert main(
        [
            "--source",
            str(source),
            "--zone-id",
            "162",
            "--zone-name",
            "Tanaris",
            "--world-map-area-id",
            "161",
            "--output",
            str(output),
            "--seed-output",
            str(seed),
        ]
    ) == 0

    exported = json.loads(output.read_text(encoding="utf-8"))
    extraction_seed = json.loads(seed.read_text(encoding="utf-8"))
    assert len(exported["records"]) == 2
    assert extraction_seed["route_nodes"] == exported["records"]


def test_parse_rejects_missing_table() -> None:
    with pytest.raises(ValueError, match="not found"):
        parse_saved_mining_nodes("GatherMate2DB = {}")
