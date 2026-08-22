import json
import struct

import numpy as np

from scripts.extract_wow_cursor_assets import (
    decode_blp2_paletted,
    seed_runtime_manifest,
)


def _paletted_blp2_fixture() -> bytes:
    data = bytearray(1172 + 8)
    data[:4] = b"BLP2"
    struct.pack_into("<I4B2I", data, 4, 1, 1, 8, 8, 0, 2, 2)
    struct.pack_into("<I", data, 20, 1172)
    struct.pack_into("<I", data, 84, 8)
    data[148 + 4 : 148 + 8] = bytes((10, 20, 30, 255))
    data[148 + 8 : 148 + 12] = bytes((40, 50, 60, 255))
    data[1172:1176] = bytes((1, 2, 1, 2))
    data[1176:1180] = bytes((255, 128, 64, 0))
    return bytes(data)


def test_decode_blp2_paletted_preserves_palette_and_alpha():
    image = decode_blp2_paletted(_paletted_blp2_fixture())

    assert image.shape == (2, 2, 4)
    assert np.array_equal(image[0, 0], (10, 20, 30, 255))
    assert np.array_equal(image[0, 1], (40, 50, 60, 128))
    assert image[1, 1, 3] == 0


def test_seed_runtime_manifest_adds_client_assets_idempotently(tmp_path):
    manifest_path = tmp_path / "cursor_templates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_size": 16,
                "labels": {"mine": [], "loot": []},
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "label": "mine",
            "hash": "abcd",
            "colorfulness": 0.25,
            "image": "Mine.png",
            "archive": "Data/enUS/locale-enUS.MPQ",
            "inner_path": "Interface/CURSOR/Mine.blp",
            "source_sha256": "1234",
        }
    ]

    assert seed_runtime_manifest(manifest_path, records) == 1
    assert seed_runtime_manifest(manifest_path, records) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["labels"]["mine"][0]["colorfulness"] == 0.25
    assert manifest["max_colorfulness_distance"] == 0.12
    assert manifest["status"] == "client_assets_seeded_live_validation_pending"
