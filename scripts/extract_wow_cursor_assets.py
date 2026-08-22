from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import mpyq
import numpy as np

from vision_bot.cursor_classifier import cursor_colorfulness, perceptual_cursor_hash


@dataclass(frozen=True)
class CursorAssetSource:
    archive: str
    inner_path: str
    label: str


DEFAULT_CURSOR_ASSETS = (
    CursorAssetSource("Data/enUS/locale-enUS.MPQ", "Interface/CURSOR/Mine.blp", "mine"),
    CursorAssetSource(
        "Data/enUS/locale-enUS.MPQ",
        "Interface/CURSOR/UnableMine.blp",
        "unable_mine",
    ),
    CursorAssetSource("Data/patch-X.MPQ", "Interface/Cursor/Mine2.blp", "mine"),
    CursorAssetSource("Data/patch-X.MPQ", "Interface/Cursor/Mine4.blp", "mine"),
    CursorAssetSource(
        "Data/patch-X.MPQ",
        "Interface/Cursor/UnableMine2.blp",
        "unable_mine",
    ),
    CursorAssetSource(
        "Data/patch-X.MPQ",
        "Interface/Cursor/UnableMine4.blp",
        "unable_mine",
    ),
)


def decode_blp2_paletted(data: bytes) -> np.ndarray:
    """Decode the paletted BLP2 cursor format used by the 3.3.5 client."""

    if len(data) < 1172 or data[:4] != b"BLP2":
        raise ValueError("Expected a paletted BLP2 image")
    _image_type, encoding, alpha_depth, _alpha_encoding, _has_mips = struct.unpack_from(
        "<I4B", data, 4
    )
    width, height = struct.unpack_from("<2I", data, 12)
    mip_offset = struct.unpack_from("<I", data, 20)[0]
    mip_length = struct.unpack_from("<I", data, 84)[0]
    pixel_count = width * height
    if encoding != 1 or alpha_depth != 8:
        raise ValueError(
            f"Unsupported BLP2 cursor encoding={encoding} alpha_depth={alpha_depth}"
        )
    if width <= 0 or height <= 0 or pixel_count > 4096 * 4096:
        raise ValueError("Invalid BLP2 dimensions")
    if mip_length < pixel_count * 2 or mip_offset + pixel_count * 2 > len(data):
        raise ValueError("Truncated BLP2 cursor mip")

    palette = np.frombuffer(data, dtype=np.uint8, count=1024, offset=148).reshape(256, 4)
    indices = np.frombuffer(
        data,
        dtype=np.uint8,
        count=pixel_count,
        offset=mip_offset,
    ).reshape(height, width)
    alpha = np.frombuffer(
        data,
        dtype=np.uint8,
        count=pixel_count,
        offset=mip_offset + pixel_count,
    ).reshape(height, width)
    image_bgra = palette[indices].copy()
    image_bgra[:, :, 3] = alpha
    return image_bgra


def extract_cursor_assets(
    client_root: Path,
    output_dir: Path,
    *,
    hash_size: int = 16,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in DEFAULT_CURSOR_ASSETS:
        archive_path = client_root / Path(source.archive)
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        archive = mpyq.MPQArchive(str(archive_path))
        data = archive.read_file(source.inner_path.replace("/", "\\"))
        if not data:
            raise FileNotFoundError(f"{archive_path}:{source.inner_path}")
        image_bgra = decode_blp2_paletted(data)
        output_path = output_dir / f"{Path(source.inner_path).stem}.png"
        if not cv2.imwrite(str(output_path), image_bgra):
            raise OSError(f"Could not write {output_path}")
        visual_hash = perceptual_cursor_hash(image_bgra, hash_size=hash_size)
        colorfulness = cursor_colorfulness(image_bgra)
        if visual_hash is None or colorfulness is None:
            raise ValueError(f"Decoded cursor has no visible pixels: {source.inner_path}")
        records.append(
            {
                "label": source.label,
                "archive": source.archive,
                "inner_path": source.inner_path,
                "image": output_path.name,
                "width": int(image_bgra.shape[1]),
                "height": int(image_bgra.shape[0]),
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "hash": format(visual_hash, "x"),
                "colorfulness": round(colorfulness, 8),
                "reviewed": True,
            }
        )
    return records


def seed_runtime_manifest(
    manifest_path: Path,
    records: list[dict[str, object]],
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported cursor template schema")
    labels = manifest.setdefault("labels", {})
    manifest.setdefault("max_colorfulness_distance", 0.12)
    added = 0
    for record in records:
        label = str(record["label"])
        entries = labels.setdefault(label, [])
        visual_hash = str(record["hash"]).lower()
        if any(
            isinstance(entry, dict)
            and str(entry.get("hash", "")).lower() == visual_hash
            for entry in entries
        ):
            continue
        entries.append(
            {
                "hash": visual_hash,
                "colorfulness": record["colorfulness"],
                "source": "ascension_client_asset",
                "asset": record["image"],
                "archive": record["archive"],
                "inner_path": record["inner_path"],
                "source_sha256": record["source_sha256"],
                "reviewed": True,
            }
        )
        added += 1
    manifest["status"] = "client_assets_seeded_live_validation_pending"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract exact Mine/UnableMine cursor assets from the local WoW client."
    )
    parser.add_argument(
        "--client-root",
        default=r"external_data\game_client",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cursor_templates/client_assets",
    )
    parser.add_argument(
        "--runtime-manifest",
        default="data/cursor_templates/cursor_templates.json",
    )
    parser.add_argument("--seed-runtime-manifest", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = extract_cursor_assets(Path(args.client_root), output_dir)
    source_manifest = output_dir / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({"schema_version": 1, "assets": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    added = 0
    if args.seed_runtime_manifest:
        added = seed_runtime_manifest(Path(args.runtime_manifest), records)
    print(f"Extracted {len(records)} cursor assets to {output_dir}")
    print(f"Runtime templates added: {added}")


if __name__ == "__main__":
    main()
