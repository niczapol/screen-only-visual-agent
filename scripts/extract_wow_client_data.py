from __future__ import annotations

import argparse
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mpyq


DEFAULT_DBC_FILES = [
    "DBFilesClient\\AreaTable.dbc",
    "DBFilesClient\\Map.dbc",
    "DBFilesClient\\WorldMapArea.dbc",
    "DBFilesClient\\WMOAreaTable.dbc",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract route-relevant WoW 3.3.5 client data from MPQs")
    parser.add_argument("--client-root", default=r"external_data\game_client")
    parser.add_argument("--route-db", default="data/routes/generated/barrens_mining_75_125_safe_cycle.json")
    parser.add_argument("--zone-name", default="Barrens")
    parser.add_argument("--map-directory", default="Kalimdor")
    parser.add_argument("--output-dir", default="data/extracted_client_data/barrens")
    parser.add_argument("--padding-tiles", type=int, default=1)
    return parser.parse_args(argv)


def build_world_targets(
    map_directory: str,
    tiles: list[tuple[int, int]],
) -> list[str]:
    normalized = map_directory.strip()
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ValueError("map_directory must be one non-empty MPQ path component")
    prefix = f"World\\Maps\\{normalized}\\{normalized}"
    return [f"{prefix}.wdt"] + [
        f"{prefix}_{tile_x}_{tile_y}.adt" for tile_x, tile_y in tiles
    ]


def main() -> None:
    args = parse_args()
    map_directory = args.map_directory.strip()

    client_root = Path(args.client_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = _client_archives(client_root)
    extracted: dict[str, str] = {}
    for inner_path, data in _extract_targets(archives, DEFAULT_DBC_FILES).items():
        output_path = output_dir / "dbc" / Path(inner_path).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        extracted[inner_path] = str(output_path)

    wma_path = output_dir / "dbc" / "WorldMapArea.dbc"
    area = _find_world_map_area(wma_path, args.zone_name) if wma_path.exists() else None
    route_data = json.loads(Path(args.route_db).read_text(encoding="utf-8"))
    tiles = _route_tiles(route_data, area, padding=args.padding_tiles) if area else []

    world_files: dict[str, str] = {}
    world_targets = build_world_targets(map_directory, tiles)
    for inner_path, data in _extract_targets(archives, world_targets).items():
        output_path = (
            output_dir / "world" / "maps" / map_directory / Path(inner_path).name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        world_files[inner_path] = str(output_path)

    manifest = {
        "client_root": str(client_root),
        "route_db": args.route_db,
        "zone_name": args.zone_name,
        "map_directory": map_directory,
        "world_map_area": area,
        "tiles": [{"x": x, "y": y} for x, y in tiles],
        "extracted_dbc": extracted,
        "extracted_world_files": world_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"WorldMapArea: {area}")
    print(f"Tiles requested: {len(tiles)}")
    print(f"World files extracted: {len(world_files)}")


def _client_archives(client_root: Path) -> list[Path]:
    data_dir = client_root / "Data"
    locale_dir = data_dir / "enUS"
    ordered_names = [
        "common.MPQ",
        "common-2.MPQ",
        "expansion.MPQ",
        "lichking.MPQ",
        "locale-enUS.MPQ",
        "patch.MPQ",
        "patch-2.MPQ",
        "patch-3.MPQ",
        "patch-enUS.MPQ",
        "patch-enUS-2.MPQ",
        "patch-enUS-3.MPQ",
    ]
    paths: list[Path] = []
    for name in ordered_names:
        candidates = [data_dir / name, locale_dir / name]
        paths.extend(path for path in candidates if path.exists())
    paths.extend(sorted(data_dir.glob("patch-*.MPQ")))
    return list(dict.fromkeys(paths))


def _read_file(mpq_path: Path, inner_path: str) -> bytes | None:
    try:
        archive = mpyq.MPQArchive(str(mpq_path))
        return archive.read_file(inner_path)
    except Exception:
        return None


def _extract_targets(archives: list[Path], targets: list[str]) -> dict[str, bytes]:
    normalized_targets = {_normalize_mpq_path(target): target for target in targets}
    extracted: dict[str, bytes] = {}
    for mpq_path in archives:
        try:
            archive = mpyq.MPQArchive(str(mpq_path))
        except Exception:
            continue

        available: dict[str, str] = {}
        for item in archive.files:
            name = item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)
            normalized = _normalize_mpq_path(name)
            if normalized in normalized_targets:
                available[normalized] = name

        for normalized, archive_name in available.items():
            try:
                extracted[normalized_targets[normalized]] = archive.read_file(archive_name)
            except Exception:
                continue

        # Some valid MPQs ship with incomplete listfiles. mpyq can still read a
        # known path directly, so probe only targets that no earlier archive has
        # supplied. Later archives may still override them through their listfile.
        for normalized, target in normalized_targets.items():
            if normalized in available or target in extracted:
                continue
            try:
                data = archive.read_file(target)
            except Exception:
                continue
            if data:
                extracted[target] = data
    return extracted


def _normalize_mpq_path(path: str) -> str:
    return path.replace("/", "\\").lower()


def _find_world_map_area(path: Path, zone_name: str) -> dict[str, Any] | None:
    dbc = _read_dbc(path)
    target = zone_name.lower()
    for row in dbc["rows"]:
        if row.get("string", "").lower() == target:
            return row
    for row in dbc["rows"]:
        if target in row.get("string", "").lower():
            return row
    return None


def _read_dbc(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"WDBC":
        raise ValueError(f"Unsupported DBC header in {path}")
    records, fields, record_size, string_size = struct.unpack_from("<4I", data, 4)
    records_offset = 20
    strings_offset = records_offset + records * record_size
    strings = data[strings_offset : strings_offset + string_size]
    rows = []
    for index in range(records):
        start = records_offset + index * record_size
        record = data[start : start + record_size]
        values = list(struct.unpack("<" + "I" * fields, record))
        floats = list(struct.unpack("<" + "f" * fields, record))
        row_string = _string_at(strings, values[3]) if fields > 3 else ""
        string_offset_field = 3 if row_string else None
        if not row_string:
            for field_index, value in enumerate(values):
                candidate = _string_at(strings, value)
                if candidate:
                    row_string = candidate
                    string_offset_field = field_index
                    break
        row = {
            "index": index,
            "values": values,
            "floats": floats,
            "string": row_string,
            "string_field": string_offset_field,
        }
        if row_string:
            # WorldMapArea 3.3.5 shape: id, map, area, name, left, right, top, bottom, ...
            row.update(
                {
                    "id": values[0] if fields > 0 else None,
                    "map_id": values[1] if fields > 1 else None,
                    "area_id": values[2] if fields > 2 else None,
                    "left": floats[4] if fields > 4 else None,
                    "right": floats[5] if fields > 5 else None,
                    "top": floats[6] if fields > 6 else None,
                    "bottom": floats[7] if fields > 7 else None,
                    "y1": floats[4] if fields > 4 else None,
                    "y2": floats[5] if fields > 5 else None,
                    "x1": floats[6] if fields > 6 else None,
                    "x2": floats[7] if fields > 7 else None,
                }
            )
        rows.append(row)
    return {"records": records, "fields": fields, "record_size": record_size, "rows": rows}


def _string_at(strings: bytes, offset: int) -> str:
    if offset >= len(strings):
        return ""
    end = strings.find(b"\x00", offset)
    if end < 0:
        return ""
    raw = strings[offset:end]
    if not raw:
        return ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if not text or any(ord(char) < 32 for char in text):
        return ""
    return text


def _route_tiles(route_data: dict[str, Any], area: dict[str, Any], *, padding: int) -> list[tuple[int, int]]:
    left = float(area["left"])
    right = float(area["right"])
    top = float(area["top"])
    bottom = float(area["bottom"])
    tiles: set[tuple[int, int]] = set()
    for waypoint in route_data.get("route_loop", []):
        # WorldMapArea fields 4/5 are y1/y2 and 6/7 are x1/x2.
        # Client UI map axes are swapped relative to world x/y.
        world_x = top + (bottom - top) * (float(waypoint["y"]) / 100.0)
        world_y = left + (right - left) * (float(waypoint["x"]) / 100.0)
        tile_x = int(math.floor(32.0 - world_y / 533.3333333333))
        tile_y = int(math.floor(32.0 - world_x / 533.3333333333))
        for dx in range(-padding, padding + 1):
            for dy in range(-padding, padding + 1):
                tx = tile_x + dx
                ty = tile_y + dy
                if 0 <= tx <= 63 and 0 <= ty <= 63:
                    tiles.add((tx, ty))
    return sorted(tiles)


if __name__ == "__main__":
    main()
