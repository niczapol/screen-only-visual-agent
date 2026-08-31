from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vision_bot.core.world import WorldSnapshot
from vision_bot.replay.codec import snapshot_from_dict, snapshot_to_dict


@dataclass(frozen=True)
class EpisodeManifest:
    name: str
    schema_version: int = 1
    source: str = "v09"
    config_fingerprint: str | None = None


@dataclass(frozen=True)
class ReplayEpisode:
    manifest: EpisodeManifest
    snapshots: tuple[WorldSnapshot, ...]


def write_episode(
    path: str | Path,
    manifest: EpisodeManifest,
    snapshots: Iterable[WorldSnapshot],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "record": "manifest",
                    "name": manifest.name,
                    "schema_version": manifest.schema_version,
                    "source": manifest.source,
                    "config_fingerprint": manifest.config_fingerprint,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for snapshot in snapshots:
            handle.write(
                json.dumps(
                    {"record": "snapshot", "snapshot": snapshot_to_dict(snapshot)},
                    sort_keys=True,
                )
                + "\n"
            )


def read_episode(path: str | Path) -> ReplayEpisode:
    source_path = Path(path)
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record") != "manifest":
        raise ValueError("replay episode must start with a manifest")
    header = records[0]
    if int(header.get("schema_version", 0)) != 1:
        raise ValueError("unsupported replay episode schema")
    snapshots = tuple(
        snapshot_from_dict(record["snapshot"])
        for record in records[1:]
        if record.get("record") == "snapshot"
    )
    previous_frame = -1
    for snapshot in snapshots:
        if snapshot.frame_id <= previous_frame:
            raise ValueError("episode frame ids must increase monotonically")
        previous_frame = snapshot.frame_id
    return ReplayEpisode(
        manifest=EpisodeManifest(
            name=str(header["name"]),
            schema_version=int(header["schema_version"]),
            source=str(header.get("source", "unknown")),
            config_fingerprint=header.get("config_fingerprint"),
        ),
        snapshots=snapshots,
    )
