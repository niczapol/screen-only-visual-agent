from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


ADDON_NAME = "ScreenVisionTelemetry"
ADDON_FILES = ("ScreenVisionTelemetry.lua", "ScreenVisionTelemetry.toc")


@dataclass(frozen=True)
class AddonInstallResult:
    source: Path
    destination: Path
    copied_files: tuple[Path, ...]
    fingerprint: str


def addon_source_dir(repository_root: str | Path) -> Path:
    return Path(repository_root).resolve() / "addons" / ADDON_NAME


def addon_destination_dir(client_root: str | Path) -> Path:
    return (
        Path(client_root).resolve()
        / "Interface"
        / "AddOns"
        / ADDON_NAME
    )


def addon_fingerprint(path: str | Path) -> str | None:
    directory = Path(path)
    if not all((directory / name).is_file() for name in ADDON_FILES):
        return None
    digest = hashlib.sha256()
    for name in ADDON_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def install_addon(
    repository_root: str | Path,
    client_root: str | Path,
) -> AddonInstallResult:
    source = addon_source_dir(repository_root)
    missing = [name for name in ADDON_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"addon source is incomplete: {', '.join(missing)}")
    destination = addon_destination_dir(client_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ADDON_FILES:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    fingerprint = addon_fingerprint(destination)
    if fingerprint is None or fingerprint != addon_fingerprint(source):
        raise OSError("installed addon fingerprint does not match repository source")
    return AddonInstallResult(source, destination, tuple(copied), fingerprint)
