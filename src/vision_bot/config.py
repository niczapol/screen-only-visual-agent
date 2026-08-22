from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path.cwd()
    return base_path / relative_path


def runtime_state_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / path
    return Path.cwd() / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _resolve_config_path(path or "config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config_path(path: str | Path) -> Path:
    requested_path = Path(path)
    candidates = [requested_path]

    if not requested_path.is_absolute():
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates.extend(
                [
                    exe_dir / requested_path,
                    exe_dir.parent / requested_path,
                    resource_path(str(requested_path)),
                ]
            )
        else:
            candidates.append(resource_path(str(requested_path)))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return requested_path
