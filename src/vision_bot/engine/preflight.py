from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vision_bot.addon_install import (
    ADDON_FILES,
    addon_destination_dir,
    addon_fingerprint,
    addon_source_dir,
)
from vision_bot.config import resource_path
from vision_bot.engine.factory import V09KernelBundle, build_v09_kernel
from vision_bot.ore_world_detector import OreWorldDetector


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    blocking_for_validation: bool
    blocking_for_production: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    schema_version: int
    generated_at_epoch: float
    fingerprint: str
    ready_for_shadow: bool
    ready_for_validation: bool
    ready_for_production: bool
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(check) for check in self.checks],
        }


def build_preflight_report(
    project_config: dict[str, Any],
    *,
    repository_root: str | Path,
    client_root: str | Path,
    now: float | None = None,
    validate_model_runtime: bool = True,
    bundle: V09KernelBundle | None = None,
) -> PreflightReport:
    root = Path(repository_root).resolve()
    checks: list[PreflightCheck] = []
    bundle = bundle or build_v09_kernel(project_config)
    checks.append(_check("v09_enabled", bundle.config.enabled, True, True, "typed v0.9 config"))
    checks.append(
        _check(
            "safe_default_mode",
            bundle.config.default_mode != "live",
            True,
            True,
            f"default={bundle.config.default_mode}",
        )
    )
    checks.append(
        _check(
            "physical_route",
            len(bundle.route.points) >= 2 and bundle.route.total_length_yards > 0.0,
            True,
            True,
            f"points={len(bundle.route.points)} length_yards={bundle.route.total_length_yards:.1f}",
        )
    )

    route_value = project_config.get("route", {}).get("database_path")
    route_path = resource_path(str(route_value)) if route_value else None
    route_data: dict[str, Any] = {}
    if route_path is not None and route_path.is_file() and route_path.suffix.lower() == ".json":
        route_data = json.loads(route_path.read_text(encoding="utf-8"))
    route_status = str(route_data.get("status", "missing"))
    production_route = route_status in {
        "live_validated",
        "accepted",
        "production",
    }
    checks.append(
        _check(
            "route_live_acceptance",
            production_route,
            False,
            True,
            f"status={route_status}",
        )
    )
    access_count = bundle.route_report.mining_node_count
    stored_node_count = (
        access_count + bundle.route_report.permanently_excluded_node_count
    )
    checks.append(
        _check(
            "mining_access_plans",
            access_count > 0,
            True,
            True,
            (
                f"effective_planned={access_count} stored={stored_node_count} "
                f"permanently_excluded={bundle.route_report.permanently_excluded_node_count}"
            ),
        )
    )

    model_value = (
        project_config.get("mining", {})
        .get("ore_world_detector", {})
        .get("model_path")
    )
    model_path = resource_path(str(model_value)) if model_value else None
    model_exists = model_path is not None and model_path.is_file()
    model_reason = "not_checked"
    if model_exists and validate_model_runtime:
        model_reason = OreWorldDetector(project_config).load().reason
    model_ready = model_exists and (
        not validate_model_runtime or model_reason == "model_ready"
    )
    checks.append(
        _check(
            "world_ore_model",
            model_ready,
            True,
            True,
            (
                f"{model_path} reason={model_reason}"
                if model_path is not None
                else "not configured"
            ),
        )
    )

    source_addon = addon_source_dir(root)
    installed_addon = addon_destination_dir(client_root)
    source_hash = addon_fingerprint(source_addon)
    installed_hash = addon_fingerprint(installed_addon)
    checks.append(
        _check(
            "addon_source_protocol_v2",
            source_hash is not None and _addon_source_supports_protocol_v2(source_addon),
            True,
            True,
            str(source_addon),
        )
    )
    checks.append(
        _check(
            "addon_installed_exactly",
            source_hash is not None and installed_hash == source_hash,
            True,
            True,
            str(installed_addon),
        )
    )
    client_path = Path(client_root).resolve()
    checks.append(
        _check(
            "client_root",
            client_path.is_dir() and (client_path / "Interface" / "AddOns").is_dir(),
            True,
            True,
            str(client_path),
        )
    )

    profile = (
        project_config.get("route", {})
        .get("entry", {})
        .get("navmesh_profiles", {})
        .get(str(bundle.config.zone.zone_id), {})
    )
    hazard_value = profile.get("hazards_path") if isinstance(profile, dict) else None
    hazard_path = resource_path(str(hazard_value)) if hazard_value else None
    exclusion_value = project_config.get("route", {}).get("permanent_exclusions_path")
    exclusion_path = resource_path(str(exclusion_value)) if exclusion_value else None
    fingerprint = _fingerprint(
        project_config,
        runtime_source_path=root / "src" / "vision_bot",
        package_metadata_path=root / "pyproject.toml",
        route_path=route_path,
        hazard_path=hazard_path,
        exclusion_path=exclusion_path,
        model_path=model_path,
        addon_source_hash=source_hash,
        addon_installed_hash=installed_hash,
    )
    ready_validation = all(
        check.passed or not check.blocking_for_validation for check in checks
    )
    ready_production = all(
        check.passed or not check.blocking_for_production for check in checks
    )
    return PreflightReport(
        schema_version=1,
        generated_at_epoch=time.time() if now is None else float(now),
        fingerprint=fingerprint,
        ready_for_shadow=ready_validation,
        ready_for_validation=ready_validation,
        ready_for_production=ready_production,
        checks=tuple(checks),
    )


def _addon_source_supports_protocol_v2(source_addon: Path) -> bool:
    """Validate capability without pinning every compatible addon patch."""

    try:
        lua = (source_addon / ADDON_FILES[0]).read_text(encoding="utf-8")
        toc = (source_addon / ADDON_FILES[1]).read_text(encoding="utf-8")
    except OSError:
        return False
    match = re.search(r"^## Version:\s*(\d+)\.(\d+)\.(\d+)\s*$", toc, re.MULTILINE)
    if match is None:
        return False
    version = tuple(int(part) for part in match.groups())
    return version >= (0, 5, 0) and "local protocolVersion = 2" in lua


def write_preflight_report(path: str | Path, report: PreflightReport) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_preflight_manifest(
    path: str | Path,
    expected: PreflightReport,
    *,
    purpose: str,
    now: float | None = None,
    max_age_seconds: float = 900.0,
) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("fingerprint") != expected.fingerprint:
        raise RuntimeError("preflight fingerprint does not match current config/artifacts")
    timestamp = time.time() if now is None else float(now)
    if timestamp - float(data.get("generated_at_epoch", 0.0)) > max_age_seconds:
        raise RuntimeError("preflight manifest is stale")
    ready_key = "ready_for_production" if purpose == "production" else "ready_for_validation"
    if not bool(data.get(ready_key)) or not getattr(expected, ready_key):
        raise RuntimeError(f"preflight is not {ready_key}")


def _check(
    name: str,
    passed: bool,
    blocking_for_validation: bool,
    blocking_for_production: bool,
    detail: str,
) -> PreflightCheck:
    return PreflightCheck(
        name,
        bool(passed),
        bool(blocking_for_validation),
        bool(blocking_for_production),
        detail,
    )


def _fingerprint(
    config: dict[str, Any],
    *,
    runtime_source_path: Path,
    package_metadata_path: Path,
    route_path: Path | None,
    hazard_path: Path | None,
    exclusion_path: Path | None,
    model_path: Path | None,
    addon_source_hash: str | None,
    addon_installed_hash: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"runtime_source")
    digest.update(_hash_tree(runtime_source_path, suffixes={".py"}))
    for label, path in (
        ("package_metadata", package_metadata_path),
        ("route", route_path),
        ("hazards", hazard_path),
        ("permanent_exclusions", exclusion_path),
        ("world_model", model_path),
    ):
        digest.update(label.encode("ascii"))
        if path is not None and path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing")
    digest.update(b"addon_source")
    digest.update((addon_source_hash or "missing").encode("ascii"))
    digest.update(b"addon_installed")
    digest.update((addon_installed_hash or "missing").encode("ascii"))
    return digest.hexdigest()


def _hash_tree(root: Path, *, suffixes: set[str]) -> bytes:
    digest = hashlib.sha256()
    if not root.is_dir():
        return b"missing"
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.digest()
