from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from vision_bot.addon_install import addon_destination_dir, install_addon
from vision_bot.engine.preflight import (
    _hash_tree,
    build_preflight_report,
    validate_preflight_manifest,
    write_preflight_report,
)


ROOT = Path(__file__).parents[1]


def _config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _config_with_stub_model(tmp_path):
    config = _config()
    model = tmp_path / "stub-model.pt"
    model.write_bytes(b"portfolio preflight fixture")
    config["mining"]["ore_world_detector"]["model_path"] = str(model)
    return config


def test_reinstalled_client_requires_exact_protocol_v2_addon(tmp_path) -> None:
    client = tmp_path / "client"
    (client / "Interface" / "AddOns").mkdir(parents=True)
    before = build_preflight_report(
        _config(), repository_root=ROOT, client_root=client, now=100.0,
        validate_model_runtime=False,
    )

    assert not before.ready_for_validation
    assert not next(
        check for check in before.checks if check.name == "addon_installed_exactly"
    ).passed

    install_addon(ROOT, client)
    after = build_preflight_report(
        _config(), repository_root=ROOT, client_root=client, now=101.0,
        validate_model_runtime=False,
    )

    # The public repository deliberately omits model weights, so installing the
    # addon alone must not make static validation ready.
    assert not after.ready_for_validation
    assert not after.ready_for_production
    assert next(
        check for check in after.checks if check.name == "addon_source_protocol_v2"
    ).passed
    assert not next(
        check for check in after.checks if check.name == "world_ore_model"
    ).passed


def test_preflight_manifest_is_bound_to_current_installed_artifacts(tmp_path) -> None:
    client = tmp_path / "client"
    (client / "Interface" / "AddOns").mkdir(parents=True)
    install_addon(ROOT, client)
    report = build_preflight_report(
        _config_with_stub_model(tmp_path), repository_root=ROOT, client_root=client, now=100.0,
        validate_model_runtime=False,
    )
    manifest = tmp_path / "preflight.json"
    write_preflight_report(manifest, report)

    validate_preflight_manifest(
        manifest,
        report,
        purpose="validation",
        now=101.0,
    )

    installed_toc = addon_destination_dir(client) / "ScreenVisionTelemetry.toc"
    installed_toc.write_text("changed", encoding="utf-8")
    changed = build_preflight_report(
        _config(), repository_root=ROOT, client_root=client, now=102.0,
        validate_model_runtime=False,
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        validate_preflight_manifest(
            manifest,
            changed,
            purpose="validation",
            now=102.0,
        )


def test_preflight_manifest_expires(tmp_path) -> None:
    client = tmp_path / "client"
    (client / "Interface" / "AddOns").mkdir(parents=True)
    install_addon(ROOT, client)
    report = build_preflight_report(
        _config_with_stub_model(tmp_path), repository_root=ROOT, client_root=client, now=100.0,
        validate_model_runtime=False,
    )
    manifest = tmp_path / "preflight.json"
    write_preflight_report(manifest, report)

    with pytest.raises(RuntimeError, match="stale"):
        validate_preflight_manifest(
            manifest,
            report,
            purpose="validation",
            now=1001.0,
            max_age_seconds=900.0,
        )


def test_preflight_fingerprint_covers_full_runtime_config(tmp_path) -> None:
    client = tmp_path / "client"
    (client / "Interface" / "AddOns").mkdir(parents=True)
    install_addon(ROOT, client)
    original = _config()
    changed = deepcopy(original)
    changed["minimap"]["x"] += 1

    before = build_preflight_report(
        original,
        repository_root=ROOT,
        client_root=client,
        now=100.0,
        validate_model_runtime=False,
    )
    after = build_preflight_report(
        changed,
        repository_root=ROOT,
        client_root=client,
        now=101.0,
        validate_model_runtime=False,
    )

    assert before.fingerprint != after.fingerprint


def test_runtime_source_tree_hash_is_ordered_and_content_sensitive(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "b.py").write_text("B = 1\n", encoding="utf-8")
    (source / "a.py").write_text("A = 1\n", encoding="utf-8")
    (source / "ignored.txt").write_text("first", encoding="utf-8")

    first = _hash_tree(source, suffixes={".py"})
    (source / "ignored.txt").write_text("second", encoding="utf-8")
    assert _hash_tree(source, suffixes={".py"}) == first

    (source / "a.py").write_text("A = 2\n", encoding="utf-8")
    assert _hash_tree(source, suffixes={".py"}) != first
