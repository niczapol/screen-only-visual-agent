from __future__ import annotations

import json
import sys
from typing import Any

from vision_bot.config import resource_path, runtime_state_path
from vision_bot.cursor_classifier import CursorTemplateClassifier
from vision_bot.mining import read_cursor_signature
from vision_bot.navmesh_route_entry import navmesh_profile_for_zone
from vision_bot.ore_world_detector import OreWorldDetector
from vision_bot.permanent_exclusions import load_permanent_exclusions
from vision_bot.route_planner import is_route_waypoint, load_nodes
from vision_bot.route_probe_routing import (
    configured_route_zone_ids as _configured_route_zone_ids,
    route_entry_strategy as _route_entry_strategy,
)


def build_route_live_preflight(config: dict[str, Any]) -> dict[str, Any]:
    route_cfg = config.get("route", {})
    mining_cfg = config.get("mining", {})
    route_path_value = route_cfg.get("database_path")
    route_path = resource_path(str(route_path_value)) if route_path_value else None
    issues: list[str] = []
    warnings: list[str] = []
    route_status: str | None = None
    route_waypoint_count = 0
    ore_node_count = 0
    ore_access_plan_count = 0
    mining_cursor_classifier_calibrated = False
    ore_world_detector_status = "not_required"
    ore_world_detector_load_ms = 0.0

    if route_path is None or not route_path.exists():
        issues.append("configured_route_database_missing")
    else:
        try:
            route_nodes = load_nodes(route_path)
            permanent_exclusions = load_permanent_exclusions(
                runtime_state_path(
                    str(
                        route_cfg.get(
                            "permanent_exclusions_path",
                            "data/permanent_exclusions.json",
                        )
                    )
                )
            )
            active_ore_nodes = [
                node
                for node in route_nodes
                if not is_route_waypoint(node)
                and node.coord not in permanent_exclusions
            ]
            ore_node_count = len(active_ore_nodes)
            ore_access_plan_count = len(
                [
                    node
                    for node in active_ore_nodes
                    if node.access_plan is not None
                ]
            )
            if route_path.suffix.lower() == ".json":
                route_data = json.loads(route_path.read_text(encoding="utf-8"))
                route_status = str(route_data.get("status")) if route_data.get("status") else None
                route_waypoint_count = len(route_data.get("route_loop", []))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            issues.append(f"configured_route_database_invalid:{type(exc).__name__}")

    if bool(route_cfg.get("follow_route_loop", False)) and route_waypoint_count == 0:
        issues.append("route_loop_empty")
    if _route_entry_strategy(config) == "navmesh_dynamic":
        route_zone_ids = _configured_route_zone_ids(route_cfg)
        if not route_zone_ids:
            route_zone_ids = {
                int(value)
                for value in route_cfg.get("locations", [])
            }
        if len(route_zone_ids) != 1:
            issues.append("navmesh_route_entry_requires_one_configured_zone")
        else:
            try:
                profile = navmesh_profile_for_zone(config, next(iter(route_zone_ids)))
                required_paths = {
                    "manifest": resource_path(str(profile["manifest_path"])),
                    "adt": resource_path(str(profile["adt_dir"])),
                    "mmap": resource_path(str(profile["mmap_dir"])),
                }
                for label, path in required_paths.items():
                    if not path.exists():
                        issues.append(f"navmesh_{label}_path_missing")
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"navmesh_profile_invalid:{type(exc).__name__}")
    if bool(mining_cfg.get("route_live_enabled", False)) and ore_node_count == 0:
        issues.append("mining_candidate_nodes_empty")
    if bool(mining_cfg.get("route_live_enabled", False)):
        if int(mining_cfg.get("bright_confirm_frames", 0)) < 1:
            issues.append("bright_confirm_frames_invalid")
        if int(mining_cfg.get("verify_clear_frames", 0)) < 1:
            issues.append("verify_clear_frames_invalid")
        if int(mining_cfg.get("scan_max_points", 0)) < 1:
            issues.append("scan_max_points_invalid")
        if bool(mining_cfg.get("require_cursor_change", True)) and read_cursor_signature() is None:
            warnings.append("win32_cursor_signature_not_currently_visible")
        recognition_cfg = config.get("recognition", {})
        raw_templates = recognition_cfg.get("bright_templates", [])
        template_paths = [
            resource_path(str(item["path"]))
            for item in raw_templates
            if isinstance(item, dict) and item.get("path")
        ]
        if not template_paths:
            legacy_template = recognition_cfg.get("bright_template_path")
            if legacy_template:
                template_paths = [resource_path(str(legacy_template))]
        if not template_paths:
            issues.append("bright_ore_templates_not_configured")
        missing_templates = [str(path) for path in template_paths if not path.exists()]
        if missing_templates:
            issues.append("bright_ore_template_assets_missing")
        if ore_access_plan_count < ore_node_count:
            warnings.append(
                f"ore_access_plans_partial:{ore_access_plan_count}/{ore_node_count}"
            )
        detector_cfg = mining_cfg.get("ore_world_detector", {})
        if bool(detector_cfg.get("enabled", False)):
            detector_result = OreWorldDetector(config).load()
            ore_world_detector_status = detector_result.reason
            ore_world_detector_load_ms = detector_result.elapsed_ms
            if detector_result.reason != "model_ready":
                issues.append(
                    f"ore_world_detector_unavailable:{detector_result.reason}"
                )
        else:
            ore_world_detector_status = "disabled"
            warnings.append("ore_world_detector_disabled")
        classifier_cfg = config.get("cursor_classifier", {})
        if bool(classifier_cfg.get("enabled", True)):
            cursor_templates_path = resource_path(
                str(
                    classifier_cfg.get(
                        "templates_path",
                        "data/cursor_templates/cursor_templates.json",
                    )
                )
            )
            try:
                cursor_classifier = CursorTemplateClassifier.load(
                    cursor_templates_path,
                    min_similarity=float(classifier_cfg.get("min_similarity", 0.90)),
                )
                mining_cursor_classifier_calibrated = cursor_classifier.has_templates(
                    "mine"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                issues.append(f"cursor_template_manifest_invalid:{type(exc).__name__}")
            if not mining_cursor_classifier_calibrated:
                warnings.append("mining_cursor_classifier_requires_live_calibration")
    loot_cfg = config.get("post_combat_loot", {})
    if bool(loot_cfg.get("enabled", False)):
        combat_cfg = config.get("safety", {}).get("combat", {})
        if not bool(
            combat_cfg.get(
                "route_live_handling_enabled",
                combat_cfg.get("enabled", True),
            )
        ):
            issues.append("post_combat_loot_requires_route_live_combat")
        if not bool(config.get("runtime_markers", {}).get("enabled", False)):
            issues.append("post_combat_loot_requires_runtime_markers")
    if route_status and "not_live_validated" in route_status:
        warnings.append(f"route_status:{route_status}")

    return {
        "ready_offline": not issues,
        "issues": issues,
        "warnings": warnings,
        "route_database": str(route_path) if route_path is not None else None,
        "follow_route_loop": bool(route_cfg.get("follow_route_loop", False)),
        "route_waypoint_count": route_waypoint_count,
        "ore_node_count": ore_node_count,
        "ore_access_plan_count": ore_access_plan_count,
        "mining_enabled": bool(mining_cfg.get("route_live_enabled", False)),
        "mining_cursor_classifier_calibrated": mining_cursor_classifier_calibrated,
        "runtime_python": sys.executable,
        "ore_world_detector_status": ore_world_detector_status,
        "ore_world_detector_load_ms": ore_world_detector_load_ms,
        "post_combat_loot_enabled": bool(loot_cfg.get("enabled", False)),
        "manual_gates": [
            "GatherMate2 historical mining pins are disabled on the minimap",
            "Find Minerals is active and bright/dark live icons are visible at the calibrated minimap zoom",
            "The game window is foreground and the route zone matches the character location",
        ],
    }
