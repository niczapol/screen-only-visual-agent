from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from vision_bot.capture import ScreenCapture
from vision_bot.config import runtime_state_path
from vision_bot.mining import MiningOutcome
from vision_bot.recognition import recognize_ore_point_classes, save_debug_artifacts
from vision_bot.route_planner import MiningNode
from vision_bot.route_probe_state import RouteLiveSummary


def capture_reached_ore_training_sample(
    capture: ScreenCapture,
    frame,
    config: dict[str, Any],
    *,
    target_node: MiningNode,
    target_coord: int,
    reached_coord: int,
    distance: float,
    index: int,
    route_output_dir: Path,
) -> dict[str, Any] | None:
    capture_cfg = config.get("training_capture", {}).get("reached_ore", {})
    if not bool(capture_cfg.get("enabled", False)):
        return None

    minimap = capture.crop_minimap(frame, config)
    ore_points, detected_dark_points = recognize_ore_point_classes(minimap, config)
    include_dark = bool(capture_cfg.get("include_dark_ore", False))
    dark_points = detected_dark_points if include_dark and not ore_points else []
    if not ore_points and not dark_points:
        return None

    output_root = runtime_state_path(
        str(capture_cfg.get("output_dir", "data/reached_ore_training_samples"))
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    node_label = target_node.node_id if target_node.node_id is not None else "manual"
    sample_dir = output_root / f"{timestamp}_{index:04d}_node_{node_label}_{target_coord}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    frame_path = sample_dir / "frame.png"
    minimap_path = sample_dir / "minimap.png"
    metadata_path = sample_dir / "metadata.json"
    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(minimap_path), minimap)

    artifacts: dict[str, str] = {}
    if bool(capture_cfg.get("save_debug_artifacts", True)):
        artifacts = {
            name: str(path)
            for name, path in save_debug_artifacts(
                minimap,
                ore_points,
                sample_dir / "minimap_debug",
                config,
            ).items()
        }

    metadata = {
        "saved_at_epoch": time.time(),
        "route_output_dir": str(route_output_dir),
        "target_coord": target_coord,
        "target_node_id": target_node.node_id,
        "target_zone_id": target_node.zone_id,
        "target_ore_type": target_node.ore_type,
        "reached_coord": reached_coord,
        "distance": distance,
        "step_index": index,
        "ore_points": [list(point) for point in ore_points],
        "dark_ore_points": [list(point) for point in dark_points],
        "frame": str(frame_path),
        "minimap": str(minimap_path),
        "artifacts": artifacts,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return {
        "saved": True,
        "sample_dir": str(sample_dir),
        "metadata": str(metadata_path),
        "ore_points": metadata["ore_points"],
        "dark_ore_points": metadata["dark_ore_points"],
    }


def should_save_route_frame(
    *,
    now: float,
    last_saved_at: float,
    interval: float,
    action: str,
    previous_action: str | None,
    force: bool = False,
) -> bool:
    action_class = route_frame_action_class(action)
    previous_class = (
        route_frame_action_class(previous_action) if previous_action is not None else None
    )
    if force or previous_class is None or action_class != previous_class:
        return True
    if interval <= 0.0:
        return True
    return now - last_saved_at >= interval


def route_frame_action_class(action: str) -> str:
    if action in {"vector_forward", "continue_forward", "course_correct_and_forward"}:
        return "route_motion"
    if action.startswith("combat_"):
        return "combat"
    if action.startswith("mining_"):
        return "mining"
    if action.startswith("recover_"):
        return "stuck_recovery"
    if action.startswith("avoid_hostile"):
        return "hostile_avoidance"
    if "local_avoid" in action:
        return "local_avoidance"
    if any(
        marker in action
        for marker in ("route_entry", "cycle_next_target", "target_blocked", "reached")
    ):
        return "route_transition"
    return action


def mining_outcome_item(outcome: MiningOutcome, *, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "success": bool(outcome.success),
        "reason": outcome.reason,
        "node_id": outcome.node_id,
        "node_coord": outcome.node_coord,
        "elapsed_seconds": round(float(outcome.elapsed_seconds), 3),
        "marker_point": (
            list(outcome.marker_point) if outcome.marker_point is not None else None
        ),
        "resume_waypoint_count": len(outcome.resume_coords),
        "resume_route_index": outcome.resume_route_index,
    }


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def summary_to_dict(summary: RouteLiveSummary) -> dict[str, Any]:
    result = dict(summary.__dict__)
    result["output_dir"] = str(summary.output_dir)
    return result
