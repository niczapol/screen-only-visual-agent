from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from vision_bot import route_probe_targeting as _route_probe_targeting
from vision_bot.artifact_writer import AsyncImageWriter, ImageWriterStats
from vision_bot.capture import ScreenCapture
from vision_bot.combat import (
    CombatFallbackState,
    CombatHealState,
    CombatState,
    CombatThreatEstimator,
    PeriodicCombatKeyState,
    detect_combat_state,
    normalize_attack_keys,
    normalize_periodic_combat_keys,
)
from vision_bot.config import resource_path, runtime_state_path
from vision_bot.control_arbitration import (
    OperationalControlOwner,
    OperationalControlSignals,
    select_operational_control_owner,
)
from vision_bot.death_recovery import (
    DeathRecoveryDecision,
    DeathRecoveryController,
    read_resurrection_sickness_remaining,
    write_resurrection_sickness_state,
)
from vision_bot.game_state import GameState, detect_game_state
from vision_bot.hard_stuck_recovery import (
    HardStuckEscapeSettings,
    perform_hard_stuck_escape,
)
from vision_bot.live_navigation import _ensure_input_allowed, _movement_was_stuck, _relative_posix, _visual_motion_delta
from vision_bot.local_navigation import analyze_local_navigation, draw_local_navigation_report
from vision_bot.mining import (
    HoverScanPhase,
    MiningCycleController,
    MiningInteractor,
    MiningPhase,
    MinimapTooltipProbeController,
    _scaled_client_point,
    minimap_marker_client_point,
    mining_target_alignment,
)
from vision_bot.mining_final_approach import (
    MiningFinalApproachAction,
    MiningFinalApproachController,
)
from vision_bot.mounted_escape import MountedEscapeState
from vision_bot.mounting import MountTravelState
from vision_bot.movement import (
    InputController,
    MouseController,
    MouseSteeringController,
    MovementNavigator,
)
from vision_bot.navmesh_route_entry import (
    NavMeshGuidanceObservation,
    NavMeshRouteEntryPlanner,
    NavMeshRouteGuidance,
)
from vision_bot.position import read_player_position_candidates
from vision_bot.position_filter import CoordinateResyncState
from vision_bot.post_combat_loot import (
    PostCombatLootController,
    PostCombatLootDecision,
)
from vision_bot.permanent_exclusions import load_permanent_exclusions
from vision_bot.recognition import recognize_ore_points
from vision_bot.route_database import project_to_loop
from vision_bot.route_following import DirectedRouteObservation
from vision_bot.route_planner import (
    MiningNode,
    MiningRoutePlanner,
    ROUTE_ENTRY_WAYPOINT_SOURCE,
    ROUTE_WAYPOINT_SOURCE,
    is_route_waypoint,
    load_mining_nodes,
    load_nodes,
    load_route_waypoints,
)
from vision_bot.route_probe_recording import (
    append_jsonl as _append_jsonl,
    capture_reached_ore_training_sample as _capture_reached_ore_training_sample,
    mining_outcome_item as _mining_outcome_item,
    route_frame_action_class as _route_frame_action_class,
    should_save_route_frame as _should_save_route_frame,
    summary_to_dict as _summary_to_dict,
)
from vision_bot.route_probe_routing import (
    allowed_route_locations as _allowed_route_locations,
    autonomous_route_entry_target as _autonomous_route_entry_target,
    build_directed_route_follower as _build_directed_route_follower,
    consume_reached_route_entry as _consume_reached_route_entry,
    configured_route_zone_ids as _configured_route_zone_ids,
    directed_route_steering_target as _directed_route_steering_target,
    load_configured_route_entry as _load_configured_route_entry,
    load_configured_route_loop as _load_configured_route_loop,
    load_configured_route_milestones as _load_configured_route_milestones,
    mining_access_resume_queue as _mining_access_resume_queue,
    mining_route_resume_target as _mining_route_resume_target,
    next_route_sort_key as _next_route_sort_key,
    resume_generated_route_entry as _resume_generated_route_entry,
    route_entry_enabled as _route_entry_enabled,
    route_entry_max_anchor_distance as _route_entry_max_anchor_distance,
    route_entry_reached_distance as _route_entry_reached_distance,
    route_entry_strategy as _route_entry_strategy,
    route_geometry_sort_key as _route_geometry_sort_key,
    should_complete_hazard_egress as _should_complete_hazard_egress,
    should_allow_forbidden_marker_during_hazard_egress as _should_allow_forbidden_marker_during_hazard_egress,
    route_sort_key as _route_sort_key,
    route_target_reached_distance as _route_target_reached_distance,
    single_allowed_route_zone as _single_allowed_route_zone,
)
from vision_bot.route_probe_preflight import build_route_live_preflight
from vision_bot.route_probe_state import (
    LocalAvoidanceCommitment,
    RouteLiveSummary,
    RouteTargetProgressWatchdog,
    SpiritSearchLegProgress,
    ThreatAvoidanceCommitment,
)
from vision_bot.route_motion import RouteMotionController
from vision_bot.route_probe_actions import (
    _advance_route_stuck_samples,
    _apply_hostile_avoidance,
    _is_combat_action,
    _is_recovery_action,
    _is_threat_action,
    _route_metadata_action,
    _should_local_avoid,
)
from vision_bot.route_probe_combat import (
    _apply_combat_fallback,
    _apply_emergency_heal_followup,
    _apply_combat_low_health_heal,
    _combat_face_search_turn_key,
    _maybe_invert_turn_key,
    _release_continuous_face_search,
    _tap_due_combat_keys,
    _tap_due_low_health_combat_key,
)
from vision_bot.route_probe_observation import observe_route_frame
from vision_bot.route_probe_position import (
    _create_navigator,
    _read_filtered_coord,
    _route_coord_jump_limit,
    _select_stable_start_coord,
    _should_read_route_coord,
)
from vision_bot.route_probe_recovery import (
    _choose_spirit_healer_coordinate_anchor,
    _resurrection_sickness_state_path,
    _should_handle_death_recovery,
    _should_handle_dismissable_modal,
    _spirit_healer_anchor_due,
    _spirit_healer_local_search_coord,
    _spirit_healer_search_turn_key,
    _spirit_search_interact_due,
    _turn_key_towards_click_point,
)
from vision_bot.route_probe_targeting import (
    _block_route_waypoint_window,
    _combined_excluded_coords,
    _record_route_node_milestone,
    _record_target_recovery,
    _record_target_route_node_milestone,
    _target_recovery_can_block,
    _target_history_item,
    _target_progress_is_good,
    _target_skipped_item,
)
from vision_bot.route_hazards import RouteHazardGuard
from vision_bot.runtime_markers import (
    detect_armor_critical_marker,
    detect_combat_loot_pending_marker,
    detect_dead_hostile_target_marker,
    detect_forbidden_subzone_marker,
    detect_loot_opened_marker,
    detect_mounted_marker,
    read_minimap_ore_tooltip_telemetry,
    read_runtime_telemetry,
)
from vision_bot.threat_detection import detect_hostile_threat
from vision_bot.window_video import WindowVideoRecorder
from vision_bot.world_zone import read_world_zone_name, resolve_world_zone_ids


def _mining_world_target_turn_key(
    turn_key: str,
    *,
    mining_config: dict[str, Any],
    route_invert_turn_direction: bool,
) -> str:
    """Resolve direct screen-space ore facing independently of route yaw."""

    if not bool(mining_config.get("world_target_face_invert_turn_direction", False)):
        return turn_key
    return _maybe_invert_turn_key(
        turn_key,
        invert_turn_direction=route_invert_turn_direction,
    )


def _load_route_nodes(route_cfg: dict[str, Any]) -> list[MiningNode]:
    return _route_probe_targeting._load_route_nodes(
        route_cfg,
        resource_resolver=resource_path,
        nodes_loader=load_nodes,
        mining_loader=load_mining_nodes,
        waypoints_loader=load_route_waypoints,
    )


def _prepare_death_recovery_input(
    route_motion: RouteMotionController,
    input_controller: InputController,
    combat_fallback: CombatFallbackState,
    mouse_steering: MouseSteeringController,
) -> None:
    """Release every movement/yaw owner before a recovery cursor action.

    Continuous combat yaw runs on a worker that owns the mouse lock for the
    duration of its RMB drag.  A death frame can arrive between combat ticks;
    stopping route input alone would then leave the recovery click blocked on
    that lock forever.  Death/modal ownership must cancel both paths first.
    """
    route_motion.suspend()
    _release_continuous_face_search(
        input_controller,
        combat_fallback,
        mouse_steering,
    )


def _load_mining_candidate_nodes(route_cfg: dict[str, Any]) -> list[MiningNode]:
    return _route_probe_targeting._load_mining_candidate_nodes(
        route_cfg,
        resource_resolver=resource_path,
        nodes_loader=load_nodes,
        mining_loader=load_mining_nodes,
        state_path_resolver=runtime_state_path,
        exclusions_loader=load_permanent_exclusions,
    )


def _covered_center_tooltip_probe_target(
    *,
    current_coord: int | None,
    nodes: list[MiningNode],
    mining_cycle: MiningCycleController,
    now: float,
    minimap_shape: tuple[int, ...],
    config: dict[str, Any],
) -> tuple[int, tuple[int, int]] | None:
    """Return a read-only center hover for a possibly player-covered DB spawn."""
    probe_cfg = config.get("mining", {}).get("minimap_tooltip_probe", {})
    if (
        current_coord is None
        or len(minimap_shape) < 2
        or not bool(probe_cfg.get("center_fallback_enabled", False))
    ):
        return None
    max_distance = max(
        0.0,
        float(probe_cfg.get("center_fallback_max_distance_coord", 0.08)),
    )
    available_nodes = [
        node
        for node in nodes
        if MovementNavigator.distance(current_coord, int(node.coord)) <= max_distance
        and mining_cycle.cooldowns.get(int(node.coord), 0.0) <= now
        and not mining_cycle.is_spatially_cooled(
            int(node.coord),
            str(node.ore_type),
            now=now,
        )
    ]
    node = min(
        available_nodes,
        key=lambda item: MovementNavigator.distance(current_coord, int(item.coord)),
        default=None,
    )
    if node is None:
        return None
    center_cfg = config.get("recognition", {}).get(
        "sensor_circle_center_fraction",
        {"x": 0.50, "y": 0.48},
    )
    height, width = minimap_shape[:2]
    track_id = -max(
        1,
        abs(int(getattr(node, "node_id", None) or node.coord)),
    )
    return track_id, (
        int(round(width * float(center_cfg.get("x", 0.50)))),
        int(round(height * float(center_cfg.get("y", 0.48)))),
    )


def _choose_route_target(
    config: dict[str, Any],
    current_coord: int,
    *,
    target_coord: int | None,
    zone_id: int | None,
    auto_zone: bool = False,
    min_target_distance: float = 0.0,
    excluded_coords: set[int] | None = None,
    auto_zone_ids: set[int] | None = None,
    route_after_sort_key: float | None = None,
    manual_source: str = "manual",
    manual_ore_type: str = "manual",
    manual_route_index: int | None = None,
    manual_route_t: float | None = None,
) -> MiningNode | None:
    return _route_probe_targeting._choose_route_target(
        config,
        current_coord,
        target_coord=target_coord,
        zone_id=zone_id,
        auto_zone=auto_zone,
        min_target_distance=min_target_distance,
        excluded_coords=excluded_coords,
        auto_zone_ids=auto_zone_ids,
        route_after_sort_key=route_after_sort_key,
        manual_source=manual_source,
        manual_ore_type=manual_ore_type,
        manual_route_index=manual_route_index,
        manual_route_t=manual_route_t,
        route_nodes_loader=_load_route_nodes,
        state_path_resolver=runtime_state_path,
        exclusions_loader=load_permanent_exclusions,
    )


def _route_reference_coords(
    config: dict[str, Any],
    *,
    zone_id: int | None,
    target_coord: int | None,
    auto_zone: bool = False,
    auto_zone_ids: set[int] | None = None,
) -> list[int]:
    return _route_probe_targeting._route_reference_coords(
        config,
        zone_id=zone_id,
        target_coord=target_coord,
        auto_zone=auto_zone,
        auto_zone_ids=auto_zone_ids,
        route_nodes_loader=_load_route_nodes,
    )


def _with_route_live_combat_enabled(config: dict[str, Any], enabled: bool | None) -> dict[str, Any]:
    if enabled is None:
        return config

    updated = dict(config)
    safety = dict(updated.get("safety", {}))
    combat = dict(safety.get("combat", {}))
    combat["route_live_handling_enabled"] = bool(enabled)
    safety["combat"] = combat
    updated["safety"] = safety
    return updated


def _with_route_live_mining_enabled(config: dict[str, Any], enabled: bool | None) -> dict[str, Any]:
    if enabled is None:
        return config
    updated = dict(config)
    mining = dict(updated.get("mining", {}))
    mining["route_live_enabled"] = bool(enabled)
    updated["mining"] = mining
    return updated


def _with_post_combat_loot_enabled(
    config: dict[str, Any],
    enabled: bool | None,
) -> dict[str, Any]:
    if enabled is None:
        return config
    updated = dict(config)
    loot = dict(updated.get("post_combat_loot", {}))
    loot["enabled"] = bool(enabled)
    updated["post_combat_loot"] = loot
    return updated


def _route_probe_requires_safe_drain(
    *,
    combat_handling_enabled: bool,
    combat_active: bool,
    death_or_blocking_modal: bool,
    mining_active: bool = False,
    post_combat_loot_active: bool = False,
    death_recovery_active: bool = False,
) -> bool:
    return bool(
        death_or_blocking_modal
        or death_recovery_active
        or mining_active
        or post_combat_loot_active
        or (combat_handling_enabled and combat_active)
    )


def _should_prewarm_ore_world_detector(
    *,
    mining_enabled: bool,
    death_or_blocking_modal: bool,
    combat_active: bool,
) -> bool:
    """Avoid blocking cold model startup while the character needs control."""
    return bool(
        mining_enabled
        and not death_or_blocking_modal
        and not combat_active
    )


def _mounted_escape_has_mining_conflict(
    *,
    mining_active: bool,
    tooltip_probe_active: bool,
    candidate_available: bool,
) -> bool:
    """Protect only ore work that has progressed beyond raw marker presence."""
    return bool(
        mining_active
        or tooltip_probe_active
        or candidate_available
    )


def _route_entry_start_on_rail_sort_key(
    start_coord: int,
    route_loop: list[int],
    *,
    max_distance: float,
) -> float | None:
    """Skip a precision entry pivot when the character already occupies the rail."""
    if len(route_loop) < 2:
        return None
    projection = project_to_loop(start_coord, route_loop)
    if projection.distance > max(0.0, float(max_distance)):
        return None
    return float(projection.sort_key)


def _external_stop_file_requested(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def collect_route_to_node_probe(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    steps: int,
    target_coord: int | None = None,
    zone_id: int | None = None,
    auto_zone: bool = False,
    min_target_distance: float = 0.0,
    cycle_targets: int = 1,
    reached_distance: float = 0.35,
    report_interval: int = 1,
    frame_interval: float | None = None,
    poll_interval: float | None = None,
    coord_interval: float | None = None,
    visual_stuck_min_delta: float = 3.0,
    visual_stuck_window: int | None = None,
    combat_enabled: bool | None = None,
    mining_enabled: bool | None = None,
    post_combat_loot_enabled: bool | None = None,
) -> RouteLiveSummary:
    config = _with_route_live_combat_enabled(config, combat_enabled)
    config = _with_route_live_mining_enabled(config, mining_enabled)
    config = _with_post_combat_loot_enabled(config, post_combat_loot_enabled)
    route_live_cfg = config.get("training_capture", {}).get("route_live", {})
    external_stop_value = route_live_cfg.get("external_stop_file")
    external_stop_path = Path(str(external_stop_value)) if external_stop_value else None
    if _external_stop_file_requested(external_stop_path):
        raise RuntimeError(f"Route-live stop file already exists: {external_stop_path}")
    external_stop_drain_seconds = max(
        0.0,
        float(route_live_cfg.get("external_stop_safe_drain_seconds", 120.0)),
    )
    external_stop_requested_at: float | None = None
    output_path = Path(output_dir)
    frame_dir = output_path / "frames"
    report_dir = output_path / "reports"
    frame_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    capture = ScreenCapture(config)
    hwnd = capture.find_window()
    if hwnd is None:
        raise RuntimeError("Game window not found")
    foreground_ok = capture.activate_window()
    if not foreground_ok:
        raise RuntimeError("Game window could not be activated; live route input aborted")
    _ensure_input_allowed(hwnd, output_path)

    input_cfg = config.get("input", {})
    input_controller = InputController(
        target_hwnd=hwnd,
        backend=str(input_cfg.get("backend", "sendinput")),
    )
    mouse_controller = MouseController(target_hwnd=hwnd)
    mouse_steering = MouseSteeringController.from_config(mouse_controller, config)
    navigator = _create_navigator(
        input_controller,
        config,
        mouse_steering=mouse_steering,
    )
    route_motion = RouteMotionController.from_config(navigator, config)
    mining_cycle = MiningCycleController(config)
    mining_final_approach = MiningFinalApproachController(config)
    minimap_tooltip_probe = MinimapTooltipProbeController(config)
    mining_interactor = MiningInteractor(capture, mouse_controller, config)
    post_combat_loot = PostCombatLootController(config)
    post_combat_loot_decision = PostCombatLootDecision(
        "post_combat_loot_idle",
        False,
    )
    post_combat_loot_outcomes: list[dict[str, Any]] = []
    mining_cfg = config.get("mining", {})
    mining_camera_reset_enabled = bool(
        mining_cfg.get("world_scan_camera_reset_enabled", True)
    )
    mining_camera_reset_key = str(
        mining_cfg.get("world_scan_camera_reset_key", "F7")
    ).upper()
    mining_camera_reset_tap_seconds = max(
        0.0,
        float(mining_cfg.get("world_scan_camera_reset_tap_seconds", 0.04)),
    )
    def start_mining_world_scan(frame_shape: tuple[int, ...], *, now: float):
        if mining_camera_reset_enabled:
            input_controller.tap_key(
                mining_camera_reset_key,
                duration=mining_camera_reset_tap_seconds,
            )
        return mining_interactor.start_hover_scan(frame_shape, now=now)
    mining_candidate_nodes = _load_mining_candidate_nodes(config.get("route", {}))
    mining_outcomes: list[dict[str, Any]] = []

    start_frame = capture.capture_client_region()
    start_game_state = detect_game_state(start_frame)
    initial_combat = detect_combat_state(start_frame, config)
    if _should_prewarm_ore_world_detector(
        mining_enabled=mining_cycle.enabled,
        death_or_blocking_modal=start_game_state.death_or_blocking_modal,
        combat_active=initial_combat.active,
    ):
        warm_up_detector = getattr(mining_interactor, "warm_up_ore_world_detector", None)
        if callable(warm_up_detector):
            warm_up_detector(start_frame)
    start_now = time.monotonic()
    start_telemetry = read_runtime_telemetry(start_frame, config)
    route_motion.observe_heading(
        start_telemetry.heading_degrees if start_telemetry is not None else None,
        now=start_now,
    )
    detected_zone_name = read_world_zone_name(start_frame, config) if auto_zone and zone_id is None else None
    route_zone_ids = resolve_world_zone_ids(detected_zone_name, config)
    hazard_guard_zone_id = zone_id or _single_allowed_route_zone(config.get("route", {}))
    route_hazard_guard = (
        RouteHazardGuard.from_config(config, zone_id=hazard_guard_zone_id)
        if hazard_guard_zone_id is not None
        else RouteHazardGuard()
    )
    route_hazard_raw_candidate = False

    reference_coords = _route_reference_coords(
        config,
        zone_id=zone_id,
        target_coord=target_coord,
        auto_zone=auto_zone,
        auto_zone_ids=set(route_zone_ids),
    )
    coord_resync_state = CoordinateResyncState()
    target_limit = max(1, cycle_targets)
    completed_target_coords: set[int] = set()
    blocked_target_coords: set[int] = set()
    target_history: list[dict[str, Any]] = []
    route_node_history: list[dict[str, Any]] = []
    completed_route_node_indices: set[int] = set()
    skipped_target_history: list[dict[str, Any]] = []
    hard_stuck_escape_history: list[dict[str, Any]] = []
    target_recovery_limit = max(0, int(config.get("movement", {}).get("target_recovery_limit", 3)))
    target_recovery_count = 0

    movement_cfg = config.get("movement", {})
    hard_stuck_escape_settings = HardStuckEscapeSettings.from_config(config)
    target_progress_watchdog = RouteTargetProgressWatchdog(
        timeout_seconds=float(movement_cfg.get("target_progress_timeout_seconds", 6.0)),
        min_improvement=float(movement_cfg.get("target_progress_min_improvement", 0.08)),
    )
    start_sample_count = max(1, int(movement_cfg.get("start_coord_sample_count", 3)))
    start_sample_interval = max(0.0, float(movement_cfg.get("start_coord_sample_interval", 0.12)))
    start_candidate_sets = [read_player_position_candidates(start_frame, config)]
    for _sample_index in range(1, start_sample_count):
        time.sleep(start_sample_interval)
        if not capture.activate_window():
            raise RuntimeError(
                "Game window lost foreground during start-coordinate sampling"
            )
        start_candidate_sets.append(
            read_player_position_candidates(capture.capture_client_region(), config)
        )
    start_candidates = [candidate for candidates in start_candidate_sets for candidate in candidates]
    start_coord = _select_stable_start_coord(
        start_candidate_sets,
        reference_coords=reference_coords,
        max_cluster_distance=float(movement_cfg.get("start_coord_consensus_distance", 0.15)),
        max_jump=float(movement_cfg.get("max_coord_jump_per_poll", 1.5)),
    )
    start_coord_inferred = start_coord is None
    if start_coord is None:
        if not start_game_state.death_or_blocking_modal:
            raise RuntimeError("Player coordinates are not readable")
        fallback_start_coord = target_coord or (reference_coords[0] if reference_coords else None)
        if fallback_start_coord is None:
            raise RuntimeError("Player coordinates are not readable")
        start_coord = fallback_start_coord

    route_loop = _load_configured_route_loop(config)
    route_milestones = _load_configured_route_milestones(config)
    route_entry_enabled = _route_entry_enabled(config, target_coord=target_coord)
    route_entry_reached_distance = _route_entry_reached_distance(config, reached_distance)
    route_entry_queue: list[MiningNode] = []
    route_entry_sort_key = (
        _route_entry_start_on_rail_sort_key(
            start_coord,
            route_loop,
            max_distance=reached_distance,
        )
        if route_entry_enabled
        else None
    )
    route_entry_strategy = _route_entry_strategy(config)
    dynamic_entry_planner: NavMeshRouteEntryPlanner | None = None
    dynamic_entry_result = None
    dynamic_entry_error: str | None = None
    hazard_egress_active = False
    generated_entry = (
        _load_configured_route_entry(config)
        if route_entry_enabled and route_entry_strategy == "validated_path"
        else None
    )
    if (
        route_entry_enabled
        and route_entry_strategy == "navmesh_dynamic"
        and route_loop
        and route_entry_sort_key is None
    ):
        entry_zone_id = zone_id or _single_allowed_route_zone(config.get("route", {}))
        if entry_zone_id is None:
            raise RuntimeError("Dynamic navmesh route entry requires one resolved route zone")
        dynamic_entry_planner = NavMeshRouteEntryPlanner.from_config(
            config,
            zone_id=entry_zone_id,
        )
        try:
            dynamic_entry_result = dynamic_entry_planner.plan(start_coord, route_loop)
        except RuntimeError as exc:
            dynamic_entry_error = str(exc)
            failure_mode = str(
                config.get("route", {}).get("entry", {}).get(
                    "dynamic_failure_mode",
                    "advisory",
                )
            ).strip().lower()
            if failure_mode == "block":
                raise
            (output_path / "route_entry_plan.json").write_text(
                json.dumps(
                    {
                        "status": "advisory_fallback",
                        "reason": dynamic_entry_error,
                        "start_coord": start_coord,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if dynamic_entry_result is not None:
            hazard_egress_active = bool(
                dynamic_entry_result.terrain_validation.get("hazard_egress_valid", False)
            )
            generated_entry = dynamic_entry_result.entry_plan
            (output_path / "route_entry_plan.json").write_text(
                json.dumps(dynamic_entry_result.to_dict(), indent=2),
                encoding="utf-8",
            )
    if generated_entry is not None:
        route_entry_sort_key = generated_entry.target_route_sort_key
        if dynamic_entry_result is not None:
            route_entry_queue = list(generated_entry.waypoints)
        else:
            max_anchor_distance = _route_entry_max_anchor_distance(config)
            route_entry_queue, entry_projection_distance = _resume_generated_route_entry(
                generated_entry,
                start_coord,
                reached_distance=route_entry_reached_distance,
            )
            if (
                not start_coord_inferred
                and not start_game_state.death_or_blocking_modal
                and entry_projection_distance > max_anchor_distance
            ):
                raise RuntimeError(
                    "Current position is outside the validated route-entry corridor "
                    f"({entry_projection_distance:.3f} > {max_anchor_distance:.3f})"
                )
        _consume_reached_route_entry(
            route_entry_queue,
            start_coord,
            reached_distance=route_entry_reached_distance,
        )
        if not route_entry_queue:
            _record_route_node_milestone(
                route_node_history,
                completed_route_node_indices,
                route_milestones,
                route_index=int(generated_entry.target_route_sort_key),
                reached_coord=start_coord,
                distance=0.0,
            )
    elif route_entry_enabled and route_loop and route_entry_sort_key is None:
        autonomous_entry = _autonomous_route_entry_target(
            config,
            start_coord,
            route_loop,
            excluded_coords=set(),
            zone_id=zone_id,
        )
        if autonomous_entry is not None:
            route_entry_sort_key = _route_sort_key(autonomous_entry)
            if MovementNavigator.distance(start_coord, autonomous_entry.coord) > route_entry_reached_distance:
                route_entry_queue = [autonomous_entry]
    route_entry_pending = bool(route_entry_queue)

    target_node = route_entry_queue[0] if route_entry_pending else _choose_route_target(
        config,
        start_coord,
        target_coord=target_coord,
        zone_id=zone_id,
        auto_zone=auto_zone,
        min_target_distance=min_target_distance,
        excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
        auto_zone_ids=set(route_zone_ids),
        route_after_sort_key=route_entry_sort_key,
    )
    if target_node is None:
        raise RuntimeError("No mining target is available for route probe")
    target = target_node.coord
    route_follower = _build_directed_route_follower(config, route_loop)
    route_following_observation: DirectedRouteObservation | None = None
    route_target_passed = False
    if route_follower is not None:
        route_follower.seed(
            route_entry_sort_key
            if route_entry_sort_key is not None
            else project_to_loop(start_coord, route_loop).sort_key
        )
    steering_target, route_target_passed, route_following_observation = (
        _directed_route_steering_target(
            route_follower,
            current_coord=start_coord,
            target_node=target_node,
            route_entry_pending=route_entry_pending,
        )
    )
    entry_cfg = config.get("route", {}).get("entry", {})
    navmesh_route_guidance = (
        NavMeshRouteGuidance(
            dynamic_entry_planner,
            reached_distance=route_entry_reached_distance,
            pass_through_distance=float(
                entry_cfg.get("waypoint_pass_through_distance", 0.25)
            ),
            replan_distance=float(entry_cfg.get("guidance_replan_distance", 0.55)),
        )
        if dynamic_entry_planner is not None
        and isinstance(entry_cfg, dict)
        and bool(entry_cfg.get("guide_route_segments", False))
        else None
    )
    navmesh_guidance_failure_mode = str(
        entry_cfg.get("guidance_failure_mode", "advisory")
    ).strip().lower()
    navmesh_guidance_observation: NavMeshGuidanceObservation | None = None
    navmesh_guidance_blocked = False
    start_distance = MovementNavigator.distance(start_coord, target)

    cv2.imwrite(str(output_path / "start_frame.png"), start_frame)
    cv2.imwrite(
        str(output_path / "start_report.png"),
        draw_local_navigation_report(start_frame, analyze_local_navigation(start_frame, config)),
    )

    interval = (
        float(config.get("movement", {}).get("poll_interval", 0.8))
        if poll_interval is None
        else max(0.05, poll_interval)
    )
    frame_save_interval = (
        float(config.get("training_capture", {}).get("route_live", {}).get("frame_interval_seconds", 1.0))
        if frame_interval is None
        else max(0.0, frame_interval)
    )
    last_frame_saved_at = time.monotonic()
    last_frame_action = "start"
    local_turn_duration = float(config.get("movement", {}).get("turn_duration", 0.22))
    local_jump_duration = float(config.get("movement", {}).get("obstacle_jump_duration", 0.08))
    local_avoidance = LocalAvoidanceCommitment(
        retry_interval=float(config.get("movement", {}).get("local_avoid_retry_seconds", 0.45)),
        clear_commit_seconds=float(
            config.get("local_navigation", {}).get("bypass_clear_seconds", 0.65)
        ),
        max_turn_pulses_per_side=max(
            1,
            int(config.get("local_navigation", {}).get("max_turn_pulses_per_side", 3)),
        ),
    )
    local_side_switch_backtrack_seconds = max(
        0.0,
        float(config.get("local_navigation", {}).get("side_switch_backtrack_seconds", 0.25)),
    )
    hostile_turn_duration = float(
        config.get("safety", {}).get("hostile_avoidance", {}).get("turn_duration", local_turn_duration)
    )
    hostile_commit_duration = float(
        config.get("safety", {}).get("hostile_avoidance", {}).get("commit_duration", 1.75)
    )
    threat_commitment = ThreatAvoidanceCommitment(hostile_commit_duration)
    combat_cfg = config.get("safety", {}).get("combat", {})
    combat_handling_enabled = bool(
        combat_cfg.get(
            "route_live_handling_enabled",
            combat_cfg.get("enabled", True),
        )
    )
    planned_steps = max(1, steps)
    safe_drain_max_steps = max(0, int(route_live_cfg.get("safe_drain_max_steps", 600)))
    max_probe_steps = planned_steps + safe_drain_max_steps
    safe_drain_started = False
    armor_critical_observed = detect_armor_critical_marker(start_frame, config)
    mounting_cfg = config.get("mounting", {})
    mount_state = MountTravelState(
        enabled=bool(mounting_cfg.get("enabled", False)),
        key=str(mounting_cfg.get("key", "1")).strip().upper(),
        cast_seconds=float(mounting_cfg.get("cast_seconds", 2.0)),
        retry_seconds=float(mounting_cfg.get("retry_seconds", 4.0)),
        max_attempts_before_fallback=int(mounting_cfg.get("max_attempts_before_fallback", 2)),
        post_combat_settle_seconds=float(
            mounting_cfg.get("post_combat_settle_seconds", 3.0)
        ),
    )
    start_mounted = detect_mounted_marker(start_frame, config)
    mounted_observed = start_mounted
    mount_casts = 0
    mounted_escape = MountedEscapeState.from_config(config)
    combat_fallback = CombatFallbackState(
        attack_keys=normalize_attack_keys(combat_cfg.get("attack_keys", ["E"])),
        attack_interval=float(combat_cfg.get("attack_interval", 1.0)),
        clear_frames=int(combat_cfg.get("clear_frames", 3)),
        clear_seconds=float(combat_cfg.get("clear_seconds", 3.0)),
        marker_clear_seconds=float(combat_cfg.get("combat_marker_clear_seconds", 0.75)),
        damage_timeout=float(combat_cfg.get("damage_timeout_seconds", 2.4)),
        face_search_cooldown=float(combat_cfg.get("face_search_cooldown_seconds", 1.0)),
        target_lock_no_hit_seconds=float(
            combat_cfg.get("target_lock_no_hit_seconds", 8.0)
        ),
        stale_soft_target_seconds=float(
            combat_cfg.get("stale_soft_target_seconds", 6.0)
        ),
        health_drop_latch_threshold=float(combat_cfg.get("health_drop_latch_threshold", 0.02)),
        face_search_turn_key=_combat_face_search_turn_key(combat_cfg.get("face_search_turn_key", "D")),
    )
    combat_threat_estimator = CombatThreatEstimator(
        window_seconds=float(combat_cfg.get("threat_hp_window_seconds", 5.0))
    )
    periodic_combat_keys = PeriodicCombatKeyState(
        keys=normalize_periodic_combat_keys(combat_cfg.get("periodic_keys", [])),
        interval=float(combat_cfg.get("periodic_interval", 15.0)),
    )
    aligned_periodic_combat_keys = PeriodicCombatKeyState(
        keys=normalize_periodic_combat_keys(
            combat_cfg.get("aligned_periodic_keys", [])
        ),
        interval=float(combat_cfg.get("aligned_periodic_interval", 10.0)),
    )
    low_health_priority_combat_keys = PeriodicCombatKeyState(
        keys=normalize_periodic_combat_keys(
            combat_cfg.get("low_health_priority_keys", ["F"])
        ),
        interval=float(combat_cfg.get("low_health_priority_interval", 15.0)),
    )
    low_health_priority_threshold = float(
        combat_cfg.get("low_health_priority_threshold", 0.20)
    )
    combat_face_turn_duration = float(combat_cfg.get("face_turn_duration", 0.15))
    combat_face_search_turn_180_duration = float(combat_cfg.get("face_search_turn_180_duration", 0.90))
    combat_face_search_turn_90_duration = float(combat_cfg.get("face_search_turn_90_duration", 0.45))
    combat_attack_tap_duration = float(combat_cfg.get("attack_tap_duration", 0.06))
    combat_key_min_interval = float(combat_cfg.get("combat_key_min_interval", 0.12))
    combat_range_approach_duration = float(combat_cfg.get("range_approach_duration", 0.35))
    combat_range_approach_cooldown = float(combat_cfg.get("range_approach_cooldown_seconds", 0.55))
    continuous_face_search_enabled = bool(combat_cfg.get("continuous_face_search_enabled", True))
    attacker_target_selection_enabled = bool(combat_cfg.get("attacker_target_selection_enabled", False))
    attacker_target_cycle_key = str(combat_cfg.get("attacker_target_cycle_key", "F8")).strip().upper()
    attacker_target_cycle_interval = float(combat_cfg.get("attacker_target_cycle_interval", 0.35))
    nameplate_facing_enabled = bool(combat_cfg.get("nameplate_facing_enabled", False))
    nameplate_face_turn_duration = float(combat_cfg.get("nameplate_face_turn_duration", 0.08))
    target_interact_facing_enabled = bool(
        combat_cfg.get("target_interact_facing_enabled", False)
    )
    target_interact_key = str(combat_cfg.get("target_interact_key", "G")).strip().upper()
    target_interact_cooldown = float(
        combat_cfg.get("target_interact_cooldown_seconds", 0.45)
    )
    target_interact_probe_delay = float(
        combat_cfg.get("target_interact_probe_delay_seconds", 0.35)
    )
    invert_turn_direction = bool(config.get("movement", {}).get("invert_turn_direction", False))
    combat_heal_key = str(combat_cfg.get("heal_key", "V")).strip().upper()
    combat_heal_tap_duration = float(combat_cfg.get("heal_tap_duration", 0.06))
    emergency_heal_followup_enabled = bool(
        combat_cfg.get("emergency_heal_followup_enabled", True)
    )
    emergency_heal_followup_delay = float(
        combat_cfg.get("emergency_heal_followup_delay_seconds", combat_key_min_interval)
    )
    combat_heal = CombatHealState(
        enabled=bool(combat_cfg.get("low_health_heal_enabled", True)),
        threshold=float(combat_cfg.get("low_health_heal_threshold", 0.50)),
        cooldown_seconds=float(combat_cfg.get("low_health_heal_cooldown_seconds", 8.0)),
        cast_seconds=float(combat_cfg.get("heal_cast_seconds", 2.0)),
    )
    death_recovery = DeathRecoveryController.from_config(config)
    death_recovery_cfg = config.get("safety", {}).get("death_recovery", {})
    death_release_click_duration = float(death_recovery_cfg.get("release_click_duration", 0.05))
    death_click_duration = float(death_recovery_cfg.get("click_duration", death_release_click_duration))
    death_click_settle_seconds = float(death_recovery_cfg.get("click_settle_seconds", 0.0))
    spirit_healer_approach_forward_seconds = float(
        death_recovery_cfg.get("spirit_healer_approach_forward_seconds", 0.65)
    )
    spirit_healer_approach_turn_seconds = float(
        death_recovery_cfg.get("spirit_healer_approach_turn_seconds", 0.18)
    )
    spirit_healer_approach_deadzone_fraction = float(
        death_recovery_cfg.get("spirit_healer_approach_deadzone_fraction", 0.12)
    )
    spirit_healer_search_turn_seconds = float(death_recovery_cfg.get("spirit_healer_search_turn_seconds", 0.35))
    spirit_healer_search_turn_key = _spirit_healer_search_turn_key(
        death_recovery_cfg.get("spirit_healer_search_turn_key", "D")
    )
    spirit_healer_local_search_enabled = bool(
        death_recovery_cfg.get("spirit_healer_local_search_enabled", True)
    )
    spirit_healer_local_search_leash_radius = max(
        float(death_recovery_cfg.get("spirit_healer_local_search_max_radius", 0.55)),
        float(death_recovery_cfg.get("spirit_healer_local_search_leash_radius", 0.65)),
    )
    spirit_healer_local_search_reached_distance = max(
        0.01,
        float(death_recovery_cfg.get("spirit_healer_local_search_reached_distance", 0.045)),
    )
    spirit_healer_local_search_poll_seconds = max(
        0.05,
        float(death_recovery_cfg.get("spirit_healer_local_search_poll_seconds", 0.20)),
    )
    spirit_healer_local_search_leg_progress = SpiritSearchLegProgress(
        reached_distance=spirit_healer_local_search_reached_distance,
        pass_distance=max(
            spirit_healer_local_search_reached_distance,
            float(death_recovery_cfg.get("spirit_healer_local_search_pass_distance", 0.18)),
        ),
        pass_margin=max(
            0.01,
            float(death_recovery_cfg.get("spirit_healer_local_search_pass_margin", 0.08)),
        ),
        max_seconds=max(
            0.25,
            float(death_recovery_cfg.get("spirit_healer_local_search_leg_max_seconds", 2.5)),
        ),
    )
    spirit_healer_search_interact_interval = max(
        0.05,
        float(death_recovery_cfg.get("spirit_healer_search_interact_interval_seconds", 0.30)),
    )
    resurrection_sickness_state_path = _resurrection_sickness_state_path(config)
    coord_read_interval = (
        float(config.get("movement", {}).get("coord_interval", 1.5))
        if coord_interval is None
        else max(0.0, coord_interval)
    )
    recovery_coord_interval = max(
        0.0,
        float(config.get("movement", {}).get("recovery_coord_interval", interval)),
    )
    stuck_window = (
        int(config.get("movement", {}).get("visual_stuck_window", config.get("movement", {}).get("stuck_check_window", 4)))
        if visual_stuck_window is None
        else max(1, visual_stuck_window)
    )
    alternate_turn_key = "D"
    final_coord = start_coord
    final_distance = start_distance
    reached = False
    stuck_events = 0
    visual_stuck_samples = 0
    previous_motion_frame = start_frame
    last_coord_read_at = time.monotonic()
    last_coord_accepted_at = last_coord_read_at
    coord_feedback_paused = False
    coord_feedback_timeout = max(
        0.0,
        float(movement_cfg.get("coord_feedback_timeout_seconds", 2.0)),
    )
    executed_steps = 0
    termination_reason = "step_limit"
    spirit_healer_local_search_origin: int | None = None
    spirit_healer_local_search_target: int | None = None
    spirit_healer_local_search_index = 0
    next_spirit_healer_search_interact_at = 0.0

    pending_resurrection_wait = read_resurrection_sickness_remaining(resurrection_sickness_state_path)
    if pending_resurrection_wait is not None:
        navigator.stop()
        _append_jsonl(
            metadata_path,
            {
                "index": -1,
                "timestamp": time.time(),
                "foreground_ok": foreground_ok,
                "frame": "start_frame.png",
                "report": "start_report.png",
                "coord": start_coord,
                "coord_candidates": start_candidates,
                "coord_resynced": False,
                "coord_fresh": True,
                "target_coord": target,
                "target_node_id": target_node.node_id,
                "target_zone_id": target_node.zone_id,
                "detected_zone_name": detected_zone_name,
                "route_zone_ids": route_zone_ids,
                "completed_targets": len(target_history),
                "skipped_targets": len(skipped_target_history),
                "skipped_target_history": skipped_target_history,
                "blocked_target_coords": sorted(blocked_target_coords),
                "target_recovery_count": target_recovery_count,
                "target_recovery_limit": target_recovery_limit,
                "distance": start_distance,
                "action": "death_recovery_resurrection_sickness_wait_resume",
                "local_turn_key": None,
                "threat": None,
                "combat": None,
                "death_recovery": {
                    "action": "death_recovery_resurrection_sickness_wait_resume",
                    "attempts": death_recovery.attempts,
                    "click_point": None,
                    "button_bbox": None,
                    "wait_seconds": pending_resurrection_wait,
                    "reason": "persisted_resurrection_sickness",
                },
                "game_state": detect_game_state(start_frame).__dict__,
                "movement": navigator.last_diagnostics.__dict__,
                "mouse_steering": (
                    mouse_steering.last_result.to_dict()
                    if mouse_steering.last_result is not None
                    else None
                ),
                "navigation": analyze_local_navigation(start_frame, config).to_dict(),
                "visual_motion_delta": None,
                "visual_stuck_samples": visual_stuck_samples,
            },
        )
        termination_reason = "persisted_resurrection_sickness_wait"
        steps = 0

    if start_game_state.death_or_blocking_modal:
        navigator.stop()
    elif route_entry_pending and start_distance <= route_entry_reached_distance:
        target_recovery_count = 0
        next_target_node = _consume_reached_route_entry(
            route_entry_queue,
            start_coord,
            reached_distance=route_entry_reached_distance,
        )
        route_entry_pending = next_target_node is not None
        if next_target_node is None:
            next_target_node = _choose_route_target(
                config,
                start_coord,
                target_coord=None,
                zone_id=zone_id,
                auto_zone=auto_zone,
                min_target_distance=min_target_distance,
                excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
                auto_zone_ids=set(route_zone_ids),
                route_after_sort_key=route_entry_sort_key,
            )
        if next_target_node is None:
            reached = True
            termination_reason = "no_target_after_route_entry"
        else:
            target_node = next_target_node
            target = target_node.coord
            steering_target, route_target_passed, route_following_observation = (
                _directed_route_steering_target(
                    route_follower,
                    current_coord=start_coord,
                    target_node=target_node,
                    route_entry_pending=route_entry_pending,
                )
            )
            final_distance = MovementNavigator.distance(start_coord, target)
            route_motion.move_towards(
                start_coord,
                steering_target,
                now=time.monotonic(),
            )
    elif not route_entry_pending and start_distance <= _route_target_reached_distance(
        config,
        reached_distance,
        target_node,
    ):
        ore_training_capture = None
        if not is_route_waypoint(target_node):
            ore_training_capture = _capture_reached_ore_training_sample(
                capture,
                start_frame,
                config,
                target_node=target_node,
                target_coord=target,
                reached_coord=start_coord,
                distance=start_distance,
                index=-1,
                route_output_dir=output_path,
            )
        target_history.append(
            _target_history_item(
                target_node,
                target,
                start_coord,
                start_distance,
                ore_training_capture=ore_training_capture,
            )
        )
        _record_target_route_node_milestone(
            route_node_history,
            completed_route_node_indices,
            route_milestones,
            target_node=target_node,
            reached_coord=start_coord,
            distance=start_distance,
        )
        if not is_route_waypoint(target_node):
            completed_target_coords.add(target)
        reached = len(target_history) >= target_limit
        if not reached:
            next_target_node = _choose_route_target(
                config,
                start_coord,
                target_coord=None,
                zone_id=zone_id,
                auto_zone=auto_zone,
                min_target_distance=min_target_distance,
                excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
                auto_zone_ids=set(route_zone_ids),
                route_after_sort_key=_next_route_sort_key(target_node),
            )
            if next_target_node is None:
                reached = True
                termination_reason = "no_target_after_start"
            else:
                target_node = next_target_node
                target = target_node.coord
                steering_target, route_target_passed, route_following_observation = (
                    _directed_route_steering_target(
                        route_follower,
                        current_coord=start_coord,
                        target_node=target_node,
                        route_entry_pending=route_entry_pending,
                    )
                )
                final_distance = MovementNavigator.distance(start_coord, target)
                route_motion.move_towards(
                    start_coord,
                    steering_target,
                    now=time.monotonic(),
                )
    else:
        route_motion.move_towards(
            start_coord,
            steering_target,
            now=time.monotonic(),
        )

    if steps > 0 and not reached and not start_game_state.death_or_blocking_modal:
        initial_mount_action = mount_state.observe(
            now=time.monotonic(),
            mounted=start_mounted,
            combat_active=initial_combat.active,
        )
        if initial_mount_action is not None:
            navigator.stop()
            if initial_mount_action == "mount_cast":
                input_controller.tap_key(mount_state.key, duration=0.06)
                mount_casts += 1

    video_recorder = WindowVideoRecorder(
        config,
        target_hwnd=hwnd,
        output_dir=output_path,
    )
    video_recorder.start()
    image_writer = AsyncImageWriter(
        max_pending=max(
            1,
            int(
                config.get("training_capture", {})
                .get("route_live", {})
                .get("image_writer_queue_size", 16)
            ),
        )
    )
    image_writer_stats = ImageWriterStats(0, 0, 0, 0)

    try:
        for index in range(max_probe_steps):
            capture.activate_window()
            frame = capture.capture_client_region()
            now = time.monotonic()
            minimap = capture.crop_minimap(frame, config) if mining_cycle.enabled else None
            observation = observe_route_frame(
                frame,
                config,
                timestamp=now,
                minimap=minimap,
                recognize_ore=mining_cycle.enabled,
            )
            frame_telemetry = observation.telemetry
            minimap_ore_tooltip = read_minimap_ore_tooltip_telemetry(frame, config)
            route_motion.observe_heading(
                frame_telemetry.heading_degrees if frame_telemetry is not None else None,
                now=now,
            )
            game_state = observation.game_state
            if game_state.death_or_blocking_modal:
                _prepare_death_recovery_input(
                    route_motion,
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
            early_combat = observation.combat
            bright_ore_points = list(observation.bright_ore_points)
            dark_ore_points = list(observation.dark_ore_points)
            ore_presence = mining_cycle.observe_minimap(bright_ore_points, dark_ore_points)
            if game_state.death_or_blocking_modal and mining_cycle.active:
                mining_cycle.fail("death_or_blocking_modal", now=now)
                mining_interactor.cancel_hover_scan()
            if game_state.death_or_blocking_modal and minimap_tooltip_probe.active:
                minimap_tooltip_probe.cancel("death_or_blocking_modal", now=now)
            if (
                external_stop_requested_at is None
                and _external_stop_file_requested(external_stop_path)
            ):
                external_stop_requested_at = now
            if external_stop_requested_at is not None:
                external_drain_required = _route_probe_requires_safe_drain(
                    combat_handling_enabled=combat_handling_enabled,
                    combat_active=early_combat.active or combat_fallback.engaged,
                    death_or_blocking_modal=game_state.death_or_blocking_modal,
                    mining_active=bool(
                        mining_cycle.active or minimap_tooltip_probe.active
                    ),
                    post_combat_loot_active=post_combat_loot.active,
                    death_recovery_active=(
                        death_recovery.death_seen
                        and not death_recovery.resurrect_sickness_wait_consumed
                    ),
                )
                if not external_drain_required:
                    termination_reason = "external_stop_requested"
                    break
                if now - external_stop_requested_at >= external_stop_drain_seconds:
                    termination_reason = "external_stop_safe_drain_timeout"
                    break
            safe_drain_active = index >= planned_steps
            if safe_drain_active:
                if not _route_probe_requires_safe_drain(
                    combat_handling_enabled=combat_handling_enabled,
                    combat_active=early_combat.active or combat_fallback.engaged,
                    death_or_blocking_modal=game_state.death_or_blocking_modal,
                    mining_active=bool(
                        mining_cycle.active or minimap_tooltip_probe.active
                    ),
                    post_combat_loot_active=post_combat_loot.active,
                    death_recovery_active=(
                        death_recovery.death_seen
                        and not death_recovery.resurrect_sickness_wait_consumed
                    ),
                ):
                    termination_reason = (
                        "step_limit_after_safe_drain" if safe_drain_started else "step_limit"
                    )
                    break
                safe_drain_started = True
            executed_steps = index + 1
            if continuous_face_search_enabled and early_combat.outgoing_damage_visible:
                _release_continuous_face_search(
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
            armor_critical = observation.armor_critical
            if armor_critical:
                armor_critical_observed = True
            mounted = observation.mounted
            mounted_observed = mounted_observed or mounted
            forbidden_subzone_visible = observation.forbidden_subzone_visible
            if (
                forbidden_subzone_visible
                and not _should_allow_forbidden_marker_during_hazard_egress(
                    hazard_egress_active=hazard_egress_active,
                    route_entry_pending=route_entry_pending,
                    planner_available=dynamic_entry_planner is not None,
                )
            ):
                navigator.stop()
                mining_interactor.cancel_hover_scan()
                termination_reason = "forbidden_underground_subzone"
                break
            if _should_handle_dismissable_modal(game_state, early_combat):
                navigator.stop()
                screen_x, screen_y = capture.client_to_screen_point(*game_state.dismissable_modal_click)
                mouse_controller.move_to(screen_x, screen_y)
                mouse_controller.left_click(duration=death_release_click_duration)
                action = "dismiss_modal"
                frame_candidate = frame_dir / f"{index:04d}.png"
                frame_path = (
                    frame_candidate
                    if image_writer.submit(frame_candidate, frame)
                    else None
                )
                if frame_path is not None:
                    last_frame_saved_at = now
                    last_frame_action = action
                navigation = analyze_local_navigation(frame, config)
                report_candidate = report_dir / f"{index:04d}.png"
                report_path = (
                    report_candidate
                    if image_writer.submit(
                        report_candidate,
                        draw_local_navigation_report(frame, navigation),
                    )
                    else None
                )
                visual_motion_delta = _visual_motion_delta(previous_motion_frame, frame)
                previous_motion_frame = frame
                _append_jsonl(
                    metadata_path,
                    {
                        "index": index,
                        "timestamp": time.time(),
                        "foreground_ok": foreground_ok,
                        "frame": _relative_posix(frame_path, output_path) if frame_path else None,
                        "report": _relative_posix(report_path, output_path) if report_path else None,
                        "coord": final_coord,
                        "coord_candidates": [],
                        "coord_resynced": False,
                        "target_coord": target,
                        "target_node_id": target_node.node_id,
                        "target_zone_id": target_node.zone_id,
                        "detected_zone_name": detected_zone_name,
                        "route_zone_ids": route_zone_ids,
                        "completed_targets": len(target_history),
                        "skipped_targets": len(skipped_target_history),
                        "skipped_target_history": skipped_target_history,
                        "blocked_target_coords": sorted(blocked_target_coords),
                        "target_recovery_count": target_recovery_count,
                        "target_recovery_limit": target_recovery_limit,
                        "distance": final_distance,
                        "action": action,
                        "local_turn_key": None,
                        "threat": None,
                        "combat": None,
                        "death_recovery": None,
                        "game_state": game_state.__dict__,
                        "movement": navigator.last_diagnostics.__dict__,
                        "mouse_steering": (
                            mouse_steering.last_result.to_dict()
                            if mouse_steering.last_result is not None
                            else None
                        ),
                        "navigation": navigation.to_dict(),
                        "visual_motion_delta": visual_motion_delta,
                        "visual_stuck_samples": visual_stuck_samples,
                        "coord_fresh": False,
                    },
                )
                time.sleep(0.5)
                continue
            if _should_handle_death_recovery(game_state, early_combat):
                death_decision = None
                action = "death_or_blocking_modal"
                death_local_turn_key: str | None = None
                spirit_healer_graveyard_distance: float | None = None
                spirit_healer_recentered = False
                death_coord_candidates: list[int] = []
                death_coord_resynced = False
                death_coord_fresh = False
                death_coord, death_coord_candidates, death_coord_resynced = _read_filtered_coord(
                    frame,
                    config,
                    previous_coord=final_coord,
                    reference_coords=reference_coords,
                    resync_state=coord_resync_state,
                    allow_distant_resync=True,
                )
                if death_coord is not None:
                    final_coord = death_coord
                    death_coord_fresh = True
                    last_coord_read_at = now
                    last_coord_accepted_at = now
                    coord_feedback_paused = False
                if death_recovery.enabled:
                    recovery_cfg = config.get("safety", {}).get("death_recovery", {})
                    # The first stable coordinate after Return to Graveyard is the
                    # only trustworthy cemetery-local origin. Target frames prove
                    # NPC identity, but contain no bearing and must never authorize
                    # an unbounded forward approach.
                    if (
                        spirit_healer_local_search_origin is None
                        and death_recovery.return_to_graveyard_confirmed
                        and final_coord is not None
                    ):
                        spirit_healer_local_search_origin = final_coord
                    if (
                        spirit_healer_local_search_origin is not None
                        and final_coord is not None
                    ):
                        spirit_healer_graveyard_distance = MovementNavigator.distance(
                            final_coord,
                            spirit_healer_local_search_origin,
                        )
                    priority_death_decision = death_recovery.decide_open_dialog(
                        frame,
                        config,
                        now=now,
                    )
                    if priority_death_decision is not None:
                        death_decision = priority_death_decision
                        action = priority_death_decision.action
                        navigator.stop()
                        spirit_healer_local_search_target = None
                        spirit_healer_local_search_leg_progress.reset()
                    elif (
                        spirit_healer_graveyard_distance is not None
                        and spirit_healer_graveyard_distance
                        > spirit_healer_local_search_leash_radius
                    ):
                        if spirit_healer_local_search_target != spirit_healer_local_search_origin:
                            spirit_healer_local_search_target = spirit_healer_local_search_origin
                            spirit_healer_local_search_leg_progress.reset()
                        spirit_healer_recentered = True
                    local_search_distance = (
                        MovementNavigator.distance(final_coord, spirit_healer_local_search_target)
                        if (
                            priority_death_decision is None
                            and final_coord is not None
                            and spirit_healer_local_search_target is not None
                        )
                        else None
                    )
                    local_search_leg_complete = False
                    if (
                        local_search_distance is not None
                        and spirit_healer_local_search_target is not None
                    ):
                        local_search_leg_complete, _local_search_leg_reason = (
                            spirit_healer_local_search_leg_progress.observe(
                                target_coord=spirit_healer_local_search_target,
                                distance=local_search_distance,
                                now=now,
                            )
                        )
                    if priority_death_decision is not None:
                        pass
                    elif (
                        spirit_healer_local_search_enabled
                        and local_search_distance is not None
                        and not local_search_leg_complete
                    ):
                        route_motion.move_towards(
                            final_coord,
                            spirit_healer_local_search_target,
                            now=now,
                        )
                        death_local_turn_key = navigator.last_diagnostics.turn_key
                        action = "death_recovery_navigate_local_spirit_search"
                        death_decision = DeathRecoveryDecision(
                            action,
                            attempts=death_recovery.spirit_healer_search_attempts,
                            wait_seconds=spirit_healer_local_search_poll_seconds,
                            reason=(
                                (
                                    "graveyard_leash_recenter:"
                                    f"distance={spirit_healer_graveyard_distance:.3f}:"
                                    f"limit={spirit_healer_local_search_leash_radius:.3f}"
                                )
                                if spirit_healer_recentered
                                else (
                                    f"local_search:index={spirit_healer_local_search_index - 1}:"
                                    f"distance={local_search_distance:.3f}:"
                                    f"best={spirit_healer_local_search_leg_progress.best_distance:.3f}"
                                )
                            ),
                        )
                    else:
                        if spirit_healer_local_search_target is not None:
                            navigator.stop()
                            spirit_healer_local_search_target = None
                            spirit_healer_local_search_leg_progress.reset()
                        death_decision = death_recovery.decide(frame, game_state, config, now=now)
                        action = death_decision.action
                        anchor = _choose_spirit_healer_coordinate_anchor(final_coord, config)
                        if (
                            anchor is not None
                            and spirit_healer_local_search_origin is not None
                            and MovementNavigator.distance(
                                spirit_healer_local_search_origin,
                                anchor[1],
                            )
                            > spirit_healer_local_search_leash_radius
                        ):
                            anchor = None
                        anchor_after_search_attempts = max(
                            0,
                            int(recovery_cfg.get("spirit_healer_anchor_after_search_attempts", 8)),
                        )
                        anchor_due = _spirit_healer_anchor_due(
                            action,
                            anchor,
                            search_attempts=death_recovery.spirit_healer_search_attempts,
                            after_attempts=anchor_after_search_attempts,
                        )
                        if anchor_due and anchor is not None:
                            anchor_name, anchor_coord, anchor_distance, arrival_distance = anchor
                            if anchor_distance > arrival_distance and final_coord is not None:
                                route_motion.move_towards(
                                    final_coord,
                                    anchor_coord,
                                    now=now,
                                )
                                death_local_turn_key = navigator.last_diagnostics.turn_key
                                action = "death_recovery_navigate_to_spirit_anchor"
                                death_decision = DeathRecoveryDecision(
                                    action,
                                    attempts=death_recovery.spirit_healer_search_attempts,
                                    wait_seconds=max(
                                        0.05,
                                        float(recovery_cfg.get("spirit_healer_anchor_poll_seconds", 0.20)),
                                    ),
                                    reason=(
                                        f"configured_anchor:{anchor_name}:"
                                        f"distance={anchor_distance:.3f}"
                                    ),
                                )
                        elif (
                            action == "death_recovery_search_spirit_healer"
                            and spirit_healer_local_search_enabled
                            and final_coord is not None
                        ):
                            if spirit_healer_local_search_origin is None:
                                spirit_healer_local_search_origin = final_coord
                            spirit_healer_local_search_target = _spirit_healer_local_search_coord(
                                spirit_healer_local_search_origin,
                                spirit_healer_local_search_index,
                                config,
                            )
                            spirit_healer_local_search_index += 1
                            local_search_distance = MovementNavigator.distance(
                                final_coord,
                                spirit_healer_local_search_target,
                            )
                            route_motion.move_towards(
                                final_coord,
                                spirit_healer_local_search_target,
                                now=now,
                            )
                            death_local_turn_key = navigator.last_diagnostics.turn_key
                            action = "death_recovery_navigate_local_spirit_search"
                            death_decision = DeathRecoveryDecision(
                                action,
                                attempts=death_recovery.spirit_healer_search_attempts,
                                wait_seconds=spirit_healer_local_search_poll_seconds,
                                reason=(
                                    f"local_search:index={spirit_healer_local_search_index - 1}:"
                                    f"distance={local_search_distance:.3f}"
                                ),
                            )
                    if action not in {
                        "death_recovery_navigate_to_spirit_anchor",
                        "death_recovery_navigate_local_spirit_search",
                    }:
                        navigator.stop()
                    else:
                        interact_due, next_spirit_healer_search_interact_at = (
                            _spirit_search_interact_due(
                                action,
                                now=now,
                                next_at=next_spirit_healer_search_interact_at,
                                interval=spirit_healer_search_interact_interval,
                            )
                        )
                    if action in {
                        "death_recovery_navigate_to_spirit_anchor",
                        "death_recovery_navigate_local_spirit_search",
                    } and interact_due:
                        capture.activate_window()
                        input_controller.tap_key(
                            death_recovery.spirit_healer_target_key,
                            duration=max(0.0, death_click_duration),
                        )
                        input_controller.tap_key(
                            death_recovery.spirit_healer_interact_key,
                            duration=max(0.0, death_click_duration),
                        )
                    if (
                        action == "death_recovery_hover_spirit_healer"
                        and death_decision.click_point is not None
                    ):
                        screen_x, screen_y = capture.client_to_screen_point(*death_decision.click_point)
                        capture.activate_window()
                        time.sleep(max(0.0, death_click_settle_seconds))
                        mouse_controller.move_to(screen_x, screen_y)
                    elif (
                        action
                        in {
                            "death_recovery_release_spirit",
                            "death_recovery_resurrect_safe_zone",
                            "death_recovery_safe_zone_destination",
                            "death_recovery_return_to_graveyard",
                            "death_recovery_accept_return_to_graveyard",
                            "death_recovery_right_click_spirit_healer",
                            "death_recovery_return_to_life",
                            "death_recovery_accept_resurrection",
                        }
                        and death_decision.click_point is not None
                    ):
                        screen_x, screen_y = capture.client_to_screen_point(*death_decision.click_point)
                        capture.activate_window()
                        time.sleep(max(0.0, death_click_settle_seconds))
                        mouse_controller.move_to(screen_x, screen_y)
                        time.sleep(max(0.0, death_click_settle_seconds))
                        if death_decision.mouse_button == "right":
                            mouse_controller.right_click(duration=death_click_duration)
                        else:
                            mouse_controller.left_click(duration=death_click_duration)
                    elif (
                        action
                        in {
                            "death_recovery_dismiss_death_recap",
                            "death_recovery_target_spirit_healer",
                            "death_recovery_interact_spirit_healer_target",
                        }
                        and death_decision.key is not None
                    ):
                        capture.activate_window()
                        time.sleep(max(0.0, death_click_settle_seconds))
                        input_controller.tap_key(death_decision.key, duration=max(0.0, death_click_duration))
                    elif action == "death_recovery_approach_spirit_healer" and death_decision.click_point is not None:
                        death_local_turn_key = _turn_key_towards_click_point(
                            death_decision.click_point,
                            frame.shape,
                            deadzone_fraction=spirit_healer_approach_deadzone_fraction,
                        )
                        if death_local_turn_key is not None:
                            navigator.turn_character(
                                death_local_turn_key,
                                duration=max(0.0, spirit_healer_approach_turn_seconds),
                                context="spirit",
                            )
                        input_controller.tap_key("W", duration=max(0.0, spirit_healer_approach_forward_seconds))
                    elif action == "death_recovery_search_spirit_healer":
                        death_local_turn_key = spirit_healer_search_turn_key
                        navigator.turn_character(
                            death_local_turn_key,
                            duration=max(0.0, spirit_healer_search_turn_seconds),
                            context="spirit",
                        )
                frame_candidate = frame_dir / f"{index:04d}.png"
                frame_path = (
                    frame_candidate
                    if image_writer.submit(frame_candidate, frame)
                    else None
                )
                if frame_path is not None:
                    last_frame_saved_at = now
                    last_frame_action = action
                report_candidate = report_dir / f"{index:04d}.png"
                navigation = analyze_local_navigation(frame, config)
                report_path = (
                    report_candidate
                    if image_writer.submit(
                        report_candidate,
                        draw_local_navigation_report(frame, navigation),
                    )
                    else None
                )
                visual_motion_delta = _visual_motion_delta(previous_motion_frame, frame)
                previous_motion_frame = frame
                _append_jsonl(
                    metadata_path,
                    {
                        "index": index,
                        "timestamp": time.time(),
                        "foreground_ok": foreground_ok,
                        "frame": _relative_posix(frame_path, output_path) if frame_path else None,
                        "report": _relative_posix(report_path, output_path) if report_path else None,
                        "coord": final_coord,
                        "coord_candidates": death_coord_candidates,
                        "coord_resynced": death_coord_resynced,
                        "target_coord": target,
                        "target_node_id": target_node.node_id,
                        "target_zone_id": target_node.zone_id,
                        "detected_zone_name": detected_zone_name,
                        "route_zone_ids": route_zone_ids,
                        "completed_targets": len(target_history),
                        "skipped_targets": len(skipped_target_history),
                        "skipped_target_history": skipped_target_history,
                        "blocked_target_coords": sorted(blocked_target_coords),
                        "target_recovery_count": target_recovery_count,
                        "target_recovery_limit": target_recovery_limit,
                        "distance": final_distance,
                        "action": action,
                        "spirit_healer_graveyard_origin": spirit_healer_local_search_origin,
                        "spirit_healer_graveyard_distance": spirit_healer_graveyard_distance,
                        "spirit_healer_graveyard_leash_radius": spirit_healer_local_search_leash_radius,
                        "local_turn_key": death_local_turn_key,
                        "threat": None,
                        "combat": None,
                        "death_recovery": death_decision.to_dict() if death_decision is not None else None,
                        "game_state": game_state.__dict__,
                        "movement": navigator.last_diagnostics.__dict__,
                        "navigation": navigation.to_dict(),
                        "visual_motion_delta": visual_motion_delta,
                        "visual_stuck_samples": visual_stuck_samples,
                        "coord_fresh": death_coord_fresh,
                    },
                )
                if action in {
                    "death_recovery_release_spirit",
                    "death_recovery_resurrect_safe_zone",
                    "death_recovery_safe_zone_destination",
                    "death_recovery_return_to_graveyard",
                    "death_recovery_accept_return_to_graveyard",
                    "death_recovery_hover_spirit_healer",
                    "death_recovery_right_click_spirit_healer",
                    "death_recovery_target_spirit_healer",
                    "death_recovery_interact_spirit_healer_target",
                    "death_recovery_approach_spirit_healer",
                    "death_recovery_search_spirit_healer",
                    "death_recovery_navigate_to_spirit_anchor",
                    "death_recovery_navigate_local_spirit_search",
                    "death_recovery_return_to_life",
                    "death_recovery_accept_resurrection",
                    "death_recovery_dismiss_death_recap",
                    "death_recovery_wait",
                }:
                    time.sleep(max(0.0, death_decision.wait_seconds if death_decision is not None else 0.0))
                    continue
                termination_reason = "death_or_blocking_modal" if action == "death_or_blocking_modal" else action
                break

            spirit_healer_local_search_origin = None
            spirit_healer_local_search_target = None
            spirit_healer_local_search_index = 0
            spirit_healer_local_search_leg_progress.reset()
            next_spirit_healer_search_interact_at = 0.0
            resurrect_wait_seconds = death_recovery.consume_resurrected_wait(game_state)
            if resurrect_wait_seconds is not None:
                navigator.stop()
                navigation = analyze_local_navigation(frame, config)
                visual_motion_delta = _visual_motion_delta(previous_motion_frame, frame)
                previous_motion_frame = frame
                frame_candidate = frame_dir / f"{index:04d}.png"
                frame_path = (
                    frame_candidate
                    if image_writer.submit(frame_candidate, frame)
                    else None
                )
                if frame_path is not None:
                    last_frame_saved_at = now
                    last_frame_action = "death_recovery_resurrection_sickness_wait"
                report_candidate = report_dir / f"{index:04d}.png"
                report_path = (
                    report_candidate
                    if image_writer.submit(
                        report_candidate,
                        draw_local_navigation_report(frame, navigation),
                    )
                    else None
                )
                _append_jsonl(
                    metadata_path,
                    {
                        "index": index,
                        "timestamp": time.time(),
                        "foreground_ok": foreground_ok,
                        "frame": _relative_posix(frame_path, output_path) if frame_path else None,
                        "report": _relative_posix(report_path, output_path) if report_path else None,
                        "coord": final_coord,
                        "coord_candidates": [],
                        "coord_resynced": False,
                        "target_coord": target,
                        "target_node_id": target_node.node_id,
                        "target_zone_id": target_node.zone_id,
                        "detected_zone_name": detected_zone_name,
                        "route_zone_ids": route_zone_ids,
                        "completed_targets": len(target_history),
                        "skipped_targets": len(skipped_target_history),
                        "skipped_target_history": skipped_target_history,
                        "blocked_target_coords": sorted(blocked_target_coords),
                        "target_recovery_count": target_recovery_count,
                        "target_recovery_limit": target_recovery_limit,
                        "distance": final_distance,
                        "action": "death_recovery_resurrection_sickness_wait",
                        "local_turn_key": None,
                        "threat": None,
                        "combat": None,
                        "death_recovery": {
                            "action": "death_recovery_resurrection_sickness_wait",
                            "attempts": death_recovery.attempts,
                            "click_point": None,
                            "button_bbox": None,
                            "wait_seconds": resurrect_wait_seconds,
                            "reason": "resurrected_after_release",
                        },
                        "game_state": game_state.__dict__,
                        "movement": navigator.last_diagnostics.__dict__,
                        "navigation": navigation.to_dict(),
                        "visual_motion_delta": visual_motion_delta,
                        "visual_stuck_samples": visual_stuck_samples,
                        "coord_fresh": False,
                    },
                )
                write_resurrection_sickness_state(resurrection_sickness_state_path, resurrect_wait_seconds)
                termination_reason = "resurrection_sickness_wait"
                break

            navigation = analyze_local_navigation(frame, config)
            visual_motion_delta = _visual_motion_delta(previous_motion_frame, frame)
            previous_motion_frame = frame
            previous_coord = final_coord
            combat = early_combat
            combat_engaged = (
                combat_fallback.observe(combat, now=now)
                if combat_handling_enabled
                else False
            )
            combat_threat_assessment = combat_threat_estimator.observe(
                combat,
                now=now,
                engaged=bool(combat_handling_enabled and combat_engaged),
            )
            # Coordinate polling happens later in this iteration. Mining uses the
            # last accepted coordinate here, which is always represented by
            # final_coord; current_coord is intentionally iteration-local below.
            mining_coord = final_coord
            mining_candidate_missing = False
            mining_candidate = None
            tooltip_probe_track_id = ore_presence.confirmed_track_id
            tooltip_probe_marker_point = ore_presence.confirmed_point
            if (
                tooltip_probe_track_id is None
                and mining_cycle.phase == MiningPhase.IDLE
            ):
                center_fallback = _covered_center_tooltip_probe_target(
                    current_coord=mining_coord,
                    nodes=mining_candidate_nodes,
                    mining_cycle=mining_cycle,
                    now=now,
                    minimap_shape=minimap.shape,
                    config=config,
                )
                if center_fallback is not None:
                    tooltip_probe_track_id, tooltip_probe_marker_point = center_fallback
            minimap_tooltip_step = minimap_tooltip_probe.observe(
                track_id=tooltip_probe_track_id,
                marker_point=tooltip_probe_marker_point,
                now=now,
                tooltip_ore_id=(
                    minimap_ore_tooltip.ore_id
                    if minimap_ore_tooltip is not None
                    else None
                ),
                tooltip_ore_type=(
                    minimap_ore_tooltip.ore_type
                    if minimap_ore_tooltip is not None
                    else None
                ),
                control_available=bool(
                    mining_cycle.phase == MiningPhase.IDLE
                    and not combat_engaged
                    and not game_state.death_or_blocking_modal
                    and not post_combat_loot.active
                ),
            )
            if minimap_tooltip_step.release_cursor:
                release_point_cfg = config.get("mining", {}).get(
                    "minimap_tooltip_probe", {}
                ).get(
                    "release_cursor_point",
                    config.get("mining", {}).get(
                        "cursor_baseline_point", {"x": 1280, "y": 170}
                    ),
                )
                release_point = _scaled_client_point(
                    release_point_cfg,
                    frame.shape,
                    config,
                )
                screen_x, screen_y = capture.client_to_screen_point(*release_point)
                mouse_controller.move_to(screen_x, screen_y)
            tooltip_candidate_confirmed = minimap_tooltip_probe.confirmed
            mining_marker_point = (
                minimap_tooltip_probe.confirmed_point
                if tooltip_candidate_confirmed
                else ore_presence.confirmed_point
            )
            if (
                mining_cycle.enabled
                and mining_cycle.phase == MiningPhase.IDLE
                and (ore_presence.confirmed_bright or tooltip_candidate_confirmed)
                and mining_coord is not None
                and (
                    tooltip_candidate_confirmed
                    or not mining_cycle.require_minimap_tooltip_confirmation
                )
            ):
                mining_candidate = mining_cycle.choose_candidate(
                    mining_coord,
                    mining_candidate_nodes,
                    now=now,
                    marker_point=mining_marker_point,
                    minimap_shape=minimap.shape,
                    tooltip_confirmed=tooltip_candidate_confirmed,
                    confirmed_ore_type=minimap_tooltip_probe.confirmed_ore_type,
                )
                mining_candidate_missing = mining_candidate is None
            mounted_escape_mining_conflict = _mounted_escape_has_mining_conflict(
                mining_active=mining_cycle.active,
                tooltip_probe_active=minimap_tooltip_probe.active,
                candidate_available=mining_candidate is not None,
            )
            mounted_escape_decision = mounted_escape.observe(
                now=now,
                combat_engaged=combat_engaged,
                mounted=mounted,
                player_health_fraction=combat.player_health_fraction,
                current_coord=final_coord,
                mining_conflict=mounted_escape_mining_conflict,
                heal_casting=combat_heal.is_casting(now),
                threat_level=combat_threat_assessment.level,
                route_hazard=(
                    route_hazard_guard.contains_coord(final_coord)
                    or route_hazard_raw_candidate
                ),
            )
            mounted_combat_bypass = mounted_escape_decision.active
            combat_blocks_route = combat_engaged and not mounted_combat_bypass
            dead_hostile_target_visible = detect_dead_hostile_target_marker(
                frame,
                config,
            )
            combat_loot_pending_visible = detect_combat_loot_pending_marker(
                frame,
                config,
            )
            combat_cleared = bool(
                combat_handling_enabled and combat_fallback.consume_clear_event()
            )
            stale_soft_target_cleared = bool(
                combat_cleared
                and combat_fallback.consume_stale_soft_target_clear_event()
            )
            combat_clear_had_confirmed_outgoing = bool(
                combat_cleared
                and combat_fallback.consume_confirmed_outgoing_clear_event()
            )
            if combat_cleared:
                _release_continuous_face_search(
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
                periodic_combat_keys.reset()
                aligned_periodic_combat_keys.reset()
                low_health_priority_combat_keys.reset()
                if stale_soft_target_cleared:
                    input_controller.tap_key("ESC", duration=0.06)
                loot_armed = bool(
                    combat_clear_had_confirmed_outgoing
                    and post_combat_loot.arm(now=now, in_combat=False)
                )
                if not loot_armed and not post_combat_loot.active:
                    mining_cycle.resume_after_interrupt(
                        target_visible=ore_presence.target_visible,
                        confirmed_bright=ore_presence.confirmed_bright,
                        now=now,
                    )
            elif combat_handling_enabled and combat_blocks_route and mining_cycle.active:
                mining_cycle.suspend(now=now)
                mining_interactor.cancel_hover_scan()
            if combat_blocks_route and combat_loot_pending_visible:
                post_combat_loot.arm(now=now, in_combat=True)
            if (
                combat_blocks_route
                and post_combat_loot.active
                and not post_combat_loot.armed_in_combat
            ):
                post_combat_loot.cancel(now=now, reason="combat_reentered")
            post_combat_loot_decision = post_combat_loot.observe(
                now=now,
                dead_hostile_target_visible=dead_hostile_target_visible,
                loot_opened_visible=detect_loot_opened_marker(frame, config),
                target_is_attacker_visible=combat.target_is_attacker,
                control_available=not (
                    combat_handling_enabled and combat_heal.is_casting(now)
                ),
            )
            post_combat_loot_outcome = post_combat_loot.consume_outcome()
            if post_combat_loot_outcome is not None:
                post_combat_loot_outcomes.append(
                    {
                        "index": index,
                        "attempted": post_combat_loot_outcome.attempted,
                        "reason": post_combat_loot_outcome.reason,
                        "elapsed_seconds": post_combat_loot_outcome.elapsed_seconds,
                        "interactions": post_combat_loot_outcome.interactions,
                        "confirmed_loots": post_combat_loot_outcome.confirmed_loots,
                    }
                )
                if not combat_blocks_route and not post_combat_loot.active:
                    mining_cycle.resume_after_interrupt(
                        target_visible=ore_presence.target_visible,
                        confirmed_bright=ore_presence.confirmed_bright,
                        now=now,
                    )
            combat_fallback.clear_tapped_keys()
            combat_heal_keys_tapped: list[str] = []
            post_combat_loot_keys_tapped: list[str] = []
            threat = None
            threat_turn_key: str | None = None
            if not combat_handling_enabled:
                threat_commitment.choose_turn_key(
                    None,
                    now=now,
                    fallback_turn_key=alternate_turn_key,
                )
            elif combat_blocks_route or combat_heal.is_casting(now) or mounted_combat_bypass:
                threat_commitment.choose_turn_key(None, now=now, fallback_turn_key=alternate_turn_key)
            else:
                threat = detect_hostile_threat(frame, config, fallback_turn_key=alternate_turn_key)
                threat_turn_key = threat_commitment.choose_turn_key(
                    threat,
                    now=now,
                    fallback_turn_key=alternate_turn_key,
                )
            current_coord: int | None = None
            coord_candidates: list[int] = []
            resynced = False
            coord_delta: float | None = None
            coord_fresh = False
            coord_feedback_age = max(0.0, now - last_coord_accepted_at)
            coord_jump_limit = _route_coord_jump_limit(
                config,
                elapsed_seconds=coord_feedback_age,
                mounted=mounted,
            )
            due_for_coord = _should_read_route_coord(
                now=now,
                last_coord_read_at=last_coord_read_at,
                coord_read_interval=coord_read_interval,
                index=index,
                held_key=navigator.held_key,
                reached=reached,
                combat_engaged=combat_blocks_route if combat_handling_enabled else False,
                combat_heal_casting=(
                    combat_heal.is_casting(now) if combat_handling_enabled else False
                ),
            )
            if (
                navigator.recovery_detour_active
                and now - last_coord_read_at >= recovery_coord_interval
            ):
                due_for_coord = True
            if (
                not combat_blocks_route
                and not combat_heal.is_casting(now)
                and mining_final_approach.precision_coord_feedback_active
                and now - last_coord_read_at
                >= mining_final_approach.precision_coord_interval_seconds
            ):
                # Final arrival needs one post-stop sample, but waiting for the
                # ordinary route OCR cadence creates a visible idle pause. This
                # bounded override exists only during the stopped recheck or
                # post-burst feedback state and adds no extra capture/input wait.
                due_for_coord = True

            if due_for_coord:
                current_coord, coord_candidates, resynced = _read_filtered_coord(
                    frame,
                    config,
                    previous_coord=final_coord,
                    reference_coords=reference_coords,
                    resync_state=coord_resync_state,
                    max_jump=coord_jump_limit,
                )
                last_coord_read_at = time.monotonic()
                route_hazard_raw_candidate = route_hazard_guard.contains_any_coord(
                    coord_candidates
                )
                coord_fresh = current_coord is not None
                if coord_fresh:
                    last_coord_accepted_at = now
                    coord_feedback_age = 0.0
                    coord_feedback_paused = False
                if previous_coord is not None and current_coord is not None:
                    coord_delta = MovementNavigator.distance(previous_coord, current_coord)
                if current_coord is not None:
                    # Position is shared by route, combat, mining and recovery. Keeping
                    # it branch-local made skipped OCR ticks fall back to an old value.
                    final_coord = current_coord
                    if (
                        dynamic_entry_planner is not None
                        and _should_complete_hazard_egress(
                            hazard_egress_active=hazard_egress_active,
                            coord_is_hazard=dynamic_entry_planner.coord_is_hazard(
                                current_coord
                            ),
                            forbidden_subzone_visible=forbidden_subzone_visible,
                        )
                    ):
                        hazard_egress_active = False
                if route_hazard_raw_candidate and mounted_escape_decision.active:
                    # The accepted-coordinate filter can deliberately reject a
                    # sudden mounted jump. Hazard safety must still see the raw
                    # captured-pixel candidates, otherwise a real 63.53,49.70
                    # reading can remain hidden behind a stale 61.03,50.27 pose.
                    mounted_escape_decision = mounted_escape.observe(
                        now=now,
                        combat_engaged=combat_engaged,
                        mounted=mounted,
                        player_health_fraction=combat.player_health_fraction,
                        current_coord=final_coord,
                        mining_conflict=mounted_escape_mining_conflict,
                        heal_casting=combat_heal.is_casting(now),
                        threat_level=combat_threat_assessment.level,
                        route_hazard=True,
                    )
                    mounted_combat_bypass = mounted_escape_decision.active
                    combat_blocks_route = combat_engaged and not mounted_combat_bypass
            # Mining decisions in this same control tick must use the coordinate
            # accepted above, not the one-tick-old snapshot captured before OCR.
            mining_coord = current_coord if current_coord is not None else final_coord

            action = "continue_forward"
            local_turn_key: str | None = None
            local_avoid_pulsed = False
            diagnostics = navigator.last_diagnostics
            previous_distance_to_target = final_distance
            distance_progress: float | None = None
            distance = final_distance
            route_target_passed = False
            navmesh_guidance_observation = None
            navmesh_guidance_blocked = False
            if current_coord is not None:
                route_steering_suspended = bool(
                    mining_cycle.active
                    or minimap_tooltip_probe.active
                    or combat_blocks_route
                    or game_state.death_or_blocking_modal
                )
                steering_target, route_target_passed, route_following_observation = (
                    _directed_route_steering_target(
                        route_follower,
                        current_coord=current_coord,
                        target_node=target_node,
                        route_entry_pending=route_entry_pending,
                        suspended=route_steering_suspended,
                    )
                )
                if navmesh_route_guidance is not None:
                    guidance_active = bool(
                        not route_entry_pending
                        and target_node.source == ROUTE_WAYPOINT_SOURCE
                        and not route_steering_suspended
                    )
                    navmesh_guidance_observation = navmesh_route_guidance.observe(
                        current_coord,
                        target,
                        target_route_index=int(target_node.route_index or 0),
                        active=guidance_active,
                    )
                    if guidance_active:
                        if not navmesh_guidance_observation.blocked:
                            steering_target = navmesh_guidance_observation.steering_coord
                        elif navmesh_guidance_failure_mode == "block":
                            navmesh_guidance_blocked = True
            stuck_min_progress = float(config.get("movement", {}).get("stuck_min_progress", 0.03))
            stuck_min_coord_delta = float(config.get("movement", {}).get("stuck_min_coord_delta", stuck_min_progress))
            mount_action = mount_state.observe(
                now=now,
                mounted=mounted,
                combat_active=bool(combat.active or combat_engaged),
                post_combat_loot_active=post_combat_loot_decision.active,
                control_available=not bool(
                    combat_heal.is_casting(now)
                    or combat_blocks_route
                    or post_combat_loot_decision.active
                    or mining_cycle.active
                    or minimap_tooltip_probe.active
                    or mining_candidate is not None
                ),
            )
            if (
                mining_candidate is not None
                and mining_cycle.phase == MiningPhase.IDLE
                and not combat_engaged
            ):
                mining_cycle.begin(
                    mining_candidate,
                    now=now,
                    current_coord=mining_coord,
                    marker_track_id=minimap_tooltip_probe.confirmed_track_id,
                    marker_point=mining_marker_point,
                )

            control_owner = select_operational_control_owner(
                OperationalControlSignals(
                    combat_heal_casting=bool(
                        combat_handling_enabled and combat_heal.is_casting(now)
                    ),
                    combat_blocks_route=bool(
                        combat_handling_enabled and combat_blocks_route
                    ),
                    post_combat_loot_active=post_combat_loot_decision.active,
                    mining_active=bool(
                        mining_cycle.enabled
                        and (
                            mining_cycle.phase != MiningPhase.IDLE
                            or minimap_tooltip_probe.active
                        )
                    ),
                    mount_requested=bool(not reached and mount_action is not None),
                )
            )
            if control_owner not in {
                OperationalControlOwner.ROUTE,
                OperationalControlOwner.MINING,
            }:
                route_motion.suspend()

            low_health_priority_keys = (
                _tap_due_low_health_combat_key(
                    input_controller,
                    combat_fallback,
                    combat,
                    low_health_priority_combat_keys,
                    now=now,
                    health_threshold=low_health_priority_threshold,
                    attack_tap_duration=combat_attack_tap_duration,
                    combat_key_min_interval=combat_key_min_interval,
                )
                if combat_handling_enabled
                and (combat_blocks_route or combat_heal.is_casting(now))
                else []
            )

            if low_health_priority_keys:
                if emergency_heal_followup_enabled:
                    combat_heal_keys_tapped = _apply_emergency_heal_followup(
                        input_controller,
                        combat_fallback,
                        combat_heal,
                        low_health_priority_keys,
                        now=now,
                        heal_key=combat_heal_key,
                        heal_tap_duration=combat_heal_tap_duration,
                        delay_seconds=emergency_heal_followup_delay,
                    )
                _release_continuous_face_search(
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                action = (
                    "combat_low_health_priority_heal"
                    if combat_heal_keys_tapped
                    else "combat_low_health_priority"
                )
            elif control_owner == OperationalControlOwner.COMBAT_HEAL:
                _release_continuous_face_search(
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                action = "combat_low_health_heal_wait"
            elif control_owner == OperationalControlOwner.COMBAT:
                if navigator.held_key is not None:
                    navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                heal_action = _apply_combat_low_health_heal(
                    input_controller,
                    combat_heal,
                    combat,
                    now=now,
                    heal_key=combat_heal_key,
                    heal_tap_duration=combat_heal_tap_duration,
                )
                if heal_action is not None:
                    action = heal_action
                    combat_heal_keys_tapped = [combat_heal_key]
                elif combat.active or combat.has_latching_evidence() or combat_fallback.last_observation_latching_evidence:
                    action, local_turn_key = _apply_combat_fallback(
                        input_controller,
                        combat_fallback,
                        combat,
                        now=now,
                        face_turn_duration=combat_face_turn_duration,
                        face_search_turn_180_duration=combat_face_search_turn_180_duration,
                        face_search_turn_90_duration=combat_face_search_turn_90_duration,
                        attack_tap_duration=combat_attack_tap_duration,
                        combat_key_min_interval=combat_key_min_interval,
                        range_approach_duration=combat_range_approach_duration,
                        range_approach_cooldown=combat_range_approach_cooldown,
                        periodic_combat_keys=periodic_combat_keys,
                        aligned_periodic_combat_keys=aligned_periodic_combat_keys,
                        continuous_face_search=continuous_face_search_enabled,
                        attacker_target_selection=attacker_target_selection_enabled,
                        attacker_target_cycle_key=attacker_target_cycle_key,
                        attacker_target_cycle_interval=attacker_target_cycle_interval,
                        nameplate_facing=nameplate_facing_enabled,
                        nameplate_face_turn_duration=nameplate_face_turn_duration,
                        target_interact_facing=target_interact_facing_enabled,
                        target_interact_key=target_interact_key,
                        target_interact_cooldown=target_interact_cooldown,
                        target_interact_probe_delay=target_interact_probe_delay,
                        invert_turn_direction=invert_turn_direction,
                        mouse_steering=mouse_steering,
                    )
                else:
                    action = "combat_clear_wait"
            elif control_owner == OperationalControlOwner.POST_COMBAT_LOOT:
                _release_continuous_face_search(
                    input_controller,
                    combat_fallback,
                    mouse_steering,
                )
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                action = post_combat_loot_decision.action
                if post_combat_loot_decision.input_key is not None:
                    input_controller.tap_key(
                        post_combat_loot_decision.input_key,
                        duration=float(
                            config.get("post_combat_loot", {}).get(
                                "interact_tap_seconds",
                                0.06,
                            )
                        ),
                    )
                    post_combat_loot_keys_tapped.append(
                        post_combat_loot_decision.input_key
                    )
            elif (
                control_owner == OperationalControlOwner.MINING
                and minimap_tooltip_probe.active
            ):
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                if minimap_tooltip_step.move_point is not None:
                    client_point = minimap_marker_client_point(
                        minimap_tooltip_step.move_point,
                        frame.shape,
                        config,
                    )
                    screen_x, screen_y = capture.client_to_screen_point(*client_point)
                    mouse_controller.move_to(screen_x, screen_y)
                action = minimap_tooltip_step.action
            elif control_owner == OperationalControlOwner.MINING:
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                mining_cycle.observe_dark_target(now=now)
                if mining_cycle.phase == MiningPhase.INTERCEPT:
                    mining_cycle.observe_intercept_marker(
                        now=now,
                        current_coord=mining_coord,
                    )
                if mining_cycle.phase != MiningPhase.INTERCEPT:
                    mining_final_approach.reset()
                if mining_cycle.phase == MiningPhase.RESUME:
                    navigator.stop()
                    mining_outcome = mining_cycle.consume_outcome()
                    if mining_outcome is None:
                        action = "mining_resume_without_outcome"
                    else:
                        mining_outcomes.append(_mining_outcome_item(mining_outcome, index=index))
                        action = (
                            "mining_verified_resume_route"
                            if mining_outcome.success
                            else f"mining_failed_resume_route:{mining_outcome.reason}"
                        )
                        if mining_coord is not None:
                            access_resume = _mining_access_resume_queue(
                                mining_outcome,
                                zone_id=target_node.zone_id,
                            )
                            if access_resume:
                                route_entry_queue = access_resume
                                route_entry_pending = True
                                route_entry_sort_key = (
                                    float(mining_outcome.resume_route_index or 0) + 0.001
                                )
                                target_node = access_resume[0]
                                target = target_node.coord
                                final_distance = MovementNavigator.distance(
                                    mining_coord,
                                    target,
                                )
                                previous_distance_to_target = None
                                action = f"{action}:terrain_access_resume"
                            else:
                                resume_target = _mining_route_resume_target(
                                    config,
                                    mining_coord,
                                    route_loop,
                                    zone_id=target_node.zone_id,
                                    route_after_sort_key=_next_route_sort_key(target_node),
                                )
                                if resume_target is not None:
                                    route_entry_queue = []
                                    route_entry_pending = False
                                    target_node = resume_target
                                    target = resume_target.coord
                                    final_distance = MovementNavigator.distance(
                                        mining_coord,
                                        target,
                                    )
                                    previous_distance_to_target = None
                                    action = f"{action}:reprojected_forward"
                    mining_interactor.cancel_hover_scan()
                elif mining_cycle.phase == MiningPhase.SUSPENDED:
                    navigator.stop()
                    action = "mining_suspended_for_combat"
                elif mining_cycle.phase == MiningPhase.INTERCEPT:
                    candidate = mining_cycle.candidate
                    if candidate is None or mining_coord is None:
                        navigator.stop()
                        mining_outcome = mining_cycle.fail("candidate_or_coordinate_missing", now=now)
                        action = f"mining_failed:{mining_outcome.reason}"
                    else:
                        mining_cycle.refresh_direct_marker_intercept(
                            mining_coord,
                            minimap.shape,
                        )
                        intercept_target = mining_cycle.current_intercept_coord
                        if intercept_target is None:
                            navigator.stop()
                            mining_cycle.mark_intercept_reached(now=now)
                            action = "mining_face_node_pending"
                        else:
                            mining_distance = MovementNavigator.distance(
                                mining_coord,
                                intercept_target,
                            )
                            final_intercept = (
                                mining_cycle.intercept_cursor
                                >= len(mining_cycle.intercept_coords) - 1
                            )
                            intercept_reached_distance = (
                                (
                                    mining_cycle.final_world_scan_distance
                                    if mining_cycle.direct_marker_intercept_active
                                    else mining_cycle.world_search_distance
                                )
                                if final_intercept
                                else route_entry_reached_distance
                            )
                            must_finish_on_centered_marker = bool(
                                final_intercept
                                and mining_cycle.requires_centered_marker_confirmation
                            )
                            database_final_approach = bool(
                                final_intercept
                                and mining_cycle.final_intercept_uses_database_anchor
                            )
                            live_marker_final_approach = bool(
                                final_intercept
                                and must_finish_on_centered_marker
                                and mining_cycle.database_anchor_arrived
                            )
                            bounded_final_approach = bool(
                                database_final_approach or live_marker_final_approach
                            )
                            if bounded_final_approach:
                                final_decision = mining_final_approach.observe(
                                    mining_coord,
                                    intercept_target,
                                    now=now,
                                    coord_fresh=coord_fresh,
                                )
                                if final_decision.action == MiningFinalApproachAction.COARSE:
                                    intercept_motion = route_motion.move_towards(
                                        mining_coord,
                                        intercept_target,
                                        now=now,
                                    )
                                    diagnostics = navigator.last_diagnostics
                                    action = _route_metadata_action(
                                        (
                                            "mining_live_marker_coarse"
                                            if live_marker_final_approach
                                            else "mining_database_anchor_coarse"
                                        ),
                                        diagnostics.action,
                                    )
                                elif final_decision.action == MiningFinalApproachAction.ALIGN:
                                    navigator.stop()
                                    face_snapshot = route_motion.face_towards(
                                        mining_coord,
                                        intercept_target,
                                        now=now,
                                        coordinate_scale_x=(
                                            mining_final_approach.map_width_yards
                                        ),
                                        coordinate_scale_y=(
                                            mining_final_approach.map_height_yards
                                        ),
                                    )
                                    diagnostics = navigator.last_diagnostics
                                    if face_snapshot.aligned:
                                        burst_decision = mining_final_approach.commit_burst(
                                            mining_coord,
                                            intercept_target,
                                            now=now,
                                            mounted=mounted,
                                        )
                                        input_controller.tap_key(
                                            "W",
                                            duration=float(
                                                burst_decision.burst_seconds or 0.0
                                            ),
                                        )
                                        action = (
                                            "mining_live_marker_single_burst"
                                            if live_marker_final_approach
                                            else "mining_database_anchor_single_burst"
                                        )
                                    elif (
                                        face_snapshot.command is not None
                                        and face_snapshot.command.turn_key is not None
                                    ):
                                        mining_cycle.note_intercept_heading_alignment(now=now)
                                        action = (
                                            (
                                                "mining_live_marker_align_"
                                                if live_marker_final_approach
                                                else "mining_database_anchor_align_"
                                            )
                                            + f"{face_snapshot.command.turn_key.lower()}"
                                        )
                                    else:
                                        action = (
                                            "mining_live_marker_wait_heading"
                                            if live_marker_final_approach
                                            else "mining_database_anchor_wait_heading"
                                        )
                                elif final_decision.action == MiningFinalApproachAction.ARRIVED:
                                    navigator.stop()
                                    if live_marker_final_approach:
                                        completed = (
                                            mining_cycle.mark_intercept_target_reached(
                                                now=now
                                            )
                                        )
                                        if not completed:
                                            mining_outcome = mining_cycle.fail(
                                                "ore_marker_not_centered_after_final_burst",
                                                now=now,
                                            )
                                    else:
                                        completed = (
                                            mining_cycle.mark_database_anchor_arrived(
                                                now=now
                                            )
                                        )
                                    if completed:
                                        mining_final_approach.reset()
                                    action = (
                                        "mining_anchor_world_validation_start"
                                        if mining_cycle.anchor_local_world_validation_active
                                        else
                                        "mining_live_marker_arrived"
                                        if completed and live_marker_final_approach
                                        else "mining_database_anchor_arrived"
                                        if completed
                                        else "mining_live_marker_final_burst_missed"
                                        if live_marker_final_approach
                                        else "mining_database_anchor_wait_marker_center"
                                    )
                                elif final_decision.action == MiningFinalApproachAction.FAILED:
                                    navigator.stop()
                                    mining_outcome = mining_cycle.observe_final_approach_failure(
                                        str(final_decision.reason or "ore_final_database_approach_failed"),
                                        now=now,
                                        current_coord=mining_coord,
                                    )
                                    mining_final_approach.reset()
                                    action = (
                                        f"mining_failed:{mining_outcome.reason}"
                                        if mining_outcome is not None
                                        else "mining_final_approach_retry_access_option"
                                    )
                                else:
                                    navigator.stop()
                                    action = (
                                        (
                                            "mining_live_marker_stop_recheck"
                                            if live_marker_final_approach
                                            else "mining_database_anchor_stop_recheck"
                                        )
                                        if final_decision.action
                                        == MiningFinalApproachAction.STOP_RECHECK
                                        else (
                                            "mining_live_marker_wait_feedback"
                                            if live_marker_final_approach
                                            else "mining_database_anchor_wait_feedback"
                                        )
                                    )
                            elif (
                                mining_distance <= intercept_reached_distance
                                and not must_finish_on_centered_marker
                            ):
                                navigator.stop()
                                mining_final_approach.reset()
                                completed = mining_cycle.mark_intercept_target_reached(now=now)
                                if completed:
                                    action = "mining_face_node_pending"
                                else:
                                    action = "mining_access_waypoint_reached"
                            else:
                                mining_final_approach.reset()
                                intercept_motion = route_motion.move_towards(
                                    mining_coord,
                                    intercept_target,
                                    now=now,
                                )
                                if (
                                    intercept_motion.command is not None
                                    and intercept_motion.command.turn_key is not None
                                    and not intercept_motion.command.hold_forward
                                ):
                                    mining_cycle.note_intercept_heading_alignment(now=now)
                                diagnostics = navigator.last_diagnostics
                                action = _route_metadata_action(
                                    (
                                        "mining_center_minimap_marker"
                                        if must_finish_on_centered_marker
                                        else "mining_access_intercept"
                                        if candidate.access_plan is not None
                                        else "mining_intercept"
                                    ),
                                    diagnostics.action,
                                )
                elif mining_cycle.phase == MiningPhase.CENTER_TOOLTIP:
                    navigator.stop()
                    mining_outcome = mining_cycle.observe_center_tooltip(
                        now=now,
                        tooltip_ore_type=(
                            minimap_ore_tooltip.ore_type
                            if minimap_ore_tooltip is not None
                            else None
                        ),
                    )
                    if mining_outcome is not None:
                        action = f"mining_failed:{mining_outcome.reason}"
                    elif mining_cycle.phase == MiningPhase.WORLD_SCAN:
                        route_motion.suspend()
                        navigator.stop()
                        start_mining_world_scan(frame.shape, now=now)
                        action = "mining_world_scan_start_after_center_tooltip"
                    elif mining_cycle.phase == MiningPhase.FACE_NODE:
                        release_point_cfg = config.get("mining", {}).get(
                            "minimap_tooltip_probe", {}
                        ).get(
                            "release_cursor_point",
                            config.get("mining", {}).get(
                                "cursor_baseline_point", {"x": 1280, "y": 170}
                            ),
                        )
                        release_point = _scaled_client_point(
                            release_point_cfg,
                            frame.shape,
                            config,
                        )
                        screen_x, screen_y = capture.client_to_screen_point(
                            *release_point
                        )
                        mouse_controller.move_to(screen_x, screen_y)
                        action = "mining_center_tooltip_confirmed"
                    else:
                        center_point = mining_cycle.center_tooltip_marker_point()
                        if center_point is None:
                            mining_outcome = mining_cycle.fail(
                                "minimap_shape_missing_at_center",
                                now=now,
                            )
                            action = f"mining_failed:{mining_outcome.reason}"
                        else:
                            client_point = minimap_marker_client_point(
                                center_point,
                                frame.shape,
                                config,
                            )
                            screen_x, screen_y = capture.client_to_screen_point(
                                *client_point
                            )
                            mouse_controller.move_to(screen_x, screen_y)
                            action = "mining_center_tooltip_probe"
                elif mining_cycle.phase == MiningPhase.FACE_NODE:
                    navigator.stop()
                    candidate = mining_cycle.candidate
                    face_target_coord = mining_cycle.face_target_coord
                    mining_outcome = mining_cycle.observe_face_node_timeout(now=now)
                    if mining_cycle.phase == MiningPhase.WORLD_SCAN:
                        route_motion.suspend()
                        navigator.stop()
                        start_mining_world_scan(frame.shape, now=now)
                        action = "mining_world_scan_start_after_face_timeout"
                    elif mining_outcome is not None:
                        action = f"mining_failed:{mining_outcome.reason}"
                    elif candidate is None or mining_coord is None or face_target_coord is None:
                        mining_outcome = mining_cycle.fail(
                            "candidate_or_coordinate_missing",
                            now=now,
                        )
                        action = f"mining_failed:{mining_outcome.reason}"
                    else:
                        face_snapshot = route_motion.face_towards(
                            mining_coord,
                            face_target_coord,
                            now=now,
                        )
                        diagnostics = navigator.last_diagnostics
                        if face_snapshot.aligned:
                            mining_cycle.mark_node_faced(now=now)
                            route_motion.suspend()
                            navigator.stop()
                            if mining_interactor.current_cursor_is_mining_ready(frame):
                                mining_interactor.right_click()
                                mining_cycle.mark_clicked((0, 0), now=now)
                                action = "mining_right_click_existing_hover"
                            else:
                                start_mining_world_scan(frame.shape, now=now)
                                action = "mining_world_scan_start"
                        elif face_snapshot.command is not None and face_snapshot.command.turn_key:
                            action = (
                                "mining_face_node_"
                                f"{face_snapshot.command.turn_key.lower()}"
                            )
                        else:
                            action = "mining_face_node_wait_heading"
                elif mining_cycle.phase == MiningPhase.WORLD_SCAN:
                    route_motion.suspend()
                    navigator.stop()
                    anchor_local_validation_was_active = bool(
                        mining_cycle.anchor_local_world_validation_active
                    )
                    mining_outcome = mining_cycle.observe_world_scan_timeout(
                        now=now,
                        current_coord=mining_coord,
                    )
                    if mining_cycle.phase == MiningPhase.INTERCEPT:
                        mining_interactor.cancel_hover_scan()
                        action = (
                            "mining_anchor_world_validation_resume_residual"
                            if anchor_local_validation_was_active
                            else "mining_retry_alternate_access"
                        )
                    elif mining_outcome is not None:
                        mining_interactor.cancel_hover_scan()
                        action = f"mining_failed:{mining_outcome.reason}"
                    else:
                        if mining_interactor.hover_scanner.phase == HoverScanPhase.IDLE:
                            hover_step = start_mining_world_scan(frame.shape, now=now)
                        else:
                            hover_step = mining_interactor.poll_hover_scan(frame, now=now)
                        if hover_step.phase == HoverScanPhase.FOUND and hover_step.point is not None:
                            alignment = mining_target_alignment(
                                hover_step.point,
                                frame.shape,
                                config,
                            )
                            if (
                                bool(mining_cfg.get("world_target_face_enabled", True))
                                and not alignment.aligned
                                and alignment.turn_key is not None
                            ):
                                turn_key = _mining_world_target_turn_key(
                                    alignment.turn_key,
                                    mining_config=mining_cfg,
                                    route_invert_turn_direction=invert_turn_direction,
                                )
                                navigator.turn_character(
                                    turn_key,
                                    duration=alignment.turn_seconds,
                                    context="mining",
                                )
                                mining_interactor.cancel_hover_scan()
                                action = f"mining_face_world_target_{turn_key.lower()}"
                            else:
                                mining_interactor.right_click()
                                mining_cycle.mark_clicked(hover_step.point, now=now)
                                action = "mining_right_click_once"
                        elif hover_step.phase == HoverScanPhase.EXHAUSTED:
                            if mining_cycle.resume_residual_after_anchor_world_validation(
                                now=now,
                                current_coord=mining_coord,
                            ):
                                mining_interactor.cancel_hover_scan()
                                action = "mining_anchor_world_validation_resume_residual"
                            else:
                                mining_outcome = mining_cycle.fail(
                                    "mining_hover_not_found",
                                    now=now,
                                )
                                action = f"mining_failed:{mining_outcome.reason}"
                        else:
                            action = f"mining_{hover_step.reason}"
                elif mining_cycle.phase in {MiningPhase.GATHER_WAIT, MiningPhase.VERIFY}:
                    navigator.stop()
                    mining_outcome = mining_cycle.observe_after_click(
                        target_visible=ore_presence.target_visible,
                        hover_mining_ready=mining_interactor.hover_target_is_mining_ready(
                            frame
                        ),
                        now=now,
                    )
                    if mining_outcome is None:
                        action = (
                            "mining_gather_wait"
                            if mining_cycle.phase == MiningPhase.GATHER_WAIT
                            else "mining_verify_wait"
                        )
                    else:
                        action = (
                            "mining_verified"
                            if mining_outcome.success
                            else f"mining_failed:{mining_outcome.reason}"
                        )
            elif control_owner == OperationalControlOwner.MOUNT:
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                action = f"route_{mount_action}"
                if mount_action == "mount_cast":
                    input_controller.tap_key(mount_state.key, duration=0.06)
                    mount_casts += 1
            elif current_coord is not None:
                if resynced:
                    navigator.stop()
                distance = MovementNavigator.distance(current_coord, target)
                if previous_distance_to_target is not None:
                    distance_progress = previous_distance_to_target - distance
                final_distance = distance
                if route_entry_pending and distance <= route_entry_reached_distance:
                    target_recovery_count = 0
                    next_target_node = _consume_reached_route_entry(
                        route_entry_queue,
                        current_coord,
                        reached_distance=route_entry_reached_distance,
                    )
                    route_entry_pending = next_target_node is not None
                    if not route_entry_pending and generated_entry is not None:
                        _record_route_node_milestone(
                            route_node_history,
                            completed_route_node_indices,
                            route_milestones,
                            route_index=int(generated_entry.target_route_sort_key),
                            reached_coord=current_coord,
                            distance=distance,
                        )
                    if next_target_node is None:
                        next_target_node = _choose_route_target(
                            config,
                            current_coord,
                            target_coord=None,
                            zone_id=zone_id,
                            auto_zone=auto_zone,
                            min_target_distance=min_target_distance,
                            excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
                            auto_zone_ids=set(route_zone_ids),
                            route_after_sort_key=route_entry_sort_key,
                        )
                    if next_target_node is None:
                        navigator.stop()
                        action = "route_entry_no_target"
                        reached = False
                        termination_reason = "no_target_after_route_entry"
                    else:
                        target_node = next_target_node
                        target = target_node.coord
                        steering_target, route_target_passed, route_following_observation = (
                            _directed_route_steering_target(
                                route_follower,
                                current_coord=current_coord,
                                target_node=target_node,
                                route_entry_pending=route_entry_pending,
                            )
                        )
                        distance = MovementNavigator.distance(current_coord, target)
                        final_distance = distance
                        threat_commitment.choose_turn_key(None, now=now, fallback_turn_key=alternate_turn_key)
                        route_motion.move_towards(
                            current_coord,
                            steering_target,
                            now=now,
                        )
                        diagnostics = navigator.last_diagnostics
                        action = _route_metadata_action("route_entry_reached", diagnostics.action)
                elif not route_entry_pending and (
                    route_target_passed
                    or distance
                    <= _route_target_reached_distance(
                        config,
                        reached_distance,
                        target_node,
                    )
                ):
                    ore_training_capture = None
                    if not is_route_waypoint(target_node):
                        ore_training_capture = _capture_reached_ore_training_sample(
                            capture,
                            frame,
                            config,
                            target_node=target_node,
                            target_coord=target,
                            reached_coord=current_coord,
                            distance=distance,
                            index=index,
                            route_output_dir=output_path,
                        )
                    target_history.append(
                        _target_history_item(
                            target_node,
                            target,
                            current_coord,
                            distance,
                            ore_training_capture=ore_training_capture,
                        )
                    )
                    _record_target_route_node_milestone(
                        route_node_history,
                        completed_route_node_indices,
                        route_milestones,
                        target_node=target_node,
                        reached_coord=current_coord,
                        distance=distance,
                    )
                    if not is_route_waypoint(target_node):
                        completed_target_coords.add(target)
                    target_recovery_count = 0
                    if len(target_history) >= target_limit:
                        navigator.stop()
                        action = "reached"
                        reached = True
                        termination_reason = "target_limit_reached"
                    else:
                        next_target_node = _choose_route_target(
                            config,
                            current_coord,
                            target_coord=None,
                            zone_id=zone_id,
                            auto_zone=auto_zone,
                            min_target_distance=min_target_distance,
                            excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
                            auto_zone_ids=set(route_zone_ids),
                            route_after_sort_key=_next_route_sort_key(target_node),
                        )
                        if next_target_node is None:
                            navigator.stop()
                            action = "cycle_no_target"
                            reached = False
                            termination_reason = "no_target"
                        else:
                            target_node = next_target_node
                            target = target_node.coord
                            steering_target, route_target_passed, route_following_observation = (
                                _directed_route_steering_target(
                                    route_follower,
                                    current_coord=current_coord,
                                    target_node=target_node,
                                    route_entry_pending=route_entry_pending,
                                )
                            )
                            distance = MovementNavigator.distance(current_coord, target)
                            final_distance = distance
                            target_recovery_count = 0
                            threat_commitment.choose_turn_key(None, now=now, fallback_turn_key=alternate_turn_key)
                            route_motion.move_towards(
                                current_coord,
                                steering_target,
                                now=now,
                            )
                            diagnostics = navigator.last_diagnostics
                            action = _route_metadata_action("cycle_next_target", diagnostics.action)
                else:
                    if threat is not None and threat_turn_key is not None:
                        action, local_turn_key = _apply_hostile_avoidance(
                            navigator,
                            input_controller,
                            current_coord,
                            steering_target,
                            threat,
                            threat_turn_key,
                            hostile_turn_duration,
                        )
                        alternate_turn_key = "A" if local_turn_key == "D" else "D"
                        diagnostics = navigator.last_diagnostics
                    else:
                        local_avoid_requested = _should_local_avoid(
                            navigation.center_blocked,
                            proactive=bool(
                                config.get("local_navigation", {}).get(
                                    "proactive_avoid_enabled", False
                                )
                            ),
                            distance_progress=distance_progress,
                            coord_delta=coord_delta,
                            visual_motion_delta=visual_motion_delta,
                            stuck_min_progress=stuck_min_progress,
                            stuck_min_coord_delta=stuck_min_coord_delta,
                            visual_stuck_min_delta=visual_stuck_min_delta,
                        )
                        local_turn_key = local_avoidance.next_turn_key(
                            blocked=local_avoid_requested,
                            now=now,
                            preferred_turn_key=navigation.recommended_turn,
                            fallback_turn_key=alternate_turn_key,
                            distance_progress=distance_progress,
                            progress_threshold=stuck_min_progress,
                        )
                    if threat is None and local_turn_key is not None:
                        local_side_switched = local_avoidance.consume_side_switch()
                        jump_only = local_avoidance.consume_jump_only()
                        if local_side_switched:
                            navigator.stop()
                            input_controller.tap_key("S", duration=local_side_switch_backtrack_seconds)
                        input_controller.tap_key("SPACE", duration=local_jump_duration)
                        if not jump_only:
                            navigator.turn_character(
                                local_turn_key,
                                duration=local_turn_duration,
                                context="local",
                            )
                            alternate_turn_key = "A" if alternate_turn_key == "D" else "D"
                        local_avoid_pulsed = True
                        action = (
                            "local_avoid_backtrack_switch"
                            if local_side_switched
                            else (
                                "local_avoid_jump"
                                if jump_only
                                else "local_avoid_jump_turn"
                            )
                        )
                    elif threat is None:
                        action = "vector_forward"

                    if threat is None:
                        local_bypass_forward = local_avoidance.should_continue_forward(now)
                        if local_avoid_pulsed or local_bypass_forward:
                            navigator.observe_position(current_coord)
                            navigator.continue_forward(current_coord, steering_target)
                            if not local_avoid_pulsed:
                                action = "local_avoid_bypass_forward"
                        else:
                            route_motion.move_towards(
                                current_coord,
                                steering_target,
                                now=now,
                            )
                        diagnostics = navigator.last_diagnostics
                        action = _route_metadata_action(action, diagnostics.action)
                    if _is_recovery_action(diagnostics.action):
                        stuck_events += 1
                        visual_stuck_samples = 0
            elif reached:
                navigator.stop()
                action = "reached"
            elif coord_feedback_paused or (
                navigator.held_key == "W"
                and coord_feedback_age >= coord_feedback_timeout
            ):
                coord_feedback_paused = True
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                local_avoidance.clear()
                action = "coord_feedback_wait"
            else:
                if threat is not None and threat_turn_key is not None:
                    action, local_turn_key = _apply_hostile_avoidance(
                        navigator,
                        input_controller,
                        final_coord,
                        steering_target,
                        threat,
                        threat_turn_key,
                        hostile_turn_duration,
                    )
                    alternate_turn_key = "A" if local_turn_key == "D" else "D"
                    diagnostics = navigator.last_diagnostics
                else:
                    local_avoid_requested = _should_local_avoid(
                        navigation.center_blocked,
                        proactive=bool(
                            config.get("local_navigation", {}).get(
                                "proactive_avoid_enabled", False
                            )
                        ),
                        distance_progress=None,
                        coord_delta=coord_delta,
                        visual_motion_delta=visual_motion_delta,
                        stuck_min_progress=stuck_min_progress,
                        stuck_min_coord_delta=stuck_min_coord_delta,
                        visual_stuck_min_delta=visual_stuck_min_delta,
                    )
                    local_turn_key = local_avoidance.next_turn_key(
                        blocked=local_avoid_requested,
                        now=now,
                        preferred_turn_key=navigation.recommended_turn,
                        fallback_turn_key=alternate_turn_key,
                        distance_progress=None,
                        progress_threshold=stuck_min_progress,
                    )
                if threat is None and local_turn_key is not None:
                    local_side_switched = local_avoidance.consume_side_switch()
                    jump_only = local_avoidance.consume_jump_only()
                    if local_side_switched:
                        navigator.stop()
                        input_controller.tap_key("S", duration=local_side_switch_backtrack_seconds)
                    input_controller.tap_key("SPACE", duration=local_jump_duration)
                    if not jump_only:
                        navigator.turn_character(
                            local_turn_key,
                            duration=local_turn_duration,
                            context="local",
                        )
                        alternate_turn_key = "A" if alternate_turn_key == "D" else "D"
                    local_avoid_pulsed = True
                    action = (
                        "local_avoid_backtrack_switch"
                        if local_side_switched
                        else (
                            "local_avoid_jump"
                            if jump_only
                            else "local_avoid_jump_turn"
                        )
                    )
                if threat is None:
                    local_bypass_forward = local_avoidance.should_continue_forward(now)
                    if local_avoid_pulsed or local_bypass_forward:
                        navigator.continue_forward(final_coord, steering_target)
                        if local_bypass_forward and not local_avoid_pulsed:
                            action = "local_avoid_bypass_forward"
                    else:
                        route_motion.move_towards(
                            final_coord,
                            steering_target,
                            now=now,
                        )
                    diagnostics = navigator.last_diagnostics
                    action = _route_metadata_action(action, diagnostics.action)

            if (
                not reached
                and not (mining_cycle.active or minimap_tooltip_probe.active)
                and navigator.held_key == "W"
                and not _is_threat_action(action)
            ):
                if navigator.recovery_suppresses_stuck:
                    visual_stuck_samples = 0
                    movement_stuck = False
                else:
                    movement_stuck = _movement_was_stuck(
                        coord_delta=coord_delta,
                        visual_motion_delta=visual_motion_delta,
                        stuck_min_delta=stuck_min_coord_delta,
                        visual_stuck_min_delta=visual_stuck_min_delta,
                    )
                visual_stuck_samples = _advance_route_stuck_samples(
                    visual_stuck_samples,
                    coord_delta=coord_delta,
                    movement_stuck=movement_stuck,
                )

                if visual_stuck_samples >= max(1, stuck_window):
                    local_turn_key = navigation.recommended_turn or alternate_turn_key
                    alternate_turn_key = "A" if alternate_turn_key == "D" else "D"
                    recovery = navigator.recover_from_obstacle(final_coord, preferred_turn_key=local_turn_key)
                    diagnostics = navigator.last_diagnostics
                    action = recovery.action
                    stuck_events += 1
                    visual_stuck_samples = 0

            progress_timeout_stalled = target_progress_watchdog.observe(
                target_coord=target,
                distance=distance,
                now=now,
                active=bool(
                    not reached
                    and not (mining_cycle.active or minimap_tooltip_probe.active)
                    and not combat_blocks_route
                    and not game_state.death_or_blocking_modal
                    and not coord_feedback_paused
                    and navigator.held_key == "W"
                    and not action.startswith(("combat_", "route_mount"))
                ),
            )
            target_stalled = bool(progress_timeout_stalled or navmesh_guidance_blocked)
            target_blocked = False
            target_blocked_reason = "target_recovery_limit"
            if (
                not reached
                and not (mining_cycle.active or minimap_tooltip_probe.active)
                and _target_recovery_can_block(
                    directed_route_active=route_follower is not None,
                    target_source=target_node.source,
                )
            ):
                target_recovery_count, target_blocked = _record_target_recovery(
                    action,
                    target_recovery_count=target_recovery_count,
                    target_recovery_limit=target_recovery_limit,
                    distance_progress=distance_progress,
                    coord_delta=coord_delta,
                    stuck_min_progress=stuck_min_progress,
                    stuck_min_coord_delta=stuck_min_coord_delta,
                )
            elif route_follower is not None and target_node.source == ROUTE_WAYPOINT_SOURCE:
                target_recovery_count = 0
            if navmesh_guidance_blocked:
                target_blocked = True
                target_blocked_reason = "navmesh_route_segment_blocked"
            elif target_stalled:
                target_blocked = True
                target_blocked_reason = "target_progress_timeout"

            if target_blocked:
                hard_stuck_action = None
                hard_stuck_turn_key = (
                    navigator.last_diagnostics.turn_key
                    or local_turn_key
                    or alternate_turn_key
                )
                if progress_timeout_stalled:
                    hard_stuck_action = perform_hard_stuck_escape(
                        navigator=navigator,
                        input_controller=input_controller,
                        mount_state=mount_state,
                        mounted=mounted,
                        turn_key=hard_stuck_turn_key,
                        settings=hard_stuck_escape_settings,
                        now=now,
                    )
                    if hard_stuck_action is not None:
                        hard_stuck_escape_history.append(
                            {
                                "index": index,
                                "coord": final_coord,
                                "target_coord": target,
                                "target_route_index": target_node.route_index,
                                "mounted": mounted,
                                "turn_key": hard_stuck_turn_key,
                                "reason": target_blocked_reason,
                            }
                        )
                skipped_target_history.append(
                    _target_skipped_item(
                        target_node,
                        target,
                        final_coord,
                        distance,
                        recovery_count=target_recovery_count,
                        reason=target_blocked_reason,
                    )
                )
                _block_route_waypoint_window(
                    blocked_target_coords,
                    target_node=target_node,
                    route_loop=route_loop,
                    forward_padding=max(
                        hard_stuck_escape_settings.forward_waypoint_padding
                        if hard_stuck_action is not None
                        else 0,
                        int(config.get("route", {}).get("blocked_waypoint_forward_padding", 2)),
                    ),
                )
                navigator.stop()
                diagnostics = navigator.last_diagnostics
                visual_stuck_samples = 0
                threat_commitment.choose_turn_key(None, now=now, fallback_turn_key=alternate_turn_key)
                if route_entry_pending and route_loop:
                    next_target_node = _autonomous_route_entry_target(
                        config,
                        final_coord,
                        route_loop,
                        excluded_coords=blocked_target_coords,
                        zone_id=zone_id,
                    )
                    route_entry_queue = [next_target_node] if next_target_node is not None else []
                    route_entry_sort_key = _route_sort_key(next_target_node) if next_target_node is not None else None
                else:
                    next_target_node = None if route_entry_pending else _choose_route_target(
                        config,
                        final_coord,
                        target_coord=None,
                        zone_id=zone_id,
                        auto_zone=auto_zone,
                        min_target_distance=min_target_distance,
                        excluded_coords=_combined_excluded_coords(completed_target_coords, blocked_target_coords),
                        auto_zone_ids=set(route_zone_ids),
                        route_after_sort_key=_next_route_sort_key(target_node),
                    )
                target_recovery_count = 0
                if next_target_node is None:
                    action = "route_entry_blocked" if route_entry_pending else "target_blocked_no_target"
                    termination_reason = action
                else:
                    target_node = next_target_node
                    target = target_node.coord
                    steering_target, route_target_passed, route_following_observation = (
                        _directed_route_steering_target(
                            route_follower,
                            current_coord=final_coord,
                            target_node=target_node,
                            route_entry_pending=route_entry_pending,
                        )
                    )
                    distance = MovementNavigator.distance(final_coord, target)
                    final_distance = distance
                    route_motion.move_towards(
                        final_coord,
                        steering_target,
                        now=now,
                    )
                    diagnostics = navigator.last_diagnostics
                    action = _route_metadata_action(
                        (
                            "hard_stuck_escape_cycle_next"
                            if hard_stuck_action is not None
                            else "target_blocked_cycle_next"
                        ),
                        diagnostics.action,
                    )

            save_frame = _should_save_route_frame(
                now=now,
                last_saved_at=last_frame_saved_at,
                interval=frame_save_interval,
                action=action,
                previous_action=last_frame_action,
                force=navigation.center_blocked,
            )
            frame_path: Path | None = None
            if save_frame:
                frame_candidate = frame_dir / f"{index:04d}.png"
                if image_writer.submit(frame_candidate, frame):
                    frame_path = frame_candidate
                    last_frame_saved_at = now
                    last_frame_action = action
            report_path: Path | None = None
            if frame_path is not None and report_interval > 0 and (
                index % report_interval == 0 or navigation.center_blocked or action != "vector_forward"
            ):
                report_candidate = report_dir / f"{index:04d}.png"
                if image_writer.submit(
                    report_candidate,
                    draw_local_navigation_report(frame, navigation),
                ):
                    report_path = report_candidate

            _append_jsonl(
                metadata_path,
                {
                    "index": index,
                    "timestamp": time.time(),
                    "foreground_ok": foreground_ok,
                    "frame": _relative_posix(frame_path, output_path) if frame_path else None,
                    "report": _relative_posix(report_path, output_path) if report_path else None,
                    "coord": current_coord if coord_fresh else final_coord,
                    "coord_candidates": coord_candidates,
                    "coord_resynced": resynced,
                    "coord_fresh": coord_fresh,
                    "coord_delta": coord_delta,
                    "coord_jump_limit": coord_jump_limit,
                    "coord_feedback_age": coord_feedback_age,
                    "coord_feedback_paused": coord_feedback_paused,
                    "distance_progress": distance_progress,
                    "target_coord": target,
                    "steering_target_coord": steering_target,
                    "route_following": (
                        route_following_observation.to_dict()
                        if route_following_observation is not None
                        else None
                    ),
                    "navmesh_guidance": (
                        navmesh_guidance_observation.to_dict()
                        if navmesh_guidance_observation is not None
                        else None
                    ),
                    "target_node_id": target_node.node_id,
                    "target_zone_id": target_node.zone_id,
                    "detected_zone_name": detected_zone_name,
                    "route_zone_ids": route_zone_ids,
                    "route_entry_pending": route_entry_pending,
                    "route_entry_remaining_waypoints": len(route_entry_queue),
                    "hazard_egress_active": hazard_egress_active,
                    "route_hazard_raw_candidate": route_hazard_raw_candidate,
                    "forbidden_subzone_visible": forbidden_subzone_visible,
                    "completed_targets": len(target_history),
                    "skipped_targets": len(skipped_target_history),
                    "skipped_target_history": skipped_target_history,
                    "blocked_target_coords": sorted(blocked_target_coords),
                    "target_recovery_count": target_recovery_count,
                    "target_recovery_limit": target_recovery_limit,
                    "distance": distance,
                    "action": action,
                    "local_turn_key": local_turn_key,
                    "threat": threat.to_dict() if threat is not None else None,
                    "combat": combat.to_dict(),
                    "combat_threat": combat_threat_assessment.to_dict(),
                    "combat_heal": combat_heal.to_dict(now),
                    "combat_keys_tapped": list(combat_fallback.last_tapped_keys),
                    "combat_stale_soft_target_cleared": stale_soft_target_cleared,
                    "combat_clear_had_confirmed_outgoing": combat_clear_had_confirmed_outgoing,
                    "combat_heal_keys_tapped": combat_heal_keys_tapped,
                    "armor_critical": armor_critical,
                    "combat_evidence_reliability": (
                        "degraded_armor_critical" if armor_critical else "normal"
                    ),
                    "mounted_combat_bypass": mounted_combat_bypass,
                    "mounted_escape": mounted_escape_decision.to_dict(),
                    "hard_stuck_escapes": len(hard_stuck_escape_history),
                    "hard_stuck_escape_history": list(hard_stuck_escape_history),
                    "safe_drain_active": safe_drain_active,
                    "mounted": mounted,
                    "mount_action": mount_action,
                    "mount_casts": mount_casts,
                    "mining": mining_cycle.snapshot(now=now),
                    "mining_final_approach": mining_final_approach.snapshot(now=now),
                    "minimap_tooltip_probe": minimap_tooltip_probe.snapshot(now=now),
                    "minimap_ore_tooltip": (
                        {
                            "ore_id": minimap_ore_tooltip.ore_id,
                            "ore_type": minimap_ore_tooltip.ore_type,
                        }
                        if minimap_ore_tooltip is not None
                        else None
                    ),
                    "mining_hover_scan": mining_interactor.hover_scanner.snapshot(),
                    "mining_candidate_missing": mining_candidate_missing,
                    "mining_outcomes": len(mining_outcomes),
                    "post_combat_loot": {
                        "enabled": post_combat_loot.enabled,
                        "active": post_combat_loot.active,
                        "action": post_combat_loot_decision.action,
                        "input_key": post_combat_loot_decision.input_key,
                        "keys_tapped": post_combat_loot_keys_tapped,
                        "in_combat": post_combat_loot.armed_in_combat,
                        "pending_marker": combat_loot_pending_visible,
                        "outcomes": len(post_combat_loot_outcomes),
                    },
                    "death_recovery": None,
                    "visual_motion_delta": visual_motion_delta,
                    "visual_stuck_samples": visual_stuck_samples,
                    "game_state": game_state.__dict__,
                    "movement": diagnostics.__dict__,
                    "route_motion": route_motion.last_snapshot.to_dict(),
                    "mouse_steering": (
                        mouse_steering.last_result.to_dict()
                        if mouse_steering.last_result is not None
                        else None
                    ),
                    "navigation": navigation.to_dict(),
                },
            )
            if reached or action in {
                "target_blocked_no_target",
                "route_entry_no_target",
                "route_entry_blocked",
            }:
                break
            time.sleep(interval)
        else:
            if safe_drain_started:
                termination_reason = "safe_drain_limit"
    finally:
        navigator.stop()
        video_recorder.stop()
        image_writer_stats = image_writer.close()

    final_frame = capture.capture_client_region()
    filtered_final_coord, _, _ = _read_filtered_coord(
        final_frame,
        config,
        previous_coord=final_coord,
        reference_coords=reference_coords,
        resync_state=coord_resync_state,
    )
    final_coord = filtered_final_coord or final_coord
    if final_coord is not None:
        final_distance = MovementNavigator.distance(final_coord, target)
    cv2.imwrite(str(output_path / "final_frame.png"), final_frame)
    cv2.imwrite(
        str(output_path / "final_report.png"),
        draw_local_navigation_report(final_frame, analyze_local_navigation(final_frame, config)),
    )
    final_mining_outcome = mining_cycle.consume_outcome()
    if final_mining_outcome is not None:
        mining_outcomes.append(
            _mining_outcome_item(final_mining_outcome, index=executed_steps)
        )
    if post_combat_loot.active:
        post_combat_loot.cancel(now=time.monotonic(), reason="probe_ended")
    final_loot_outcome = post_combat_loot.consume_outcome()
    if final_loot_outcome is not None:
        post_combat_loot_outcomes.append(
            {
                "index": executed_steps,
                "attempted": final_loot_outcome.attempted,
                "reason": final_loot_outcome.reason,
                "elapsed_seconds": final_loot_outcome.elapsed_seconds,
                "interactions": final_loot_outcome.interactions,
                "confirmed_loots": final_loot_outcome.confirmed_loots,
            }
        )

    summary = RouteLiveSummary(
        output_dir=output_path,
        target_coord=target,
        target_node_id=target_node.node_id,
        target_zone_id=target_node.zone_id,
        target_ore_type=target_node.ore_type,
        start_coord=start_coord,
        final_coord=final_coord,
        start_distance=start_distance,
        final_distance=final_distance,
        steps=executed_steps,
        reached=bool(len(target_history) >= target_limit),
        stuck_events=stuck_events,
        completed_targets=len(target_history),
        target_history=target_history,
        completed_route_nodes=len(route_node_history),
        route_node_history=route_node_history,
        skipped_targets=len(skipped_target_history),
        skipped_target_history=skipped_target_history,
        blocked_target_coords=sorted(blocked_target_coords),
        detected_zone_name=detected_zone_name,
        route_zone_ids=route_zone_ids,
        armor_critical_observed=armor_critical_observed,
        mounted_observed=mounted_observed,
        mount_casts=mount_casts,
        mounted_escape_episodes=mounted_escape.escape_episodes,
        mounted_escape_escalations=mounted_escape.escalations,
        mounted_escape_last_reason=mounted_escape.last_escalation_reason,
        hard_stuck_escapes=len(hard_stuck_escape_history),
        hard_stuck_escape_history=hard_stuck_escape_history,
        mining_attempts=len(mining_outcomes),
        mining_successes=sum(1 for item in mining_outcomes if item["success"]),
        mining_failures=sum(1 for item in mining_outcomes if not item["success"]),
        mining_outcomes=mining_outcomes,
        post_combat_loot_attempts=sum(
            1 for item in post_combat_loot_outcomes if item["attempted"]
        ),
        post_combat_loot_outcomes=post_combat_loot_outcomes,
        image_writes_submitted=image_writer_stats.submitted,
        image_writes_written=image_writer_stats.written,
        image_writes_dropped=image_writer_stats.dropped,
        image_write_failures=image_writer_stats.failed,
        termination_reason=termination_reason,
    )
    (output_path / "summary.json").write_text(json.dumps(_summary_to_dict(summary), indent=2), encoding="utf-8")
    return summary
