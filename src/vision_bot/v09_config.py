from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vision_bot.core.geometry import MapPoint, ZoneGeometry


class V09ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FreshnessConfig:
    critical_seconds: float
    operational_seconds: float
    heavy_sensor_seconds: float


@dataclass(frozen=True)
class PerformanceConfig:
    control_hz: float
    critical_preemption_p95_ms: float
    controller_p95_ms: float
    max_frame_backlog: int
    idle_ore_scan_interval_frames: int
    shutdown_guard_quiet_seconds: float
    shutdown_guard_max_seconds: float


@dataclass(frozen=True)
class NavigationConfig:
    lookahead_min_yards: float
    lookahead_cruise_yards: float
    lookahead_max_yards: float
    corridor_radius_yards: float
    turn_engage_degrees: float
    turn_release_degrees: float
    pivot_degrees: float
    yaw_pixels_per_degree: float
    yaw_pixels_per_second: float
    moving_drag_max_seconds: float
    pivot_drag_max_seconds: float
    turn_reversal_guard_seconds: float
    stall_soft_seconds: float
    stall_hard_seconds: float
    stall_min_progress_yards: float
    mounted_escape_seconds: float
    mounted_escape_attacker_count: int


@dataclass(frozen=True)
class MiningApproachConfig:
    acquisition_radius_yards: float
    capture_radius_yards: float
    braking_radius_yards: float
    centered_radius_pixels: float
    minimap_sensor_radius_pixels: float
    minimap_radius_x_yards: float
    minimap_radius_y_yards: float
    live_marker_weight: float
    max_fusion_disagreement_yards: float
    target_max_age_seconds: float
    max_approach_seconds: float
    visual_search_seconds: float
    verify_seconds: float
    out_of_range_recovery_seconds: float
    max_out_of_range_recoveries: int
    identity_confirm_frames: int
    identity_timeout_seconds: float
    marker_track_max_jump_pixels: float
    database_match_radius_yards: float
    require_access_plan: bool
    access_attachment_radius_yards: float
    failure_suppression_seconds: float


@dataclass(frozen=True)
class CombatConfig:
    attack_key: str
    attack_interval_seconds: float
    key_tap_seconds: float
    priority_key: str
    priority_interval_seconds: float
    aligned_key: str
    aligned_interval_seconds: float
    emergency_key: str
    emergency_hp_fraction: float
    emergency_interval_seconds: float
    emergency_followup_key: str
    emergency_followup_gap_seconds: float
    heal_key: str
    heal_hp_fraction: float
    heal_interval_seconds: float
    face_search_cooldown_seconds: float
    face_search_drag_seconds: float
    bearing_lock_seconds: float


@dataclass(frozen=True)
class RecoveryConfig:
    target_key: str
    interact_key: str
    approach_key: str
    key_tap_seconds: float
    click_seconds: float
    target_interval_seconds: float
    interact_interval_seconds: float
    approach_seconds: float
    approach_settle_seconds: float
    max_approach_attempts: int
    healer_anchor_activation_radius_yards: float
    healer_interact_radius_yards: float
    healer_anchors: tuple[MapPoint, ...]
    release_retry_seconds: float
    release_timeout_seconds: float
    healer_search_timeout_seconds: float
    dialog_timeout_seconds: float
    max_release_attempts: int
    max_target_attempts: int
    max_interact_attempts: int


@dataclass(frozen=True)
class MountingConfig:
    enabled: bool
    key: str
    cast_seconds: float
    retry_seconds: float
    max_attempts_per_cycle: int
    cycle_cooldown_seconds: float
    post_combat_settle_seconds: float


@dataclass(frozen=True)
class V09Config:
    enabled: bool
    default_mode: str
    protocol_version: int
    freshness: FreshnessConfig
    performance: PerformanceConfig
    zone: ZoneGeometry
    navigation: NavigationConfig
    mining: MiningApproachConfig
    combat: CombatConfig
    recovery: RecoveryConfig
    mounting: MountingConfig

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> "V09Config":
        section = _mapping(root.get("v09"), "v09")
        _reject_unknown(
            section,
            {
                "enabled",
                "default_mode",
                "protocol_version",
                "freshness",
                "performance",
                "zone",
                "navigation",
                "mining",
                "combat",
                "recovery",
                "mounting",
            },
            "v09",
        )
        freshness = _mapping(section.get("freshness"), "v09.freshness")
        performance = _mapping(section.get("performance"), "v09.performance")
        zone = _mapping(section.get("zone"), "v09.zone")
        navigation = _mapping(section.get("navigation"), "v09.navigation")
        mining = _mapping(section.get("mining"), "v09.mining")
        combat = _mapping(section.get("combat"), "v09.combat")
        recovery = _mapping(section.get("recovery"), "v09.recovery")
        mounting = _mapping(section.get("mounting"), "v09.mounting")

        _reject_unknown(
            freshness,
            {"critical_seconds", "operational_seconds", "heavy_sensor_seconds"},
            "v09.freshness",
        )
        _reject_unknown(
            performance,
            {
                "control_hz",
                "critical_preemption_p95_ms",
                "controller_p95_ms",
                "max_frame_backlog",
                "idle_ore_scan_interval_frames",
                "shutdown_guard_quiet_seconds",
                "shutdown_guard_max_seconds",
            },
            "v09.performance",
        )
        _reject_unknown(zone, {"zone_id", "width_yards", "height_yards"}, "v09.zone")
        _reject_unknown(
            navigation,
            {
                "lookahead_min_yards",
                "lookahead_cruise_yards",
                "lookahead_max_yards",
                "corridor_radius_yards",
                "turn_engage_degrees",
                "turn_release_degrees",
                "pivot_degrees",
                "yaw_pixels_per_degree",
                "yaw_pixels_per_second",
                "moving_drag_max_seconds",
                "pivot_drag_max_seconds",
                "turn_reversal_guard_seconds",
                "stall_soft_seconds",
                "stall_hard_seconds",
                "stall_min_progress_yards",
                "mounted_escape_seconds",
                "mounted_escape_attacker_count",
            },
            "v09.navigation",
        )
        _reject_unknown(
            mining,
            {
                "acquisition_radius_yards",
                "capture_radius_yards",
                "braking_radius_yards",
                "centered_radius_pixels",
                "minimap_sensor_radius_pixels",
                "minimap_radius_x_yards",
                "minimap_radius_y_yards",
                "live_marker_weight",
                "max_fusion_disagreement_yards",
                "target_max_age_seconds",
                "max_approach_seconds",
                "visual_search_seconds",
                "verify_seconds",
                "out_of_range_recovery_seconds",
                "max_out_of_range_recoveries",
                "identity_confirm_frames",
                "identity_timeout_seconds",
                "marker_track_max_jump_pixels",
                "database_match_radius_yards",
                "require_access_plan",
                "access_attachment_radius_yards",
                "failure_suppression_seconds",
            },
            "v09.mining",
        )
        _reject_unknown(
            combat,
            {
                "attack_key",
                "attack_interval_seconds",
                "key_tap_seconds",
                "priority_key",
                "priority_interval_seconds",
                "aligned_key",
                "aligned_interval_seconds",
                "emergency_key",
                "emergency_hp_fraction",
                "emergency_interval_seconds",
                "emergency_followup_key",
                "emergency_followup_gap_seconds",
                "heal_key",
                "heal_hp_fraction",
                "heal_interval_seconds",
                "face_search_cooldown_seconds",
                "face_search_drag_seconds",
                "bearing_lock_seconds",
            },
            "v09.combat",
        )
        _reject_unknown(
            recovery,
            {
                "target_key",
                "interact_key",
                "approach_key",
                "key_tap_seconds",
                "click_seconds",
                "target_interval_seconds",
                "interact_interval_seconds",
                "approach_seconds",
                "approach_settle_seconds",
                "max_approach_attempts",
                "healer_anchor_activation_radius_yards",
                "healer_interact_radius_yards",
                "healer_anchors",
                "release_retry_seconds",
                "release_timeout_seconds",
                "healer_search_timeout_seconds",
                "dialog_timeout_seconds",
                "max_release_attempts",
                "max_target_attempts",
                "max_interact_attempts",
            },
            "v09.recovery",
        )
        _reject_unknown(
            mounting,
            {
                "enabled",
                "key",
                "cast_seconds",
                "retry_seconds",
                "max_attempts_per_cycle",
                "cycle_cooldown_seconds",
                "post_combat_settle_seconds",
            },
            "v09.mounting",
        )

        default_mode = _choice(
            section,
            "default_mode",
            choices={"offline", "replay", "shadow", "live"},
        )
        if default_mode == "live":
            raise V09ConfigError("v09.default_mode must fail closed and cannot be live")

        critical_seconds = _positive_float(freshness, "critical_seconds")
        operational_seconds = _positive_float(freshness, "operational_seconds")
        heavy_sensor_seconds = _positive_float(freshness, "heavy_sensor_seconds")
        if operational_seconds < critical_seconds:
            raise V09ConfigError(
                "v09.freshness.operational_seconds must be >= critical_seconds"
            )

        lookahead_min = _positive_float(navigation, "lookahead_min_yards")
        lookahead_cruise = _positive_float(navigation, "lookahead_cruise_yards")
        lookahead_max = _positive_float(navigation, "lookahead_max_yards")
        if not lookahead_min <= lookahead_cruise <= lookahead_max:
            raise V09ConfigError("v09 navigation lookahead must satisfy min <= cruise <= max")
        turn_release = _non_negative_float(navigation, "turn_release_degrees")
        turn_engage = _positive_float(navigation, "turn_engage_degrees")
        pivot = _positive_float(navigation, "pivot_degrees")
        if not turn_release <= turn_engage <= pivot <= 180.0:
            raise V09ConfigError(
                "v09 navigation turns must satisfy release <= engage <= pivot <= 180"
            )
        soft_stall = _positive_float(navigation, "stall_soft_seconds")
        hard_stall = _positive_float(navigation, "stall_hard_seconds")
        if hard_stall <= soft_stall:
            raise V09ConfigError("v09 hard stall timeout must exceed soft stall timeout")

        acquisition_radius = _positive_float(mining, "acquisition_radius_yards")
        braking_radius = _positive_float(mining, "braking_radius_yards")
        capture_radius = _positive_float(mining, "capture_radius_yards")
        if not capture_radius < braking_radius <= acquisition_radius:
            raise V09ConfigError(
                "v09 mining radii must satisfy capture < braking <= acquisition"
            )
        live_marker_weight = float(_required(mining, "live_marker_weight"))
        if not 0.0 <= live_marker_weight <= 1.0:
            raise V09ConfigError("live_marker_weight must be between 0 and 1")

        max_frame_backlog = _integer(performance, "max_frame_backlog", minimum=1)
        if max_frame_backlog != 1:
            raise V09ConfigError("v09.max_frame_backlog must remain 1")

        return cls(
            enabled=_boolean(section, "enabled"),
            default_mode=default_mode,
            protocol_version=_integer(section, "protocol_version", minimum=1),
            freshness=FreshnessConfig(
                critical_seconds=critical_seconds,
                operational_seconds=operational_seconds,
                heavy_sensor_seconds=heavy_sensor_seconds,
            ),
            performance=PerformanceConfig(
                control_hz=_positive_float(performance, "control_hz"),
                critical_preemption_p95_ms=_positive_float(
                    performance, "critical_preemption_p95_ms"
                ),
                controller_p95_ms=_positive_float(performance, "controller_p95_ms"),
                max_frame_backlog=max_frame_backlog,
                idle_ore_scan_interval_frames=_integer(
                    performance,
                    "idle_ore_scan_interval_frames",
                    minimum=1,
                ),
                shutdown_guard_quiet_seconds=_positive_float(
                    performance, "shutdown_guard_quiet_seconds"
                ),
                shutdown_guard_max_seconds=_positive_float(
                    performance, "shutdown_guard_max_seconds"
                ),
            ),
            zone=ZoneGeometry(
                zone_id=_integer(zone, "zone_id", minimum=1),
                width_yards=_positive_float(zone, "width_yards"),
                height_yards=_positive_float(zone, "height_yards"),
            ),
            navigation=NavigationConfig(
                lookahead_min_yards=lookahead_min,
                lookahead_cruise_yards=lookahead_cruise,
                lookahead_max_yards=lookahead_max,
                corridor_radius_yards=_positive_float(
                    navigation, "corridor_radius_yards"
                ),
                turn_engage_degrees=turn_engage,
                turn_release_degrees=turn_release,
                pivot_degrees=pivot,
                yaw_pixels_per_degree=_positive_float(
                    navigation, "yaw_pixels_per_degree"
                ),
                yaw_pixels_per_second=_positive_float(
                    navigation, "yaw_pixels_per_second"
                ),
                moving_drag_max_seconds=_positive_float(
                    navigation, "moving_drag_max_seconds"
                ),
                pivot_drag_max_seconds=_positive_float(
                    navigation, "pivot_drag_max_seconds"
                ),
                turn_reversal_guard_seconds=_positive_float(
                    navigation, "turn_reversal_guard_seconds"
                ),
                stall_soft_seconds=soft_stall,
                stall_hard_seconds=hard_stall,
                stall_min_progress_yards=_positive_float(
                    navigation, "stall_min_progress_yards"
                ),
                mounted_escape_seconds=_positive_float(
                    navigation, "mounted_escape_seconds"
                ),
                mounted_escape_attacker_count=_integer(
                    navigation, "mounted_escape_attacker_count", minimum=1
                ),
            ),
            mining=MiningApproachConfig(
                acquisition_radius_yards=acquisition_radius,
                capture_radius_yards=capture_radius,
                braking_radius_yards=braking_radius,
                centered_radius_pixels=_positive_float(
                    mining, "centered_radius_pixels"
                ),
                minimap_sensor_radius_pixels=_positive_float(
                    mining, "minimap_sensor_radius_pixels"
                ),
                minimap_radius_x_yards=_positive_float(
                    mining, "minimap_radius_x_yards"
                ),
                minimap_radius_y_yards=_positive_float(
                    mining, "minimap_radius_y_yards"
                ),
                live_marker_weight=live_marker_weight,
                max_fusion_disagreement_yards=_positive_float(
                    mining, "max_fusion_disagreement_yards"
                ),
                target_max_age_seconds=_positive_float(
                    mining, "target_max_age_seconds"
                ),
                max_approach_seconds=_positive_float(
                    mining, "max_approach_seconds"
                ),
                visual_search_seconds=_positive_float(
                    mining, "visual_search_seconds"
                ),
                verify_seconds=_positive_float(mining, "verify_seconds"),
                out_of_range_recovery_seconds=_positive_float(
                    mining, "out_of_range_recovery_seconds"
                ),
                max_out_of_range_recoveries=_integer(
                    mining, "max_out_of_range_recoveries", minimum=0
                ),
                identity_confirm_frames=_integer(
                    mining, "identity_confirm_frames", minimum=1
                ),
                identity_timeout_seconds=_positive_float(
                    mining, "identity_timeout_seconds"
                ),
                marker_track_max_jump_pixels=_positive_float(
                    mining, "marker_track_max_jump_pixels"
                ),
                database_match_radius_yards=_positive_float(
                    mining, "database_match_radius_yards"
                ),
                require_access_plan=_boolean(mining, "require_access_plan"),
                access_attachment_radius_yards=_positive_float(
                    mining, "access_attachment_radius_yards"
                ),
                failure_suppression_seconds=_positive_float(
                    mining, "failure_suppression_seconds"
                ),
            ),
            combat=CombatConfig(
                attack_key=_key(combat, "attack_key"),
                attack_interval_seconds=_positive_float(
                    combat, "attack_interval_seconds"
                ),
                key_tap_seconds=_positive_float(combat, "key_tap_seconds"),
                priority_key=_key(combat, "priority_key"),
                priority_interval_seconds=_positive_float(
                    combat, "priority_interval_seconds"
                ),
                aligned_key=_key(combat, "aligned_key"),
                aligned_interval_seconds=_positive_float(
                    combat, "aligned_interval_seconds"
                ),
                emergency_key=_key(combat, "emergency_key"),
                emergency_hp_fraction=_fraction(combat, "emergency_hp_fraction"),
                emergency_interval_seconds=_positive_float(
                    combat, "emergency_interval_seconds"
                ),
                emergency_followup_key=_key(combat, "emergency_followup_key"),
                emergency_followup_gap_seconds=_positive_float(
                    combat, "emergency_followup_gap_seconds"
                ),
                heal_key=_key(combat, "heal_key"),
                heal_hp_fraction=_fraction(combat, "heal_hp_fraction"),
                heal_interval_seconds=_positive_float(
                    combat, "heal_interval_seconds"
                ),
                face_search_cooldown_seconds=_positive_float(
                    combat, "face_search_cooldown_seconds"
                ),
                face_search_drag_seconds=_positive_float(
                    combat, "face_search_drag_seconds"
                ),
                bearing_lock_seconds=_positive_float(
                    combat, "bearing_lock_seconds"
                ),
            ),
            recovery=RecoveryConfig(
                target_key=_key(recovery, "target_key"),
                interact_key=_key(recovery, "interact_key"),
                approach_key=_key(recovery, "approach_key"),
                key_tap_seconds=_positive_float(recovery, "key_tap_seconds"),
                click_seconds=_positive_float(recovery, "click_seconds"),
                target_interval_seconds=_positive_float(
                    recovery, "target_interval_seconds"
                ),
                interact_interval_seconds=_positive_float(
                    recovery, "interact_interval_seconds"
                ),
                approach_seconds=_positive_float(
                    recovery, "approach_seconds"
                ),
                approach_settle_seconds=_positive_float(
                    recovery, "approach_settle_seconds"
                ),
                max_approach_attempts=_integer(
                    recovery, "max_approach_attempts", minimum=1
                ),
                healer_anchor_activation_radius_yards=_positive_float(
                    recovery, "healer_anchor_activation_radius_yards"
                ),
                healer_interact_radius_yards=_positive_float(
                    recovery, "healer_interact_radius_yards"
                ),
                healer_anchors=_map_points(recovery, "healer_anchors"),
                release_retry_seconds=_positive_float(
                    recovery, "release_retry_seconds"
                ),
                release_timeout_seconds=_positive_float(
                    recovery, "release_timeout_seconds"
                ),
                healer_search_timeout_seconds=_positive_float(
                    recovery, "healer_search_timeout_seconds"
                ),
                dialog_timeout_seconds=_positive_float(
                    recovery, "dialog_timeout_seconds"
                ),
                max_release_attempts=_integer(
                    recovery, "max_release_attempts", minimum=1
                ),
                max_target_attempts=_integer(
                    recovery, "max_target_attempts", minimum=1
                ),
                max_interact_attempts=_integer(
                    recovery, "max_interact_attempts", minimum=1
                ),
            ),
            mounting=MountingConfig(
                enabled=_boolean(mounting, "enabled"),
                key=_key(mounting, "key"),
                cast_seconds=_positive_float(mounting, "cast_seconds"),
                retry_seconds=_positive_float(mounting, "retry_seconds"),
                max_attempts_per_cycle=_integer(
                    mounting, "max_attempts_per_cycle", minimum=1
                ),
                cycle_cooldown_seconds=_positive_float(
                    mounting, "cycle_cooldown_seconds"
                ),
                post_combat_settle_seconds=_non_negative_float(
                    mounting, "post_combat_settle_seconds"
                ),
            ),
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V09ConfigError(f"{path} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise V09ConfigError(f"{path} contains unknown keys: {', '.join(unknown)}")


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise V09ConfigError(f"missing required v0.9 setting: {key}")
    return value[key]


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    raw = _required(value, key)
    if not isinstance(raw, bool):
        raise V09ConfigError(f"{key} must be a boolean")
    return raw


def _positive_float(value: Mapping[str, Any], key: str) -> float:
    result = float(_required(value, key))
    if result <= 0.0:
        raise V09ConfigError(f"{key} must be positive")
    return result


def _non_negative_float(value: Mapping[str, Any], key: str) -> float:
    result = float(_required(value, key))
    if result < 0.0:
        raise V09ConfigError(f"{key} must be non-negative")
    return result


def _integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    raw = _required(value, key)
    if isinstance(raw, bool) or int(raw) != raw or int(raw) < minimum:
        raise V09ConfigError(f"{key} must be an integer >= {minimum}")
    return int(raw)


def _choice(value: Mapping[str, Any], key: str, *, choices: set[str]) -> str:
    result = str(_required(value, key)).strip().lower()
    if result not in choices:
        raise V09ConfigError(f"{key} must be one of: {', '.join(sorted(choices))}")
    return result


def _key(value: Mapping[str, Any], key: str) -> str:
    result = str(_required(value, key)).strip().upper()
    if not result:
        raise V09ConfigError(f"{key} must not be empty")
    return result


def _fraction(value: Mapping[str, Any], key: str) -> float:
    result = float(_required(value, key))
    if not 0.0 <= result <= 1.0:
        raise V09ConfigError(f"{key} must be between 0 and 1")
    return result


def _map_points(value: Mapping[str, Any], key: str) -> tuple[MapPoint, ...]:
    raw = _required(value, key)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise V09ConfigError(f"{key} must be a non-empty list")
    points: list[MapPoint] = []
    for index, item in enumerate(raw):
        point = _mapping(item, f"{key}[{index}]")
        _reject_unknown(point, {"x", "y"}, f"{key}[{index}]")
        x = float(_required(point, "x"))
        y = float(_required(point, "y"))
        if not 0.0 <= x <= 100.0 or not 0.0 <= y <= 100.0:
            raise V09ConfigError(f"{key}[{index}] must stay within map percentages")
        points.append(MapPoint(x, y))
    return tuple(points)
