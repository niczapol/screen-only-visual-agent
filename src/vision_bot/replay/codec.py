from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping, TypeVar

from vision_bot.core.evidence import Evidence, EvidenceState
from vision_bot.core.geometry import MapPoint
from vision_bot.core.world import (
    CombatView,
    LifeState,
    MiningPerception,
    MiningSignal,
    ModalState,
    PlayerPose,
    RecoveryPerception,
    RecoverySignal,
    WorldSnapshot,
)


T = TypeVar("T")


def snapshot_to_dict(snapshot: WorldSnapshot) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "frame_id": snapshot.frame_id,
        "captured_at": snapshot.captured_at,
        "input_ready": _evidence_to_dict(snapshot.input_ready),
        "pose": _evidence_to_dict(snapshot.pose),
        "life": _evidence_to_dict(snapshot.life),
        "modal": _evidence_to_dict(snapshot.modal),
        "combat": _evidence_to_dict(snapshot.combat),
        "mounted": _evidence_to_dict(snapshot.mounted),
        "mining": _evidence_to_dict(snapshot.mining),
        "loot_pending": _evidence_to_dict(snapshot.loot_pending),
        "forbidden_subzone": _evidence_to_dict(snapshot.forbidden_subzone),
        "recovery": _evidence_to_dict(snapshot.recovery),
        "metadata": _json_value(snapshot.metadata),
    }


def snapshot_from_dict(value: Mapping[str, Any]) -> WorldSnapshot:
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported replay snapshot schema")
    frame_id = int(value["frame_id"])
    return WorldSnapshot(
        frame_id=frame_id,
        captured_at=float(value["captured_at"]),
        input_ready=_evidence_from_dict(value["input_ready"], _bool_value),
        pose=_evidence_from_dict(value["pose"], _player_pose),
        life=_evidence_from_dict(value["life"], lambda item: LifeState(str(item))),
        modal=_evidence_from_dict(value["modal"], lambda item: ModalState(str(item))),
        combat=_evidence_from_dict(value["combat"], _combat_view),
        mounted=_evidence_from_dict(value["mounted"], _bool_value),
        mining=_evidence_from_dict(value["mining"], _mining_perception),
        loot_pending=_evidence_from_dict(value["loot_pending"], _bool_value),
        forbidden_subzone=_evidence_from_dict(
            value["forbidden_subzone"], _bool_value
        ),
        recovery=_evidence_from_dict(value["recovery"], _recovery_perception),
        metadata=dict(value.get("metadata", {})),
    )


def _evidence_to_dict(evidence: Evidence[Any]) -> dict[str, Any]:
    return {
        "state": evidence.state.value,
        "value": _json_value(evidence.value),
        "confidence": evidence.confidence,
        "observed_at": evidence.observed_at,
        "frame_id": evidence.frame_id,
        "source": evidence.source,
        "geometry": _json_value(evidence.geometry),
        "reason": evidence.reason,
    }


def _evidence_from_dict(
    value: Mapping[str, Any],
    decoder: Callable[[Any], T],
) -> Evidence[T]:
    state = EvidenceState(str(value["state"]))
    raw_value = value.get("value")
    decoded = decoder(raw_value) if state is EvidenceState.PRESENT else None
    return Evidence(
        state=state,
        value=decoded,
        confidence=float(value["confidence"]),
        observed_at=float(value["observed_at"]),
        frame_id=int(value["frame_id"]),
        source=str(value["source"]),
        geometry=dict(value.get("geometry", {})),
        reason=value.get("reason"),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if all(hasattr(value, field) for field in ("x", "y", "width", "height")):
        return {
            "x": int(value.x),
            "y": int(value.y),
            "width": int(value.width),
            "height": int(value.height),
        }
    raise TypeError(f"value is not replay serializable: {type(value).__name__}")


def _bool_value(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("replay boolean evidence must contain a boolean")
    return value


def _player_pose(value: Any) -> PlayerPose:
    mapping = _require_mapping(value, "player pose")
    position = _require_mapping(mapping["position"], "player position")
    return PlayerPose(
        zone_id=int(mapping["zone_id"]),
        position=MapPoint(
            x_percent=float(position["x_percent"]),
            y_percent=float(position["y_percent"]),
        ),
        heading_degrees=float(mapping["heading_degrees"]),
    )


def _combat_view(value: Any) -> CombatView:
    mapping = _require_mapping(value, "combat view")
    return CombatView(
        active=bool(mapping["active"]),
        hp_fraction=_optional_float(mapping.get("hp_fraction")),
        target_confirmed=bool(mapping.get("target_confirmed", False)),
        outgoing_hit=bool(mapping.get("outgoing_hit", False)),
        wrong_facing=bool(mapping.get("wrong_facing", False)),
        out_of_range=bool(mapping.get("out_of_range", False)),
        attacker_count=(
            int(mapping["attacker_count"])
            if mapping.get("attacker_count") is not None
            else None
        ),
        outcome_sequence=(
            int(mapping["outcome_sequence"])
            if mapping.get("outcome_sequence") is not None
            else None
        ),
        dead_hostile_target=bool(mapping.get("dead_hostile_target", False)),
        loot_opened=bool(mapping.get("loot_opened", False)),
        target_is_attacker=bool(mapping.get("target_is_attacker", False)),
    )


def _mining_perception(value: Any) -> MiningPerception:
    mapping = _require_mapping(value, "mining perception")
    candidate_position_raw = mapping.get("candidate_position")
    candidate_position = None
    if candidate_position_raw is not None:
        candidate_mapping = _require_mapping(candidate_position_raw, "candidate position")
        candidate_position = MapPoint(
            float(candidate_mapping["x_percent"]),
            float(candidate_mapping["y_percent"]),
        )
    return MiningPerception(
        signal=MiningSignal(str(mapping.get("signal", MiningSignal.IDLE.value))),
        candidate_id=(
            int(mapping["candidate_id"])
            if mapping.get("candidate_id") is not None
            else None
        ),
        candidate_position=candidate_position,
        minimap_offset_px=_optional_pair(mapping.get("minimap_offset_px"), float),
        minimap_centered=bool(mapping.get("minimap_centered", False)),
        bright_minimap_points=_pairs(mapping.get("bright_minimap_points", []), int),
        dark_minimap_points=_pairs(mapping.get("dark_minimap_points", []), int),
        tooltip_ore_type=mapping.get("tooltip_ore_type"),
        world_candidate_points=_pairs(mapping.get("world_candidate_points", []), int),
        interaction_authority=bool(mapping.get("interaction_authority", False)),
        interaction_point=_optional_pair(mapping.get("interaction_point"), int),
        interaction_outcome=mapping.get("interaction_outcome"),
    )


def _recovery_perception(value: Any) -> RecoveryPerception:
    mapping = _require_mapping(value, "recovery perception")
    return RecoveryPerception(
        signal=RecoverySignal(str(mapping.get("signal", RecoverySignal.NONE.value))),
        click_point=_optional_pair(mapping.get("click_point"), int),
        visual_bearing_point=_optional_pair(
            mapping.get("visual_bearing_point"), int
        ),
    )


def _pairs(value: Any, cast: Callable[[Any], T]) -> tuple[tuple[T, T], ...]:
    return tuple((cast(item[0]), cast(item[1])) for item in value)


def _optional_pair(
    value: Any,
    cast: Callable[[Any], T],
) -> tuple[T, T] | None:
    if value is None:
        return None
    return cast(value[0]), cast(value[1])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"replay {label} must be a mapping")
    return value
