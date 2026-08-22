from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from vision_bot.coords import coord_to_xy


@dataclass(frozen=True)
class NodeAccessOption:
    rank: int
    candidate_index: int
    primary: bool
    approach_coord: int
    sector: int
    bearing_degrees: float
    objective: float
    inbound_coords: tuple[int, ...]
    return_coords: tuple[int, ...]
    resume_coords: tuple[int, ...]
    approach_distance_yards: float | None = None
    node_facing_aligned: bool = False
    facing_stage_coord: int | None = None


@dataclass(frozen=True)
class NodeAccessPlan:
    attachment_route_index: int
    resume_route_index: int
    primary_option_rank: int
    options: tuple[NodeAccessOption, ...]


@dataclass(frozen=True)
class MiningNode:
    zone_id: int
    coord: int
    ore_type: str
    node_id: int | None
    ore_id: int | None = None
    route_index: int | None = None
    route_t: float | None = None
    route_order: int | None = None
    route_distance: float | None = None
    source: str = "mining_data"
    access_plan: NodeAccessPlan | None = None


ROUTE_WAYPOINT_SOURCE = "route_loop_waypoint"
ROUTE_ENTRY_WAYPOINT_SOURCE = "route_entry_waypoint"


@dataclass(frozen=True)
class RouteEntryPlan:
    waypoints: tuple[MiningNode, ...]
    target_route_sort_key: float


@dataclass
class MiningRoutePlanner:
    nodes: list[MiningNode]
    allowed_locations: set[int] | None = None
    allowed_ores: set[str] | None = None
    permanent_exclusions: set[int] | None = None
    current_position: int | None = None
    cooldowns: dict[int, datetime] = field(default_factory=dict)
    route_mode: str = "nearest"
    _route_cursor: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.allowed_locations = set(self.allowed_locations or set())
        self.allowed_ores = set(self.allowed_ores or set())
        self.permanent_exclusions = set(self.permanent_exclusions or set())

    def set_current_position(self, coord: int) -> None:
        self.current_position = coord

    def choose_next_node(self) -> MiningNode | None:
        now = datetime.now()
        candidates = self.available_nodes(now)
        if not candidates:
            return None

        if self.route_mode == "cyclic":
            return self._choose_cyclic_node(candidates)

        if self.current_position is None:
            return candidates[0]

        return min(
            candidates,
            key=lambda node: self._distance(node.coord, self.current_position),
        )

    def get_available_node(self, node_id: int | None, now: datetime | None = None) -> MiningNode | None:
        if node_id is None:
            return None
        for node in self.available_nodes(now):
            if node.node_id == node_id:
                return node
        return None

    def mark_mined(self, node_id: int | None, cooldown_seconds: int = 180) -> None:
        if node_id is None:
            return
        self.cooldowns[node_id] = datetime.now() + timedelta(seconds=cooldown_seconds)

    def mark_absent(self, node_id: int | None, cooldown_seconds: int = 300) -> None:
        if node_id is None:
            return
        self.cooldowns[node_id] = datetime.now() + timedelta(seconds=cooldown_seconds)

    def exclude_permanently(self, coord: int) -> None:
        if self.permanent_exclusions is None:
            self.permanent_exclusions = set()
        self.permanent_exclusions.add(coord)

    def distance_from_current(self, coord: int) -> float | None:
        if self.current_position is None:
            return None
        return self.distance(coord, self.current_position)

    def available_nodes(self, now: datetime | None = None) -> list[MiningNode]:
        now = now or datetime.now()
        return [
            node
            for node in self.nodes
            if (not self.allowed_locations or node.zone_id in self.allowed_locations)
            and (not self.allowed_ores or node.ore_type in self.allowed_ores)
            and node.coord not in self.permanent_exclusions
            and self._is_available(node, now)
        ]

    def available_coords(self, now: datetime | None = None) -> list[int]:
        return [node.coord for node in self.available_nodes(now)]

    def _is_available(self, node: MiningNode, now: datetime) -> bool:
        if node.node_id is None:
            return True
        expires_at = self.cooldowns.get(node.node_id)
        return expires_at is None or now >= expires_at

    def _choose_cyclic_node(self, candidates: list[MiningNode]) -> MiningNode:
        routed = [node for node in candidates if _cyclic_order(node) is not None]
        if not routed:
            if self.current_position is None:
                return candidates[0]
            return min(candidates, key=lambda node: self._distance(node.coord, self.current_position))

        if self._route_cursor is None:
            if self.current_position is None:
                selected = min(routed, key=_cyclic_order_value)
            else:
                selected = min(routed, key=lambda node: self._distance(node.coord, self.current_position))
            self._route_cursor = _cyclic_order_value(selected)
            return selected

        ordered = sorted(routed, key=_cyclic_order_value)
        selected = next((node for node in ordered if _cyclic_order_value(node) > self._route_cursor), ordered[0])
        self._route_cursor = _cyclic_order_value(selected)
        return selected

    @staticmethod
    def _distance(a: int, b: int) -> float:
        return MiningRoutePlanner.distance(a, b)

    @staticmethod
    def distance(a: int, b: int) -> float:
        ax, ay = MiningRoutePlanner._decode_coord(a)
        bx, by = MiningRoutePlanner._decode_coord(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    @staticmethod
    def _decode_coord(coord: int) -> tuple[float, float]:
        return coord_to_xy(coord)


def load_nodes(path: str | Path) -> list[MiningNode]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".json":
        return load_route_nodes(source_path)
    return load_mining_nodes(source_path)


def load_route_nodes(path: str | Path) -> list[MiningNode]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes: list[MiningNode] = []
    for index, item in enumerate(data.get("route_nodes", []), start=1):
        ore_id = item.get("ore_id")
        ore_type = item.get("ore_type")
        if not ore_type:
            ore_type = ore_name(int(ore_id)) if ore_id is not None else "Ore"
        nodes.append(
            MiningNode(
                zone_id=int(item["zone_id"]),
                coord=int(item["coord"]),
                ore_type=str(ore_type),
                node_id=int(item.get("node_id") or index),
                ore_id=int(ore_id) if ore_id is not None else None,
                route_index=_optional_int(item.get("route_index")),
                route_t=_optional_float(item.get("route_t")),
                route_order=_optional_int(item.get("route_order")),
                route_distance=_optional_float(item.get("distance_to_route", item.get("route_distance"))),
                source=str(item.get("source") or data.get("name") or "route_database"),
                access_plan=_parse_node_access_plan(item.get("terrain_access_plan")),
            )
        )
    return nodes


def _parse_node_access_plan(value: object) -> NodeAccessPlan | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("terrain_access_plan must be an object")
    raw_options = value.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("terrain_access_plan must contain at least one option")

    options: list[NodeAccessOption] = []
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            raise ValueError("terrain access option must be an object")
        options.append(
            NodeAccessOption(
                rank=int(raw_option["rank"]),
                candidate_index=int(raw_option["candidate_index"]),
                primary=bool(raw_option.get("primary", False)),
                approach_coord=int(raw_option["approach_coord"]),
                sector=int(raw_option.get("sector", -1)),
                bearing_degrees=float(raw_option.get("bearing_degrees", 0.0)),
                objective=float(raw_option.get("objective", 0.0)),
                inbound_coords=_parse_access_leg_coords(raw_option.get("inbound")),
                return_coords=_parse_access_leg_coords(raw_option.get("return")),
                resume_coords=_parse_access_leg_coords(raw_option.get("resume")),
                approach_distance_yards=(
                    float(raw_option["approach_distance_yards"])
                    if raw_option.get("approach_distance_yards") is not None
                    else None
                ),
                node_facing_aligned=bool(
                    raw_option.get("node_facing_aligned", False)
                ),
                facing_stage_coord=(
                    int(raw_option["facing_stage_coord"])
                    if raw_option.get("facing_stage_coord") is not None
                    else None
                ),
            )
        )

    primary_rank = int(value.get("primary_option_rank", 0))
    if not any(option.rank == primary_rank and option.primary for option in options):
        raise ValueError("terrain_access_plan primary option is missing")
    return NodeAccessPlan(
        attachment_route_index=int(value["attachment_route_index"]),
        resume_route_index=int(value["resume_route_index"]),
        primary_option_rank=primary_rank,
        options=tuple(sorted(options, key=lambda option: option.rank)),
    )


def _parse_access_leg_coords(value: object) -> tuple[int, ...]:
    if not isinstance(value, dict):
        raise ValueError("terrain access leg must be an object")
    waypoints = value.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("terrain access leg must contain waypoints")
    return tuple(int(waypoint["coord"]) for waypoint in waypoints)


def load_route_waypoints(path: str | Path) -> list[MiningNode]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zone_id = int(data.get("zone_id", data.get("zone", {}).get("id", 0)))
    waypoints: list[MiningNode] = []
    for fallback_index, item in enumerate(data.get("route_loop", [])):
        index = int(item.get("index", fallback_index))
        waypoints.append(
            MiningNode(
                zone_id=zone_id,
                coord=int(item["coord"]),
                ore_type="Route Waypoint",
                node_id=-(index + 1),
                route_index=index,
                route_t=-0.5,
                route_order=index,
                source=ROUTE_WAYPOINT_SOURCE,
            )
        )
    return waypoints


def load_route_entry_plan(path: str | Path) -> RouteEntryPlan | None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = data.get("entry_route")
    if not isinstance(entry, dict):
        return None
    raw_waypoints = entry.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        return None
    zone_id = int(data.get("zone_id", data.get("zone", {}).get("id", 0)))
    waypoints: list[MiningNode] = []
    for fallback_index, item in enumerate(raw_waypoints):
        if not isinstance(item, dict) or "coord" not in item:
            continue
        index = int(item.get("index", fallback_index))
        waypoints.append(
            MiningNode(
                zone_id=zone_id,
                coord=int(item["coord"]),
                ore_type="Route Entry Waypoint",
                node_id=-(1_000_000 + index),
                route_index=index,
                route_t=-0.75,
                route_order=index,
                source=ROUTE_ENTRY_WAYPOINT_SOURCE,
            )
        )
    if not waypoints:
        return None
    return RouteEntryPlan(
        waypoints=tuple(waypoints),
        target_route_sort_key=float(entry.get("target_route_index", 0)),
    )


def is_route_waypoint(node: MiningNode) -> bool:
    return node.source in {ROUTE_WAYPOINT_SOURCE, ROUTE_ENTRY_WAYPOINT_SOURCE}


def load_mining_nodes(path: str | Path) -> list[MiningNode]:
    text = Path(path).read_text(encoding="utf-8")

    zone_id: int | None = None
    pending_zone_id: int | None = None
    grouped_ore_id: int | None = None
    nodes: list[MiningNode] = []
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        if stripped.startswith(("GatherMateData2MineDB", "GatherMateDataMineDB")):
            continue

        if pending_zone_id is not None and stripped == "{":
            zone_id = pending_zone_id
            pending_zone_id = None
            continue

        if stripped.startswith("},"):
            if grouped_ore_id is not None:
                grouped_ore_id = None
            elif zone_id is not None:
                zone_id = None
            continue

        if stripped == "}":
            grouped_ore_id = None
            pending_zone_id = None
            zone_id = None
            continue

        if zone_id is None:
            zone_match = re.match(r"^\[(\d+)\]\s*=\s*(\{)?\s*$", stripped)
            if zone_match:
                if zone_match.group(2) == "{":
                    zone_id = int(zone_match.group(1))
                else:
                    pending_zone_id = int(zone_match.group(1))
            continue

        if grouped_ore_id is not None:
            for coord_match in re.finditer(r"\b(\d{7,10})\b", stripped):
                _append_mining_node(nodes, zone_id, int(coord_match.group(1)), grouped_ore_id)
            continue

        grouped_match = re.match(r"^\[(\d+)\]\s*=\s*\{\s*$", stripped)
        if grouped_match:
            grouped_ore_id = int(grouped_match.group(1))
            continue

        match = re.match(r"^\[(\d{7,10})\]\s*=\s*(\d+)", stripped)
        if match:
            _append_mining_node(nodes, zone_id, int(match.group(1)), int(match.group(2)))

    return nodes


def ore_name(ore_id: int) -> str:
    mapping = {
        201: "Copper",
        202: "Tin",
        203: "Iron",
        204: "Silver",
        205: "Gold",
        206: "Mithril",
        207: "Ooze Covered Mithril",
        208: "Truesilver",
        209: "Ooze Covered Silver",
        210: "Ooze Covered Gold",
        211: "Ooze Covered Truesilver",
        212: "Ooze Covered Rich Thorium",
        213: "Ooze Covered Thorium",
        214: "Small Thorium",
        215: "Rich Thorium",
        217: "Dark Iron",
        218: "Lesser Bloodstone",
        219: "Incendicite",
        220: "Indurium",
        221: "Fel Iron",
        222: "Adamantite",
        223: "Rich Adamantite",
        224: "Khorium",
        228: "Cobalt",
        229: "Rich Cobalt",
        230: "Titanium",
        231: "Saronite",
        232: "Rich Saronite",
    }
    return mapping.get(ore_id, f"Ore{ore_id}")


def _append_mining_node(nodes: list[MiningNode], zone_id: int, coord: int, ore_id: int) -> None:
    nodes.append(
        MiningNode(
            zone_id=zone_id,
            coord=coord,
            ore_type=ore_name(ore_id),
            node_id=len(nodes) + 1,
            ore_id=ore_id,
        )
    )


def _cyclic_order(node: MiningNode) -> int | None:
    if node.route_order is not None:
        return node.route_order
    return node.route_index


def _cyclic_order_value(node: MiningNode) -> int:
    order = _cyclic_order(node)
    return order if order is not None else 0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
