from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from vision_bot.coords import coord_to_xy
from vision_bot.route_database import project_to_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a route-live compatible slice from a larger route_loop JSON. "
            "This is intended for short mining-acceptance runs that should start "
            "near an ore-dense part of an already generated route."
        )
    )
    parser.add_argument(
        "--source-route",
        default="data/routes/generated/tanaris_terrain_coverage_cycle_v4.json",
    )
    parser.add_argument(
        "--output",
        default="data/routes/generated/tanaris_mining_acceptance_slice_v1.json",
    )
    parser.add_argument("--start-index", type=int, default=278)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument(
        "--auto-best",
        action="store_true",
        help="Ignore --start-index and choose the covered-node densest window.",
    )
    parser.add_argument(
        "--min-node-distance-to-route",
        type=float,
        default=1.25,
        help="Keep covered route nodes whose source-loop projection is this close in UI units.",
    )
    return parser.parse_args()


def _in_cyclic_window(index: int, *, start: int, size: int, route_size: int) -> bool:
    if route_size <= 0 or size <= 0:
        return False
    normalized = int(index) % route_size
    return ((normalized - start) % route_size) < size


def _local_route_index(index: int, *, start: int, route_size: int) -> int:
    return (int(index) % route_size - start) % route_size


def _window_density(
    data: dict[str, Any],
    *,
    start: int,
    size: int,
    max_distance: float,
) -> tuple[int, int, int]:
    source_loop = [int(item["coord"]) for item in data.get("route_loop", [])]
    route_size = len(source_loop)
    covered = 0
    human = 0
    with_access = 0
    for node in data.get("route_nodes", []):
        if node.get("coverage_status") != "covered":
            continue
        projection = project_to_loop(int(node["coord"]), source_loop)
        if projection.distance > max_distance:
            continue
        if not _in_cyclic_window(
            projection.segment_index,
            start=start,
            size=size,
            route_size=route_size,
        ):
            continue
        covered += 1
        if bool(node.get("human_reference_covered")):
            human += 1
        if isinstance(node.get("terrain_access_plan"), dict):
            with_access += 1
    return covered, human, with_access


def choose_best_window(
    data: dict[str, Any],
    *,
    size: int,
    max_distance: float,
) -> int:
    route_size = len(data.get("route_loop", []))
    if route_size <= 0:
        raise ValueError("source route has no route_loop")
    ranked = [
        (_window_density(data, start=index, size=size, max_distance=max_distance), index)
        for index in range(route_size)
    ]
    ranked.sort(key=lambda item: (*item[0], -item[1]), reverse=True)
    return ranked[0][1]


def build_route_slice(
    data: dict[str, Any],
    *,
    start_index: int,
    window_size: int,
    max_node_distance_to_route: float,
    output_name: str,
) -> dict[str, Any]:
    source_loop_items = data.get("route_loop", [])
    if not source_loop_items:
        raise ValueError("source route has no route_loop")
    route_size = len(source_loop_items)
    if not (1 <= window_size <= route_size):
        raise ValueError("window_size must be within the source route length")

    start = start_index % route_size
    source_loop = [int(item["coord"]) for item in source_loop_items]
    source_indexes = [(start + offset) % route_size for offset in range(window_size)]
    route_loop: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(source_indexes):
        source_item = source_loop_items[source_index]
        coord = int(source_item["coord"])
        x, y = coord_to_xy(coord)
        route_loop.append(
            {
                "index": local_index,
                "source_index": source_index,
                "coord": coord,
                "x": round(float(x), 6),
                "y": round(float(y), 6),
            }
        )

    route_nodes: list[dict[str, Any]] = []
    rejected_nodes: list[dict[str, Any]] = []
    for source_node in data.get("route_nodes", []):
        node = copy.deepcopy(source_node)
        if node.get("coverage_status") != "covered":
            node["slice_reject_reason"] = str(node.get("coverage_status") or "not_covered")
            rejected_nodes.append(node)
            continue
        projection = project_to_loop(int(node["coord"]), source_loop)
        if projection.distance > max_node_distance_to_route:
            node["slice_reject_reason"] = "too_far_from_source_route"
            node["source_route_index"] = projection.segment_index
            node["distance_to_route"] = round(projection.distance, 6)
            rejected_nodes.append(node)
            continue
        if not _in_cyclic_window(
            projection.segment_index,
            start=start,
            size=window_size,
            route_size=route_size,
        ):
            node["slice_reject_reason"] = "outside_route_slice"
            node["source_route_index"] = projection.segment_index
            rejected_nodes.append(node)
            continue

        local_index = _local_route_index(
            projection.segment_index,
            start=start,
            route_size=route_size,
        )
        node["source_route_index"] = projection.segment_index
        node["route_index"] = local_index
        node["route_t"] = round(projection.segment_t, 6)
        node["route_order"] = len(route_nodes)
        node["distance_to_route"] = round(projection.distance, 6)
        node["source"] = "route_slice_node"
        access_plan = node.get("terrain_access_plan")
        if isinstance(access_plan, dict):
            _remap_access_plan(
                access_plan,
                start=start,
                route_size=route_size,
                window_size=window_size,
            )
        route_nodes.append(node)

    for node_id, node in enumerate(route_nodes, start=1):
        node["node_id"] = node_id

    covered, human, with_access = _window_density(
        data,
        start=start,
        size=window_size,
        max_distance=max_node_distance_to_route,
    )
    return {
        "schema_version": 1,
        "name": output_name,
        "zone_id": int(data.get("zone_id", data.get("zone", {}).get("id", 0))),
        "world_map_area_id": data.get("world_map_area_id"),
        "status": "offline_mining_acceptance_slice_not_live_validated",
        "source": {
            "route": str(data.get("name") or "unknown"),
            "start_index": start,
            "window_size": window_size,
            "max_node_distance_to_route": max_node_distance_to_route,
        },
        "coverage_policy": data.get("coverage_policy", {}),
        "route_loop": route_loop,
        "route_nodes": route_nodes,
        "rejected_nodes": rejected_nodes,
        "metrics": {
            "source_route_waypoints": route_size,
            "route_waypoint_count": len(route_loop),
            "source_start_index": start,
            "window_size": window_size,
            "accepted_node_count": len(route_nodes),
            "rejected_node_count": len(rejected_nodes),
            "covered_node_density": covered,
            "human_reference_covered_count": human,
            "access_plan_count": with_access,
        },
    }


def _remap_access_plan(
    access_plan: dict[str, Any],
    *,
    start: int,
    route_size: int,
    window_size: int,
) -> None:
    for key in ("attachment_route_index", "resume_route_index"):
        source_value = int(access_plan[key])
        local_value = _local_route_index(source_value, start=start, route_size=route_size)
        access_plan[f"source_{key}"] = source_value
        access_plan[key] = min(local_value, window_size - 1)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_route)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    start_index = (
        choose_best_window(
            data,
            size=args.window_size,
            max_distance=args.min_node_distance_to_route,
        )
        if args.auto_best
        else args.start_index
    )
    output_path = Path(args.output)
    output = build_route_slice(
        data,
        start_index=start_index,
        window_size=args.window_size,
        max_node_distance_to_route=args.min_node_distance_to_route,
        output_name=f"{data.get('name', source_path.stem)} slice {start_index}:{args.window_size}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = output["metrics"]
    print(f"Route slice: {output_path}")
    print(
        "waypoints={route_waypoint_count} nodes={accepted_node_count} "
        "access_plans={access_plan_count} start={source_start_index} window={window_size}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
