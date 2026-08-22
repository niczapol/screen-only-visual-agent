from __future__ import annotations

import argparse
from pathlib import Path

from vision_bot.route_database import (
    build_safe_route_database,
    load_wotlk_routes_from_markdown,
    write_route_csv,
    write_route_database,
    write_route_svg,
)
from vision_bot.route_planner import load_mining_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a safe cyclic mining route database")
    parser.add_argument("--mining-data", default="data/MiningData.lua")
    parser.add_argument("--routes-markdown", default="debug_output/route_research/smooth_wotlk_routes_readme.md")
    parser.add_argument("--zone-id", type=int, required=True)
    parser.add_argument("--zone-name", required=True)
    parser.add_argument("--route-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--corridor-radius", type=float, default=2.0)
    parser.add_argument("--min-node-spacing", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--svg-output", default=None)
    args = parser.parse_args()

    routes = load_wotlk_routes_from_markdown(args.routes_markdown)
    matching_routes = [
        route for route in routes if route.zone_name == args.zone_name and route.name == args.route_name
    ]
    if not matching_routes:
        available = ", ".join(f"{route.zone_name}/{route.name}" for route in routes)
        raise SystemExit(f"Route not found: {args.zone_name}/{args.route_name}. Available: {available}")

    nodes = load_mining_nodes(args.mining_data)
    data = build_safe_route_database(
        route=matching_routes[0],
        nodes=nodes,
        zone_id=args.zone_id,
        corridor_radius=args.corridor_radius,
        min_node_spacing=args.min_node_spacing,
        name=args.name,
        source_mining_data=args.mining_data,
        source_route_data=args.routes_markdown,
    )

    write_route_database(data, args.output)
    if args.csv_output:
        write_route_csv(data, args.csv_output)
    if args.svg_output:
        write_route_svg(data, args.svg_output)

    generation = data["generation"]
    print(f"Wrote {Path(args.output)}")
    print(
        "accepted={accepted_nodes} rejected={rejected_nodes} waypoints={route_waypoints} ores={ore_counts}".format(
            **generation
        )
    )


if __name__ == "__main__":
    main()
