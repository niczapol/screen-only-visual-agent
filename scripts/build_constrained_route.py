from __future__ import annotations

import argparse
from pathlib import Path

from vision_bot.route_database import (
    RouteBounds,
    build_constrained_route_database,
    write_route_csv,
    write_route_database,
    write_route_svg,
)
from vision_bot.route_planner import load_mining_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded cyclic mining route database")
    parser.add_argument("--mining-data", default="data/MiningData.lua")
    parser.add_argument("--zone-id", type=int, required=True)
    parser.add_argument("--zone-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--min-x", type=float, required=True)
    parser.add_argument("--max-x", type=float, required=True)
    parser.add_argument("--min-y", type=float, required=True)
    parser.add_argument("--max-y", type=float, required=True)
    parser.add_argument("--ore", action="append", default=None, help="Allowed ore type; repeat for multiple ores")
    parser.add_argument("--min-node-spacing", type=float, default=0.0)
    parser.add_argument("--max-two-opt-iterations", type=int, default=80)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--svg-output", default=None)
    args = parser.parse_args()

    nodes = load_mining_nodes(args.mining_data)
    data = build_constrained_route_database(
        nodes=nodes,
        zone_id=args.zone_id,
        zone_name=args.zone_name,
        bounds=RouteBounds(args.min_x, args.max_x, args.min_y, args.max_y),
        name=args.name,
        source_mining_data=args.mining_data,
        allowed_ores=set(args.ore or []),
        min_node_spacing=args.min_node_spacing,
        max_two_opt_iterations=args.max_two_opt_iterations,
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
