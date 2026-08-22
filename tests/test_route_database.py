import base64
from pathlib import Path

from vision_bot.route_database import RouteBounds, build_constrained_route_database, load_route_loop_coords, load_wotlk_routes_from_markdown, project_to_loop, project_to_path
from vision_bot.route_planner import MiningNode, load_nodes


def test_wotlk_routes_decode_synthetic_loop(tmp_path):
    raw = "^N46672843^N48002843^N50003000"
    payload = base64.b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    source = tmp_path / "routes.md"
    source.write_text(
        f"WOTLKC:Routes:Synthetic Mining Loop:Example Zone:{payload}\n",
        encoding="utf-8",
    )
    routes = load_wotlk_routes_from_markdown(source)
    matching = [
        route
        for route in routes
        if route.zone_name == "Example Zone" and route.name == "Synthetic Mining Loop"
    ]

    assert len(matching) == 1
    assert len(matching[0].coords) == 3
    assert matching[0].coords[0] == 4667284300


def test_barrens_safe_route_nodes_are_snapped_to_loop():
    route_path = Path("data/routes/generated/barrens_mining_75_125_safe_cycle.json")
    nodes = load_nodes(route_path)
    route_coords = load_route_loop_coords(route_path)

    assert nodes
    assert all(project_to_loop(node.coord, route_coords).distance <= 2.0 for node in nodes)


def test_barrens_bounded_route_nodes_stay_inside_configured_bounds():
    nodes = load_nodes(Path("data/routes/generated/barrens_central_east_mining_safe_cycle.json"))
    bounds = RouteBounds(min_x=50.0, max_x=65.0, min_y=30.0, max_y=68.0)

    assert len(nodes) == 21
    assert all(node.zone_id == 5 for node in nodes)
    assert all(node.ore_type == "Copper" for node in nodes)
    assert all(bounds.contains(node.coord) for node in nodes)


def test_project_to_loop_returns_nearest_segment_coord():
    projection = project_to_loop(5000150000, [4000100000, 6000100000])

    assert round(projection.distance, 2) == 5.0
    assert projection.segment_index == 0
    assert projection.segment_t == 0.5
    assert projection.nearest_coord == 5000100000


def test_project_to_path_does_not_close_last_waypoint_to_first():
    projection = project_to_path(
        4500300000,
        [4000100000, 6000100000, 6000900000],
    )

    assert projection.segment_index == 1
    assert projection.nearest_coord == 6000300000
    assert round(projection.distance, 2) == 15.0
    assert project_to_loop(4500300000, [4000100000, 6000100000, 6000900000]).distance == 0.0


def test_build_constrained_route_database_keeps_only_nodes_inside_bounds():
    nodes = [
        MiningNode(zone_id=5, coord=5200400000, ore_type="Copper", node_id=1, ore_id=1731),
        MiningNode(zone_id=5, coord=5400420000, ore_type="Copper", node_id=2, ore_id=1731),
        MiningNode(zone_id=5, coord=4300610000, ore_type="Copper", node_id=3, ore_id=1731),
        MiningNode(zone_id=4, coord=5500430000, ore_type="Copper", node_id=4, ore_id=1731),
        MiningNode(zone_id=5, coord=5600440000, ore_type="Tin", node_id=5, ore_id=1732),
    ]

    data = build_constrained_route_database(
        nodes=nodes,
        zone_id=5,
        zone_name="The Barrens",
        bounds=RouteBounds(min_x=50.0, max_x=60.0, min_y=35.0, max_y=50.0),
        name="test_barrens_bounds",
        source_mining_data="test.lua",
        allowed_ores={"Copper"},
    )

    assert [item["coord"] for item in data["route_nodes"]] == [5200400000, 5400420000]
    assert len(data["route_loop"]) == 2
    assert data["generation"]["accepted_nodes"] == 2
    assert data["generation"]["ore_counts"] == {"Copper": 2}
    assert {item["reason"] for item in data["rejected_nodes"]} == {"outside_route_bounds", "ore_not_allowed"}


def test_build_constrained_route_database_assigns_cyclic_route_order():
    nodes = [
        MiningNode(zone_id=5, coord=5200400000, ore_type="Copper", node_id=1),
        MiningNode(zone_id=5, coord=5800400000, ore_type="Copper", node_id=2),
        MiningNode(zone_id=5, coord=5800480000, ore_type="Copper", node_id=3),
        MiningNode(zone_id=5, coord=5200480000, ore_type="Copper", node_id=4),
    ]

    data = build_constrained_route_database(
        nodes=nodes,
        zone_id=5,
        zone_name="The Barrens",
        bounds=RouteBounds(min_x=50.0, max_x=60.0, min_y=35.0, max_y=50.0),
        name="test_barrens_cycle",
        source_mining_data="test.lua",
        min_node_spacing=0.0,
    )

    route_nodes = data["route_nodes"]
    assert [item["route_order"] for item in route_nodes] == [0, 1, 2, 3]
    assert [item["route_index"] for item in route_nodes] == [0, 1, 2, 3]
    assert all(item["distance_to_route"] == 0.0 for item in route_nodes)
    assert [item["coord"] for item in data["route_loop"]] == [item["coord"] for item in route_nodes]
