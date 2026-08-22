from pathlib import Path

from vision_bot.route_planner import (
    MiningRoutePlanner,
    is_route_waypoint,
    load_mining_nodes,
    load_nodes,
    load_route_waypoints,
    load_route_entry_plan,
)


def test_load_mining_nodes_parses_lua_database():
    nodes = load_mining_nodes(Path("data/MiningData.lua"))

    assert nodes
    assert any(node.ore_type == "Copper" for node in nodes)
    assert any(node.zone_id == 27 for node in nodes)


def test_load_route_waypoints_preserves_dense_loop_order(tmp_path):
    route_path = tmp_path / "route.json"
    route_path.write_text(
        '{"zone":{"id":102},"route_loop":['
        '{"index":0,"coord":1001},{"index":1,"coord":1002}]}'
    )

    waypoints = load_route_waypoints(route_path)

    assert [node.coord for node in waypoints] == [1001, 1002]
    assert [node.route_index for node in waypoints] == [0, 1]
    assert [node.route_t for node in waypoints] == [-0.5, -0.5]
    assert all(node.zone_id == 102 for node in waypoints)
    assert all(is_route_waypoint(node) for node in waypoints)


def test_load_route_entry_plan_preserves_safe_connector_order(tmp_path):
    route_path = tmp_path / "route.json"
    route_path.write_text(
        '{"zone":{"id":102},"entry_route":{"target_route_index":7,'
        '"waypoints":[{"index":0,"coord":1001},{"index":1,"coord":1002}]}}',
        encoding="utf-8",
    )

    plan = load_route_entry_plan(route_path)

    assert plan is not None
    assert [node.coord for node in plan.waypoints] == [1001, 1002]
    assert plan.target_route_sort_key == 7.0
    assert all(is_route_waypoint(node) for node in plan.waypoints)


def test_load_mining_nodes_parses_gathermate_v1_coordinates(tmp_path):
    mining_data = tmp_path / "MiningData.lua"
    mining_data.write_text(
        "\n".join(
            [
                "GatherMateDataMineDB = {",
                "[5] = {",
                "[46672843] = 201,",
                "},",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    nodes = load_mining_nodes(mining_data)

    assert len(nodes) == 1
    assert nodes[0].zone_id == 5
    assert nodes[0].coord == 46672843
    assert nodes[0].ore_type == "Copper"
    assert MiningRoutePlanner.distance(nodes[0].coord, 4667284300) == 0


def test_load_mining_nodes_parses_grouped_gathermate2_coordinates(tmp_path):
    mining_data = tmp_path / "MiningData.lua"
    mining_data.write_text(
        "\n".join(
            [
                "GatherMateData2MineDB = {",
                "  [5] = {",
                "    [201] = {",
                "      3560355000,3600263000,",
                "    },",
                "    [202] = {",
                "      4800688000",
                "    },",
                "  },",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    nodes = load_mining_nodes(mining_data)

    assert [(node.zone_id, node.coord, node.ore_type) for node in nodes] == [
        (5, 3560355000, "Copper"),
        (5, 3600263000, "Copper"),
        (5, 4800688000, "Tin"),
    ]


def test_load_route_database_json():
    nodes = load_nodes(Path("data/routes/generated/barrens_mining_75_125_safe_cycle.json"))

    assert len(nodes) == 59
    assert all(node.zone_id == 5 for node in nodes)
    assert all(node.route_order is not None for node in nodes)
    assert max(node.route_distance or 0.0 for node in nodes) <= 2.0


def test_load_desolace_route_exposes_generated_node_access_plans():
    nodes = load_nodes(
        Path("data/routes/generated/desolace_shadowprey_to_kodo_north_topographic_cycle.json")
    )

    assert len(nodes) == 6
    assert sum(len(node.access_plan.options) for node in nodes if node.access_plan) == 17
    for node in nodes:
        assert node.access_plan is not None
        plan = node.access_plan
        assert plan.options[plan.primary_option_rank].primary
        for option in plan.options:
            assert option.inbound_coords
            assert option.return_coords == tuple(reversed(option.inbound_coords))
            assert option.resume_coords


def test_load_tanaris_coverage_route_exposes_safe_access_plan_subset():
    nodes = load_nodes(
        Path("data/routes/generated/tanaris_terrain_coverage_cycle_v1.json")
    )

    planned = [node for node in nodes if node.access_plan is not None]
    assert len(nodes) == 263
    assert len(planned) == 212
    assert sum(len(node.access_plan.options) for node in planned) == 572
    assert all(node.access_plan.options for node in planned)
    assert all(
        option.approach_distance_yards is not None
        for node in planned
        for option in node.access_plan.options
    )
    assert sum(
        option.node_facing_aligned
        for node in planned
        for option in node.access_plan.options
    ) == 454


def test_route_planner_selects_nearest_node_and_honors_exclusions():
    planner = MiningRoutePlanner(
        nodes=load_mining_nodes(Path("data/MiningData.lua")),
        allowed_locations={27},
        allowed_ores={"Copper"},
    )
    planner.set_current_position(2850495000)

    first = planner.choose_next_node()
    assert first is not None
    planner.mark_mined(first.node_id, cooldown_seconds=60)

    remaining = planner.choose_next_node()
    assert remaining is not None
    assert remaining.node_id != first.node_id


def test_route_planner_marks_absent_nodes_with_cooldown():
    planner = MiningRoutePlanner(
        nodes=load_mining_nodes(Path("data/MiningData.lua")),
        allowed_locations={640},
        allowed_ores=set(),
    )
    planner.set_current_position(3752521200)

    first = planner.choose_next_node()
    assert first is not None
    planner.mark_absent(first.node_id, cooldown_seconds=300)

    remaining = planner.choose_next_node()
    assert remaining is not None
    assert remaining.node_id != first.node_id


def test_route_planner_returns_available_node_by_id():
    planner = MiningRoutePlanner(
        nodes=load_mining_nodes(Path("data/MiningData.lua")),
        allowed_locations={640},
        allowed_ores=set(),
    )
    planner.set_current_position(3752521200)

    first = planner.choose_next_node()
    assert first is not None
    assert planner.get_available_node(first.node_id) == first

    planner.mark_absent(first.node_id, cooldown_seconds=300)

    assert planner.get_available_node(first.node_id) is None


def test_route_planner_skips_permanent_excluded_coords():
    all_nodes = load_mining_nodes(Path("data/MiningData.lua"))
    planner = MiningRoutePlanner(
        nodes=all_nodes,
        allowed_locations={640},
        allowed_ores=set(),
    )
    planner.set_current_position(3752521200)

    first = planner.choose_next_node()
    assert first is not None

    excluded_planner = MiningRoutePlanner(
        nodes=all_nodes,
        allowed_locations={640},
        allowed_ores=set(),
        permanent_exclusions={first.coord},
    )
    excluded_planner.set_current_position(3752521200)

    remaining = excluded_planner.choose_next_node()
    assert remaining is not None
    assert remaining.coord != first.coord


def test_route_planner_can_follow_cyclic_route_order():
    nodes = load_nodes(Path("data/routes/generated/barrens_mining_75_125_safe_cycle.json"))
    planner = MiningRoutePlanner(
        nodes=nodes,
        allowed_locations={5},
        allowed_ores=set(),
        route_mode="cyclic",
    )
    planner.set_current_position(nodes[10].coord)

    first = planner.choose_next_node()
    assert first is not None
    planner.mark_mined(first.node_id, cooldown_seconds=60)

    second = planner.choose_next_node()
    assert second is not None
    assert second.route_order is not None
    assert first.route_order is not None
    assert second.route_order == first.route_order + 1
