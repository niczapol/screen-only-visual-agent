from vision_bot.permanent_exclusions import add_permanent_exclusion, load_permanent_exclusions
from vision_bot.route_planner import MiningNode


def test_permanent_exclusions_are_saved_and_loaded(tmp_path):
    path = tmp_path / "permanent_exclusions.json"
    node = MiningNode(zone_id=640, coord=3750520000, ore_type="Small Thorium", node_id=1)

    add_permanent_exclusion(path, node, "dark_minimap_icon")

    assert load_permanent_exclusions(path) == {3750520000}


def test_permanent_exclusion_add_is_idempotent(tmp_path):
    path = tmp_path / "permanent_exclusions.json"
    node = MiningNode(zone_id=640, coord=3750520000, ore_type="Small Thorium", node_id=1)

    add_permanent_exclusion(path, node, "dark_minimap_icon")
    add_permanent_exclusion(path, node, "dark_minimap_icon")

    assert load_permanent_exclusions(path) == {3750520000}
