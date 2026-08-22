from vision_bot.world_zone import resolve_world_zone_ids


def test_resolve_world_zone_ids_maps_durotar_subzone_to_gathermate_zone():
    assert resolve_world_zone_ids("Southfury River") == [4]


def test_resolve_world_zone_ids_maps_barrens_to_gathermate_zone():
    assert resolve_world_zone_ids("The Barrens") == [11]


def test_resolve_world_zone_ids_maps_crossroads_subzone_to_barrens():
    assert resolve_world_zone_ids("The Crossroads") == [11]


def test_resolve_world_zone_ids_maps_ratchet_ocr_variants_to_barrens():
    assert resolve_world_zone_ids("Ratchet") == [11]
    assert resolve_world_zone_ids("MRAtchet") == [11]
    assert resolve_world_zone_ids("Ratchet es") == [11]
    assert resolve_world_zone_ids("Ratch et") == [11]


def test_resolve_world_zone_ids_accepts_config_mapping_override():
    config = {"world_zone": {"zone_mappings": {"Custom Place": [123, 456]}}}

    assert resolve_world_zone_ids("Custom Place", config) == [123, 456]


def test_resolve_world_zone_ids_returns_empty_for_unknown_zone():
    assert resolve_world_zone_ids("Unknown Test Zone") == []
