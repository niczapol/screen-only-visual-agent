from scripts.record_player_demonstration import (
    ALL_KEY_CODES,
    build_recording_config,
    state_signature,
    virtual_key_name,
)


def test_demonstration_config_enables_segmented_video_without_mutating_source():
    source = {"training_capture": {"route_live": {"video": {"enabled": False}}}}

    result = build_recording_config(source, fps=12.0, max_width=1280, segment_seconds=20.0)

    assert source["training_capture"]["route_live"]["video"]["enabled"] is False
    video = result["training_capture"]["route_live"]["video"]
    assert video == {
        "enabled": True,
        "fps": 12.0,
        "max_width": 1280,
        "segment_seconds": 20.0,
        "filename": "demonstration.mp4",
    }


def test_demonstration_state_signature_is_stable_for_equivalent_lists():
    first = {"keys_down": ["W"], "mouse_down": [], "cursor_client": [10, 20], "foreground": True}
    second = {"keys_down": ["W"], "mouse_down": [], "cursor_client": [10, 20], "foreground": True}

    assert state_signature(first) == state_signature(second)


def test_demonstration_tracks_full_keyboard_range_with_readable_common_names():
    assert virtual_key_name(0x41) == "A"
    assert virtual_key_name(0x70) == "F1"
    assert virtual_key_name(0xBA) == "VK_BA"
    assert ALL_KEY_CODES["SPACE"] == 0x20
    assert ALL_KEY_CODES["F24"] == 0x87
    assert 0x01 not in ALL_KEY_CODES.values()
    assert 0x05 not in ALL_KEY_CODES.values()
    assert 0x06 not in ALL_KEY_CODES.values()
    assert len(ALL_KEY_CODES) == 250
