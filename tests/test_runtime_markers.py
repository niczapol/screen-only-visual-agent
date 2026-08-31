import cv2
import numpy as np

from vision_bot.runtime_markers import (
    detect_armor_critical_marker,
    detect_combat_loot_pending_marker,
    detect_dead_hostile_target_marker,
    detect_forbidden_subzone_marker,
    detect_loot_opened_marker,
    detect_mounted_marker,
    detect_spirit_healer_dialog_marker,
    detect_spirit_healer_target_marker,
    read_minimap_ore_tooltip_telemetry,
    read_runtime_telemetry,
    _runtime_telemetry_checksum,
)


def _config():
    return {
        "display": {"reference_width": 100, "reference_height": 100},
        "runtime_markers": {
            "enabled": True,
            "mounted_region": {"x": 0, "y": 0, "width": 50, "height": 50},
            "mounted_min_pixels": 100,
            "armor_critical_region": {"x": 50, "y": 50, "width": 50, "height": 50},
            "armor_critical_min_pixels": 100,
        },
    }


def test_runtime_markers_detect_only_their_fixed_colors():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mounted_hsv = np.uint8([[[108, 255, 255]]])
    mounted_bgr = cv2.cvtColor(mounted_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    frame[10:30, 10:30] = mounted_bgr
    frame[60:80, 60:80] = (0, 0, 255)

    assert detect_mounted_marker(frame, _config())
    assert detect_armor_critical_marker(frame, _config())


def test_runtime_markers_fail_closed_when_disabled():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)

    assert not detect_mounted_marker(frame, {})
    assert not detect_armor_critical_marker(frame, {})
    assert not detect_combat_loot_pending_marker(frame, {})
    assert not detect_forbidden_subzone_marker(frame, {})
    assert not detect_dead_hostile_target_marker(frame, {})
    assert not detect_loot_opened_marker(frame, {})
    assert not detect_spirit_healer_target_marker(frame, {})
    assert not detect_spirit_healer_dialog_marker(frame, {})


def test_forbidden_subzone_marker_uses_fixed_yellow_signal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 10:30] = (0, 217, 255)
    config = _config()
    config["runtime_markers"]["forbidden_subzone_region"] = {
        "x": 0,
        "y": 0,
        "width": 50,
        "height": 50,
    }
    config["runtime_markers"]["forbidden_subzone_min_pixels"] = 100

    assert detect_forbidden_subzone_marker(frame, config)


def test_dead_hostile_target_marker_uses_fixed_lime_signal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    marker_hsv = np.uint8([[[70, 230, 255]]])
    marker_bgr = cv2.cvtColor(marker_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    frame[10:30, 10:30] = marker_bgr
    config = _config()
    config["runtime_markers"]["dead_hostile_target_region"] = {
        "x": 0,
        "y": 0,
        "width": 50,
        "height": 50,
    }
    config["runtime_markers"]["dead_hostile_target_min_pixels"] = 100

    assert detect_dead_hostile_target_marker(frame, config)


def test_loot_opened_marker_uses_fixed_white_signal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 10:30] = (255, 255, 255)
    config = _config()
    config["runtime_markers"]["loot_opened_region"] = {
        "x": 0,
        "y": 0,
        "width": 50,
        "height": 50,
    }
    config["runtime_markers"]["loot_opened_min_pixels"] = 100

    assert detect_loot_opened_marker(frame, config)


def test_combat_loot_pending_marker_uses_fixed_violet_signal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    marker_bgr = cv2.cvtColor(
        np.uint8([[[137, 204, 255]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    frame[10:30, 10:30] = marker_bgr
    config = _config()
    config["runtime_markers"].update(
        {
            "combat_loot_pending_region": {
                "x": 0,
                "y": 0,
                "width": 50,
                "height": 50,
            },
            "combat_loot_pending_min_pixels": 100,
        }
    )

    assert detect_combat_loot_pending_marker(frame, config)


def test_default_loot_rois_contain_addon_markers_at_reference_resolution():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    violet_bgr = cv2.cvtColor(
        np.uint8([[[137, 204, 255]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    # UI scale maps addon offsets -216 and +576 near these screen positions.
    frame[13:51, 892:981] = violet_bgr
    frame[13:51, 2151:2240] = (255, 255, 255)
    config = {
        "display": {"reference_width": 2560, "reference_height": 1440},
        "runtime_markers": {"enabled": True},
    }

    assert detect_combat_loot_pending_marker(frame, config)
    assert detect_loot_opened_marker(frame, config)


def test_spirit_healer_markers_use_distinct_fixed_signals():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target_bgr = cv2.cvtColor(
        np.uint8([[[166, 217, 255]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    dialog_bgr = cv2.cvtColor(
        np.uint8([[[74, 166, 255]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    frame[10:30, 10:30] = target_bgr
    frame[60:80, 60:80] = dialog_bgr
    config = _config()
    config["runtime_markers"].update(
        {
            "spirit_healer_target_region": {"x": 0, "y": 0, "width": 50, "height": 50},
            "spirit_healer_target_min_pixels": 100,
            "spirit_healer_dialog_region": {"x": 50, "y": 50, "width": 50, "height": 50},
            "spirit_healer_dialog_min_pixels": 100,
        }
    )

    assert detect_spirit_healer_target_marker(frame, config)
    assert detect_spirit_healer_dialog_marker(frame, config)


def test_default_mounted_roi_contains_the_addon_marker_at_reference_resolution():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    mounted_hsv = np.uint8([[[108, 255, 255]]])
    mounted_bgr = cv2.cvtColor(mounted_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    # UIParent uses the active UI scale, so xOffset=216 renders here at 2560x1440.
    frame[13:51, 1580:1669] = mounted_bgr

    config = {
        "display": {"reference_width": 2560, "reference_height": 1440},
        "runtime_markers": {"enabled": True},
    }

    assert detect_mounted_marker(frame, config)


def _draw_telemetry_value(frame, value, start_index, bit_count, color):
    for bit in range(bit_count):
        if value & (1 << bit):
            x = 14 + (start_index + bit) * 4
            frame[6:18, x : x + 4] = color


def test_runtime_telemetry_decodes_visible_coordinate_and_heading_bits():
    frame = np.zeros((24, 320, 3), dtype=np.uint8)
    frame[6:18, 4:10] = (255, 0, 255)
    frame[6:18, 306:312] = (255, 255, 0)
    protocol = 2
    frame_sequence = 17
    x_value = 5388
    y_value = 2874
    heading_value = 255
    event_sequence = 9
    status_bits = 0b10010011
    checksum = _runtime_telemetry_checksum(
        protocol,
        frame_sequence,
        x_value,
        y_value,
        heading_value,
        event_sequence,
        status_bits,
    )
    _draw_telemetry_value(frame, protocol, 0, 3, (0, 0, 255))
    _draw_telemetry_value(frame, frame_sequence, 3, 8, (0, 0, 255))
    _draw_telemetry_value(frame, x_value, 11, 14, (0, 0, 255))
    _draw_telemetry_value(frame, y_value, 25, 14, (0, 255, 0))
    _draw_telemetry_value(frame, heading_value, 39, 9, (255, 0, 0))
    _draw_telemetry_value(frame, event_sequence, 48, 8, (0, 0, 255))
    _draw_telemetry_value(frame, status_bits, 56, 8, (0, 255, 0))
    _draw_telemetry_value(frame, checksum, 64, 8, (0, 0, 255))
    config = {
        "screen": {
            "reference_width": 320,
            "reference_height": 24,
            "scale_regions": False,
        },
        "runtime_markers": {
            "enabled": True,
            "telemetry": {
                "enabled": True,
                "region": {"x": 0, "y": 0, "width": 320, "height": 24},
                "bit_threshold": 150,
            },
        },
    }

    telemetry = read_runtime_telemetry(frame, config)

    assert telemetry is not None
    assert telemetry.coord == 5388287400
    assert telemetry.x == 53.88
    assert telemetry.y == 28.74
    assert abs(telemetry.heading_degrees - 179.65) < 0.1
    assert telemetry.protocol_version == 2
    assert telemetry.frame_sequence == 17
    assert telemetry.event_sequence == 9
    assert telemetry.status_bits == status_bits


def test_runtime_telemetry_decodes_centered_strip_after_ui_scale_change():
    strip = np.zeros((24, 320, 3), dtype=np.uint8)
    strip[6:18, 4:10] = (255, 0, 255)
    strip[6:18, 306:312] = (255, 255, 0)
    protocol = 2
    frame_sequence = 188
    x_value = 5165
    y_value = 2606
    heading_value = 63
    event_sequence = 0
    status_bits = 1 << 7
    checksum = _runtime_telemetry_checksum(
        protocol,
        frame_sequence,
        x_value,
        y_value,
        heading_value,
        event_sequence,
        status_bits,
    )
    _draw_telemetry_value(strip, protocol, 0, 3, (0, 0, 255))
    _draw_telemetry_value(strip, frame_sequence, 3, 8, (0, 0, 255))
    _draw_telemetry_value(strip, x_value, 11, 14, (0, 0, 255))
    _draw_telemetry_value(strip, y_value, 25, 14, (0, 255, 0))
    _draw_telemetry_value(strip, heading_value, 39, 9, (255, 0, 0))
    _draw_telemetry_value(strip, event_sequence, 48, 8, (0, 0, 255))
    _draw_telemetry_value(strip, status_bits, 56, 8, (0, 255, 0))
    _draw_telemetry_value(strip, checksum, 64, 8, (0, 0, 255))

    scaled = cv2.resize(strip, (506, 38), interpolation=cv2.INTER_NEAREST)
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    frame[60:98, 1027:1533] = scaled
    config = {
        "screen": {
            "reference_width": 2560,
            "reference_height": 1440,
            "scale_regions": True,
        },
        "v09": {"protocol_version": 2},
        "runtime_markers": {
            "enabled": True,
            "telemetry": {"enabled": True, "bit_threshold": 150},
        },
    }

    telemetry = read_runtime_telemetry(frame, config)

    assert telemetry is not None
    assert (telemetry.x, telemetry.y) == (51.65, 26.06)
    assert telemetry.frame_sequence == 188
    assert telemetry.status_bits == 1 << 7


def test_runtime_telemetry_fails_closed_on_checksum_mismatch():
    frame = np.zeros((24, 320, 3), dtype=np.uint8)
    frame[6:18, 4:10] = (255, 0, 255)
    frame[6:18, 306:312] = (255, 255, 0)
    _draw_telemetry_value(frame, 2, 0, 3, (0, 0, 255))
    _draw_telemetry_value(frame, 1, 3, 8, (0, 0, 255))
    _draw_telemetry_value(frame, 5000, 11, 14, (0, 0, 255))
    _draw_telemetry_value(frame, 5000, 25, 14, (0, 255, 0))
    _draw_telemetry_value(frame, 100, 39, 9, (255, 0, 0))
    config = {
        "screen": {"scale_regions": False},
        "v09": {"protocol_version": 2},
        "runtime_markers": {
            "enabled": True,
            "telemetry": {
                "enabled": True,
                "region": {"x": 0, "y": 0, "width": 320, "height": 24},
            },
        },
    }

    assert read_runtime_telemetry(frame, config) is None


def test_runtime_telemetry_fails_closed_without_both_sentinels():
    frame = np.zeros((24, 320, 3), dtype=np.uint8)
    frame[6:18, 4:10] = (255, 0, 255)

    assert read_runtime_telemetry(
        frame,
        {
            "screen": {"scale_regions": False},
            "runtime_markers": {
                "enabled": True,
                "telemetry": {
                    "enabled": True,
                    "region": {"x": 0, "y": 0, "width": 320, "height": 24},
                },
            },
        },
    ) is None


def _ore_tooltip_config():
    return {
        "screen": {"scale_regions": False},
        "runtime_markers": {
            "enabled": True,
            "minimap_ore_tooltip": {
                "enabled": True,
                "region": {"x": 0, "y": 0, "width": 70, "height": 24},
                "bit_threshold": 150,
            },
        },
    }


def test_minimap_ore_tooltip_decodes_known_ore_and_explicit_source_bit():
    frame = np.zeros((24, 70, 3), dtype=np.uint8)
    frame[6:18, 4:10] = (0, 255, 255)
    frame[6:18, 56:62] = (255, 0, 0)
    _draw_telemetry_value(frame, 8, 0, 8, (0, 0, 255))
    frame[6:18, 46:50] = (0, 255, 0)

    telemetry = read_minimap_ore_tooltip_telemetry(frame, _ore_tooltip_config())

    assert telemetry is not None
    assert telemetry.ore_id == 8
    assert telemetry.ore_type == "Small Thorium"


def test_minimap_ore_tooltip_fails_closed_without_minimap_source_bit():
    frame = np.zeros((24, 70, 3), dtype=np.uint8)
    frame[6:18, 4:10] = (0, 255, 255)
    frame[6:18, 56:62] = (255, 0, 0)
    _draw_telemetry_value(frame, 8, 0, 8, (0, 0, 255))

    assert read_minimap_ore_tooltip_telemetry(
        frame, _ore_tooltip_config()
    ) is None
