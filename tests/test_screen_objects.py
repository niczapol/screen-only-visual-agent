import numpy as np
import cv2

from vision_bot.screen_objects import (
    BoundingBox,
    DEFAULT_CLASS_NAMES,
    ScreenDetection,
    detect_nameplate_mobs,
    detect_walkability_candidates,
    detections_to_yolo_lines,
    _filter_by_class_min_scores,
    yolo_results_to_detections,
)
from vision_bot.threat_detection import detect_hostile_threat


def test_detect_nameplate_mobs_finds_red_and_yellow_bars():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 120), (220, 132), (0, 0, 255), -1)
    cv2.rectangle(frame, (300, 180), (430, 193), (0, 220, 220), -1)
    config = {
        "screen": {"reference_width": 1280, "reference_height": 720, "scale_regions": False},
        "screen_objects": {
            "world_region": {"x": 0, "y": 0, "width": 1280, "height": 720},
            "nameplates": {
                "min_area": 20,
                "max_area": 10000,
                "min_width": 20,
                "max_width": 300,
                "min_height": 4,
                "max_height": 40,
                "min_aspect": 2.0,
                "max_aspect": 40.0,
            },
        },
    }

    detections = detect_nameplate_mobs(frame, config)
    labels = {detection.label for detection in detections}

    assert "aggressive_mob" in labels
    assert "peaceful_mob" in labels


def test_detections_to_yolo_lines_exports_normalized_boxes():
    detections = [
        ScreenDetection("aggressive_mob", BoundingBox(100, 50, 200, 100), 0.9, "test"),
        ScreenDetection("terrain_obstacle", BoundingBox(0, 0, 20, 20), 0.1, "test"),
    ]

    lines = detections_to_yolo_lines(
        detections,
        (1000, 2000, 3),
        ["aggressive_mob", "terrain_obstacle"],
        min_score=0.5,
    )

    assert lines == ["0 0.100000 0.100000 0.100000 0.100000"]


def test_yolo_results_to_detections_exports_screen_detections():
    class Boxes:
        xyxy = np.array([[10, 20, 30, 45], [100, 50, 160, 90]], dtype=np.float32)
        conf = np.array([0.8, 0.6], dtype=np.float32)
        cls = np.array([0, 4], dtype=np.float32)

    class Result:
        boxes = Boxes()
        names = {0: "aggressive_mob", 4: "terrain_obstacle"}

    detections = yolo_results_to_detections([Result()], (100, 200, 3))

    assert [detection.label for detection in detections] == ["aggressive_mob", "terrain_obstacle"]
    assert detections[0].bbox == BoundingBox(10, 20, 20, 25)
    assert detections[0].source == "yolo_model"


def test_filter_by_class_min_scores_keeps_low_terrain_but_filters_low_ore():
    detections = [
        ScreenDetection("terrain_obstacle", BoundingBox(0, 0, 10, 10), 0.16, "test"),
        ScreenDetection("ore_vein", BoundingBox(20, 20, 10, 10), 0.16, "test"),
    ]

    filtered = _filter_by_class_min_scores(
        detections,
        {
            "terrain_obstacle": 0.15,
            "ore_vein": 0.35,
        },
    )

    assert [detection.label for detection in filtered] == ["terrain_obstacle"]


def test_default_class_names_append_walkability_classes_without_shifting_existing_ids():
    assert DEFAULT_CLASS_NAMES[:6] == [
        "aggressive_mob",
        "peaceful_mob",
        "ore_vein",
        "minimap_ore_icon",
        "terrain_obstacle",
        "terrain_edge",
    ]
    assert DEFAULT_CLASS_NAMES[6:] == ["movement_blocker", "walkable_corridor"]


def test_detect_walkability_candidates_marks_clear_corridor_and_center_blocker():
    frame = np.full((240, 360, 3), 96, dtype=np.uint8)
    rng = np.random.default_rng(1337)
    noisy_blocker = rng.integers(20, 230, size=(95, 90, 3), dtype=np.uint8)
    frame[125:220, 135:225] = noisy_blocker
    cv2.rectangle(frame, (142, 132), (218, 214), (20, 20, 20), 2)
    config = {
        "screen": {"reference_width": 360, "reference_height": 240, "scale_regions": False},
        "screen_objects": {
            "navigation_region": {"x": 0, "y": 0, "width": 360, "height": 240},
            "walkability": {
                "enabled": True,
                "columns": 9,
                "lower_fraction": 0.45,
                "blocker_threshold": 0.36,
                "free_threshold": 0.24,
            },
        },
    }

    detections = detect_walkability_candidates(frame, config)
    labels = {detection.label for detection in detections}
    blockers = [detection for detection in detections if detection.label == "movement_blocker"]

    assert "walkable_corridor" in labels
    assert blockers
    assert any(blocker.bbox.x <= 180 <= blocker.bbox.x2 for blocker in blockers)


def test_detect_hostile_threat_finds_top_left_target_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (155, 34), (330, 46), (0, 0, 255), -1)

    threat = detect_hostile_threat(frame, _threat_test_config(), fallback_turn_key="A")

    assert threat is not None
    assert threat.reason == "hostile_target_frame"
    assert threat.turn_key == "A"
    assert threat.bbox.x <= 155


def test_detect_hostile_threat_finds_central_nameplate_and_turns_away():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 0, 255), -1)

    threat = detect_hostile_threat(frame, _threat_test_config(), fallback_turn_key="A")

    assert threat is not None
    assert threat.reason == "aggressive_nameplate"
    assert threat.turn_key == "D"


def test_detect_hostile_threat_ignores_sparse_text_like_name_overlay():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 204), (0, 0, 255), 2)
    cv2.line(frame, (490, 197), (605, 197), (120, 0, 255), 1)

    threat = detect_hostile_threat(frame, _threat_test_config(), fallback_turn_key="D")

    assert threat is None


def test_detect_hostile_threat_ignores_sparse_combat_warning_noise():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (880, 450), (1030, 464), (0, 0, 255), 2)
    cv2.line(frame, (900, 457), (1015, 457), (120, 0, 255), 1)
    config = _threat_test_config()
    avoidance = config["safety"]["hostile_avoidance"]
    avoidance["combat_warning_enabled"] = True
    avoidance["combat_warning_region"] = {"x": 820, "y": 420, "width": 360, "height": 160}
    avoidance["combat_warning_min_red_area"] = 50
    avoidance["combat_warning_component_min_area"] = 50
    avoidance["combat_warning_min_fill_ratio"] = 0.45

    threat = detect_hostile_threat(frame, config, fallback_turn_key="D")

    assert threat is None


def test_detect_hostile_threat_ignores_neutral_nameplate_red_level_fragment():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (470, 190), (620, 202), (0, 220, 220), -1)
    cv2.rectangle(frame, (640, 192), (664, 200), (0, 0, 255), -1)

    threat = detect_hostile_threat(frame, _threat_test_config(), fallback_turn_key="D")

    assert threat is None


def test_detect_hostile_threat_ignores_bottom_action_bar_and_right_ui():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (560, 650), (760, 664), (0, 0, 255), -1)
    cv2.rectangle(frame, (1100, 90), (1230, 104), (0, 0, 255), -1)

    threat = detect_hostile_threat(frame, _threat_test_config(), fallback_turn_key="D")

    assert threat is None


def _threat_test_config():
    return {
        "screen": {"reference_width": 1280, "reference_height": 720, "scale_regions": False},
        "safety": {
            "hostile_avoidance": {
                "enabled": True,
                "target_frame_region": {"x": 110, "y": 0, "width": 300, "height": 110},
                "central_threat_region": {"x": 260, "y": 90, "width": 660, "height": 340},
                "combat_warning_enabled": False,
                "target_frame_min_area": 50,
                "target_frame_min_width": 28,
                "target_frame_min_fill_ratio": 0.55,
                "nameplate_min_area": 50,
                "nameplate_min_width": 40,
                "nameplate_min_fill_ratio": 0.60,
                "ignore_bottom_ui_fraction": 0.72,
                "ignore_right_ui_fraction": 0.78,
            }
        },
    }
