from types import SimpleNamespace

import numpy as np

from vision_bot.ore_world_detector import (
    _MODEL_CACHE,
    OreWorldDetection,
    OreWorldDetector,
    OreWorldDetectorResult,
    build_ore_detector_probe_points,
    deduplicate_ore_detections,
    filter_ore_detections_by_geometry,
    filter_ore_detections_by_self_region,
)


def test_disabled_detector_does_not_load_model():
    detector = OreWorldDetector(
        {"mining": {"ore_world_detector": {"enabled": False}}},
        model_loader=lambda _path: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result.reason == "disabled"
    assert result.detections == ()


def test_prewarm_can_be_disabled_without_loading_model():
    detector = OreWorldDetector(
        {
            "mining": {
                "ore_world_detector": {
                    "enabled": True,
                    "prewarm": False,
                }
            }
        },
        model_loader=lambda _path: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    result = detector.warm_up(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result.reason == "prewarm_disabled"


def test_detector_deduplicates_overlapping_predictions(tmp_path):
    model_path = tmp_path / "ore.pt"
    model_path.write_bytes(b"test")

    class FakeModel:
        def predict(self, **_kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=np.asarray(
                            [[10, 20, 80, 90], [12, 18, 82, 92], [120, 10, 180, 60]],
                            dtype=np.float32,
                        ),
                        conf=np.asarray([0.92, 0.70, 0.55], dtype=np.float32),
                    )
                )
            ]

    detector = OreWorldDetector(
        {
            "mining": {
                "ore_world_detector": {
                    "enabled": True,
                    "model_path": str(model_path),
                    "max_candidates": 3,
                    "max_area_fraction": 1.0,
                }
            }
        },
        model_loader=lambda _path: FakeModel(),
    )

    result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result.reason == "ok"
    assert len(result.detections) == 2
    assert result.detections[0].center == (45, 55)
    assert result.detections[1].center == (150, 35)


def test_detector_fails_closed_when_model_is_missing(tmp_path):
    detector = OreWorldDetector(
        {
            "mining": {
                "ore_world_detector": {
                    "enabled": True,
                    "model_path": str(tmp_path / "missing.pt"),
                }
            }
        }
    )

    assert detector.detect(np.zeros((20, 20, 3), dtype=np.uint8)).reason == "model_file_missing"


def test_detector_load_preflight_does_not_run_inference(tmp_path):
    model_path = tmp_path / "ore.pt"
    model_path.write_bytes(b"test")
    model = SimpleNamespace(predict=lambda **_kwargs: (_ for _ in ()).throw(AssertionError))
    detector = OreWorldDetector(
        {
            "mining": {
                "ore_world_detector": {
                    "enabled": True,
                    "model_path": str(model_path),
                }
            }
        },
        model_loader=lambda _path: model,
    )

    result = detector.load()

    assert result.reason == "model_ready"
    assert result.detections == ()


def test_detector_load_preflight_preserves_actionable_loader_error(tmp_path):
    model_path = tmp_path / "ore.pt"
    model_path.write_bytes(b"test")
    detector = OreWorldDetector(
        {
            "mining": {
                "ore_world_detector": {
                    "enabled": True,
                    "model_path": str(model_path),
                }
            }
        },
        model_loader=lambda _path: (_ for _ in ()).throw(
            RuntimeError("optional package is missing")
        ),
    )

    assert detector.load().reason == (
        "model_load_error:RuntimeError:optional_package_is_missing"
    )


def test_default_loader_cache_reuses_preflight_model(monkeypatch, tmp_path):
    model_path = tmp_path / "ore.pt"
    model_path.write_bytes(b"test")
    loaded = []
    model = object()
    monkeypatch.setattr(
        "vision_bot.ore_world_detector._load_ultralytics_model",
        lambda path: loaded.append(path) or model,
    )
    _MODEL_CACHE.clear()
    config = {
        "mining": {
            "ore_world_detector": {
                "enabled": True,
                "model_path": str(model_path),
            }
        }
    }

    assert OreWorldDetector(config).load().reason == "model_ready"
    assert OreWorldDetector(config).load().reason == "model_ready"
    assert loaded == [model_path.resolve()]
    _MODEL_CACHE.clear()


def test_probe_points_start_at_box_center_then_cover_horizontal_body():
    result = OreWorldDetectorResult(
        detections=(OreWorldDetection(100, 200, 200, 300, 0.9),),
        reason="ok",
    )
    config = {
        "mining": {
            "ore_world_detector": {"probe_offset_fractions": [0.0, -0.2, 0.2]}
        }
    }

    assert build_ore_detector_probe_points(result, (400, 500, 3), config) == [
        (150, 250),
        (130, 250),
        (170, 250),
    ]


def test_deduplication_keeps_highest_confidence_box():
    detections = [
        OreWorldDetection(0, 0, 100, 100, 0.4),
        OreWorldDetection(2, 2, 98, 98, 0.9),
    ]

    assert deduplicate_ore_detections(detections, iou_threshold=0.45) == [detections[1]]


def test_geometry_filter_rejects_tiny_ui_and_huge_terrain_boxes():
    detections = [
        OreWorldDetection(0, 0, 10, 10, 0.9),
        OreWorldDetection(100, 100, 200, 200, 0.8),
        OreWorldDetection(0, 0, 900, 900, 0.7),
    ]

    assert filter_ore_detections_by_geometry(
        detections,
        (1000, 1000, 3),
        min_area_fraction=0.001,
        max_area_fraction=0.05,
    ) == [detections[1]]


def test_self_region_filter_rejects_player_body_but_keeps_ore_above_it():
    player = OreWorldDetection(1200, 730, 1360, 900, 0.9)
    ore_above = OreWorldDetection(1200, 430, 1400, 650, 0.8)

    assert filter_ore_detections_by_self_region(
        [player, ore_above],
        (1440, 2560, 3),
        enabled=True,
        center_x_fraction=0.50,
        center_y_fraction=0.62,
        width_fraction=0.14,
        height_fraction=0.26,
    ) == [ore_above]
