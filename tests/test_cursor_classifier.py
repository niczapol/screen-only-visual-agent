import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_bot.cursor_classifier import (
    CursorSnapshot,
    CursorTemplateClassifier,
    cursor_colorfulness,
    cursor_hash_similarity,
    perceptual_cursor_hash,
    promote_cursor_calibration_sample,
    save_cursor_calibration_sample,
)


def _cursor_shape(offset=0):
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    image[5 + offset : 30 + offset, 10:14] = (220, 220, 220, 255)
    image[22 + offset : 27 + offset, 6:28] = (30, 170, 240, 255)
    return image


def test_cursor_hash_ignores_empty_margin_translation():
    first = perceptual_cursor_hash(_cursor_shape())
    second = perceptual_cursor_hash(_cursor_shape(3))

    assert first is not None
    assert first == second
    assert cursor_hash_similarity(first, second) == 1.0


def test_template_classifier_labels_matching_hash_and_rejects_unknown():
    mine_hash = perceptual_cursor_hash(_cursor_shape())
    classifier = CursorTemplateClassifier(
        {"mine": (mine_hash,), "loot": ()},
        min_similarity=0.95,
    )

    matched = classifier.classify(CursorSnapshot(7, _cursor_shape(), mine_hash))
    unknown = classifier.classify(CursorSnapshot(8, None, 0))

    assert matched.label == "mine"
    assert matched.calibrated
    assert matched.similarity == 1.0
    assert unknown.label is None


def test_template_classifier_uses_colorfulness_to_reject_grayscale_shape_match():
    colored = _cursor_shape()
    grayscale = colored.copy()
    gray = np.mean(grayscale[:, :, :3], axis=2).astype(np.uint8)
    grayscale[:, :, :3] = gray[:, :, None]
    shared_hash = perceptual_cursor_hash(colored)
    classifier = CursorTemplateClassifier(
        {"mine": (shared_hash,), "unable_mine": (shared_hash,)},
        template_colorfulness={
            "mine": {shared_hash: cursor_colorfulness(colored)},
            "unable_mine": {shared_hash: cursor_colorfulness(grayscale)},
        },
        min_similarity=0.95,
        max_colorfulness_distance=0.10,
    )

    colored_result = classifier.classify(
        CursorSnapshot(1, colored, shared_hash)
    )
    grayscale_result = classifier.classify(
        CursorSnapshot(2, grayscale, shared_hash)
    )

    assert colored_result.label == "mine"
    assert grayscale_result.label == "unable_mine"


def test_template_manifest_loads_reviewed_hashes(tmp_path):
    mine_hash = perceptual_cursor_hash(_cursor_shape())
    manifest = tmp_path / "cursor_templates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_size": 16,
                "min_similarity": 0.91,
                "labels": {
                    "mine": [{"hash": format(mine_hash, "x"), "reviewed": True}],
                    "loot": [],
                },
            }
        ),
        encoding="utf-8",
    )

    classifier = CursorTemplateClassifier.load(manifest)

    assert classifier.has_templates("mine")
    assert not classifier.has_templates("loot")
    assert classifier.classify(CursorSnapshot(1, None, mine_hash)).label == "mine"


def test_empty_manifest_is_uncalibrated(tmp_path):
    manifest = tmp_path / "cursor_templates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_size": 16,
                "labels": {"mine": [], "loot": []},
            }
        ),
        encoding="utf-8",
    )

    result = CursorTemplateClassifier.load(manifest).classify(
        CursorSnapshot(1, None, 123)
    )

    assert result.label is None
    assert not result.calibrated


def test_unreviewed_manifest_entry_is_ignored(tmp_path):
    manifest = tmp_path / "cursor_templates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_size": 16,
                "labels": {"mine": [{"hash": "abcd", "reviewed": False}]},
            }
        ),
        encoding="utf-8",
    )

    assert not CursorTemplateClassifier.load(manifest).has_templates("mine")


def test_calibration_sample_is_saved_unreviewed(tmp_path):
    image = _cursor_shape()
    visual_hash = perceptual_cursor_hash(image)

    metadata = save_cursor_calibration_sample(
        tmp_path,
        label="mine",
        snapshot_reader=lambda: CursorSnapshot(44, image, visual_hash),
    )

    assert metadata["label"] == "mine"
    assert metadata["reviewed"] is False
    assert list(tmp_path.glob("mine_*.png"))
    assert list(tmp_path.glob("mine_*.json"))


def test_reviewed_sample_promotion_is_explicit_and_idempotent(tmp_path):
    image = _cursor_shape()
    visual_hash = perceptual_cursor_hash(image)
    sample = tmp_path / "mine_sample.json"
    sample.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "label": "mine",
                "hash_size": 16,
                "hash": format(visual_hash, "x"),
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "templates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_size": 16,
                "labels": {"mine": [], "loot": []},
            }
        ),
        encoding="utf-8",
    )

    assert promote_cursor_calibration_sample(sample, manifest)
    assert not promote_cursor_calibration_sample(sample, manifest)
    classifier = CursorTemplateClassifier.load(manifest)
    assert classifier.classify(CursorSnapshot(1, None, visual_hash)).label == "mine"


def test_extracted_client_mine_and_unable_assets_are_separated():
    root = Path(__file__).resolve().parents[1] / "data" / "cursor_templates"
    if not (root / "cursor_templates.json").exists():
        pytest.skip("client-derived cursor fixtures are not distributed")
    classifier = CursorTemplateClassifier.load(root / "cursor_templates.json")

    for expected_label, names in {
        "mine": ("Mine.png", "Mine2.png", "Mine4.png"),
        "unable_mine": ("UnableMine.png", "UnableMine2.png", "UnableMine4.png"),
    }.items():
        for index, name in enumerate(names):
            image = cv2.imread(
                str(root / "client_assets" / name),
                cv2.IMREAD_UNCHANGED,
            )
            visual_hash = perceptual_cursor_hash(image)

            result = classifier.classify(
                CursorSnapshot(index, image, visual_hash)
            )

            assert result.label == expected_label
            assert result.similarity >= 0.99
