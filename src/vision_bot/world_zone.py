from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.regions import crop_region


DEFAULT_ZONE_MAPPINGS = {
    "durotar": [4],
    "echo isles": [4],
    "razor hill": [4],
    "razormane grounds": [4],
    "ratchet": [11],
    "senjin village": [4],
    "southfury river": [4],
    "the barrens": [11],
    "the crossroads": [11],
    "valley of trials": [4],
}


def read_world_zone_name(frame: np.ndarray, config: dict[str, Any] | None = None) -> str | None:
    cfg = config or {}
    zone_cfg = cfg.get("world_zone", {})
    region_cfg = zone_cfg.get("region", {"x": 2280, "y": 4, "width": 250, "height": 34})
    region = crop_region(frame, region_cfg, cfg)
    if region.size == 0:
        return None

    texts = _extract_zone_text_candidates(region, cfg)
    return texts[0] if texts else None


def resolve_world_zone_ids(zone_name: str | None, config: dict[str, Any] | None = None) -> list[int]:
    if not zone_name:
        return []

    cfg = config or {}
    mappings = dict(DEFAULT_ZONE_MAPPINGS)
    for key, value in cfg.get("world_zone", {}).get("zone_mappings", {}).items():
        if isinstance(value, int):
            mappings[_normalize_zone_name(str(key))] = [value]
        elif isinstance(value, list):
            mappings[_normalize_zone_name(str(key))] = [int(item) for item in value]

    normalized = _normalize_zone_name(zone_name)
    if normalized in mappings:
        return mappings[normalized]

    compact_normalized = normalized.replace(" ", "")
    for key, value in mappings.items():
        compact_key = key.replace(" ", "")
        if compact_key and compact_normalized == compact_key:
            return value

    for key, value in mappings.items():
        if key and (key in normalized or normalized in key):
            return value
        compact_key = key.replace(" ", "")
        if compact_key and (compact_key in compact_normalized or compact_normalized in compact_key):
            return value

    return []


def _extract_zone_text_candidates(region: np.ndarray, config: dict[str, Any]) -> list[str]:
    try:
        import pytesseract
    except ImportError:
        return []

    ocr_cfg = config.get("position_ocr", {})
    tesseract_cmd = str(ocr_cfg.get("tesseract_cmd", "")).strip()
    if tesseract_cmd:
        tesseract_path = Path(tesseract_cmd)
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

    zone_cfg = config.get("world_zone", {})
    scale = max(1, int(zone_cfg.get("scale", 3)))
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array(zone_cfg.get("hsv_lower", [35, 50, 80]), dtype=np.uint8),
        np.array(zone_cfg.get("hsv_upper", [95, 255, 255]), dtype=np.uint8),
    )

    variants: list[np.ndarray] = [region, green_mask]
    if len(region.shape) == 3:
        variants.append(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))

    texts: list[str] = []
    custom_config = "--oem 3 --psm 7"
    for variant in variants:
        interpolation = cv2.INTER_CUBIC if len(variant.shape) == 3 else cv2.INTER_NEAREST
        ocr_image = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=interpolation)
        try:
            text = pytesseract.image_to_string(ocr_image, config=custom_config).strip()
        except Exception:
            continue
        cleaned = _clean_zone_text(text)
        if cleaned and cleaned not in texts:
            texts.append(cleaned)

    return texts


def _clean_zone_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|'")
    return text


def _normalize_zone_name(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
