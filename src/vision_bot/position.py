from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from .regions import crop_region
from .runtime_markers import read_runtime_telemetry


@dataclass(frozen=True)
class PlayerPoseObservation:
    coord_candidates: tuple[int, ...]
    heading_degrees: float | None
    source: str

    @property
    def coord(self) -> int | None:
        return self.coord_candidates[0] if self.coord_candidates else None


def read_player_position(frame: np.ndarray, config: dict[str, Any] | None = None, direct_region: bool = False) -> int | None:
    candidates = read_player_position_candidates(frame, config, direct_region)
    return candidates[0] if candidates else None


def read_player_position_candidates(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
    direct_region: bool = False,
) -> list[int]:
    return list(read_player_pose_observation(frame, config, direct_region).coord_candidates)


def read_player_pose_observation(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
    direct_region: bool = False,
) -> PlayerPoseObservation:
    cfg = config or {}
    if direct_region:
        return PlayerPoseObservation(
            coord_candidates=tuple(_extract_coordinate_candidates(frame, cfg)),
            heading_degrees=None,
            source="direct_ocr",
        )

    telemetry = read_runtime_telemetry(frame, cfg)
    if telemetry is not None:
        return PlayerPoseObservation(
            coord_candidates=(telemetry.coord,),
            heading_degrees=telemetry.heading_degrees,
            source="runtime_telemetry",
        )

    pos_cfg = cfg.get("player_position", {})
    regions = [pos_cfg, *list(pos_cfg.get("fallback_regions", []))]

    coords: list[int] = []
    for region_cfg in regions:
        region = crop_region(frame, region_cfg, cfg)
        if region.size == 0:
            continue
        region_coords = _extract_coordinate_candidates(region, cfg)
        for coord in region_coords:
            if coord not in coords:
                coords.append(coord)
        if region_coords and bool(
            cfg.get("position_ocr", {}).get("stop_after_first_valid_region", False)
        ):
            break

    return PlayerPoseObservation(
        coord_candidates=tuple(coords),
        heading_degrees=None,
        source="ocr" if coords else "unavailable",
    )


def _extract_text(region: np.ndarray, config: dict[str, Any] | None = None) -> str:
    texts = _extract_text_candidates(region, config)
    for text in texts:
        if _parse_coordinates(text) is not None:
            return text
    return texts[0] if texts else ""


def _extract_coordinate_candidates(region: np.ndarray, config: dict[str, Any] | None = None) -> list[int]:
    coords: list[int] = []
    for text in _extract_text_candidates(region, config):
        coord = _parse_coordinates(text)
        if coord is not None and coord not in coords:
            coords.append(coord)
    return coords


def _extract_text_candidates(region: np.ndarray, config: dict[str, Any] | None = None) -> list[str]:
    cfg = config or {}
    mask = _prepare_coordinate_mask(region, config)
    if mask.size == 0:
        return []

    template_text = _extract_text_with_templates(mask)
    template_coord = _parse_coordinates(template_text)
    if template_coord is not None and bool(
        cfg.get("position_ocr", {}).get("template_first_fast_path", False)
    ):
        return [template_text]

    tesseract_texts = _extract_tesseract_variant_texts(region, mask, cfg)
    tesseract_coord_texts = [text for text in tesseract_texts if _parse_coordinates(text) is not None]

    if tesseract_coord_texts:
        if template_coord is not None and template_coord in {
            _parse_coordinates(text) for text in tesseract_coord_texts
        }:
            return _unique_texts([template_text, *tesseract_coord_texts, *tesseract_texts])
        return _unique_texts([*tesseract_coord_texts, template_text, *tesseract_texts])

    return _unique_texts([template_text, *tesseract_texts])


def _unique_texts(texts: list[str]) -> list[str]:
    unique: list[str] = []
    for text in texts:
        if text and text not in unique:
            unique.append(text)
    return unique


def _extract_text_with_tesseract_variants(region: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> str:
    for text in _extract_tesseract_variant_texts(region, mask, config):
        if _parse_coordinates(text) is not None:
            return text
    return ""


def _extract_tesseract_variant_texts(region: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> list[str]:
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

    ocr_scale = int((config or {}).get("position_ocr", {}).get("scale", 5))
    ocr_scale = max(1, ocr_scale)
    custom_config = r"--oem 3 -c tessedit_char_whitelist=0123456789., "

    if bool(ocr_cfg.get("fast_grayscale_first", False)) and len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_image = cv2.resize(
            gray_mask,
            None,
            fx=ocr_scale,
            fy=ocr_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        try:
            text = pytesseract.image_to_string(
                ocr_image,
                config=f"{custom_config} --psm 7",
            ).strip()
        except Exception:
            text = ""
        if _parse_coordinates(text) is not None:
            return [text]

    variants: list[np.ndarray] = [mask]
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(gray_mask)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        bright_values = ocr_cfg.get("bright_fallback_values", [140, 150, 170])
        for bright_value in bright_values:
            bright_lower = list(ocr_cfg.get("bright_fallback_hsv_lower", [14, 50, 150]))
            while len(bright_lower) < 3:
                bright_lower.append(0)
            bright_lower[2] = int(bright_value)
            bright_yellow_mask = cv2.inRange(
                hsv,
                np.array(bright_lower, dtype=np.uint8),
                np.array(ocr_cfg.get("bright_fallback_hsv_upper", [55, 255, 255]), dtype=np.uint8),
            )
            variants.append(bright_yellow_mask)
            height = bright_yellow_mask.shape[0]
            for top_ratio in (0.43, 0.52):
                top = int(round(height * top_ratio))
                if 0 < top < height:
                    variants.append(bright_yellow_mask[top:, :])

        yellow_mask = cv2.inRange(
            hsv,
            np.array(ocr_cfg.get("fallback_hsv_lower", [14, 50, 120]), dtype=np.uint8),
            np.array(ocr_cfg.get("fallback_hsv_upper", [55, 255, 255]), dtype=np.uint8),
        )
        variants.append(yellow_mask)

        height = yellow_mask.shape[0]
        for top_ratio in (0.43, 0.52, 0.35, 0.26):
            top = int(round(height * top_ratio))
            if 0 < top < height:
                variants.append(yellow_mask[top:, :])

    texts: list[str] = []
    for psm in (7, 6):
        for variant in variants:
            if variant.size == 0:
                continue
            ocr_image = cv2.resize(
                variant,
                None,
                fx=ocr_scale,
                fy=ocr_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            try:
                text = pytesseract.image_to_string(
                    ocr_image,
                    config=f"{custom_config} --psm {psm}",
                ).strip()
            except Exception:
                continue
            if text and text not in texts:
                texts.append(text)
        if any(_parse_coordinates(text) is not None for text in texts):
            return texts

    return texts


def _prepare_coordinate_mask(region: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    ocr_cfg = cfg.get("position_ocr", {})
    min_area = int(ocr_cfg.get("min_area", 2))
    max_area = int(ocr_cfg.get("max_area", 120))
    min_height = int(ocr_cfg.get("min_height", 3))
    max_height = int(ocr_cfg.get("max_height", 20))
    max_width = int(ocr_cfg.get("max_width", 14))

    if len(region.shape) == 2:
        _, mask = cv2.threshold(region, 1, 255, cv2.THRESH_BINARY)
    else:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array(ocr_cfg.get("hsv_lower", [15, 40, 100]), dtype=np.uint8)
        hsv_upper = np.array(ocr_cfg.get("hsv_upper", [50, 255, 255]), dtype=np.uint8)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if min_area <= area <= max_area and min_height <= height <= max_height and width <= max_width:
            filtered[labels == label_index] = 255

    return filtered


def _parse_coordinates(text: str) -> int | None:
    if not text:
        return None

    normalized = text.replace(" ", "")
    matches = re.findall(r"\d{1,3}[.,]\d{1,2}", normalized)
    if len(matches) >= 2:
        parts = matches[:2]
    else:
        return None

    try:
        x = float(parts[0].replace(",", ".").replace(".00", "."))
        y = float(parts[1].replace(",", ".").replace(".00", "."))
    except ValueError:
        return None

    if not (0.0 <= x <= 100.0 and 0.0 <= y <= 100.0):
        return None

    return _to_coord_id(x, y)


def _to_coord_id(x: float, y: float) -> int:
    x_value = int(round(x * 100))
    y_value = int(round(y * 100))
    return int(f"{x_value:04d}{y_value:04d}00")


def _extract_text_with_templates(mask: np.ndarray) -> str:
    glyphs = _segment_glyphs(mask)
    if not glyphs:
        return ""

    chars = [_classify_glyph(glyph) for glyph in glyphs]
    return _normalize_coordinate_chars(chars)


def _normalize_coordinate_chars(chars: list[str]) -> str:
    chars = [char for char in chars if char in "0123456789.,"]
    if not chars:
        return ""

    while chars and chars[0] in ".,":
        chars.pop(0)
    while chars and chars[-1] in ".,":
        chars.pop()

    if not chars:
        return ""

    separator_index: int | None = None
    for index, char in enumerate(chars):
        if char not in ".,":
            continue
        digits_before = sum(item.isdigit() for item in chars[:index])
        digits_after = sum(item.isdigit() for item in chars[index + 1 :])
        if digits_before >= 3 and digits_after >= 3:
            separator_index = index
            break

    if separator_index is not None:
        left = chars[:separator_index]
        right = chars[separator_index + 1 :]
    else:
        digits = [char for char in chars if char.isdigit()]
        if len(digits) < 6:
            return "".join(chars)
        midpoint = len(digits) // 2
        left = digits[:midpoint]
        right = digits[midpoint:]

    left_text = _format_coordinate_group(left)
    right_text = _format_coordinate_group(right)
    if not left_text or not right_text:
        return "".join(chars)
    return f"{left_text},{right_text}"


def _format_coordinate_group(chars: list[str]) -> str:
    digits = "".join(char for char in chars if char.isdigit())
    if len(digits) < 3:
        return digits
    return f"{digits[:-2]}.{digits[-2:]}"


def _segment_glyphs(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int, np.ndarray]] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if area < 2:
            continue
        glyph = np.zeros((height, width), dtype=np.uint8)
        glyph[labels[y : y + height, x : x + width] == label_index] = 255
        components.append((int(x), int(y), glyph))

    return [glyph for _, _, glyph in sorted(components, key=lambda item: (item[0], item[1]))]


def _classify_glyph(glyph: np.ndarray) -> str:
    if glyph.shape[0] <= 5 and glyph.shape[1] <= 4:
        return "," if glyph.shape[0] >= 5 else "."

    best_char = "?"
    best_score = 1.0
    for char, template in _coordinate_templates().items():
        resized = cv2.resize(glyph, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = cv2.bitwise_xor(resized, template)
        score = float(np.count_nonzero(diff)) / float(template.size)
        if score < best_score:
            best_char = char
            best_score = score

    return best_char if best_score <= 0.36 else "?"


@lru_cache(maxsize=1)
def _coordinate_templates() -> dict[str, np.ndarray]:
    patterns = {
        "1": [
            "..##..",
            ".###..",
            "####..",
            "####..",
            "#.##..",
            "..##..",
            "..##..",
            "..##..",
            "..##..",
            "..##..",
            "######",
            "######",
        ],
        "0": [
            "..###..",
            ".#####.",
            ".##.###",
            "###..##",
            "###..##",
            "###..##",
            "###..##",
            "###..##",
            "###..##",
            "###.###",
            ".#####.",
            ".#####.",
        ],
        "2": [
            ".###..",
            "#####.",
            "##.##.",
            "...##.",
            "...##.",
            "...##.",
            "..###.",
            "..##..",
            ".###..",
            ".##...",
            "######",
            "######",
        ],
        "3": [
            "#####.",
            "######",
            "#####.",
            "..##..",
            ".###..",
            ".###..",
            ".#####",
            "...###",
            "...###",
            "...###",
            "#####.",
            "####..",
        ],
        "4": [
            "....##..",
            "...###..",
            "...###..",
            "..####..",
            "..####..",
            ".##.##..",
            ".##.##..",
            "########",
            "########",
            "....##..",
            "....##..",
            "....##..",
        ],
        "5": [
            "#####.",
            "######",
            "#####.",
            "##....",
            "##....",
            "#####.",
            "#####.",
            "...###",
            "...###",
            "...###",
            "#####.",
            "####..",
        ],
        "6": [
            ".####.",
            "#####.",
            "###...",
            "##....",
            "##....",
            "#####.",
            "######",
            "##..##",
            "##..##",
            "##..##",
            "#####.",
            ".####.",
        ],
        "7": [
            ".######",
            "#######",
            "#######",
            "....##.",
            "....##.",
            "...###.",
            "...##..",
            "..###..",
            "..###..",
            "..##...",
            ".###...",
            ".###...",
        ],
        "8": [
            "..###..",
            ".#####.",
            ".##.###",
            ".##..##",
            ".##.##.",
            ".#####.",
            "..####.",
            ".##.###",
            ".##..##",
            "###..##",
            ".######",
            ".#####.",
        ],
        "9": [
            ".####.",
            "######",
            "##..##",
            "##..##",
            "##..##",
            "######",
            ".#####",
            "...##.",
            "...##.",
            "..###.",
            "#####.",
            "####..",
        ],
    }
    return {char: _pattern_to_mask(pattern) for char, pattern in patterns.items()}


def _pattern_to_mask(pattern: list[str]) -> np.ndarray:
    rows = [[255 if char == "#" else 0 for char in row] for row in pattern]
    return np.array(rows, dtype=np.uint8)
