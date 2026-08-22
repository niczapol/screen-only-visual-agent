from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.config import resource_path


_TEMPLATE_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class OreTemplateSpec:
    path: Path
    reference_width: float
    reference_height: float


def _load_image(image_source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_source, np.ndarray):
        return image_source

    image_path = Path(image_source)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def recognize_ore_points(
    image_source: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    bright, _dark = recognize_ore_point_classes(image_source, config)
    return bright


def recognize_ore_point_classes(
    image_source: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return surface and dim/underground ore icons from one template pass."""
    image = _load_image(image_source)
    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    if bool(rec_cfg.get("bright_template_enabled", False)):
        template_points = recognize_bright_ore_template_points(image, cfg)
        if template_points or not bool(rec_cfg.get("bright_template_fallback_hsv", False)):
            dark_cfg = rec_cfg.get("dark_icon", {})
            if bool(dark_cfg.get("template_classification_enabled", True)):
                return _classify_template_points_by_luminance(
                    image,
                    template_points,
                    rec_cfg,
                )
            return template_points, []

    bright = _recognize_points_from_mask(image, build_ore_mask(image, cfg), cfg)
    dark = _recognize_generic_dark_points(image, cfg)
    return bright, dark


def recognize_bright_ore_template_points(
    image_source: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    image = _load_image(image_source)
    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    if not bool(rec_cfg.get("bright_template_enabled", False)):
        return []
    specs = _bright_template_specs(rec_cfg)
    if not specs:
        return []

    scale_offsets = rec_cfg.get("bright_template_scale_offsets", [0.94, 1.0, 1.06])
    threshold = float(rec_cfg.get("bright_template_threshold", 0.60))
    matches: list[tuple[float, int, int, int, int]] = []
    image_feature = _ore_icon_feature(image)
    nominal_scales: list[float] = []
    for spec in specs:
        template = _load_template(spec.path)
        nominal_scale = min(
            image.shape[1] / spec.reference_width,
            image.shape[0] / spec.reference_height,
        )
        nominal_scales.append(nominal_scale)
        for offset in scale_offsets:
            scale = nominal_scale * float(offset)
            width = max(5, int(round(template.shape[1] * scale)))
            height = max(5, int(round(template.shape[0] * scale)))
            if width > image.shape[1] or height > image.shape[0]:
                continue
            resized = cv2.resize(
                template,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            template_feature = _ore_icon_feature(resized)
            result = cv2.matchTemplate(
                image_feature,
                template_feature,
                cv2.TM_CCOEFF_NORMED,
            )
            ys, xs = np.where(result >= threshold)
            for x, y in zip(xs.tolist(), ys.tolist()):
                center_x = int(x + width // 2)
                center_y = int(y + height // 2)
                if not _inside_minimap_sensor_circle(
                    center_x,
                    center_y,
                    image.shape,
                    rec_cfg,
                ):
                    continue
                if _inside_recognition_ignore_region(
                    center_x,
                    center_y,
                    image.shape,
                    cfg,
                ):
                    continue
                if not _bright_icon_shape_is_compact(
                    image,
                    (center_x, center_y),
                    rec_cfg,
                ):
                    continue
                matches.append(
                    (float(result[y, x]), center_x, center_y, width, height)
                )

    nominal_scale = min(nominal_scales) if nominal_scales else 1.0
    min_separation = max(
        2.0,
        float(rec_cfg.get("bright_template_nms_radius", 10.0)) * nominal_scale,
    )
    selected: list[tuple[float, int, int, int, int]] = []
    for match in sorted(matches, reverse=True):
        _score, x, y, _width, _height = match
        if any((x - sx) ** 2 + (y - sy) ** 2 < min_separation**2 for _s, sx, sy, _w, _h in selected):
            continue
        selected.append(match)
    return [(x, y) for _score, x, y, _width, _height in selected]


def _load_template(path: Path) -> np.ndarray:
    key = str(path.resolve())
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Unable to read bright ore template: {path}")
    _TEMPLATE_CACHE[key] = template
    return template


def _ore_icon_feature(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([18, 160, 150], dtype=np.uint8),
        np.array([36, 255, 255], dtype=np.uint8),
    )


def _bright_icon_shape_is_compact(
    image: np.ndarray,
    point: tuple[int, int],
    rec_cfg: dict[str, Any],
) -> bool:
    """Reject tall yellow UI glyphs that match the lossy ore color template."""
    if not bool(rec_cfg.get("bright_icon_shape_filter_enabled", True)):
        return True
    radius = max(3, int(rec_cfg.get("bright_icon_shape_radius", 12)))
    min_pixels = max(1, int(rec_cfg.get("bright_icon_shape_min_pixels", 8)))
    min_aspect = max(0.01, float(rec_cfg.get("bright_icon_min_component_aspect", 0.72)))
    max_aspect = max(min_aspect, float(rec_cfg.get("bright_icon_max_component_aspect", 1.45)))
    x, y = point
    height, width = image.shape[:2]
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    feature = _ore_icon_feature(image[top:bottom, left:right])
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature,
        connectivity=8,
    )
    candidates = [
        stats[index]
        for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    if not candidates:
        return False
    component = max(candidates, key=lambda item: int(item[cv2.CC_STAT_AREA]))
    component_width = max(1, int(component[cv2.CC_STAT_WIDTH]))
    component_height = max(1, int(component[cv2.CC_STAT_HEIGHT]))
    aspect = component_width / component_height
    return min_aspect <= aspect <= max_aspect


def _classify_template_points_by_luminance(
    image: np.ndarray,
    points: list[tuple[int, int]],
    rec_cfg: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    radius = max(2, int(rec_cfg.get("bright_icon_luminance_radius", 7)))
    min_yellow_pixels = max(1, int(rec_cfg.get("bright_icon_min_yellow_pixels", 20)))
    min_value = float(rec_cfg.get("bright_icon_min_yellow_value", 175.0))
    bright: list[tuple[int, int]] = []
    dark: list[tuple[int, int]] = []
    height, width = image.shape[:2]

    for x, y in points:
        left = max(0, x - radius)
        right = min(width, x + radius + 1)
        top = max(0, y - radius)
        bottom = min(height, y + radius + 1)
        patch = hsv[top:bottom, left:right]
        yellow = cv2.inRange(
            patch,
            np.array([15, 70, 30], dtype=np.uint8),
            np.array([45, 255, 255], dtype=np.uint8),
        )
        values = patch[:, :, 2][yellow > 0]
        # Weak color evidence must never create a permanent underground exclusion.
        if values.size < min_yellow_pixels or float(np.median(values)) >= min_value:
            bright.append((x, y))
        else:
            dark.append((x, y))
    return bright, dark


def _bright_template_specs(rec_cfg: dict[str, Any]) -> list[OreTemplateSpec]:
    raw_specs = rec_cfg.get("bright_templates")
    if isinstance(raw_specs, list):
        specs = [
            spec
            for value in raw_specs
            if isinstance(value, dict)
            and (spec := _parse_bright_template_spec(value)) is not None
        ]
        if specs:
            return specs

    path_value = rec_cfg.get("bright_template_path")
    if not path_value:
        return []
    reference = rec_cfg.get("bright_template_reference_minimap", {})
    return [
        OreTemplateSpec(
            path=resource_path(str(path_value)),
            reference_width=max(1.0, float(reference.get("width", 291))),
            reference_height=max(1.0, float(reference.get("height", 275))),
        )
    ]


def _parse_bright_template_spec(value: dict[str, Any]) -> OreTemplateSpec | None:
    path_value = value.get("path")
    if not path_value:
        return None
    reference = value.get("reference_minimap", {})
    if not isinstance(reference, dict):
        reference = {}
    return OreTemplateSpec(
        path=resource_path(str(path_value)),
        reference_width=max(1.0, float(reference.get("width", 293))),
        reference_height=max(1.0, float(reference.get("height", 278))),
    )


def _inside_recognition_ignore_region(
    x: int,
    y: int,
    image_shape: tuple[int, ...],
    config: dict[str, Any],
) -> bool:
    scale_x, scale_y = _recognition_scales_for_shape(image_shape, config)
    for region in config.get("recognition", {}).get("ignore_regions", []):
        left = int(round(float(region.get("x", 0)) * scale_x))
        top = int(round(float(region.get("y", 0)) * scale_y))
        width = int(round(float(region.get("width", 0)) * scale_x))
        height = int(round(float(region.get("height", 0)) * scale_y))
        if left <= x < left + width and top <= y < top + height:
            return True
    return False


def _inside_minimap_sensor_circle(
    x: int,
    y: int,
    image_shape: tuple[int, ...],
    rec_cfg: dict[str, Any],
) -> bool:
    height, width = image_shape[:2]
    center_cfg = rec_cfg.get("sensor_circle_center_fraction", {"x": 0.50, "y": 0.48})
    center_x = width * float(center_cfg.get("x", 0.50))
    center_y = height * float(center_cfg.get("y", 0.48))
    radius = min(width, height) * float(rec_cfg.get("sensor_circle_radius_fraction", 0.43))
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def recognize_dark_ore_points(
    image_source: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    image = _load_image(image_source)
    rec_cfg = (config or {}).get("recognition", {})
    if bool(rec_cfg.get("bright_template_enabled", False)):
        _bright, dark = recognize_ore_point_classes(image, config)
        return dark
    return _recognize_generic_dark_points(image, config)


def _recognize_generic_dark_points(
    image: np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    points = _recognize_points_from_mask(image, build_dark_ore_mask(image, config), config)
    rec_cfg = (config or {}).get("recognition", {})
    dark_cfg = rec_cfg.get("dark_icon", {})
    scale_x, scale_y = _recognition_scales(image, config or {})
    edge_margin = int(round(float(dark_cfg.get("edge_margin", 12)) * min(scale_x, scale_y)))
    if edge_margin <= 0:
        return points

    height, width = image.shape[:2]
    return [
        (x, y)
        for x, y in points
        if edge_margin <= x <= width - edge_margin and edge_margin <= y <= height - edge_margin
    ]


def _recognize_points_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    scale_x, scale_y = _recognition_scales(image, cfg)
    area_scale = scale_x * scale_y
    min_area = max(1, int(rec_cfg.get("min_area", 8)))
    max_area = max(min_area, int(round(float(rec_cfg.get("max_area", 160)) * area_scale)))
    min_width = max(1, int(rec_cfg.get("min_width", 3)))
    max_width = max(min_width, int(round(float(rec_cfg.get("max_width", 16)) * scale_x)))
    min_height = max(1, int(rec_cfg.get("min_height", 3)))
    max_height = max(min_height, int(round(float(rec_cfg.get("max_height", 16)) * scale_y)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points: list[tuple[int, int]] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if (
            min_area <= area <= max_area
            and min_width <= width <= max_width
            and min_height <= height <= max_height
        ):
            center_x = int(x + width // 2)
            center_y = int(y + height // 2)
            points.append((center_x, center_y))

    return points


def build_ore_mask(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    hsv_lower = np.array(rec_cfg.get("hsv_lower", [22, 120, 130]), dtype=np.uint8)
    hsv_upper = np.array(rec_cfg.get("hsv_upper", [38, 255, 255]), dtype=np.uint8)
    close_kernel_size = int(rec_cfg.get("close_kernel", 1))
    open_kernel_size = int(rec_cfg.get("open_kernel", 1))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if open_kernel_size > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    scale_x, scale_y = _recognition_scales(image, cfg)
    for ignore_region in rec_cfg.get("ignore_regions", []):
        x = int(round(float(ignore_region.get("x", 0)) * scale_x))
        y = int(round(float(ignore_region.get("y", 0)) * scale_y))
        width = int(round(float(ignore_region.get("width", 0)) * scale_x))
        height = int(round(float(ignore_region.get("height", 0)) * scale_y))
        if width <= 0 or height <= 0:
            continue
        mask[y : y + height, x : x + width] = 0

    return mask


def build_dark_ore_mask(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    dark_cfg = rec_cfg.get("dark_icon", {})
    hsv_lower = np.array(dark_cfg.get("hsv_lower", [18, 70, 30]), dtype=np.uint8)
    hsv_upper = np.array(dark_cfg.get("hsv_upper", [45, 255, 129]), dtype=np.uint8)
    close_kernel_size = int(dark_cfg.get("close_kernel", rec_cfg.get("close_kernel", 1)))
    open_kernel_size = int(dark_cfg.get("open_kernel", rec_cfg.get("open_kernel", 1)))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if open_kernel_size > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    scale_x, scale_y = _recognition_scales(image, cfg)
    for ignore_region in rec_cfg.get("ignore_regions", []):
        x = int(round(float(ignore_region.get("x", 0)) * scale_x))
        y = int(round(float(ignore_region.get("y", 0)) * scale_y))
        width = int(round(float(ignore_region.get("width", 0)) * scale_x))
        height = int(round(float(ignore_region.get("height", 0)) * scale_y))
        if width <= 0 or height <= 0:
            continue
        mask[y : y + height, x : x + width] = 0

    return mask


def _recognition_scales(image: np.ndarray, config: dict[str, Any]) -> tuple[float, float]:
    return _recognition_scales_for_shape(image.shape, config)


def _recognition_scales_for_shape(
    image_shape: tuple[int, ...],
    config: dict[str, Any],
) -> tuple[float, float]:
    minimap_cfg = config.get("minimap", {})
    reference_width = max(1.0, float(minimap_cfg.get("width", image_shape[1])))
    reference_height = max(1.0, float(minimap_cfg.get("height", image_shape[0])))
    return image_shape[1] / reference_width, image_shape[0] / reference_height


def save_debug_artifacts(
    image: np.ndarray,
    points: list[tuple[int, int]],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original_path = output_path / "original.png"
    processed_path = output_path / "processed.png"
    mask_path = output_path / "mask.png"
    dark_mask_path = output_path / "dark_mask.png"
    debug_path = output_path / "debug_overlay.png"

    cv2.imwrite(str(original_path), image)

    cfg = config or {}
    rec_cfg = cfg.get("recognition", {})
    min_area = int(rec_cfg.get("min_area", 8))
    max_area = int(rec_cfg.get("max_area", 160))
    min_width = int(rec_cfg.get("min_width", 3))
    max_width = int(rec_cfg.get("max_width", 16))
    min_height = int(rec_cfg.get("min_height", 3))
    max_height = int(rec_cfg.get("max_height", 16))
    mask = build_ore_mask(image, cfg)
    dark_mask = build_dark_ore_mask(image, cfg)

    processed = image.copy()
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    cv2.circle(processed, (center_x, center_y), 5, (255, 255, 255), 2)
    cv2.putText(processed, "center", (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if area <= 0:
            continue
        color = (0, 255, 0) if (
            min_area <= area <= max_area
            and min_width <= w <= max_width
            and min_height <= h <= max_height
        ) else (0, 128, 255)
        cv2.rectangle(processed, (x, y), (x + w, y + h), color, 1)

    for x, y in points:
        cv2.circle(processed, (x, y), 4, (0, 0, 255), 2)

    cv2.imwrite(str(processed_path), processed)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(dark_mask_path), dark_mask)
    cv2.imwrite(str(debug_path), processed)

    return {
        "original": original_path,
        "processed": processed_path,
        "mask": mask_path,
        "dark_mask": dark_mask_path,
        "debug": debug_path,
    }
