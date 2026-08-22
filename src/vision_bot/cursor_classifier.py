from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


CURSOR_TEMPLATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CursorSnapshot:
    handle: int | None
    image_bgra: np.ndarray | None
    visual_hash: int | None
    hash_size: int = 16
    colorfulness: float | None = None


@dataclass(frozen=True)
class CursorClassification:
    label: str | None
    similarity: float
    visual_hash: int | None
    handle: int | None
    calibrated: bool


def perceptual_cursor_hash(image: np.ndarray, *, hash_size: int = 16) -> int | None:
    """Hash the visible cursor shape after removing transparent/empty margins."""

    if image.size == 0 or hash_size < 2:
        return None
    if image.ndim == 2:
        gray = image
        visible = image > 0
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        visible = (image[:, :, 3] > 0) | np.any(image[:, :, :3] > 0, axis=2)
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        visible = np.any(image[:, :, :3] > 0, axis=2)

    ys, xs = np.where(visible)
    if not len(xs):
        return None
    cropped = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    normalized = cv2.resize(cropped, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    threshold = float(np.mean(normalized))
    bits = normalized >= threshold
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def cursor_hash_similarity(first: int, second: int, *, hash_size: int = 16) -> float:
    bit_count = hash_size * hash_size
    distance = (int(first) ^ int(second)).bit_count()
    return max(0.0, 1.0 - distance / bit_count)


def cursor_colorfulness(image: np.ndarray) -> float | None:
    """Return mean HSV saturation over visible cursor pixels in the 0..1 range."""

    if image.size == 0 or image.ndim != 3 or image.shape[2] < 3:
        return None
    color = image[:, :, :3]
    if image.shape[2] >= 4 and np.any(image[:, :, 3] > 0):
        visible = image[:, :, 3] > 0
    else:
        visible = np.any(color > 0, axis=2)
    if not np.any(visible):
        return None
    saturation = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.mean(saturation[visible]) / 255.0)


class CursorTemplateClassifier:
    """Classify a Win32 cursor bitmap against reviewed perceptual-hash templates."""

    def __init__(
        self,
        templates: dict[str, tuple[int, ...]] | None = None,
        *,
        template_colorfulness: dict[str, dict[int, float]] | None = None,
        hash_size: int = 16,
        min_similarity: float = 0.90,
        max_colorfulness_distance: float = 0.12,
    ) -> None:
        self.templates = templates or {}
        self.template_colorfulness = template_colorfulness or {}
        self.hash_size = max(2, int(hash_size))
        self.min_similarity = min(1.0, max(0.0, float(min_similarity)))
        self.max_colorfulness_distance = min(
            1.0,
            max(0.0, float(max_colorfulness_distance)),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        min_similarity: float | None = None,
    ) -> CursorTemplateClassifier:
        template_path = Path(path)
        if not template_path.exists():
            return cls(min_similarity=min_similarity if min_similarity is not None else 0.90)
        raw = json.loads(template_path.read_text(encoding="utf-8"))
        if int(raw.get("schema_version", 0)) != CURSOR_TEMPLATE_SCHEMA_VERSION:
            raise ValueError("Unsupported cursor template schema")
        hash_size = max(2, int(raw.get("hash_size", 16)))
        templates: dict[str, tuple[int, ...]] = {}
        template_colorfulness: dict[str, dict[int, float]] = {}
        for label, entries in raw.get("labels", {}).items():
            parsed: list[int] = []
            parsed_colorfulness: dict[int, float] = {}
            for entry in entries or []:
                if not isinstance(entry, dict) or entry.get("reviewed") is not True:
                    continue
                value = entry.get("hash")
                if isinstance(value, str):
                    parsed_value = int(value, 16)
                elif isinstance(value, int):
                    parsed_value = value
                else:
                    continue
                parsed.append(parsed_value)
                colorfulness = entry.get("colorfulness")
                if isinstance(colorfulness, (int, float)):
                    parsed_colorfulness[parsed_value] = min(
                        1.0,
                        max(0.0, float(colorfulness)),
                    )
            templates[str(label)] = tuple(parsed)
            template_colorfulness[str(label)] = parsed_colorfulness
        threshold = (
            float(min_similarity)
            if min_similarity is not None
            else float(raw.get("min_similarity", 0.90))
        )
        return cls(
            templates,
            template_colorfulness=template_colorfulness,
            hash_size=hash_size,
            min_similarity=threshold,
            max_colorfulness_distance=float(
                raw.get("max_colorfulness_distance", 0.12)
            ),
        )

    def has_templates(self, label: str | None = None) -> bool:
        if label is None:
            return any(self.templates.values())
        return bool(self.templates.get(label, ()))

    def classify(self, snapshot: CursorSnapshot | None) -> CursorClassification:
        visual_hash = snapshot.visual_hash if snapshot is not None else None
        handle = snapshot.handle if snapshot is not None else None
        colorfulness = snapshot.colorfulness if snapshot is not None else None
        if colorfulness is None and snapshot is not None and snapshot.image_bgra is not None:
            colorfulness = cursor_colorfulness(snapshot.image_bgra)
        if visual_hash is None or not self.has_templates():
            return CursorClassification(None, 0.0, visual_hash, handle, self.has_templates())

        best_label: str | None = None
        best_similarity = 0.0
        for label, hashes in self.templates.items():
            for template_hash in hashes:
                template_colorfulness = self.template_colorfulness.get(label, {}).get(
                    template_hash
                )
                if (
                    colorfulness is not None
                    and template_colorfulness is not None
                    and abs(colorfulness - template_colorfulness)
                    > self.max_colorfulness_distance
                ):
                    continue
                similarity = cursor_hash_similarity(
                    visual_hash,
                    template_hash,
                    hash_size=self.hash_size,
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = label
        if best_similarity < self.min_similarity:
            best_label = None
        return CursorClassification(
            best_label,
            best_similarity,
            visual_hash,
            handle,
            True,
        )


def capture_win32_cursor_snapshot(*, hash_size: int = 16) -> CursorSnapshot | None:
    """Capture the public Win32 cursor image; no game process memory is accessed."""

    if os.name != "nt":
        return None
    try:
        return _capture_win32_cursor_snapshot(hash_size=hash_size)
    except (AttributeError, OSError, ValueError, ctypes.ArgumentError):
        return None


def _capture_win32_cursor_snapshot(*, hash_size: int) -> CursorSnapshot | None:
    from ctypes import wintypes

    class CursorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("hCursor", wintypes.HANDLE),
            ("ptScreenPos", wintypes.POINT),
        ]

    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BitmapInfo(ctypes.Structure):
        _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    info = CursorInfo()
    info.cbSize = ctypes.sizeof(CursorInfo)
    if not user32.GetCursorInfo(ctypes.byref(info)) or not info.hCursor:
        return None

    width = max(32, int(user32.GetSystemMetrics(13)))
    height = max(32, int(user32.GetSystemMetrics(14)))
    screen_dc = user32.GetDC(None)
    memory_dc = gdi32.CreateCompatibleDC(screen_dc)
    bitmap_info = BitmapInfo()
    bitmap_info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
    bitmap_info.bmiHeader.biWidth = width
    bitmap_info.bmiHeader.biHeight = -height
    bitmap_info.bmiHeader.biPlanes = 1
    bitmap_info.bmiHeader.biBitCount = 32
    bitmap_info.bmiHeader.biCompression = 0
    bits = ctypes.c_void_p()
    bitmap = gdi32.CreateDIBSection(
        memory_dc,
        ctypes.byref(bitmap_info),
        0,
        ctypes.byref(bits),
        None,
        0,
    )
    if not screen_dc or not memory_dc or not bitmap or not bits.value:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        if screen_dc:
            user32.ReleaseDC(None, screen_dc)
        return None

    old_bitmap = gdi32.SelectObject(memory_dc, bitmap)
    try:
        ctypes.memset(bits, 0, width * height * 4)
        if not user32.DrawIconEx(memory_dc, 0, 0, info.hCursor, width, height, 0, None, 0x0003):
            return CursorSnapshot(int(info.hCursor), None, None, hash_size)
        raw = ctypes.string_at(bits, width * height * 4)
        image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4)).copy()
        return CursorSnapshot(
            int(info.hCursor),
            image,
            perceptual_cursor_hash(image, hash_size=hash_size),
            hash_size,
            cursor_colorfulness(image),
        )
    finally:
        gdi32.SelectObject(memory_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)


def save_cursor_calibration_sample(
    output_dir: str | Path,
    *,
    label: str,
    snapshot_reader: Callable[[], CursorSnapshot | None] = capture_win32_cursor_snapshot,
) -> dict[str, Any]:
    snapshot = snapshot_reader()
    if snapshot is None or snapshot.image_bgra is None or snapshot.visual_hash is None:
        raise RuntimeError("No visible Win32 cursor bitmap could be captured")
    safe_label = "".join(character for character in label.lower() if character.isalnum() or character in "-_")
    if not safe_label:
        raise ValueError("Cursor calibration label is empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    stem = f"{safe_label}_{timestamp}"
    image_path = destination / f"{stem}.png"
    metadata_path = destination / f"{stem}.json"
    if not cv2.imwrite(str(image_path), snapshot.image_bgra):
        raise OSError(f"Could not write cursor image: {image_path}")
    metadata = {
        "schema_version": CURSOR_TEMPLATE_SCHEMA_VERSION,
        "label": safe_label,
        "hash_size": snapshot.hash_size,
        "hash": format(snapshot.visual_hash, "x"),
        "colorfulness": (
            snapshot.colorfulness
            if snapshot.colorfulness is not None
            else cursor_colorfulness(snapshot.image_bgra)
        ),
        "handle": snapshot.handle,
        "image": image_path.name,
        "reviewed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def promote_cursor_calibration_sample(
    sample_path: str | Path,
    manifest_path: str | Path,
) -> bool:
    """Promote one explicitly reviewed sample; return False when already present."""

    sample_file = Path(sample_path)
    manifest_file = Path(manifest_path)
    sample = json.loads(sample_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if int(sample.get("schema_version", 0)) != CURSOR_TEMPLATE_SCHEMA_VERSION:
        raise ValueError("Unsupported cursor sample schema")
    if int(manifest.get("schema_version", 0)) != CURSOR_TEMPLATE_SCHEMA_VERSION:
        raise ValueError("Unsupported cursor template schema")
    if int(sample.get("hash_size", 0)) != int(manifest.get("hash_size", 0)):
        raise ValueError("Cursor sample and manifest hash sizes differ")
    label = str(sample.get("label", "")).strip()
    hash_value = str(sample.get("hash", "")).strip().lower()
    if not label or not hash_value:
        raise ValueError("Cursor sample is missing label or hash")
    int(hash_value, 16)
    labels = manifest.setdefault("labels", {})
    entries = labels.setdefault(label, [])
    existing_hashes = {
        str(entry.get("hash") if isinstance(entry, dict) else entry).lower()
        for entry in entries
    }
    if hash_value in existing_hashes:
        return False
    entries.append(
        {
            "hash": hash_value,
            "colorfulness": sample.get("colorfulness"),
            "source": sample_file.name,
            "reviewed": True,
        }
    )
    manifest["status"] = "calibrated"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return True
