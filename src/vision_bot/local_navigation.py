from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from vision_bot.regions import resolve_region
from vision_bot.screen_objects import BoundingBox


SECTOR_NAMES = ("left", "slight_left", "center", "slight_right", "right")
DEFAULT_SOURCE_DIRS = (
    Path("data/vision_dataset/images/raw"),
    Path("data/vision_dataset/images/bootstrap"),
    Path("data/vision_dataset_walkability_live/images/bootstrap"),
    Path("data/vision_live_raw_walkability_20260728/images/raw"),
)
BLOCKED_OBSTACLE_SCORE = 0.64
BLOCKED_PASSABLE_SCORE = 0.46


@dataclass(frozen=True)
class NavigationSector:
    name: str
    index: int
    bbox: BoundingBox
    passable_score: float
    obstacle_score: float
    unknown_score: float
    confidence: float
    edge_density: float
    vertical_density: float
    texture_score: float
    bottom_edge_density: float
    valid_fraction: float

    @property
    def blocked(self) -> bool:
        return self.obstacle_score >= BLOCKED_OBSTACLE_SCORE and self.passable_score <= BLOCKED_PASSABLE_SCORE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "index": self.index,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "passable_score": self.passable_score,
            "obstacle_score": self.obstacle_score,
            "unknown_score": self.unknown_score,
            "confidence": self.confidence,
            "edge_density": self.edge_density,
            "vertical_density": self.vertical_density,
            "texture_score": self.texture_score,
            "bottom_edge_density": self.bottom_edge_density,
            "valid_fraction": self.valid_fraction,
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class NavigationFrame:
    region: BoundingBox
    analysis_region: BoundingBox
    self_ignore_region: BoundingBox | None
    sectors: tuple[NavigationSector, ...]
    best_sector: str | None
    center_blocked: bool
    recommended_turn: str | None
    confidence: float

    def center_sector(self) -> NavigationSector | None:
        return next((sector for sector in self.sectors if sector.name == "center"), None)

    def best(self) -> NavigationSector | None:
        if self.best_sector is None:
            return None
        return next((sector for sector in self.sectors if sector.name == self.best_sector), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": _bbox_to_dict(self.region),
            "analysis_region": _bbox_to_dict(self.analysis_region),
            "self_ignore_region": _bbox_to_dict(self.self_ignore_region),
            "sectors": [sector.to_dict() for sector in self.sectors],
            "best_sector": self.best_sector,
            "center_blocked": self.center_blocked,
            "recommended_turn": self.recommended_turn,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LocalNavigationDatasetSummary:
    output_dir: Path
    frame_count: int
    report_count: int
    source_count: int
    copied_image_count: int


@dataclass(frozen=True)
class LocalNavigationOutcomeDatasetSummary:
    output_dir: Path
    sample_count: int
    source_metadata_count: int
    episode_count: int
    label_counts: dict[str, int]
    split_counts: dict[str, int]
    skipped_count: int
    report_count: int
    copied_image_count: int


@dataclass(frozen=True)
class LocalNavigationReviewSummary:
    output_dir: Path
    item_count: int
    report_count: int
    bucket_counts: dict[str, int]


def analyze_local_navigation(frame: np.ndarray, config: dict[str, Any] | None = None) -> NavigationFrame:
    cfg = config or {}
    nav_cfg = cfg.get("local_navigation", {})
    region = _navigation_region(frame, cfg)
    roi = frame[region.y : region.y2, region.x : region.x2]
    if roi.size == 0:
        return NavigationFrame(region, region, None, tuple(), None, False, None, 0.0)

    analysis_region = _analysis_region(region, nav_cfg)
    local_analysis = BoundingBox(
        analysis_region.x - region.x,
        analysis_region.y - region.y,
        analysis_region.width,
        analysis_region.height,
    )
    self_ignore = _self_ignore_region(region, nav_cfg)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        int(nav_cfg.get("canny_low", 35)),
        int(nav_cfg.get("canny_high", 105)),
    )
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    vertical_edges = (np.abs(sobel_x) > int(nav_cfg.get("vertical_edge_threshold", 38))).astype(np.uint8)

    valid_mask = np.zeros(gray.shape, dtype=bool)
    valid_mask[
        local_analysis.y : local_analysis.y2,
        local_analysis.x : local_analysis.x2,
    ] = True
    if self_ignore is not None:
        local_ignore = BoundingBox(
            self_ignore.x - region.x,
            self_ignore.y - region.y,
            self_ignore.width,
            self_ignore.height,
        ).clipped(roi.shape)
        valid_mask[local_ignore.y : local_ignore.y2, local_ignore.x : local_ignore.x2] = False

    sectors = tuple(
        _build_sector(
            index=index,
            name=name,
            region=region,
            analysis_region=analysis_region,
            roi_gray=gray,
            edges=edges,
            vertical_edges=vertical_edges,
            valid_mask=valid_mask,
            config=nav_cfg,
        )
        for index, name in enumerate(SECTOR_NAMES)
    )
    sectors = _smooth_sector_scores(sectors)
    best_sector = _choose_best_sector(sectors, nav_cfg)
    center_sector = next((sector for sector in sectors if sector.name == "center"), None)
    center_blocked = bool(center_sector and center_sector.blocked)
    recommended_turn = _recommended_turn(best_sector.name if best_sector else None, center_blocked)
    confidence = float(np.mean([sector.confidence for sector in sectors])) if sectors else 0.0
    return NavigationFrame(
        region=region,
        analysis_region=analysis_region,
        self_ignore_region=self_ignore,
        sectors=sectors,
        best_sector=best_sector.name if best_sector else None,
        center_blocked=center_blocked,
        recommended_turn=recommended_turn,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
    )


def draw_local_navigation_report(frame: np.ndarray, navigation: NavigationFrame) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()

    cv2.rectangle(
        annotated,
        (navigation.region.x, navigation.region.y),
        (navigation.region.x2, navigation.region.y2),
        (235, 235, 235),
        2,
    )
    cv2.rectangle(
        annotated,
        (navigation.analysis_region.x, navigation.analysis_region.y),
        (navigation.analysis_region.x2, navigation.analysis_region.y2),
        (190, 190, 190),
        1,
    )

    for sector in navigation.sectors:
        color = _sector_color(sector, navigation.best_sector == sector.name)
        cv2.rectangle(overlay, (sector.bbox.x, sector.bbox.y), (sector.bbox.x2, sector.bbox.y2), color, -1)
        cv2.rectangle(annotated, (sector.bbox.x, sector.bbox.y), (sector.bbox.x2, sector.bbox.y2), color, 2)
        _draw_label(
            annotated,
            f"{sector.name} p={sector.passable_score:.2f} o={sector.obstacle_score:.2f}",
            sector.bbox.x + 6,
            sector.bbox.y + 22,
            color,
            scale=0.5,
        )

    if navigation.self_ignore_region is not None:
        bbox = navigation.self_ignore_region
        cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (90, 90, 90), -1)
        cv2.rectangle(annotated, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (180, 180, 180), 2)
        _draw_label(annotated, "self-ignore", bbox.x + 6, max(18, bbox.y - 6), (210, 210, 210), scale=0.5)

    annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0.0)
    panel = _draw_report_panel(frame.shape[0], navigation)
    return np.hstack([annotated, panel])


def prepare_local_navigation_dataset(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    *,
    max_frames: int | None = None,
    save_reports: bool = True,
    copy_images: bool = True,
) -> LocalNavigationDatasetSummary:
    cfg = config or {}
    output_path = Path(output_dir)
    raw_dir = output_path / "images" / "raw"
    report_dir = output_path / "reports"
    if copy_images:
        raw_dir.mkdir(parents=True, exist_ok=True)
    if save_reports:
        report_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = list(dict.fromkeys(Path(path) for path in image_paths if Path(path).exists()))
    if max_frames is not None and max_frames > 0:
        selected_paths = selected_paths[:max_frames]

    metadata_path = output_path / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    frame_count = 0
    report_count = 0
    copied_image_count = 0
    for index, image_path in enumerate(selected_paths):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        stem = f"{index:06d}_{image_path.stem}"
        raw_path: Path | None = None
        if copy_images:
            raw_path = raw_dir / f"{stem}.png"
            cv2.imwrite(str(raw_path), frame)
            copied_image_count += 1

        navigation = analyze_local_navigation(frame, cfg)
        report_path: Path | None = None
        if save_reports:
            report_path = report_dir / f"{stem}.png"
            cv2.imwrite(str(report_path), draw_local_navigation_report(frame, navigation))
            report_count += 1

        _append_jsonl(
            metadata_path,
            {
                "stem": stem,
                "source": str(image_path),
                "image": _relative_posix(raw_path, output_path) if raw_path is not None else str(image_path),
                "report": _relative_posix(report_path, output_path) if report_path is not None else None,
                "navigation": navigation.to_dict(),
            },
        )
        frame_count += 1

    return LocalNavigationDatasetSummary(output_path, frame_count, report_count, len(selected_paths), copied_image_count)


def collect_route_probe_metadata_sources(source_paths: Iterable[str | Path] | None = None) -> list[Path]:
    if source_paths is None:
        roots = sorted(Path("data").glob("live_route_probe*"))
    else:
        roots = [Path(path) for path in source_paths]

    metadata_paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "metadata.jsonl":
            metadata_paths.append(root)
        elif root.is_file() and root.suffix.lower() == ".jsonl":
            metadata_paths.append(root)
        elif root.is_dir():
            metadata_paths.extend(sorted(root.rglob("metadata.jsonl")))
    return list(dict.fromkeys(path for path in metadata_paths if path.exists()))


def prepare_local_navigation_outcome_dataset(
    metadata_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    *,
    max_samples: int | None = None,
    save_reports: bool = True,
    copy_images: bool = True,
    min_coord_delta: float = 0.015,
    min_distance_progress: float = 0.03,
    min_visual_motion_delta: float = 3.0,
    min_visual_stuck_samples: int = 2,
    val_fraction: float = 0.20,
    test_fraction: float = 0.10,
    seed: int = 1337,
) -> LocalNavigationOutcomeDatasetSummary:
    cfg = config or {}
    output_path = Path(output_dir)
    image_root = output_path / "images"
    report_root = output_path / "reports"
    output_path.mkdir(parents=True, exist_ok=True)
    if copy_images:
        image_root.mkdir(parents=True, exist_ok=True)
    if save_reports:
        report_root.mkdir(parents=True, exist_ok=True)

    metadata_output = output_path / "metadata.jsonl"
    if metadata_output.exists():
        metadata_output.unlink()

    resolved_metadata_paths = [Path(path) for path in metadata_paths if Path(path).exists()]
    samples: list[dict[str, Any]] = []
    skipped_count = 0
    for metadata_path in resolved_metadata_paths:
        episode_id = _episode_id_from_metadata(metadata_path)
        base_dir = metadata_path.parent
        rows = _load_metadata_rows(metadata_path)
        for row in rows:
            if _row_should_skip_outcome(row):
                skipped_count += 1
                continue

            attempted_sector = _attempted_sector_from_route_row(row)
            if attempted_sector is None:
                skipped_count += 1
                continue

            label, confidence, reason = _label_outcome_row(
                row,
                min_coord_delta=min_coord_delta,
                min_distance_progress=min_distance_progress,
                min_visual_motion_delta=min_visual_motion_delta,
                min_visual_stuck_samples=min_visual_stuck_samples,
            )
            frame_path = _resolve_route_frame_path(base_dir, row.get("frame"))
            samples.append(
                {
                    "episode_id": episode_id,
                    "row_index": row.get("index"),
                    "source_metadata": str(metadata_path),
                    "source_frame": str(frame_path) if frame_path is not None else None,
                    "attempted_sector": attempted_sector,
                    "label": label,
                    "confidence": confidence,
                    "reason": reason,
                    "action": row.get("action"),
                    "local_turn_key": row.get("local_turn_key"),
                    "coord": row.get("coord"),
                    "coord_fresh": row.get("coord_fresh"),
                    "coord_delta": _optional_float(row.get("coord_delta")),
                    "distance_progress": _optional_float(row.get("distance_progress")),
                    "visual_motion_delta": _optional_float(row.get("visual_motion_delta")),
                    "visual_stuck_samples": int(row.get("visual_stuck_samples") or 0),
                    "movement": row.get("movement"),
                    "navigation": row.get("navigation"),
                }
            )

    samples = _limit_outcome_samples(samples, max_samples)
    split_by_episode = _assign_episode_splits(
        [str(sample["episode_id"]) for sample in samples],
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    report_count = 0
    copied_image_count = 0
    for ordinal, sample in enumerate(samples):
        label = str(sample["label"])
        split = split_by_episode.get(str(sample["episode_id"]), "train")
        sample["split"] = split
        label_counts[label] = label_counts.get(label, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1

        source_frame = Path(str(sample["source_frame"])) if sample.get("source_frame") else None
        if source_frame is not None and source_frame.exists():
            stem = _outcome_stem(ordinal, sample)
            if copy_images:
                image_dir = image_root / split / label
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{stem}{source_frame.suffix.lower()}"
                shutil.copy2(source_frame, image_path)
                sample["image"] = _relative_posix(image_path, output_path)
                copied_image_count += 1
            else:
                sample["image"] = str(source_frame)

            if save_reports:
                frame = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
                if frame is not None:
                    report_dir = report_root / split / label
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = report_dir / f"{stem}.png"
                    navigation = analyze_local_navigation(frame, cfg)
                    cv2.imwrite(str(report_path), draw_local_navigation_report(frame, navigation))
                    sample["report"] = _relative_posix(report_path, output_path)
                    report_count += 1
                else:
                    sample["report"] = None
            else:
                sample["report"] = None
        else:
            sample["image"] = None
            sample["report"] = None

        _append_jsonl(metadata_output, sample)

    split_payload = {
        "episodes": split_by_episode,
        "splits": {
            split: sorted(episode for episode, assigned in split_by_episode.items() if assigned == split)
            for split in ("train", "val", "test")
        },
        "label_counts": label_counts,
        "split_counts": split_counts,
        "thresholds": {
            "min_coord_delta": min_coord_delta,
            "min_distance_progress": min_distance_progress,
            "min_visual_motion_delta": min_visual_motion_delta,
            "min_visual_stuck_samples": min_visual_stuck_samples,
        },
    }
    (output_path / "episode_splits.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return LocalNavigationOutcomeDatasetSummary(
        output_dir=output_path,
        sample_count=len(samples),
        source_metadata_count=len(resolved_metadata_paths),
        episode_count=len(split_by_episode),
        label_counts=label_counts,
        split_counts=split_counts,
        skipped_count=skipped_count,
        report_count=report_count,
        copied_image_count=copied_image_count,
    )


def prepare_local_navigation_review_pack(
    metadata_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    *,
    max_per_bucket: int = 8,
) -> LocalNavigationReviewSummary:
    cfg = config or {}
    source_path = Path(metadata_path)
    rows = _load_metadata_rows(source_path)
    output_path = Path(output_dir)
    report_dir = output_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for old_report in report_dir.glob("*.png"):
        old_report.unlink()

    output_metadata = output_path / "metadata.jsonl"
    if output_metadata.exists():
        output_metadata.unlink()

    selected = _select_review_rows(rows, max_per_bucket=max_per_bucket)
    bucket_counts: dict[str, int] = {}
    report_count = 0
    for index, (bucket, row) in enumerate(selected):
        image_path = Path(str(row.get("source") or row.get("image") or ""))
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        navigation = analyze_local_navigation(frame, cfg)
        stem = f"{index:04d}_{bucket}_{image_path.stem}"
        report_path = report_dir / f"{stem}.png"
        cv2.imwrite(str(report_path), draw_local_navigation_report(frame, navigation))
        _append_jsonl(
            output_metadata,
            {
                "bucket": bucket,
                "stem": stem,
                "source": str(image_path),
                "report": _relative_posix(report_path, output_path),
                "navigation": navigation.to_dict(),
            },
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        report_count += 1

    return LocalNavigationReviewSummary(output_path, sum(bucket_counts.values()), report_count, bucket_counts)


def collect_local_navigation_sources(source_dirs: Iterable[str | Path] | None = None) -> list[Path]:
    roots = [Path(path) for path in source_dirs] if source_dirs else list(DEFAULT_SOURCE_DIRS)
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    return list(dict.fromkeys(paths))


def _load_metadata_rows(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Local-navigation metadata not found: {metadata_path}")
    rows: list[dict[str, Any]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _select_review_rows(rows: list[dict[str, Any]], *, max_per_bucket: int) -> list[tuple[str, dict[str, Any]]]:
    bucketed: dict[str, list[dict[str, Any]]] = {
        "center_nominal": [],
        "center_risky": [],
        "center_blocked": [],
        "left_escape": [],
        "right_escape": [],
        "slight_escape": [],
    }
    for row in rows:
        navigation = row.get("navigation", {})
        center = _center_from_navigation(navigation)
        if center is None:
            continue

        best_sector = str(navigation.get("best_sector"))
        center_blocked = bool(navigation.get("center_blocked", False))
        center_obstacle = float(center.get("obstacle_score", 0.0))
        center_passable = float(center.get("passable_score", 0.0))

        if center_blocked:
            bucketed["center_blocked"].append(row)
            if best_sector in {"left", "slight_left"}:
                bucketed["left_escape"].append(row)
            elif best_sector in {"right", "slight_right"}:
                bucketed["right_escape"].append(row)
        elif center_obstacle >= 0.55 or center_passable <= 0.50:
            bucketed["center_risky"].append(row)
        elif best_sector == "center":
            bucketed["center_nominal"].append(row)

        if best_sector in {"slight_left", "slight_right"}:
            bucketed["slight_escape"].append(row)

    selected: list[tuple[str, dict[str, Any]]] = []
    bucket_order = ["center_nominal", "center_risky", "left_escape", "right_escape", "slight_escape", "center_blocked"]
    for bucket in bucket_order:
        bucket_rows = bucketed[bucket]
        for row in _dedupe_rows(_sort_review_bucket(bucket, bucket_rows))[:max(1, max_per_bucket)]:
            selected.append((bucket, row))
    return selected


def _center_from_navigation(navigation: dict[str, Any]) -> dict[str, Any] | None:
    sectors = navigation.get("sectors", [])
    for sector in sectors:
        if sector.get("name") == "center":
            return sector
    return None


def _sort_review_bucket(bucket: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if bucket == "center_nominal":
        return sorted(rows, key=lambda row: _center_metric(row, "passable_score"), reverse=True)
    if bucket == "center_risky":
        return sorted(rows, key=lambda row: abs(_center_metric(row, "obstacle_score") - BLOCKED_OBSTACLE_SCORE))
    if bucket in {"center_blocked", "left_escape", "right_escape", "slight_escape"}:
        return sorted(rows, key=lambda row: _center_metric(row, "obstacle_score"), reverse=True)
    return rows


def _center_metric(row: dict[str, Any], metric: str) -> float:
    center = _center_from_navigation(row.get("navigation", {}))
    if center is None:
        return 0.0
    return float(center.get(metric, 0.0))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = _row_source_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _row_source_key(row: dict[str, Any]) -> str:
    source = Path(str(row.get("source") or row.get("image") or ""))
    stem = source.stem
    if stem.startswith("live_"):
        stem = stem[5:]
    return stem


def _episode_id_from_metadata(metadata_path: Path) -> str:
    return metadata_path.parent.name or metadata_path.stem


def _row_should_skip_outcome(row: dict[str, Any]) -> bool:
    game_state = row.get("game_state") if isinstance(row.get("game_state"), dict) else {}
    if game_state.get("death_or_blocking_modal") or game_state.get("dismissable_modal_click"):
        return True
    if row.get("death_recovery") is not None:
        return True
    if row.get("threat") is not None:
        return True
    combat = row.get("combat")
    if isinstance(combat, dict):
        return bool(
            combat.get("active")
            or combat.get("target_present")
            or combat.get("hostile_target_hint")
            or combat.get("nameplate_visible")
        )
    return bool(combat)


def _attempted_sector_from_route_row(row: dict[str, Any]) -> str | None:
    movement = row.get("movement") if isinstance(row.get("movement"), dict) else {}
    action = str(row.get("action") or "")
    movement_action = str(movement.get("action") or "")
    local_turn_key = _normalize_turn_key(row.get("local_turn_key") or movement.get("turn_key"))
    held_key = str(movement.get("held_key") or "").upper()

    if action.startswith("recover_") or movement_action.startswith("recover_"):
        return "center"
    if held_key != "W":
        return None
    if local_turn_key == "A":
        return "slight_left"
    if local_turn_key == "D":
        return "slight_right"
    return "center"


def _label_outcome_row(
    row: dict[str, Any],
    *,
    min_coord_delta: float,
    min_distance_progress: float,
    min_visual_motion_delta: float,
    min_visual_stuck_samples: int,
) -> tuple[str, float, str]:
    movement = row.get("movement") if isinstance(row.get("movement"), dict) else {}
    action = str(row.get("action") or "")
    movement_action = str(movement.get("action") or "")
    coord_delta = _optional_float(row.get("coord_delta"))
    distance_progress = _optional_float(row.get("distance_progress"))
    visual_motion_delta = _optional_float(row.get("visual_motion_delta"))
    visual_stuck_samples = int(row.get("visual_stuck_samples") or 0)
    coord_fresh = bool(row.get("coord_fresh", coord_delta is not None or distance_progress is not None))

    if action.startswith("recover_") or movement_action.startswith("recover_"):
        return "blocked", 0.85, "recovery_action"
    if visual_stuck_samples >= max(1, min_visual_stuck_samples):
        return "blocked", 0.76, "visual_stuck_samples"
    if _has_forward_progress(coord_delta, distance_progress, min_coord_delta, min_distance_progress):
        confidence = 0.92 if coord_delta is not None and distance_progress is not None else 0.82
        return "passable", confidence, "coordinate_progress"
    if (
        coord_fresh
        and coord_delta is not None
        and distance_progress is not None
        and visual_motion_delta is not None
        and coord_delta < min_coord_delta
        and distance_progress < min_distance_progress
        and visual_motion_delta < min_visual_motion_delta
    ):
        return "blocked", 0.70, "fresh_no_progress_low_visual_motion"
    return "unknown", 0.40, "insufficient_outcome_evidence"


def _has_forward_progress(
    coord_delta: float | None,
    distance_progress: float | None,
    min_coord_delta: float,
    min_distance_progress: float,
) -> bool:
    if coord_delta is not None and coord_delta >= min_coord_delta:
        return True
    if distance_progress is not None and distance_progress >= min_distance_progress:
        return True
    return False


def _resolve_route_frame_path(base_dir: Path, frame_value: Any) -> Path | None:
    if not frame_value:
        return None
    frame_path = Path(str(frame_value))
    if frame_path.is_absolute():
        return frame_path
    return base_dir / frame_path


def _limit_outcome_samples(samples: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (str(sample["label"]), str(sample["episode_id"]))
        buckets.setdefault(key, []).append(sample)

    selected: list[dict[str, Any]] = []
    labels = ["blocked", "unknown", "passable"]
    while len(selected) < max_samples and buckets:
        made_progress = False
        for label in labels:
            keys = sorted(key for key in buckets if key[0] == label)
            for key in keys:
                if len(selected) >= max_samples:
                    break
                bucket = buckets.get(key)
                if not bucket:
                    buckets.pop(key, None)
                    continue
                selected.append(bucket.pop(0))
                made_progress = True
                if not bucket:
                    buckets.pop(key, None)
            if len(selected) >= max_samples:
                break
        if not made_progress:
            break
    return selected


def _assign_episode_splits(
    episode_ids: Sequence[str],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    unique = sorted(set(episode_ids))
    if not unique:
        return {}
    rng = random.Random(seed)
    rng.shuffle(unique)

    total = len(unique)
    if total == 1:
        val_count = 0
        test_count = 0
    elif total == 2:
        val_count = 1
        test_count = 0
    else:
        test_count = max(1, int(round(total * max(0.0, min(0.8, test_fraction)))))
        val_count = max(1, int(round(total * max(0.0, min(0.8, val_fraction)))))
        while val_count + test_count >= total:
            if test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break

    test_episodes = set(unique[:test_count])
    val_episodes = set(unique[test_count : test_count + val_count])
    return {
        episode_id: "test" if episode_id in test_episodes else "val" if episode_id in val_episodes else "train"
        for episode_id in unique
    }


def _outcome_stem(ordinal: int, sample: dict[str, Any]) -> str:
    episode = str(sample.get("episode_id") or "episode")
    row_index = sample.get("row_index")
    row_text = f"{int(row_index):04d}" if isinstance(row_index, int) else f"{ordinal:04d}"
    return f"{episode}_{row_text}_{sample.get('attempted_sector')}_{sample.get('label')}"


def _normalize_turn_key(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().upper()
    return key if key in {"A", "D"} else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _navigation_region(frame: np.ndarray, config: dict[str, Any]) -> BoundingBox:
    nav_cfg = config.get("local_navigation", {})
    if "region" in nav_cfg:
        x, y, width, height = resolve_region(nav_cfg["region"], frame.shape, config)
        return BoundingBox(x, y, width, height)

    frame_height, frame_width = frame.shape[:2]
    width = int(frame_width * 0.58)
    height = int(frame_height * 0.58)
    x = (frame_width - width) // 2
    y = int(frame_height * 0.18)
    return BoundingBox(x, y, width, min(height, frame_height - y))


def _analysis_region(region: BoundingBox, nav_cfg: dict[str, Any]) -> BoundingBox:
    top_fraction = max(0.0, min(0.90, float(nav_cfg.get("analysis_top_fraction", 0.30))))
    bottom_fraction = max(top_fraction + 0.05, min(1.0, float(nav_cfg.get("analysis_bottom_fraction", 0.88))))
    y = region.y + int(round(region.height * top_fraction))
    y2 = region.y + int(round(region.height * bottom_fraction))
    return BoundingBox(region.x, y, region.width, max(1, y2 - y))


def _self_ignore_region(region: BoundingBox, nav_cfg: dict[str, Any]) -> BoundingBox | None:
    ignore_cfg = nav_cfg.get("self_ignore", {})
    if not bool(ignore_cfg.get("enabled", True)):
        return None

    width_fraction = max(0.02, min(0.80, float(ignore_cfg.get("width_fraction", 0.18))))
    height_fraction = max(0.02, min(0.80, float(ignore_cfg.get("height_fraction", 0.28))))
    center_x_fraction = max(0.0, min(1.0, float(ignore_cfg.get("center_x_fraction", 0.50))))
    bottom_fraction = max(0.0, min(1.0, float(ignore_cfg.get("bottom_fraction", 0.98))))

    width = max(1, int(round(region.width * width_fraction)))
    height = max(1, int(round(region.height * height_fraction)))
    center_x = region.x + int(round(region.width * center_x_fraction))
    bottom_y = region.y + int(round(region.height * bottom_fraction))
    x = center_x - width // 2
    y = bottom_y - height
    return BoundingBox(x, y, width, height).clipped((region.y2 + 1, region.x2 + 1, 3))


def _build_sector(
    *,
    index: int,
    name: str,
    region: BoundingBox,
    analysis_region: BoundingBox,
    roi_gray: np.ndarray,
    edges: np.ndarray,
    vertical_edges: np.ndarray,
    valid_mask: np.ndarray,
    config: dict[str, Any],
) -> NavigationSector:
    sector_width = analysis_region.width / float(len(SECTOR_NAMES))
    x1 = analysis_region.x + int(round(index * sector_width))
    x2 = analysis_region.x + int(round((index + 1) * sector_width))
    if index == len(SECTOR_NAMES) - 1:
        x2 = analysis_region.x2
    bbox = BoundingBox(x1, analysis_region.y, max(1, x2 - x1), analysis_region.height)

    local_x1 = bbox.x - region.x
    local_x2 = bbox.x2 - region.x
    local_y1 = bbox.y - region.y
    local_y2 = bbox.y2 - region.y
    segment_valid = valid_mask[local_y1:local_y2, local_x1:local_x2]
    segment_area = max(1, segment_valid.size)
    valid_area = int(np.count_nonzero(segment_valid))
    valid_fraction = valid_area / float(segment_area)
    if valid_area <= 0:
        return NavigationSector(name, index, bbox, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    edge_segment = edges[local_y1:local_y2, local_x1:local_x2]
    vertical_segment = vertical_edges[local_y1:local_y2, local_x1:local_x2]
    gray_segment = roi_gray[local_y1:local_y2, local_x1:local_x2]
    edge_density = float(np.count_nonzero(edge_segment[segment_valid])) / float(valid_area)
    vertical_density = float(np.count_nonzero(vertical_segment[segment_valid])) / float(valid_area)
    texture_score = min(1.0, float(np.std(gray_segment[segment_valid])) / 64.0)

    bottom_height = max(4, int(round((local_y2 - local_y1) * 0.28)))
    bottom_y1 = max(local_y1, local_y2 - bottom_height)
    bottom_valid = valid_mask[bottom_y1:local_y2, local_x1:local_x2]
    bottom_area = max(1, int(np.count_nonzero(bottom_valid)))
    bottom_edges = edges[bottom_y1:local_y2, local_x1:local_x2]
    bottom_edge_density = float(np.count_nonzero(bottom_edges[bottom_valid])) / float(bottom_area)

    obstacle_score = (
        edge_density * float(config.get("edge_weight", 1.35))
        + vertical_density * float(config.get("vertical_weight", 2.55))
        + texture_score * float(config.get("texture_weight", 0.10))
        + bottom_edge_density * float(config.get("bottom_edge_weight", 1.65))
    )
    obstacle_score = max(0.0, min(1.0, obstacle_score))
    confidence = max(0.0, min(1.0, valid_fraction * (0.65 + min(0.35, edge_density + texture_score * 0.20))))
    unknown_score = max(0.0, min(1.0, 1.0 - confidence))
    passable_score = max(0.0, min(1.0, (1.0 - obstacle_score) * (0.70 + 0.30 * confidence)))
    return NavigationSector(
        name=name,
        index=index,
        bbox=bbox,
        passable_score=round(passable_score, 4),
        obstacle_score=round(obstacle_score, 4),
        unknown_score=round(unknown_score, 4),
        confidence=round(confidence, 4),
        edge_density=round(edge_density, 4),
        vertical_density=round(vertical_density, 4),
        texture_score=round(texture_score, 4),
        bottom_edge_density=round(bottom_edge_density, 4),
        valid_fraction=round(valid_fraction, 4),
    )


def _smooth_sector_scores(sectors: tuple[NavigationSector, ...]) -> tuple[NavigationSector, ...]:
    if len(sectors) < 3:
        return sectors

    smoothed: list[NavigationSector] = []
    for index, sector in enumerate(sectors):
        neighbor_scores = [sector.obstacle_score]
        if index > 0:
            neighbor_scores.append(sectors[index - 1].obstacle_score * 0.45)
        if index < len(sectors) - 1:
            neighbor_scores.append(sectors[index + 1].obstacle_score * 0.45)
        obstacle_score = max(sector.obstacle_score, min(1.0, sum(neighbor_scores) / (1.0 + 0.45 * (len(neighbor_scores) - 1))))
        passable_score = max(0.0, min(1.0, sector.passable_score * (1.0 - max(0.0, obstacle_score - sector.obstacle_score) * 0.35)))
        smoothed.append(
            NavigationSector(
                sector.name,
                sector.index,
                sector.bbox,
                round(passable_score, 4),
                round(obstacle_score, 4),
                sector.unknown_score,
                sector.confidence,
                sector.edge_density,
                sector.vertical_density,
                sector.texture_score,
                sector.bottom_edge_density,
                sector.valid_fraction,
            )
        )
    return tuple(smoothed)


def _choose_best_sector(sectors: tuple[NavigationSector, ...], config: dict[str, Any]) -> NavigationSector | None:
    if not sectors:
        return None
    center_bias = float(config.get("center_bias", 0.16))
    slight_bias = float(config.get("slight_sector_bias", 0.03))
    penalties = {
        "left": 0.0,
        "slight_left": slight_bias,
        "center": center_bias,
        "slight_right": slight_bias,
        "right": 0.0,
    }
    best = max(
        sectors,
        key=lambda sector: (
            sector.passable_score
            - sector.obstacle_score * 0.45
            + penalties.get(sector.name, 0.0)
            - sector.unknown_score * 0.18
        ),
    )
    center = next((sector for sector in sectors if sector.name == "center"), None)
    if center is not None and not center.blocked:
        return center
    return best


def _recommended_turn(best_sector_name: str | None, center_blocked: bool) -> str | None:
    if best_sector_name is None:
        return None
    if not center_blocked:
        return None
    if best_sector_name in {"left", "slight_left"}:
        return "A"
    if best_sector_name in {"right", "slight_right"}:
        return "D"
    return None if not center_blocked else None


def _sector_color(sector: NavigationSector, is_best: bool) -> tuple[int, int, int]:
    if is_best:
        return (80, 255, 80)
    if sector.blocked:
        return (35, 35, 235)
    if sector.unknown_score >= 0.55:
        return (0, 190, 255)
    return (90, 175, 90)


def _draw_report_panel(height: int, navigation: NavigationFrame) -> np.ndarray:
    panel_width = 560
    panel = np.full((height, panel_width, 3), 24, dtype=np.uint8)
    y = 42
    _panel_text(panel, "Local navigation 0.2", 24, y, scale=0.78, color=(245, 245, 245), thickness=2)
    y += 42
    _panel_text(panel, f"Best sector: {navigation.best_sector}", 24, y, scale=0.62, color=(225, 225, 225))
    y += 30
    _panel_text(panel, f"Center blocked: {navigation.center_blocked}", 24, y, scale=0.62, color=(225, 225, 225))
    y += 30
    _panel_text(panel, f"Recommended turn: {navigation.recommended_turn}", 24, y, scale=0.62, color=(225, 225, 225))
    y += 30
    _panel_text(panel, f"Confidence: {navigation.confidence:.2f}", 24, y, scale=0.62, color=(225, 225, 225))
    y += 42
    _panel_text(panel, "Sectors", 24, y, scale=0.68, color=(245, 245, 245), thickness=2)
    y += 34
    for sector in navigation.sectors:
        color = _sector_color(sector, navigation.best_sector == sector.name)
        cv2.rectangle(panel, (24, y - 17), (46, y + 5), color, -1)
        _panel_text(
            panel,
            f"{sector.name}: pass={sector.passable_score:.2f} obstacle={sector.obstacle_score:.2f}",
            58,
            y,
            scale=0.48,
            color=(230, 230, 230),
        )
        y += 24
        _panel_text(
            panel,
            f"edge={sector.edge_density:.2f} vertical={sector.vertical_density:.2f} bottom={sector.bottom_edge_density:.2f}",
            58,
            y,
            scale=0.42,
            color=(170, 170, 170),
        )
        y += 30

    y += 12
    _panel_text(panel, "Ore/minimap signals disabled for this stage", 24, y, scale=0.48, color=(185, 185, 185))
    return panel


def _draw_label(
    output: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, max(0, output.shape[1] - text_width - 8)))
    y = max(text_height + 8, min(y, output.shape[0] - baseline - 4))
    cv2.rectangle(output, (x, y - text_height - 7), (x + text_width + 8, y + baseline + 5), (18, 18, 18), -1)
    cv2.rectangle(output, (x, y - text_height - 7), (x + text_width + 8, y + baseline + 5), color, 1)
    cv2.putText(output, text, (x + 4, y), font, scale, color, thickness, cv2.LINE_AA)


def _panel_text(
    panel: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _relative_posix(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _bbox_to_dict(bbox: BoundingBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}
