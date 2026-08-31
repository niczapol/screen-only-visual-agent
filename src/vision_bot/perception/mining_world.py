from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from vision_bot.controllers.mining_approach import MiningApproachPhase
from vision_bot.core.world import CombatView, MiningPerception
from vision_bot.mining import is_mining_hover_ready
from vision_bot.ore_world_detector import OreWorldDetector, OreWorldDetectorResult
from vision_bot.regions import resolve_region


@dataclass(frozen=True)
class MiningWorldSensorResult:
    perception: MiningPerception
    detector_reason: str
    detector_elapsed_ms: float
    detection_frame_id: int | None


class MiningWorldSensor:
    """On-demand world-ore proposals and direct native hover authority.

    Model inference never authorises a click.  In production it runs on one
    background worker with a one-frame backlog; the control loop consumes only
    the newest completed result and continues capturing while inference runs.
    """

    def __init__(
        self,
        project_config: dict,
        *,
        detector: OreWorldDetector | None = None,
        hover_authority: Callable[[np.ndarray, dict, tuple[int, int] | None], bool]
        | None = None,
        async_inference: bool = True,
        refresh_interval_seconds: float = 0.75,
        result_max_age_seconds: float = 1.50,
        gather_wait_seconds: float = 3.0,
        clear_frames_required: int = 2,
    ) -> None:
        self.project_config = project_config
        self.detector = detector or OreWorldDetector(project_config)
        self.hover_authority = hover_authority or (
            lambda frame, config, point: is_mining_hover_ready(frame, config, point)
        )
        self.async_inference = bool(async_inference)
        self.refresh_interval_seconds = max(0.05, float(refresh_interval_seconds))
        self.result_max_age_seconds = max(0.05, float(result_max_age_seconds))
        self.gather_wait_seconds = max(0.0, float(gather_wait_seconds))
        self.clear_frames_required = max(1, int(clear_frames_required))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ore-world-v09") if self.async_inference else None
        self._future: Future[OreWorldDetectorResult] | None = None
        self._future_frame_id: int | None = None
        self._future_generation: int | None = None
        self._last_submit_at: float | None = None
        self._last_result = OreWorldDetectorResult(reason="not_requested")
        self._last_result_at: float | None = None
        self._last_result_frame_id: int | None = None
        self._verify_clear_frames = 0
        self._previous_phase = MiningApproachPhase.IDLE
        self._search_generation = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def reset(self) -> None:
        self._verify_clear_frames = 0
        self._last_result = OreWorldDetectorResult(reason="not_requested")
        self._last_result_at = None
        self._last_result_frame_id = None
        self._last_submit_at = None
        self._previous_phase = MiningApproachPhase.IDLE
        self._search_generation += 1

    def observe(
        self,
        frame: np.ndarray,
        perception: MiningPerception,
        *,
        phase: MiningApproachPhase,
        frame_id: int,
        now: float,
        cursor_client_point: tuple[int, int] | None,
        target_marker_point: tuple[int, int] | None,
        clicked_at: float | None,
        combat: CombatView | None = None,
    ) -> MiningWorldSensorResult:
        if (
            phase is MiningApproachPhase.VISUAL_SEARCH
            and self._previous_phase is not MiningApproachPhase.VISUAL_SEARCH
        ):
            self._search_generation += 1
            self._last_result = OreWorldDetectorResult(reason="new_visual_search")
            self._last_result_at = None
            self._last_result_frame_id = None
            self._last_submit_at = None
            if self._future is not None and self._future.cancel():
                self._future = None
                self._future_frame_id = None
                self._future_generation = None
        self._collect_async_result(now=now)
        if phase is MiningApproachPhase.VISUAL_SEARCH:
            self._request_detection(frame, frame_id=frame_id, now=now)

        candidates: tuple[tuple[int, int], ...] = ()
        if (
            phase is MiningApproachPhase.VISUAL_SEARCH
            and self._last_result_at is not None
            and now - self._last_result_at <= self.result_max_age_seconds
        ):
            candidates = tuple(item.center for item in self._last_result.detections)

        authority = bool(
            phase is MiningApproachPhase.VISUAL_SEARCH
            and cursor_client_point is not None
            and _cursor_is_current_world_probe(
                frame,
                self.project_config,
                cursor_client_point,
                candidates,
            )
            and self.hover_authority(
                frame,
                self.project_config,
                cursor_client_point,
            )
        )
        outcome: str | None = None
        if phase is MiningApproachPhase.VERIFY:
            if combat is not None and combat.out_of_range:
                outcome = "out_of_range"
            elif combat is not None and combat.loot_opened:
                outcome = "loot_opened"
            elif combat is not None and combat.active:
                # Combat can hide the minimap marker and preempt the mining
                # transaction.  Marker disappearance during that interruption
                # is not gather evidence.
                self._verify_clear_frames = 0
            elif clicked_at is not None and now - clicked_at >= self.gather_wait_seconds:
                target_visible = _marker_near(
                    target_marker_point,
                    perception.bright_minimap_points,
                    max_distance=24.0,
                ) or _marker_near(
                    target_marker_point,
                    perception.dark_minimap_points,
                    max_distance=24.0,
                )
                if target_visible or authority:
                    self._verify_clear_frames = 0
                else:
                    self._verify_clear_frames += 1
                if self._verify_clear_frames >= self.clear_frames_required:
                    outcome = "gathered"
        else:
            self._verify_clear_frames = 0

        enriched = replace(
            perception,
            world_candidate_points=candidates,
            interaction_authority=authority,
            interaction_point=cursor_client_point if authority else None,
            interaction_outcome=outcome,
        )
        self._previous_phase = phase
        return MiningWorldSensorResult(
            perception=enriched,
            detector_reason=self._last_result.reason,
            detector_elapsed_ms=self._last_result.elapsed_ms,
            detection_frame_id=self._last_result_frame_id,
        )

    def _request_detection(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        now: float,
    ) -> None:
        if self._future is not None and not self._future.done():
            return
        if (
            self._last_submit_at is not None
            and now - self._last_submit_at < self.refresh_interval_seconds
        ):
            return
        self._last_submit_at = now
        if self._pool is None:
            self._last_result = self.detector.detect(frame)
            self._last_result_at = now
            self._last_result_frame_id = frame_id
            return
        self._future_frame_id = frame_id
        self._future_generation = self._search_generation
        self._future = self._pool.submit(self.detector.detect, frame.copy())

    def _collect_async_result(self, *, now: float) -> None:
        if self._future is None or not self._future.done():
            return
        future_generation = self._future_generation
        try:
            result = self._future.result()
        except Exception as exc:
            message = "_".join(str(exc).strip().split())
            suffix = f":{message[:120]}" if message else ""
            result = OreWorldDetectorResult(
                reason=f"inference_error:{type(exc).__name__}{suffix}"
            )
        if future_generation == self._search_generation:
            self._last_result = result
            self._last_result_at = now
            self._last_result_frame_id = self._future_frame_id
        self._future = None
        self._future_frame_id = None
        self._future_generation = None


def _marker_near(
    target: tuple[int, int] | None,
    candidates: tuple[tuple[int, int], ...],
    *,
    max_distance: float,
) -> bool:
    if target is None:
        return bool(candidates)
    return any(
        ((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2) ** 0.5
        <= max_distance
        for point in candidates
    )


def _cursor_is_current_world_probe(
    frame: np.ndarray,
    config: dict,
    cursor: tuple[int, int],
    candidates: tuple[tuple[int, int], ...],
) -> bool:
    """Reject stale UI/minimap cursor state as world-click authority."""

    if not candidates:
        return False
    mining_cfg = config.get("mining", {})
    region_cfg = mining_cfg.get(
        "world_interaction_region",
        {"x": 0, "y": 96, "width": 2220, "height": 1090},
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, config)
    cursor_x, cursor_y = cursor
    if not (x <= cursor_x < x + width and y <= cursor_y < y + height):
        return False
    radius = max(
        1.0,
        float(mining_cfg.get("world_candidate_authority_radius_pixels", 48.0)),
    )
    return _marker_near(cursor, candidates, max_distance=radius)
