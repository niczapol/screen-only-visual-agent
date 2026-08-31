from __future__ import annotations

from threading import Event

import numpy as np

from vision_bot.controllers.mining_approach import MiningApproachPhase
from vision_bot.core.world import CombatView, MiningPerception
from vision_bot.ore_world_detector import OreWorldDetection, OreWorldDetectorResult
from vision_bot.perception.mining_world import MiningWorldSensor


class _Detector:
    def __init__(self, result: OreWorldDetectorResult) -> None:
        self.result = result
        self.calls = 0

    def detect(self, frame: np.ndarray) -> OreWorldDetectorResult:
        self.calls += 1
        return self.result


class _BlockingDetector(_Detector):
    def __init__(self, result: OreWorldDetectorResult) -> None:
        super().__init__(result)
        self.started = Event()
        self.release = Event()

    def detect(self, frame: np.ndarray) -> OreWorldDetectorResult:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2.0)
        return self.result


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def _result() -> OreWorldDetectorResult:
    return OreWorldDetectorResult(
        detections=(OreWorldDetection(600, 300, 640, 360, 0.88),),
        elapsed_ms=12.5,
        reason="detected",
    )


def test_world_model_only_proposes_cursor_point_and_never_authorizes_click() -> None:
    detector = _Detector(_result())
    sensor = MiningWorldSensor(
        {},
        detector=detector,
        hover_authority=lambda *_args: False,
        async_inference=False,
    )

    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VISUAL_SEARCH,
        frame_id=10,
        now=1.0,
        cursor_client_point=(620, 330),
        target_marker_point=None,
        clicked_at=None,
    )

    assert detector.calls == 1
    assert observed.perception.world_candidate_points == ((620, 330),)
    assert not observed.perception.interaction_authority
    assert observed.perception.interaction_point is None


def test_native_hover_is_the_only_interaction_authority() -> None:
    sensor = MiningWorldSensor(
        {},
        detector=_Detector(_result()),
        hover_authority=lambda _frame, _config, point: point == (620, 330),
        async_inference=False,
    )

    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VISUAL_SEARCH,
        frame_id=11,
        now=1.0,
        cursor_client_point=(620, 330),
        target_marker_point=None,
        clicked_at=None,
    )

    assert observed.perception.interaction_authority
    assert observed.perception.interaction_point == (620, 330)


def test_stale_hover_without_current_world_proposal_cannot_authorize_click() -> None:
    sensor = MiningWorldSensor(
        {},
        detector=_Detector(OreWorldDetectorResult(reason="no_detection")),
        hover_authority=lambda *_args: True,
        async_inference=False,
    )

    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VISUAL_SEARCH,
        frame_id=12,
        now=1.0,
        cursor_client_point=(620, 330),
        target_marker_point=None,
        clicked_at=None,
    )

    assert not observed.perception.interaction_authority
    assert observed.perception.interaction_point is None


def test_minimap_side_cursor_cannot_authorize_world_click() -> None:
    detector = _Detector(
        OreWorldDetectorResult(
            detections=(OreWorldDetection(1120, 90, 1160, 130, 0.90),),
            reason="detected",
        )
    )
    sensor = MiningWorldSensor(
        {
            "screen": {"reference_width": 1280, "reference_height": 720},
            "mining": {
                "world_interaction_region": {
                    "x": 0,
                    "y": 48,
                    "width": 1110,
                    "height": 545,
                }
            },
        },
        detector=detector,
        hover_authority=lambda *_args: True,
        async_inference=False,
    )

    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VISUAL_SEARCH,
        frame_id=13,
        now=1.0,
        cursor_client_point=(1140, 110),
        target_marker_point=None,
        clicked_at=None,
    )

    assert not observed.perception.interaction_authority


def test_async_sensor_allows_only_one_pending_inference() -> None:
    detector = _BlockingDetector(_result())
    sensor = MiningWorldSensor(
        {},
        detector=detector,
        hover_authority=lambda *_args: False,
        async_inference=True,
        refresh_interval_seconds=0.05,
    )
    try:
        sensor.observe(
            FRAME,
            MiningPerception(),
            phase=MiningApproachPhase.VISUAL_SEARCH,
            frame_id=1,
            now=1.0,
            cursor_client_point=None,
            target_marker_point=None,
            clicked_at=None,
        )
        assert detector.started.wait(timeout=1.0)
        sensor.observe(
            FRAME,
            MiningPerception(),
            phase=MiningApproachPhase.VISUAL_SEARCH,
            frame_id=2,
            now=1.2,
            cursor_client_point=None,
            target_marker_point=None,
            clicked_at=None,
        )
        assert detector.calls == 1
    finally:
        detector.release.set()
        sensor.close()


def test_verify_requires_two_clear_frames_after_gather_wait() -> None:
    sensor = MiningWorldSensor(
        {},
        detector=_Detector(_result()),
        hover_authority=lambda *_args: False,
        async_inference=False,
        gather_wait_seconds=3.0,
        clear_frames_required=2,
    )
    first = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VERIFY,
        frame_id=20,
        now=4.1,
        cursor_client_point=None,
        target_marker_point=(100, 100),
        clicked_at=1.0,
    )
    second = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VERIFY,
        frame_id=21,
        now=4.2,
        cursor_client_point=None,
        target_marker_point=(100, 100),
        clicked_at=1.0,
    )

    assert first.perception.interaction_outcome is None
    assert second.perception.interaction_outcome == "gathered"


def test_confirmed_out_of_range_preempts_clear_frame_success() -> None:
    sensor = MiningWorldSensor(
        {},
        detector=_Detector(_result()),
        hover_authority=lambda *_args: False,
        async_inference=False,
        gather_wait_seconds=0.0,
        clear_frames_required=1,
    )
    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VERIFY,
        frame_id=22,
        now=2.0,
        cursor_client_point=None,
        target_marker_point=None,
        clicked_at=1.0,
        combat=CombatView(active=False, out_of_range=True),
    )

    assert observed.perception.interaction_outcome == "out_of_range"


def test_active_combat_cannot_turn_marker_clear_into_gather_success() -> None:
    sensor = MiningWorldSensor(
        {},
        detector=_Detector(_result()),
        hover_authority=lambda *_args: False,
        async_inference=False,
        gather_wait_seconds=0.0,
        clear_frames_required=1,
    )
    observed = sensor.observe(
        FRAME,
        MiningPerception(),
        phase=MiningApproachPhase.VERIFY,
        frame_id=23,
        now=2.0,
        cursor_client_point=None,
        target_marker_point=None,
        clicked_at=1.0,
        combat=CombatView(active=True),
    )

    assert observed.perception.interaction_outcome is None
