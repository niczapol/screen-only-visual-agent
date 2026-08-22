from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vision_bot.capture import ScreenCapture


class WindowVideoRecorder:
    """Record the selected game window independently from the controller loop."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        target_hwnd: int,
        output_dir: str | Path,
        capture_factory: Callable[[dict[str, Any]], ScreenCapture] = ScreenCapture,
    ) -> None:
        route_live_cfg = config.get("training_capture", {}).get("route_live", {})
        video_cfg = route_live_cfg.get("video", {})
        self.enabled = bool(video_cfg.get("enabled", False))
        self.config = config
        self.target_hwnd = int(target_hwnd)
        self.output_dir = Path(output_dir)
        self.fps = max(1.0, float(video_cfg.get("fps", 10.0)))
        self.max_width = max(0, int(video_cfg.get("max_width", 1600)))
        self.segment_seconds = max(5.0, float(video_cfg.get("segment_seconds", 30.0)))
        self.capture_factory = capture_factory
        self.video_path = self.output_dir / str(video_cfg.get("filename", "window_capture.mp4"))
        self.timeline_path = self.output_dir / "window_capture_timeline.jsonl"
        self.summary_path = self.output_dir / "window_capture_summary.json"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._started_at: float | None = None
        self._video_paths: list[Path] = []

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._record_loop,
            name="game-window-video",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            self._error = self._error or "recorder_stop_timeout"
        self._write_summary()

    @property
    def error(self) -> str | None:
        return self._error

    def _record_loop(self) -> None:
        capture = self.capture_factory(self.config)
        capture.window_handle = self.target_hwnd
        writer: cv2.VideoWriter | None = None
        self._started_at = time.time()
        next_frame_at = time.monotonic()
        frame_index = 0
        segment_index = 0
        segment_frame_index = 0
        max_segment_frames = max(1, int(round(self.fps * self.segment_seconds)))
        try:
            with self.timeline_path.open("w", encoding="utf-8") as timeline:
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    wait_seconds = next_frame_at - now
                    if wait_seconds > 0:
                        self._stop_event.wait(wait_seconds)
                        continue
                    next_frame_at += 1.0 / self.fps
                    if now - next_frame_at > 1.0:
                        next_frame_at = now

                    try:
                        frame = capture.capture_client_region()
                    except Exception as exc:
                        self._frames_dropped += 1
                        self._error = f"capture_error:{type(exc).__name__}:{exc}"
                        continue

                    output_frame = resize_for_recording(frame, self.max_width)
                    if writer is not None and segment_frame_index >= max_segment_frames:
                        writer.release()
                        writer = None
                        segment_index += 1
                        segment_frame_index = 0
                    if writer is None:
                        segment_path = self.video_path.with_name(
                            f"{self.video_path.stem}_{segment_index:04d}{self.video_path.suffix}"
                        )
                        writer, actual_path = open_video_writer(
                            segment_path,
                            output_frame.shape,
                            self.fps,
                        )
                        self._video_paths.append(actual_path)
                    writer.write(output_frame)
                    timestamp = time.time()
                    timeline.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "segment_index": segment_index,
                                "segment_frame_index": segment_frame_index,
                                "timestamp": timestamp,
                                "monotonic": now,
                            }
                        )
                        + "\n"
                    )
                    frame_index += 1
                    segment_frame_index += 1
                    self._frames_written = frame_index
                    if frame_index % max(1, int(round(self.fps))) == 0:
                        timeline.flush()
        except Exception as exc:
            self._error = f"recorder_error:{type(exc).__name__}:{exc}"
        finally:
            if writer is not None:
                writer.release()
            self._write_summary()

    def _write_summary(self) -> None:
        summary = {
            "enabled": self.enabled,
            "video_path": str(self.video_path),
            "video_paths": [str(path) for path in self._video_paths],
            "timeline_path": str(self.timeline_path),
            "fps": self.fps,
            "max_width": self.max_width,
            "segment_seconds": self.segment_seconds,
            "frames_written": self._frames_written,
            "frames_dropped": self._frames_dropped,
            "started_at": self._started_at,
            "finished_at": time.time(),
            "error": self._error,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resize_for_recording(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame
    scale = max_width / float(width)
    size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def open_video_writer(
    preferred_path: Path,
    frame_shape: tuple[int, ...],
    fps: float,
) -> tuple[cv2.VideoWriter, Path]:
    height, width = frame_shape[:2]
    candidates = (
        (preferred_path.with_suffix(".mp4"), "mp4v"),
        (preferred_path.with_suffix(".avi"), "MJPG"),
    )
    for path, codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("No supported OpenCV video writer is available")
