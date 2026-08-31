from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LatestFrame(Generic[T]):
    frame_id: int
    captured_at: float
    value: T


class LatestFrameMailbox(Generic[T]):
    """A one-item mailbox: new perception never queues behind stale frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: LatestFrame[T] | None = None
        self._last_consumed_id = -1
        self.replaced_count = 0

    def publish(self, frame: LatestFrame[T]) -> None:
        with self._lock:
            if frame.frame_id <= self._last_consumed_id:
                raise ValueError("cannot publish a frame older than the consumed frame")
            if self._latest is not None:
                if frame.frame_id <= self._latest.frame_id:
                    raise ValueError("frame ids must increase monotonically")
                self.replaced_count += 1
            self._latest = frame

    def consume_latest(self) -> LatestFrame[T] | None:
        with self._lock:
            frame = self._latest
            self._latest = None
            if frame is not None:
                self._last_consumed_id = frame.frame_id
            return frame
