from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageWriterStats:
    submitted: int
    written: int
    dropped: int
    failed: int


class AsyncImageWriter:
    """Bounded background encoder for noncritical diagnostic images."""

    _STOP = object()

    def __init__(
        self,
        *,
        max_pending: int = 16,
        write_image: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self._queue: Queue[tuple[Path, np.ndarray] | object] = Queue(maxsize=max_pending)
        self._write_image = write_image
        self._lock = Lock()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._failed = 0
        self._closed = False
        self._thread = Thread(target=self._run, name="route-image-writer", daemon=True)
        self._thread.start()

    def submit(self, path: str | Path, image: np.ndarray) -> bool:
        """Queue an immutable image copy without blocking the control loop."""

        with self._lock:
            if self._closed:
                raise RuntimeError("image writer is closed")
            if self._queue.full():
                self._dropped += 1
                return False

        job = (Path(path), np.ascontiguousarray(image).copy())
        try:
            self._queue.put_nowait(job)
        except Full:
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            self._submitted += 1
        return True

    def close(self) -> ImageWriterStats:
        """Flush accepted jobs and stop the worker. Calling twice is harmless."""

        with self._lock:
            if self._closed:
                return ImageWriterStats(
                    submitted=self._submitted,
                    written=self._written,
                    dropped=self._dropped,
                    failed=self._failed,
                )
            self._closed = True
        self._queue.join()
        self._queue.put(self._STOP)
        self._thread.join()
        return self.stats

    @property
    def stats(self) -> ImageWriterStats:
        with self._lock:
            return ImageWriterStats(
                submitted=self._submitted,
                written=self._written,
                dropped=self._dropped,
                failed=self._failed,
            )

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is self._STOP:
                    return
                path, image = job
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    written = bool(self._write_image(str(path), image))
                except Exception:
                    written = False
                with self._lock:
                    if written:
                        self._written += 1
                    else:
                        self._failed += 1
            finally:
                self._queue.task_done()
