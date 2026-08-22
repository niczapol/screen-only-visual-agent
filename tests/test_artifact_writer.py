from pathlib import Path
from threading import Event

import numpy as np

from vision_bot.artifact_writer import AsyncImageWriter


def test_async_image_writer_flushes_accepted_images(tmp_path: Path) -> None:
    written: list[tuple[str, int]] = []

    def write_image(path: str, image: np.ndarray) -> bool:
        written.append((path, int(image[0, 0, 0])))
        return True

    writer = AsyncImageWriter(max_pending=2, write_image=write_image)
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    assert writer.submit(tmp_path / "frame.png", image)
    image[:] = 99

    stats = writer.close()

    assert written == [(str(tmp_path / "frame.png"), 7)]
    assert stats.submitted == 1
    assert stats.written == 1
    assert stats.dropped == 0
    assert stats.failed == 0
    assert writer.close() == stats


def test_async_image_writer_drops_when_bounded_queue_is_full(tmp_path: Path) -> None:
    release = Event()
    started = Event()

    def slow_write(_path: str, _image: np.ndarray) -> bool:
        started.set()
        release.wait(timeout=2.0)
        return True

    writer = AsyncImageWriter(max_pending=1, write_image=slow_write)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert writer.submit(tmp_path / "first.png", image)
    assert started.wait(timeout=1.0)
    assert writer.submit(tmp_path / "second.png", image)
    assert not writer.submit(tmp_path / "third.png", image)

    release.set()
    stats = writer.close()

    assert stats.submitted == 2
    assert stats.written == 2
    assert stats.dropped == 1
