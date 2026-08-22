import numpy as np

from vision_bot.window_video import resize_for_recording


def test_resize_for_recording_preserves_aspect_ratio():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

    resized = resize_for_recording(frame, 1600)

    assert resized.shape == (900, 1600, 3)


def test_resize_for_recording_keeps_small_frame_unchanged():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized = resize_for_recording(frame, 1600)

    assert resized is frame
