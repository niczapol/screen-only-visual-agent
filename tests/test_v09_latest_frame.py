from __future__ import annotations

import pytest

from vision_bot.engine.latest_frame import LatestFrame, LatestFrameMailbox


def test_latest_frame_mailbox_drops_backlog_and_returns_newest() -> None:
    mailbox = LatestFrameMailbox[str]()
    mailbox.publish(LatestFrame(frame_id=1, captured_at=1.0, value="old"))
    mailbox.publish(LatestFrame(frame_id=2, captured_at=1.1, value="new"))

    latest = mailbox.consume_latest()

    assert latest is not None
    assert latest.frame_id == 2
    assert latest.value == "new"
    assert mailbox.replaced_count == 1
    assert mailbox.consume_latest() is None


def test_latest_frame_mailbox_rejects_regression() -> None:
    mailbox = LatestFrameMailbox[str]()
    mailbox.publish(LatestFrame(frame_id=2, captured_at=1.0, value="new"))

    with pytest.raises(ValueError, match="increase monotonically"):
        mailbox.publish(LatestFrame(frame_id=1, captured_at=0.9, value="old"))
