from __future__ import annotations

import numpy as np

from vision_bot.controllers.recovery import RecoveryPhase
from vision_bot.core.world import LifeState, RecoverySignal
from vision_bot.perception.recovery import RecoverySensor


FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def _none(*_args):
    return None


def _false(*_args):
    return False


def _button(point):
    return lambda *_args: (point, object())


def _sensor(**overrides) -> RecoverySensor:
    values = {
        "release_detector": _none,
        "graveyard_detector": _none,
        "graveyard_accept_detector": _none,
        "return_detector": _none,
        "accept_detector": _none,
        "healer_target_detector": _false,
        "healer_dialog_detector": _false,
    }
    values.update(overrides)
    return RecoverySensor({}, **values)


def test_dead_state_uses_visible_graveyard_confirmation_as_release_step() -> None:
    sensor = _sensor(graveyard_accept_detector=_button((40, 50)))

    result = sensor.observe(
        FRAME,
        life=LifeState.DEAD,
        phase=RecoveryPhase.WAIT_FOR_GHOST,
    )

    assert result.signal is RecoverySignal.RELEASE_SPIRIT
    assert result.click_point == (40, 50)


def test_visible_accept_is_restart_safe_authority_while_ghost() -> None:
    sensor = _sensor(accept_detector=_button((60, 70)))

    result = sensor.observe(
        FRAME,
        life=LifeState.GHOST,
        phase=RecoveryPhase.SEEK_HEALER,
    )

    assert result.signal is RecoverySignal.ACCEPT_RESURRECTION
    assert result.click_point == (60, 70)


def test_return_to_life_fallback_is_only_consulted_after_visible_dialog_state() -> None:
    calls = []

    def return_detector(*_args):
        calls.append(True)
        return ((20, 30), object())

    sensor = _sensor(return_detector=return_detector)
    searching = sensor.observe(
        FRAME,
        life=LifeState.GHOST,
        phase=RecoveryPhase.SEEK_HEALER,
    )
    waiting = sensor.observe(
        FRAME,
        life=LifeState.GHOST,
        phase=RecoveryPhase.WAIT_FOR_RETURN,
    )

    assert searching.signal is RecoverySignal.GHOST_SEARCH
    assert len(calls) == 1
    assert waiting.signal is RecoverySignal.RETURN_TO_LIFE
    assert waiting.click_point == (20, 30)
