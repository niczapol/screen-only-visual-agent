from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vision_bot.controllers.recovery import RecoveryPhase
from vision_bot.core.world import LifeState, RecoveryPerception, RecoverySignal
from vision_bot.death_recovery import (
    detect_release_spirit_button,
    detect_resurrection_accept_button,
    detect_return_to_graveyard_accept_button,
    detect_return_to_graveyard_button,
    detect_return_to_life_button,
)
from vision_bot.runtime_markers import (
    detect_spirit_healer_dialog_marker,
    detect_spirit_healer_target_marker,
)


ButtonDetector = Callable[
    [np.ndarray, dict[str, Any] | None],
    tuple[tuple[int, int], object] | None,
]
MarkerDetector = Callable[[np.ndarray, dict[str, Any] | None], bool]


class RecoverySensor:
    """Translate existing visible recovery detectors into v0.9 evidence.

    Potentially broad red-button detectors are life/geometry gated.  A visible
    upper-center resurrection Accept remains actionable after a runtime restart;
    the Return-to-Life fallback is never consulted during ordinary ghost search.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        release_detector: ButtonDetector = detect_release_spirit_button,
        graveyard_detector: ButtonDetector = detect_return_to_graveyard_button,
        graveyard_accept_detector: ButtonDetector = detect_return_to_graveyard_accept_button,
        return_detector: ButtonDetector = detect_return_to_life_button,
        accept_detector: ButtonDetector = detect_resurrection_accept_button,
        healer_target_detector: MarkerDetector = detect_spirit_healer_target_marker,
        healer_dialog_detector: MarkerDetector = detect_spirit_healer_dialog_marker,
    ) -> None:
        self.config = config
        self.release_detector = release_detector
        self.graveyard_detector = graveyard_detector
        self.graveyard_accept_detector = graveyard_accept_detector
        self.return_detector = return_detector
        self.accept_detector = accept_detector
        self.healer_target_detector = healer_target_detector
        self.healer_dialog_detector = healer_dialog_detector

    def observe(
        self,
        frame: np.ndarray,
        *,
        life: LifeState | None,
        phase: RecoveryPhase,
    ) -> RecoveryPerception:
        if life is LifeState.RESURRECTION_SICKNESS:
            return RecoveryPerception(signal=RecoverySignal.SICKNESS)
        if life is LifeState.ALIVE or life is None:
            return RecoveryPerception()

        if life is LifeState.DEAD:
            # The target client may require Return to Graveyard followed by Yes. Both
            # are represented as the same bounded release transition.
            for detector in (
                self.graveyard_accept_detector,
                self.graveyard_detector,
                self.release_detector,
            ):
                result = detector(frame, self.config)
                if result is not None:
                    return RecoveryPerception(
                        signal=RecoverySignal.RELEASE_SPIRIT,
                        click_point=result[0],
                    )
            return RecoveryPerception(signal=RecoverySignal.RELEASE_SPIRIT)

        dialog_visible = self.healer_dialog_detector(frame, self.config)
        healer_targeted = self.healer_target_detector(frame, self.config)
        # The centered resurrection confirmation is authoritative while the
        # character is visibly a ghost, even after a process restart has lost
        # the controller phase that opened it.  Its detector is constrained to
        # the upper-center modal-button geometry, away from normal action bars
        # and the server poll prompt.
        accepted = self.accept_detector(frame, self.config)
        if accepted is not None:
            return RecoveryPerception(
                signal=RecoverySignal.ACCEPT_RESURRECTION,
                click_point=accepted[0],
            )

        if phase in {RecoveryPhase.WAIT_FOR_RETURN, RecoveryPhase.RETURN_TO_LIFE}:
            returned = self.return_detector(frame, self.config)
            if returned is not None:
                return RecoveryPerception(
                    signal=RecoverySignal.RETURN_TO_LIFE,
                    click_point=returned[0],
                )
        if dialog_visible:
            return RecoveryPerception(signal=RecoverySignal.HEALER_DIALOG)
        if healer_targeted:
            return RecoveryPerception(signal=RecoverySignal.HEALER_TARGETED)
        return RecoveryPerception(signal=RecoverySignal.GHOST_SEARCH)
