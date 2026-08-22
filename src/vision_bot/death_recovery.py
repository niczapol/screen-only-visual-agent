from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_bot.game_state import GameState
from vision_bot.regions import resolve_region
from vision_bot.runtime_markers import (
    detect_spirit_healer_dialog_marker,
    detect_spirit_healer_target_marker,
)
from vision_bot.screen_objects import BoundingBox


@dataclass(frozen=True)
class DeathRecoveryDecision:
    action: str
    attempts: int
    click_point: tuple[int, int] | None = None
    button_bbox: BoundingBox | None = None
    wait_seconds: float = 0.0
    reason: str = "none"
    mouse_button: str = "left"
    key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "attempts": self.attempts,
            "click_point": list(self.click_point) if self.click_point is not None else None,
            "button_bbox": _bbox_to_dict(self.button_bbox),
            "wait_seconds": self.wait_seconds,
            "reason": self.reason,
            "mouse_button": self.mouse_button,
            "key": self.key,
        }


@dataclass
class DeathRecoveryController:
    enabled: bool
    max_attempts: int
    release_wait_seconds: float
    resurrection_mode: str
    safe_zone_resurrect_enabled: bool
    safe_zone_resurrect_max_attempts: int
    safe_zone_resurrect_wait_seconds: float
    safe_zone_destination_max_attempts: int
    safe_zone_destination_wait_seconds: float
    return_to_graveyard_enabled: bool
    return_to_graveyard_max_attempts: int
    return_to_graveyard_wait_seconds: float
    return_to_graveyard_accept_max_attempts: int
    return_to_graveyard_accept_wait_seconds: float
    spirit_healer_enabled: bool
    spirit_healer_max_attempts: int
    spirit_healer_wait_seconds: float
    spirit_healer_approach_max_attempts: int
    spirit_healer_approach_wait_seconds: float
    spirit_healer_search_max_attempts: int
    spirit_healer_search_wait_seconds: float
    spirit_healer_target_interact_enabled: bool
    spirit_healer_target_interact_max_attempts: int
    spirit_healer_target_interact_wait_seconds: float
    spirit_healer_target_key: str
    spirit_healer_target_wait_seconds: float
    spirit_healer_interact_key: str
    spirit_healer_follow_target_enabled: bool
    spirit_healer_follow_target_key: str
    spirit_healer_follow_target_wait_seconds: float
    return_to_life_max_attempts: int
    return_to_life_wait_seconds: float
    accept_resurrection_max_attempts: int
    accept_resurrection_wait_seconds: float
    resurrect_sickness_wait_seconds: float
    resurrect_sickness_wait_enabled: bool = True
    attempts: int = 0
    safe_zone_resurrect_attempts: int = 0
    safe_zone_destination_attempts: int = 0
    return_to_graveyard_attempts: int = 0
    return_to_graveyard_accept_attempts: int = 0
    spirit_healer_attempts: int = 0
    spirit_healer_approach_attempts: int = 0
    spirit_healer_search_attempts: int = 0
    spirit_healer_target_interact_attempts: int = 0
    return_to_life_attempts: int = 0
    accept_resurrection_attempts: int = 0
    waiting_until: float = 0.0
    death_seen: bool = False
    release_clicked: bool = False
    safe_zone_resurrect_clicked: bool = False
    safe_zone_destination_clicked: bool = False
    return_to_graveyard_clicked: bool = False
    return_to_graveyard_confirmed: bool = False
    spirit_healer_clicked: bool = False
    spirit_healer_targeted: bool = False
    spirit_healer_hovered: bool = False
    spirit_healer_hover_target: tuple[tuple[int, int], BoundingBox] | None = None
    spirit_healer_confirmed_visual_target: tuple[tuple[int, int], BoundingBox] | None = None
    return_to_life_clicked: bool = False
    accept_resurrection_clicked: bool = False
    resurrect_sickness_wait_consumed: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeathRecoveryController:
        recovery_cfg = _recovery_cfg(config)
        return cls(
            enabled=bool(recovery_cfg.get("enabled", True)),
            max_attempts=max(0, int(recovery_cfg.get("max_attempts", 2))),
            release_wait_seconds=max(0.0, float(recovery_cfg.get("release_wait_seconds", 2.5))),
            resurrection_mode=_normalize_resurrection_mode(recovery_cfg.get("resurrection_mode", "spirit_healer")),
            safe_zone_resurrect_enabled=bool(recovery_cfg.get("safe_zone_resurrect_enabled", False)),
            safe_zone_resurrect_max_attempts=max(
                0,
                int(recovery_cfg.get("safe_zone_resurrect_max_attempts", 2)),
            ),
            safe_zone_resurrect_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("safe_zone_resurrect_wait_seconds", 2.5)),
            ),
            safe_zone_destination_max_attempts=max(
                0,
                int(recovery_cfg.get("safe_zone_destination_max_attempts", 2)),
            ),
            safe_zone_destination_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("safe_zone_destination_wait_seconds", 2.5)),
            ),
            return_to_graveyard_enabled=bool(
                recovery_cfg.get("return_to_graveyard_enabled", True)
            ),
            return_to_graveyard_max_attempts=max(
                0,
                int(recovery_cfg.get("return_to_graveyard_max_attempts", 2)),
            ),
            return_to_graveyard_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("return_to_graveyard_wait_seconds", 3.0)),
            ),
            return_to_graveyard_accept_max_attempts=max(
                0,
                int(recovery_cfg.get("return_to_graveyard_accept_max_attempts", 2)),
            ),
            return_to_graveyard_accept_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("return_to_graveyard_accept_wait_seconds", 3.0)),
            ),
            spirit_healer_enabled=bool(recovery_cfg.get("spirit_healer_enabled", True)),
            spirit_healer_max_attempts=max(
                0,
                int(recovery_cfg.get("spirit_healer_max_attempts", 4)),
            ),
            spirit_healer_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_wait_seconds", 1.0)),
            ),
            spirit_healer_approach_max_attempts=max(
                0,
                int(recovery_cfg.get("spirit_healer_approach_max_attempts", 3)),
            ),
            spirit_healer_approach_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_approach_wait_seconds", 0.45)),
            ),
            spirit_healer_search_max_attempts=max(
                0,
                int(recovery_cfg.get("spirit_healer_search_max_attempts", 8)),
            ),
            spirit_healer_search_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_search_wait_seconds", 0.35)),
            ),
            spirit_healer_target_interact_enabled=bool(
                recovery_cfg.get("spirit_healer_target_interact_enabled", True)
            ),
            spirit_healer_target_interact_max_attempts=max(
                0,
                int(recovery_cfg.get("spirit_healer_target_interact_max_attempts", 8)),
            ),
            spirit_healer_target_interact_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_target_interact_wait_seconds", 0.65)),
            ),
            spirit_healer_target_key=str(
                recovery_cfg.get("spirit_healer_target_key", "C")
            ).strip().upper()
            or "C",
            spirit_healer_target_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_target_wait_seconds", 0.15)),
            ),
            spirit_healer_interact_key=str(recovery_cfg.get("spirit_healer_interact_key", "G")).strip().upper()
            or "G",
            spirit_healer_follow_target_enabled=bool(
                recovery_cfg.get("spirit_healer_follow_target_enabled", False)
            ),
            spirit_healer_follow_target_key=str(
                recovery_cfg.get("spirit_healer_follow_target_key", "F6")
            ).strip().upper()
            or "F6",
            spirit_healer_follow_target_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("spirit_healer_follow_target_wait_seconds", 1.25)),
            ),
            return_to_life_max_attempts=max(
                0,
                int(recovery_cfg.get("return_to_life_max_attempts", 3)),
            ),
            return_to_life_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("return_to_life_wait_seconds", 1.0)),
            ),
            accept_resurrection_max_attempts=max(
                0,
                int(recovery_cfg.get("accept_resurrection_max_attempts", 2)),
            ),
            accept_resurrection_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("accept_resurrection_wait_seconds", 1.25)),
            ),
            resurrect_sickness_wait_seconds=max(
                0.0,
                float(recovery_cfg.get("resurrect_sickness_wait_seconds", 600.0)),
            ),
            resurrect_sickness_wait_enabled=bool(recovery_cfg.get("resurrect_sickness_wait_enabled", True)),
        )

    def decide(
        self,
        frame: np.ndarray,
        game_state: GameState,
        config: dict[str, Any],
        *,
        now: float | None = None,
    ) -> DeathRecoveryDecision:
        timestamp = time.monotonic() if now is None else now
        if not self.enabled:
            return DeathRecoveryDecision("death_recovery_disabled", attempts=self.attempts, reason="disabled")

        self.death_seen = True
        if self.resurrection_mode == "spirit_healer":
            dialog_decision = self.decide_open_dialog(frame, config, now=timestamp)
            if dialog_decision is not None:
                return dialog_decision
        if timestamp < self.waiting_until:
            return DeathRecoveryDecision(
                "death_recovery_wait",
                attempts=self.attempts,
                wait_seconds=max(0.0, self.waiting_until - timestamp),
                reason="release_wait",
            )

        ghost_action_buttons_visible = int(getattr(game_state, "ghost_button_count", 0)) >= 2

        # The target client corpse screen uses a yellow "Return to Graveyard" action
        # instead of the red release button used by the stock client. Prefer
        # that explicit left-side action before looking for generic red UI;
        # the latter can otherwise match the Death Recap window below it.
        if self.resurrection_mode == "spirit_healer" and not self.return_to_graveyard_confirmed:
            if (
                self.return_to_graveyard_enabled
                and detect_return_to_graveyard_accept_button(frame, config) is not None
            ):
                return self._decide_spirit_healer(frame, config, timestamp)
            graveyard_target = (
                detect_return_to_graveyard_button(frame, config)
                if self.return_to_graveyard_enabled
                else None
            )
            if graveyard_target is not None:
                return self._return_to_graveyard(graveyard_target, timestamp)

        if self.resurrection_mode == "safe_zone":
            safe_zone_resurrect_target = detect_safe_zone_resurrect_button(frame, config)
            if game_state.ghost_visual or ghost_action_buttons_visible or safe_zone_resurrect_target is not None:
                return self._decide_safe_zone(frame, config, timestamp, safe_zone_resurrect_target)

        if game_state.ghost_visual or ghost_action_buttons_visible:
            if self.resurrection_mode == "spirit_healer":
                return self._decide_spirit_healer(frame, config, timestamp)
            return DeathRecoveryDecision(
                "death_recovery_wait",
                attempts=self.attempts,
                wait_seconds=self.release_wait_seconds,
                reason="ghost_visual",
            )

        if self.attempts >= self.max_attempts:
            return DeathRecoveryDecision("death_recovery_failed", attempts=self.attempts, reason="max_attempts")

        target = detect_release_spirit_button(frame, config)
        if target is None:
            target = _fallback_release_click(frame, config)
        if target is None:
            return DeathRecoveryDecision("death_recovery_failed", attempts=self.attempts, reason="release_button_missing")

        self.attempts += 1
        self.release_clicked = True
        self.waiting_until = timestamp + self.release_wait_seconds
        click_point, button_bbox = target
        return DeathRecoveryDecision(
            "death_recovery_release_spirit",
            attempts=self.attempts,
            click_point=click_point,
            button_bbox=button_bbox,
            wait_seconds=self.release_wait_seconds,
            reason="release_button",
        )

    def decide_open_dialog(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        *,
        now: float | None = None,
    ) -> DeathRecoveryDecision | None:
        """Handle visible resurrection dialogs before any movement or wait."""
        if not self.enabled or self.resurrection_mode != "spirit_healer":
            return None
        timestamp = time.monotonic() if now is None else now
        recovery_cfg = _recovery_cfg(config)
        if bool(recovery_cfg.get("death_recap_dismiss_enabled", True)) and detect_death_recap_panel(
            frame,
            config,
        ):
            return DeathRecoveryDecision(
                "death_recovery_dismiss_death_recap",
                attempts=self.attempts,
                wait_seconds=max(
                    0.0,
                    float(recovery_cfg.get("death_recap_dismiss_wait_seconds", 0.25)),
                ),
                reason="death_recap_visible",
                key="ESC",
            )
        spirit_dialog_visible = detect_spirit_healer_dialog_marker(frame, config)

        graveyard_accept_target = (
            detect_return_to_graveyard_accept_button(frame, config)
            if self.return_to_graveyard_enabled
            else None
        )
        if graveyard_accept_target is not None:
            if self.return_to_graveyard_accept_attempts > 0 and timestamp < self.waiting_until:
                return DeathRecoveryDecision(
                    "death_recovery_wait",
                    attempts=self.return_to_graveyard_accept_attempts,
                    wait_seconds=max(0.0, self.waiting_until - timestamp),
                    reason="return_to_graveyard_accept_settle",
                )
            if self.return_to_graveyard_accept_attempts >= self.return_to_graveyard_accept_max_attempts:
                return DeathRecoveryDecision(
                    "death_recovery_failed",
                    attempts=self.return_to_graveyard_accept_attempts,
                    reason="return_to_graveyard_accept_max_attempts",
                )
            self.return_to_graveyard_accept_attempts += 1
            self.return_to_graveyard_clicked = True
            self.return_to_graveyard_confirmed = True
            self.waiting_until = timestamp + self.return_to_graveyard_accept_wait_seconds
            click_point, button_bbox = graveyard_accept_target
            return DeathRecoveryDecision(
                "death_recovery_accept_return_to_graveyard",
                attempts=self.return_to_graveyard_accept_attempts,
                click_point=click_point,
                button_bbox=button_bbox,
                wait_seconds=self.return_to_graveyard_accept_wait_seconds,
                reason="return_to_graveyard_yes_button",
            )

        if self.return_to_graveyard_enabled and not self.return_to_graveyard_confirmed:
            graveyard_target = detect_return_to_graveyard_button(frame, config)
            if graveyard_target is not None:
                if self.return_to_graveyard_attempts > 0 and timestamp < self.waiting_until:
                    return DeathRecoveryDecision(
                        "death_recovery_wait",
                        attempts=self.return_to_graveyard_attempts,
                        wait_seconds=max(0.0, self.waiting_until - timestamp),
                        reason="return_to_graveyard_settle",
                    )
                return self._return_to_graveyard(graveyard_target, timestamp)

        healer_interaction_confirmed = (
            self.return_to_graveyard_confirmed
            or self.spirit_healer_clicked
            or self.return_to_life_clicked
            or spirit_dialog_visible
        )
        # An interact attempt is not proof that the Return to Life stage was
        # reached: the client can answer "need to be closer" while unrelated
        # red UI (for example Death Recap) resembles an Accept button.  Accept
        # is valid only after the actual healer dialog was visible or its
        # Return to Life action was clicked.
        accept_stage_confirmed = self.return_to_life_clicked or spirit_dialog_visible
        accept_target = (
            detect_resurrection_accept_button(frame, config)
            if accept_stage_confirmed
            else None
        )
        if accept_target is not None:
            if self.accept_resurrection_attempts > 0 and timestamp < self.waiting_until:
                return DeathRecoveryDecision(
                    "death_recovery_wait",
                    attempts=self.accept_resurrection_attempts,
                    wait_seconds=max(0.0, self.waiting_until - timestamp),
                    reason="accept_resurrection_settle",
                )
            if self.accept_resurrection_attempts >= self.accept_resurrection_max_attempts:
                return DeathRecoveryDecision(
                    "death_recovery_failed",
                    attempts=self.accept_resurrection_attempts,
                    reason="accept_resurrection_max_attempts",
                )
            self.accept_resurrection_attempts += 1
            self.return_to_life_clicked = True
            self.accept_resurrection_clicked = True
            self.waiting_until = timestamp + self.accept_resurrection_wait_seconds
            click_point, button_bbox = accept_target
            return DeathRecoveryDecision(
                "death_recovery_accept_resurrection",
                attempts=self.accept_resurrection_attempts,
                click_point=click_point,
                button_bbox=button_bbox,
                wait_seconds=self.accept_resurrection_wait_seconds,
                reason="accept_resurrection_button",
            )

        return_target = (
            detect_return_to_life_button(frame, config)
            if healer_interaction_confirmed
            else None
        )
        if return_target is not None:
            if self.return_to_life_attempts > 0 and timestamp < self.waiting_until:
                return DeathRecoveryDecision(
                    "death_recovery_wait",
                    attempts=self.return_to_life_attempts,
                    wait_seconds=max(0.0, self.waiting_until - timestamp),
                    reason="return_to_life_settle",
                )
            if self.return_to_life_attempts >= self.return_to_life_max_attempts:
                return DeathRecoveryDecision(
                    "death_recovery_failed",
                    attempts=self.return_to_life_attempts,
                    reason="return_to_life_max_attempts",
                )
            self.return_to_life_attempts += 1
            self.return_to_graveyard_confirmed = True
            self.spirit_healer_clicked = True
            self.return_to_life_clicked = True
            self.waiting_until = timestamp + self.return_to_life_wait_seconds
            click_point, button_bbox = return_target
            return DeathRecoveryDecision(
                "death_recovery_return_to_life",
                attempts=self.return_to_life_attempts,
                click_point=click_point,
                button_bbox=button_bbox,
                wait_seconds=self.return_to_life_wait_seconds,
                reason="return_to_life_button",
            )

        if spirit_dialog_visible:
            return DeathRecoveryDecision(
                "death_recovery_wait",
                attempts=self.spirit_healer_attempts,
                wait_seconds=0.10,
                reason="spirit_healer_dialog_visible_button_pending",
            )
        return None

    def _decide_safe_zone(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        timestamp: float,
        safe_zone_resurrect_target: tuple[tuple[int, int], BoundingBox] | None,
    ) -> DeathRecoveryDecision:
        destination_target = detect_safe_zone_destination_button(frame, config)
        if destination_target is not None:
            if self.safe_zone_destination_attempts >= self.safe_zone_destination_max_attempts:
                return DeathRecoveryDecision(
                    "death_recovery_failed",
                    attempts=self.safe_zone_destination_attempts,
                    reason="safe_zone_destination_max_attempts",
                )
            self.safe_zone_destination_attempts += 1
            self.safe_zone_destination_clicked = True
            self.waiting_until = timestamp + self.safe_zone_destination_wait_seconds
            click_point, button_bbox = destination_target
            return DeathRecoveryDecision(
                "death_recovery_safe_zone_destination",
                attempts=self.safe_zone_destination_attempts,
                click_point=click_point,
                button_bbox=button_bbox,
                wait_seconds=self.safe_zone_destination_wait_seconds,
                reason="safe_zone_destination_button",
            )

        if (
            self.safe_zone_resurrect_enabled
            and self.safe_zone_resurrect_attempts < self.safe_zone_resurrect_max_attempts
        ):
            target = safe_zone_resurrect_target
            if target is not None:
                self.safe_zone_resurrect_attempts += 1
                self.safe_zone_resurrect_clicked = True
                self.waiting_until = timestamp + self.safe_zone_resurrect_wait_seconds
                click_point, button_bbox = target
                return DeathRecoveryDecision(
                    "death_recovery_resurrect_safe_zone",
                    attempts=self.safe_zone_resurrect_attempts,
                    click_point=click_point,
                    button_bbox=button_bbox,
                    wait_seconds=self.safe_zone_resurrect_wait_seconds,
                    reason="safe_zone_resurrect_button",
                )
        return DeathRecoveryDecision(
            "death_recovery_wait",
            attempts=self.attempts,
            wait_seconds=self.release_wait_seconds,
            reason="ghost_visual",
        )

    def _decide_spirit_healer(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        timestamp: float,
    ) -> DeathRecoveryDecision:
        if not self.spirit_healer_enabled:
            return DeathRecoveryDecision(
                "death_recovery_failed",
                attempts=self.spirit_healer_attempts,
                reason="spirit_healer_disabled",
            )

        recovery_cfg = _recovery_cfg(config)
        graveyard_visual_candidate = None
        if bool(recovery_cfg.get("spirit_healer_graveyard_visual_fallback_enabled", True)):
            graveyard_visual_candidate = detect_spirit_healer_graveyard_candidate(frame, config)
        spirit_healer_target_frame = detect_spirit_healer_target_frame(frame, config)
        spirit_healer_tooltip = detect_spirit_healer_tooltip(frame, config)
        if spirit_healer_target_frame is not None:
            self.return_to_graveyard_confirmed = True

        hover_failed = False
        if self.spirit_healer_hovered:
            if spirit_healer_tooltip is not None and self.spirit_healer_hover_target is not None:
                self.spirit_healer_confirmed_visual_target = self.spirit_healer_hover_target
                self.return_to_graveyard_confirmed = True
            else:
                hover_failed = True
            self.spirit_healer_hovered = False
            self.spirit_healer_hover_target = None

        spirit_healer_target_marker = detect_spirit_healer_target_marker(frame, config)
        if spirit_healer_target_marker:
            self.return_to_graveyard_confirmed = True

        if self.spirit_healer_clicked:
            interaction_too_far = detect_spirit_healer_interaction_error(frame, config)
            if interaction_too_far and self.spirit_healer_confirmed_visual_target is not None:
                approach_decision = self._approach_spirit_healer_if_possible(
                    frame,
                    config,
                    timestamp,
                    reason="spirit_healer_interaction_too_far",
                    spirit_target=self.spirit_healer_confirmed_visual_target,
                )
                if approach_decision is not None:
                    self.spirit_healer_confirmed_visual_target = None
                    return approach_decision
            if (
                interaction_too_far
                and (spirit_healer_target_frame is not None or spirit_healer_target_marker)
            ):
                # The selected target proves identity. When the same frame also
                # contains the high-confidence graveyard silhouette, its visible
                # click point supplies the missing bearing and makes a bounded
                # forward approach safe without requiring a second tooltip pass.
                if graveyard_visual_candidate is not None:
                    approach_decision = self._approach_spirit_healer_if_possible(
                        frame,
                        config,
                        timestamp,
                        reason="spirit_healer_target_confirmed_visual_approach",
                        spirit_target=graveyard_visual_candidate,
                    )
                    if approach_decision is not None:
                        return approach_decision
                # A selected target proves identity, but WoW does not expose its
                # bearing through the visible target frame. Moving straight ahead
                # here is therefore blind and can carry the ghost out of the
                # graveyard. Let the live loop perform its coordinate-bounded
                # cemetery search instead; directional approach remains available
                # above when a visually localized healer supplied a click point.
                return self._search_spirit_healer(
                    timestamp,
                    reason="spirit_healer_target_confirmed_too_far_no_direction",
                )
            if _attempt_available(
                self.spirit_healer_search_attempts,
                self.spirit_healer_search_max_attempts,
            ):
                return self._search_spirit_healer(
                    timestamp,
                    reason=(
                        "spirit_healer_interaction_error"
                        if interaction_too_far
                        else "spirit_healer_macro_no_dialog"
                    ),
                )
            self.spirit_healer_clicked = False
            self.spirit_healer_targeted = False

        if (
            bool(_recovery_cfg(config).get("spirit_healer_interact_selected_target_enabled", True))
            and self.spirit_healer_targeted
            and (spirit_healer_target_frame is not None or spirit_healer_target_marker)
        ):
            self.spirit_healer_clicked = True
            self.waiting_until = timestamp + self.spirit_healer_target_interact_wait_seconds
            return DeathRecoveryDecision(
                "death_recovery_interact_spirit_healer_target",
                attempts=self.spirit_healer_target_interact_attempts,
                wait_seconds=self.spirit_healer_target_interact_wait_seconds,
                reason="spirit_healer_target_frame_interact",
                key=self.spirit_healer_interact_key,
            )

        if bool(_recovery_cfg(config).get("spirit_healer_visual_primary", False)):
            visual_decision = self._right_click_visible_spirit_healer(frame, config, timestamp)
            if visual_decision is not None:
                return visual_decision

        if (
            self.return_to_graveyard_confirmed
            and bool(recovery_cfg.get("spirit_healer_graveyard_visual_fallback_enabled", True))
            and self.spirit_healer_target_interact_attempts
            >= int(recovery_cfg.get("spirit_healer_graveyard_visual_fallback_after_attempts", 3))
        ):
            graveyard_visual_target = graveyard_visual_candidate
            if graveyard_visual_target is not None:
                if hover_failed:
                    return self._search_spirit_healer(
                        timestamp,
                        reason="spirit_healer_visual_hover_unconfirmed",
                    )
                if self.spirit_healer_confirmed_visual_target is None:
                    self.spirit_healer_hovered = True
                    self.spirit_healer_hover_target = graveyard_visual_target
                    self.waiting_until = timestamp + float(
                        recovery_cfg.get("spirit_healer_visual_hover_wait_seconds", 0.30)
                    )
                    click_point, button_bbox = graveyard_visual_target
                    return DeathRecoveryDecision(
                        "death_recovery_hover_spirit_healer",
                        attempts=self.spirit_healer_attempts,
                        click_point=click_point,
                        button_bbox=button_bbox,
                        wait_seconds=float(
                            recovery_cfg.get("spirit_healer_visual_hover_wait_seconds", 0.30)
                        ),
                        reason="graveyard_visual_candidate_needs_tooltip_confirmation",
                    )
                self.spirit_healer_attempts += 1
                self.spirit_healer_clicked = True
                self.spirit_healer_targeted = False
                self.waiting_until = timestamp + self.spirit_healer_wait_seconds
                click_point, button_bbox = self.spirit_healer_confirmed_visual_target
                return DeathRecoveryDecision(
                    "death_recovery_right_click_spirit_healer",
                    attempts=self.spirit_healer_attempts,
                    click_point=click_point,
                    button_bbox=button_bbox,
                    wait_seconds=self.spirit_healer_wait_seconds,
                    reason="graveyard_high_confidence_visual_fallback",
                    mouse_button="right",
                )

        if (
            self.spirit_healer_target_interact_enabled
            and (
                self.spirit_healer_targeted
                or _attempt_available(
                    self.spirit_healer_target_interact_attempts,
                    self.spirit_healer_target_interact_max_attempts,
                )
            )
        ):
            if not self.spirit_healer_targeted:
                self.spirit_healer_target_interact_attempts += 1
                self.spirit_healer_targeted = True
                self.waiting_until = timestamp + self.spirit_healer_target_wait_seconds
                return DeathRecoveryDecision(
                    "death_recovery_target_spirit_healer",
                    attempts=self.spirit_healer_target_interact_attempts,
                    wait_seconds=self.spirit_healer_target_wait_seconds,
                    reason="spirit_healer_target_macro_primary",
                    key=self.spirit_healer_target_key,
                )

            self.spirit_healer_targeted = False
            self.spirit_healer_clicked = True
            self.waiting_until = timestamp + self.spirit_healer_target_interact_wait_seconds
            return DeathRecoveryDecision(
                "death_recovery_interact_spirit_healer_target",
                attempts=self.spirit_healer_target_interact_attempts,
                wait_seconds=self.spirit_healer_target_interact_wait_seconds,
                reason="spirit_healer_interact_primary",
                key=self.spirit_healer_interact_key,
            )

        visual_decision = self._right_click_visible_spirit_healer(frame, config, timestamp)
        if visual_decision is None:
            if self.spirit_healer_attempts >= self.spirit_healer_max_attempts:
                return DeathRecoveryDecision(
                    "death_recovery_failed",
                    attempts=self.spirit_healer_attempts,
                    reason="spirit_healer_max_attempts",
                )
            if _attempt_available(
                self.spirit_healer_search_attempts,
                self.spirit_healer_search_max_attempts,
            ):
                return self._search_spirit_healer(timestamp, reason="spirit_healer_missing")
            return DeathRecoveryDecision(
                "death_recovery_failed",
                attempts=self.spirit_healer_attempts,
                reason="spirit_healer_missing_after_search",
            )
        return visual_decision

    def _return_to_graveyard(
        self,
        target: tuple[tuple[int, int], BoundingBox],
        timestamp: float,
    ) -> DeathRecoveryDecision:
        if self.return_to_graveyard_attempts >= self.return_to_graveyard_max_attempts:
            return DeathRecoveryDecision(
                "death_recovery_failed",
                attempts=self.return_to_graveyard_attempts,
                reason="return_to_graveyard_max_attempts",
            )
        self.return_to_graveyard_attempts += 1
        self.return_to_graveyard_clicked = True
        self.waiting_until = timestamp + self.return_to_graveyard_wait_seconds
        click_point, button_bbox = target
        return DeathRecoveryDecision(
            "death_recovery_return_to_graveyard",
            attempts=self.return_to_graveyard_attempts,
            click_point=click_point,
            button_bbox=button_bbox,
            wait_seconds=self.return_to_graveyard_wait_seconds,
            reason="return_to_graveyard_button",
        )

    def _right_click_visible_spirit_healer(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        timestamp: float,
    ) -> DeathRecoveryDecision | None:
        if self.spirit_healer_attempts >= self.spirit_healer_max_attempts:
            return None
        spirit_target = detect_spirit_healer_target(frame, config)
        if spirit_target is None:
            return None
        self.spirit_healer_attempts += 1
        self.spirit_healer_clicked = True
        self.waiting_until = timestamp + self.spirit_healer_wait_seconds
        click_point, button_bbox = spirit_target
        return DeathRecoveryDecision(
            "death_recovery_right_click_spirit_healer",
            attempts=self.spirit_healer_attempts,
            click_point=click_point,
            button_bbox=button_bbox,
            wait_seconds=self.spirit_healer_wait_seconds,
            reason="spirit_healer_target",
            mouse_button="right",
        )

    def _approach_spirit_healer_if_possible(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        timestamp: float,
        *,
        reason: str = "return_to_life_missing_after_spirit_click",
        spirit_target: tuple[tuple[int, int], BoundingBox] | None = None,
    ) -> DeathRecoveryDecision | None:
        if not _attempt_available(
            self.spirit_healer_approach_attempts,
            self.spirit_healer_approach_max_attempts,
        ):
            return None

        if spirit_target is None:
            spirit_target = detect_spirit_healer_target(frame, config)
        if spirit_target is None:
            return None

        self.spirit_healer_approach_attempts += 1
        self.spirit_healer_clicked = False
        self.spirit_healer_targeted = False
        self.waiting_until = timestamp + self.spirit_healer_approach_wait_seconds
        click_point, button_bbox = spirit_target
        return DeathRecoveryDecision(
            "death_recovery_approach_spirit_healer",
            attempts=self.spirit_healer_approach_attempts,
            click_point=click_point,
            button_bbox=button_bbox,
            wait_seconds=self.spirit_healer_approach_wait_seconds,
            reason=reason,
        )

    def _search_spirit_healer(
        self,
        timestamp: float,
        *,
        reason: str,
    ) -> DeathRecoveryDecision:
        self.spirit_healer_clicked = False
        self.spirit_healer_targeted = False
        self.spirit_healer_hovered = False
        self.spirit_healer_hover_target = None
        self.spirit_healer_confirmed_visual_target = None
        self.spirit_healer_search_attempts += 1
        self.waiting_until = timestamp + self.spirit_healer_search_wait_seconds
        return DeathRecoveryDecision(
            "death_recovery_search_spirit_healer",
            attempts=self.spirit_healer_search_attempts,
            wait_seconds=self.spirit_healer_search_wait_seconds,
            reason=reason,
        )

    def consume_resurrected_wait(self, game_state: GameState) -> float | None:
        if (
            not self.enabled
            or not self.resurrect_sickness_wait_enabled
            or self.resurrect_sickness_wait_consumed
            or not self.death_seen
            or not (
                self.release_clicked
                or self.safe_zone_resurrect_clicked
                or self.safe_zone_destination_clicked
                or self.accept_resurrection_clicked
            )
            or game_state.death_or_blocking_modal
        ):
            return None
        self.resurrect_sickness_wait_consumed = True
        return self.resurrect_sickness_wait_seconds


def detect_release_spirit_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    region_cfg = recovery_cfg.get("release_button_region", {"x": 640, "y": 70, "width": 1280, "height": 450})
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 30]), np.array([14, 255, 230])),
        cv2.inRange(hsv, np.array([168, 70, 30]), np.array([179, 255, 230])),
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    candidates = _release_button_candidates(red_mask, offset=(roi_x, roi_y), frame_shape=frame.shape, config=cfg)
    if not candidates:
        return None

    candidates.sort(key=lambda bbox: (bbox.y, bbox.x))
    button = candidates[0]
    return ((button.x + button.width // 2, button.y + button.height // 2), button)


def detect_safe_zone_resurrect_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    region_cfg = recovery_cfg.get(
        "safe_zone_resurrect_button_region",
        {"x": 800, "y": 95, "width": 900, "height": 190},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array(recovery_cfg.get("safe_zone_resurrect_hsv_lower", [12, 70, 70]), dtype=np.uint8),
        np.array(recovery_cfg.get("safe_zone_resurrect_hsv_upper", [50, 255, 255]), dtype=np.uint8),
    )
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((13, 41), dtype=np.uint8))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, np.ones((5, 15), dtype=np.uint8))

    candidates = _safe_zone_resurrect_candidates(
        yellow_mask,
        offset=(roi_x, roi_y),
        frame_shape=frame.shape,
        config=cfg,
    )
    if not candidates:
        fallback = _fallback_safe_zone_resurrect_click(frame, config=cfg)
        if fallback is not None:
            return fallback
        return None

    candidates.sort(key=lambda bbox: (bbox.x + bbox.width // 2), reverse=True)
    button_text = candidates[0]
    return ((button_text.x + button_text.width // 2, button_text.y + button_text.height // 2), button_text)


def detect_safe_zone_destination_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    return detect_release_spirit_button(frame, config)


def detect_return_to_graveyard_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None
    return _detect_ocr_phrase_button(
        frame,
        config or {},
        region_key="return_to_graveyard_region",
        default_region={"x": 1020, "y": 120, "width": 300, "height": 120},
        words=("return", "to", "graveyard"),
    )


def detect_death_recap_panel(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    """Recognize the optional Death Recap panel before dialog detection."""

    if frame.size == 0:
        return False
    cfg = config or {}
    return (
        _detect_ocr_phrase_button(
            frame,
            cfg,
            region_key="death_recap_region",
            default_region={"x": 780, "y": 390, "width": 850, "height": 260},
            words=("death", "recap"),
        )
        is not None
    )


def detect_return_to_graveyard_accept_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None
    cfg = config or {}
    return _detect_ocr_phrase_button(
        frame,
        cfg,
        region_key="return_to_graveyard_accept_region",
        default_region={"x": 1000, "y": 250, "width": 320, "height": 120},
        words=("yes",),
    )


def detect_spirit_healer_target(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    region_cfg = recovery_cfg.get(
        "spirit_healer_region",
        {"x": 0, "y": 220, "width": 2205, "height": 920},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where(
        (saturation <= int(recovery_cfg.get("spirit_healer_max_saturation", 82)))
        & (value >= int(recovery_cfg.get("spirit_healer_min_value", 182))),
        255,
        0,
    ).astype(np.uint8)
    close_kernel = int(recovery_cfg.get("spirit_healer_close_kernel", 9))
    open_kernel = int(recovery_cfg.get("spirit_healer_open_kernel", 5))
    if close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), dtype=np.uint8))
    if open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), dtype=np.uint8))

    candidates = _spirit_healer_component_candidates(
        mask,
        offset=(roi_x, roi_y),
        frame_shape=frame.shape,
        config=cfg,
    )
    if not candidates:
        return None

    candidates = [
        (bbox, area)
        for bbox, area in candidates
        if not _looks_like_own_ghost_for_spirit_healer(bbox, frame.shape, recovery_cfg)
    ]
    if not candidates:
        return None

    seed_bbox, _seed_area = max(candidates, key=lambda item: _spirit_healer_score(item[0], item[1], frame.shape))
    seed_center_x = seed_bbox.x + seed_bbox.width / 2.0
    seed_center_y = seed_bbox.y + seed_bbox.height / 2.0
    cluster_dx = float(recovery_cfg.get("spirit_healer_cluster_dx", 560.0))
    cluster_dy = float(recovery_cfg.get("spirit_healer_cluster_dy", 420.0))
    clustered = [
        (bbox, area)
        for bbox, area in candidates
        if abs((bbox.x + bbox.width / 2.0) - seed_center_x) <= cluster_dx
        and abs((bbox.y + bbox.height / 2.0) - seed_center_y) <= cluster_dy
    ]
    if not clustered:
        clustered = [(seed_bbox, _seed_area)]

    min_x = min(bbox.x for bbox, _area in clustered)
    min_y = min(bbox.y for bbox, _area in clustered)
    max_x = max(bbox.x + bbox.width for bbox, _area in clustered)
    max_y = max(bbox.y + bbox.height for bbox, _area in clustered)
    bbox = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
    if bbox.x <= int(recovery_cfg.get("spirit_healer_left_edge_click_margin_px", 12)):
        click_fraction_x = float(recovery_cfg.get("spirit_healer_left_edge_click_x_fraction", 0.22))
    else:
        click_fraction_x = float(recovery_cfg.get("spirit_healer_click_x_fraction", 0.50))
    click_x = int(round(bbox.x + bbox.width * click_fraction_x))
    click_y = int(round(bbox.y + bbox.height * float(recovery_cfg.get("spirit_healer_click_y_fraction", 0.52))))
    return ((click_x, click_y), bbox)


def detect_spirit_healer_graveyard_candidate(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    frame_height, frame_width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where(
        (saturation <= int(recovery_cfg.get("spirit_healer_graveyard_max_saturation", 70)))
        & (value >= int(recovery_cfg.get("spirit_healer_graveyard_min_value", 235))),
        255,
        0,
    ).astype(np.uint8)

    min_x = int(round(frame_width * float(recovery_cfg.get("spirit_healer_graveyard_min_x_fraction", 0.28))))
    max_x = int(round(frame_width * float(recovery_cfg.get("spirit_healer_graveyard_max_x_fraction", 0.72))))
    min_y = int(round(frame_height * float(recovery_cfg.get("spirit_healer_graveyard_min_y_fraction", 0.15))))
    max_y = int(round(frame_height * float(recovery_cfg.get("spirit_healer_graveyard_max_y_fraction", 0.72))))
    bounded_mask = np.zeros_like(mask)
    bounded_mask[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
    bounded_mask = cv2.morphologyEx(bounded_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    bounded_mask = cv2.morphologyEx(bounded_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    min_area = int(recovery_cfg.get("spirit_healer_graveyard_min_area", 7000))
    min_width = int(recovery_cfg.get("spirit_healer_graveyard_min_width", 80))
    max_width = int(recovery_cfg.get("spirit_healer_graveyard_max_width", 700))
    min_height = int(recovery_cfg.get("spirit_healer_graveyard_min_height", 180))
    max_height = int(recovery_cfg.get("spirit_healer_graveyard_max_height", 700))
    min_aspect = float(recovery_cfg.get("spirit_healer_graveyard_min_aspect", 0.45))
    max_aspect = float(recovery_cfg.get("spirit_healer_graveyard_max_aspect", 1.80))
    max_bbox_y = frame_height * float(recovery_cfg.get("spirit_healer_graveyard_max_bbox_y_fraction", 0.25))
    max_center_distance = float(
        recovery_cfg.get("spirit_healer_graveyard_max_center_distance_fraction", 0.18)
    )

    num_labels, _labels, stats, _centers = cv2.connectedComponentsWithStats(bounded_mask, connectivity=8)
    candidates: list[tuple[float, int, BoundingBox]] = []
    for label_index in range(1, num_labels):
        x, y, width, height, area = (int(item) for item in stats[label_index])
        if width <= 0 or height <= 0:
            continue
        aspect = width / float(height)
        if not (
            area >= min_area
            and min_width <= width <= max_width
            and min_height <= height <= max_height
            and min_aspect <= aspect <= max_aspect
            and y <= max_bbox_y
        ):
            continue
        bbox = BoundingBox(x, y, width, height)
        center_distance = abs((x + width / 2.0) - frame_width / 2.0) / max(1.0, frame_width)
        if center_distance > max_center_distance:
            continue
        candidates.append((center_distance, -area, bbox))

    if not candidates:
        return None
    _center_distance, _negative_area, bbox = min(candidates, key=lambda item: (item[0], item[1]))
    return ((bbox.x + bbox.width // 2, bbox.y + bbox.height // 2), bbox)


def detect_spirit_healer_target_frame(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    return _detect_ocr_phrase_button(
        frame,
        cfg,
        region_key="spirit_healer_target_frame_region",
        default_region={"x": 300, "y": 10, "width": 450, "height": 120},
        words=("spirit", "healer"),
    )


def detect_spirit_healer_tooltip(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    return _detect_ocr_phrase_button(
        frame,
        cfg,
        region_key="spirit_healer_tooltip_region",
        default_region={"x": 2100, "y": 1060, "width": 420, "height": 300},
        words=("spirit", "healer"),
    )


def detect_spirit_healer_interaction_error(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> bool:
    if frame.size == 0:
        return False

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    region_cfg = recovery_cfg.get(
        "spirit_healer_interaction_error_region",
        {"x": 720, "y": 120, "width": 1120, "height": 190},
    )
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return False

    blue, green, red = cv2.split(roi)
    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    mask = (
        (red_i >= int(recovery_cfg.get("spirit_healer_interaction_error_min_red", 100)))
        & (green_i <= int(recovery_cfg.get("spirit_healer_interaction_error_max_green", 75)))
        & (blue_i <= int(recovery_cfg.get("spirit_healer_interaction_error_max_blue", 75)))
        & (
            (red_i - green_i)
            >= int(recovery_cfg.get("spirit_healer_interaction_error_min_red_green_delta", 60))
        )
        & (
            (red_i - blue_i)
            >= int(recovery_cfg.get("spirit_healer_interaction_error_min_red_blue_delta", 60))
        )
    )
    ys, xs = np.where(mask)
    if len(xs) < int(recovery_cfg.get("spirit_healer_interaction_error_min_pixels", 500)):
        return False
    span_width = int(xs.max() - xs.min() + 1)
    return span_width >= int(recovery_cfg.get("spirit_healer_interaction_error_min_span_width", 300))


def detect_return_to_life_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    result = _detect_ocr_phrase_button(
        frame,
        cfg,
        region_key="return_to_life_region",
        default_region={"x": 40, "y": 120, "width": 960, "height": 720},
        words=("return", "me", "to", "life"),
    )
    if result is not None:
        return result

    recovery_cfg = _recovery_cfg(cfg)
    if not bool(recovery_cfg.get("return_to_life_fallback_click_enabled", True)):
        return None
    region_cfg = recovery_cfg.get(
        "return_to_life_region",
        {"x": 40, "y": 120, "width": 960, "height": 720},
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, cfg)
    if width <= 0 or height <= 0:
        return None
    click_fraction = recovery_cfg.get("return_to_life_fallback_click_fraction", {"x": 0.38, "y": 0.46})
    click_x = x + int(round(width * float(click_fraction.get("x", 0.38))))
    click_y = y + int(round(height * float(click_fraction.get("y", 0.46))))
    return ((click_x, click_y), BoundingBox(click_x, click_y, 1, 1))


def detect_resurrection_accept_button(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], BoundingBox] | None:
    if frame.size == 0:
        return None

    cfg = config or {}
    recovery_cfg = _recovery_cfg(cfg)
    region_cfg = recovery_cfg.get("accept_button_region", {"x": 640, "y": 70, "width": 1280, "height": 520})
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, cfg)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 30]), np.array([14, 255, 230])),
        cv2.inRange(hsv, np.array([168, 70, 30]), np.array([179, 255, 230])),
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    candidates = _accept_button_candidates(red_mask, offset=(roi_x, roi_y), frame_shape=frame.shape, config=cfg)
    if not candidates:
        return None
    candidates.sort(key=lambda bbox: (bbox.y, bbox.x))
    button = candidates[0]
    return ((button.x + button.width // 2, button.y + button.height // 2), button)


def read_resurrection_sickness_remaining(path: str | Path, *, now: float | None = None) -> float | None:
    state_path = Path(path)
    if not state_path.exists():
        return None

    timestamp = time.time() if now is None else now
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        wait_until = float(data.get("wait_until_epoch", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    remaining = wait_until - timestamp
    if remaining <= 0.0:
        clear_resurrection_sickness_state(state_path)
        return None
    return remaining


def write_resurrection_sickness_state(
    path: str | Path,
    wait_seconds: float,
    *,
    now: float | None = None,
) -> None:
    wait_duration = max(0.0, float(wait_seconds))
    state_path = Path(path)
    if wait_duration <= 0.0:
        clear_resurrection_sickness_state(state_path)
        return

    timestamp = time.time() if now is None else now
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "saved_at_epoch": timestamp,
                "wait_seconds": wait_duration,
                "wait_until_epoch": timestamp + wait_duration,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def clear_resurrection_sickness_state(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _release_button_candidates(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[BoundingBox]:
    recovery_cfg = _recovery_cfg(config)
    frame_height, frame_width = frame_shape[:2]
    offset_x, offset_y = offset
    num_labels, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[BoundingBox] = []
    min_area = int(recovery_cfg.get("release_button_min_area", 2500))
    max_area = int(recovery_cfg.get("release_button_max_area", 9000))
    min_width = int(recovery_cfg.get("release_button_min_width", 120))
    max_width = int(recovery_cfg.get("release_button_max_width", 360))
    min_height = int(recovery_cfg.get("release_button_min_height", 18))
    max_height = int(recovery_cfg.get("release_button_max_height", 60))
    min_aspect = float(recovery_cfg.get("release_button_min_aspect", 2.5))
    min_fill_ratio = float(recovery_cfg.get("release_button_min_fill_ratio", 0.65))
    center_min_x = frame_width * float(recovery_cfg.get("release_button_center_min_x_fraction", 0.32))
    center_max_x = frame_width * float(recovery_cfg.get("release_button_center_max_x_fraction", 0.68))
    center_min_y = frame_height * float(recovery_cfg.get("release_button_center_min_y_fraction", 0.12))
    center_max_y = frame_height * float(recovery_cfg.get("release_button_center_max_y_fraction", 0.34))

    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        center_x = offset_x + float(centers[label_index][0])
        center_y = offset_y + float(centers[label_index][1])
        fill_ratio = float(area) / float(width * height)
        aspect = float(width) / float(height)
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and aspect >= min_aspect
            and fill_ratio >= min_fill_ratio
            and center_min_x <= center_x <= center_max_x
            and center_min_y <= center_y <= center_max_y
        ):
            continue
        candidates.append(BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)))
    return candidates


def _safe_zone_resurrect_candidates(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[BoundingBox]:
    recovery_cfg = _recovery_cfg(config)
    frame_height, frame_width = frame_shape[:2]
    offset_x, offset_y = offset
    num_labels, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[BoundingBox] = []
    min_area = int(recovery_cfg.get("safe_zone_resurrect_min_area", 1200))
    max_area = int(recovery_cfg.get("safe_zone_resurrect_max_area", 9000))
    min_width = int(recovery_cfg.get("safe_zone_resurrect_min_width", 70))
    max_width = int(recovery_cfg.get("safe_zone_resurrect_max_width", 260))
    min_height = int(recovery_cfg.get("safe_zone_resurrect_min_height", 20))
    max_height = int(recovery_cfg.get("safe_zone_resurrect_max_height", 70))
    min_fill_ratio = float(recovery_cfg.get("safe_zone_resurrect_min_fill_ratio", 0.35))
    center_min_x = frame_width * float(recovery_cfg.get("safe_zone_resurrect_center_min_x_fraction", 0.38))
    center_max_x = frame_width * float(recovery_cfg.get("safe_zone_resurrect_center_max_x_fraction", 0.68))
    center_min_y = frame_height * float(recovery_cfg.get("safe_zone_resurrect_center_min_y_fraction", 0.08))
    center_max_y = frame_height * float(recovery_cfg.get("safe_zone_resurrect_center_max_y_fraction", 0.20))

    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        center_x = offset_x + float(centers[label_index][0])
        center_y = offset_y + float(centers[label_index][1])
        fill_ratio = float(area) / float(width * height)
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and fill_ratio >= min_fill_ratio
            and center_min_x <= center_x <= center_max_x
            and center_min_y <= center_y <= center_max_y
        ):
            continue
        candidates.append(BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)))
    return candidates


def _spirit_healer_component_candidates(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[tuple[BoundingBox, int]]:
    recovery_cfg = _recovery_cfg(config)
    frame_height, frame_width = frame_shape[:2]
    offset_x, offset_y = offset
    num_labels, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[BoundingBox, int]] = []
    min_area = int(recovery_cfg.get("spirit_healer_min_area", 2000))
    max_area = int(recovery_cfg.get("spirit_healer_max_area", 220000))
    min_width = int(recovery_cfg.get("spirit_healer_min_width", 35))
    max_width = int(recovery_cfg.get("spirit_healer_max_width", 900))
    min_height = int(recovery_cfg.get("spirit_healer_min_height", 150))
    max_height = int(recovery_cfg.get("spirit_healer_max_height", 850))
    center_min_x = frame_width * float(recovery_cfg.get("spirit_healer_center_min_x_fraction", 0.0))
    center_max_x = frame_width * float(recovery_cfg.get("spirit_healer_center_max_x_fraction", 0.88))
    center_min_y = frame_height * float(recovery_cfg.get("spirit_healer_center_min_y_fraction", 0.25))
    center_max_y = frame_height * float(recovery_cfg.get("spirit_healer_center_max_y_fraction", 0.78))
    top_margin = int(recovery_cfg.get("spirit_healer_roi_top_margin", 8))

    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        global_x = offset_x + int(x)
        global_y = offset_y + int(y)
        center_x = offset_x + float(centers[label_index][0])
        center_y = offset_y + float(centers[label_index][1])
        if global_y <= offset_y + top_margin:
            continue
        if _point_in_recovery_regions(
            (center_x, center_y),
            frame_shape,
            config,
            "spirit_healer_ignore_regions",
        ):
            continue
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and center_min_x <= center_x <= center_max_x
            and center_min_y <= center_y <= center_max_y
        ):
            continue
        candidates.append((BoundingBox(global_x, global_y, int(width), int(height)), int(area)))
    return candidates


def _point_in_recovery_regions(
    point: tuple[float, float],
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
    region_key: str,
) -> bool:
    recovery_cfg = _recovery_cfg(config)
    regions = recovery_cfg.get(region_key, [])
    if not isinstance(regions, list):
        return False
    point_x, point_y = point
    for region in regions:
        if not isinstance(region, dict):
            continue
        x, y, width, height = resolve_region(region, frame_shape, config)
        if x <= point_x <= x + width and y <= point_y <= y + height:
            return True
    return False


def _looks_like_own_ghost_for_spirit_healer(
    bbox: BoundingBox,
    frame_shape: tuple[int, ...],
    recovery_cfg: dict[str, Any],
) -> bool:
    if not bool(recovery_cfg.get("spirit_healer_self_ignore_enabled", True)):
        return False

    frame_height, frame_width = frame_shape[:2]
    center_x = (bbox.x + bbox.width / 2.0) / float(frame_width)
    center_y = (bbox.y + bbox.height / 2.0) / float(frame_height)
    return (
        bbox.width <= int(recovery_cfg.get("spirit_healer_self_ignore_max_width", 260))
        and bbox.height <= int(recovery_cfg.get("spirit_healer_self_ignore_max_height", 380))
        and float(recovery_cfg.get("spirit_healer_self_ignore_center_min_x_fraction", 0.38)) <= center_x
        and center_x <= float(recovery_cfg.get("spirit_healer_self_ignore_center_max_x_fraction", 0.62))
        and float(recovery_cfg.get("spirit_healer_self_ignore_center_min_y_fraction", 0.42)) <= center_y
        and center_y <= float(recovery_cfg.get("spirit_healer_self_ignore_center_max_y_fraction", 0.78))
    )


def _spirit_healer_score(
    bbox: BoundingBox,
    area: int,
    frame_shape: tuple[int, ...],
) -> tuple[bool, int, int, int]:
    frame_width = frame_shape[1]
    left_visible = bbox.x <= frame_width * 0.18
    return (left_visible, bbox.height, area, bbox.width)


def _accept_button_candidates(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
    frame_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[BoundingBox]:
    recovery_cfg = _recovery_cfg(config)
    frame_height, frame_width = frame_shape[:2]
    offset_x, offset_y = offset
    num_labels, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[BoundingBox] = []
    min_area = int(recovery_cfg.get("accept_button_min_area", 1800))
    max_area = int(recovery_cfg.get("accept_button_max_area", 9000))
    min_width = int(recovery_cfg.get("accept_button_min_width", 100))
    max_width = int(recovery_cfg.get("accept_button_max_width", 360))
    min_height = int(recovery_cfg.get("accept_button_min_height", 18))
    max_height = int(recovery_cfg.get("accept_button_max_height", 60))
    min_aspect = float(recovery_cfg.get("accept_button_min_aspect", 2.4))
    min_fill_ratio = float(recovery_cfg.get("accept_button_min_fill_ratio", 0.60))
    center_min_x = frame_width * float(recovery_cfg.get("accept_button_center_min_x_fraction", 0.30))
    center_max_x = frame_width * float(recovery_cfg.get("accept_button_center_max_x_fraction", 0.70))
    center_min_y = frame_height * float(recovery_cfg.get("accept_button_center_min_y_fraction", 0.10))
    center_max_y = frame_height * float(recovery_cfg.get("accept_button_center_max_y_fraction", 0.38))

    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if width <= 0 or height <= 0:
            continue
        center_x = offset_x + float(centers[label_index][0])
        center_y = offset_y + float(centers[label_index][1])
        fill_ratio = float(area) / float(width * height)
        aspect = float(width) / float(height)
        if not (
            min_area <= int(area) <= max_area
            and min_width <= int(width) <= max_width
            and min_height <= int(height) <= max_height
            and aspect >= min_aspect
            and fill_ratio >= min_fill_ratio
            and center_min_x <= center_x <= center_max_x
            and center_min_y <= center_y <= center_max_y
        ):
            continue
        candidates.append(BoundingBox(offset_x + int(x), offset_y + int(y), int(width), int(height)))
    return candidates


def _detect_ocr_phrase_button(
    frame: np.ndarray,
    config: dict[str, Any],
    *,
    region_key: str,
    default_region: dict[str, int],
    words: tuple[str, ...],
) -> tuple[tuple[int, int], BoundingBox] | None:
    try:
        import pytesseract
    except ImportError:
        return None

    recovery_cfg = _recovery_cfg(config)
    ocr_cfg = config.get("position_ocr", {})
    tesseract_cmd = str(ocr_cfg.get("tesseract_cmd", "")).strip()
    if tesseract_cmd:
        tesseract_path = Path(tesseract_cmd)
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

    region_cfg = recovery_cfg.get(region_key, default_region)
    roi_x, roi_y, roi_width, roi_height = resolve_region(region_cfg, frame.shape, config)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    if roi.size == 0:
        return None

    scale = max(1, int(recovery_cfg.get("spirit_healer_ocr_scale", 2)))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ocr_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    custom_config = str(recovery_cfg.get("spirit_healer_ocr_config", "--psm 6"))
    try:
        data = pytesseract.image_to_data(ocr_image, config=custom_config, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    cleaned_words: list[tuple[str, int, int, int, int]] = []
    total = len(data.get("text", []))
    for index in range(total):
        token = _clean_ocr_token(str(data["text"][index]))
        if not token:
            continue
        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        cleaned_words.append((token, left, top, width, height))

    target_words = tuple(_clean_ocr_token(word) for word in words)
    for start in range(0, len(cleaned_words) - len(target_words) + 1):
        window = cleaned_words[start : start + len(target_words)]
        if not all(_ocr_token_matches(item[0], target) for item, target in zip(window, target_words, strict=True)):
            continue
        min_x = min(item[1] for item in window)
        min_y = min(item[2] for item in window)
        max_x = max(item[1] + item[3] for item in window)
        max_y = max(item[2] + item[4] for item in window)
        bbox = BoundingBox(
            roi_x + int(round(min_x / scale)),
            roi_y + int(round(min_y / scale)),
            max(1, int(round((max_x - min_x) / scale))),
            max(1, int(round((max_y - min_y) / scale))),
        )
        click_point = (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)
        return (click_point, bbox)
    return None


def _clean_ocr_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _ocr_token_matches(token: str, target: str) -> bool:
    if token == target:
        return True
    if len(target) < 4 or len(token) < 3:
        return False
    if token.endswith(target) and len(token) <= len(target) + 3:
        return True
    if token.endswith(target[1:]) and len(token) <= len(target) + 3:
        return True
    if len(token) == len(target) - 1 and target.endswith(token):
        return True
    if len(token) == len(target):
        differences = sum(1 for left, right in zip(token, target, strict=True) if left != right)
        return differences <= 1
    return False


def _fallback_release_click(
    frame: np.ndarray,
    config: dict[str, Any],
) -> tuple[tuple[int, int], BoundingBox] | None:
    recovery_cfg = _recovery_cfg(config)
    if not bool(recovery_cfg.get("fallback_click_enabled", True)):
        return None
    region_cfg = recovery_cfg.get("release_button_region", {"x": 640, "y": 70, "width": 1280, "height": 450})
    x, y, width, height = resolve_region(region_cfg, frame.shape, config)
    if width <= 0 or height <= 0:
        return None
    click_fraction = recovery_cfg.get("fallback_click_fraction", {"x": 0.41, "y": 0.49})
    click_x = x + int(round(width * float(click_fraction.get("x", 0.41))))
    click_y = y + int(round(height * float(click_fraction.get("y", 0.49))))
    return ((click_x, click_y), BoundingBox(click_x, click_y, 1, 1))


def _fallback_safe_zone_resurrect_click(
    frame: np.ndarray,
    *,
    config: dict[str, Any],
) -> tuple[tuple[int, int], BoundingBox] | None:
    recovery_cfg = _recovery_cfg(config)
    if not bool(recovery_cfg.get("safe_zone_resurrect_fallback_click_enabled", False)):
        return None
    region_cfg = recovery_cfg.get(
        "safe_zone_resurrect_button_region",
        {"x": 800, "y": 95, "width": 900, "height": 190},
    )
    x, y, width, height = resolve_region(region_cfg, frame.shape, config)
    if width <= 0 or height <= 0:
        return None
    click_fraction = recovery_cfg.get("safe_zone_resurrect_fallback_click_fraction", {"x": 0.68, "y": 0.45})
    click_x = x + int(round(width * float(click_fraction.get("x", 0.68))))
    click_y = y + int(round(height * float(click_fraction.get("y", 0.45))))
    return ((click_x, click_y), BoundingBox(click_x, click_y, 1, 1))


def _recovery_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("safety", {}).get("death_recovery", {})


def _attempt_available(current: int, maximum: int) -> bool:
    return maximum <= 0 or current < maximum


def _normalize_resurrection_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"safe_zone", "safezone", "city", "closest_city", "closest_town", "town"}:
        return "safe_zone"
    return "spirit_healer"


def _bbox_to_dict(bbox: BoundingBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }
