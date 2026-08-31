from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


class EvidenceState(str, Enum):
    """Whether a sensor positively observed, disproved, or could not decide."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Evidence(Generic[T]):
    """One timestamped perception result.

    `ABSENT` is a real observation and must not be used for failures to capture,
    decode, or classify.  Those cases are `UNKNOWN`.  This distinction prevents
    a missing/stale ROI from silently authorising route or interaction actions.
    """

    state: EvidenceState
    value: T | None
    confidence: float
    observed_at: float
    frame_id: int
    source: str
    geometry: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        observed_at = float(self.observed_at)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("evidence confidence must be finite and between 0 and 1")
        if not math.isfinite(observed_at):
            raise ValueError("evidence observed_at must be finite")
        if int(self.frame_id) < 0:
            raise ValueError("evidence frame_id must be non-negative")
        if not str(self.source).strip():
            raise ValueError("evidence source must not be empty")
        if self.state is EvidenceState.PRESENT and self.value is None:
            raise ValueError("present evidence requires a value")
        if self.state is EvidenceState.UNKNOWN and self.value is not None:
            raise ValueError("unknown evidence must not carry a value")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "geometry", MappingProxyType(dict(self.geometry)))

    @classmethod
    def present(
        cls,
        value: T,
        *,
        observed_at: float,
        frame_id: int,
        source: str,
        confidence: float = 1.0,
        geometry: Mapping[str, Any] | None = None,
        reason: str | None = None,
    ) -> "Evidence[T]":
        return cls(
            state=EvidenceState.PRESENT,
            value=value,
            confidence=confidence,
            observed_at=observed_at,
            frame_id=frame_id,
            source=source,
            geometry=geometry or {},
            reason=reason,
        )

    @classmethod
    def absent(
        cls,
        *,
        observed_at: float,
        frame_id: int,
        source: str,
        confidence: float = 1.0,
        geometry: Mapping[str, Any] | None = None,
        reason: str | None = None,
    ) -> "Evidence[T]":
        return cls(
            state=EvidenceState.ABSENT,
            value=None,
            confidence=confidence,
            observed_at=observed_at,
            frame_id=frame_id,
            source=source,
            geometry=geometry or {},
            reason=reason,
        )

    @classmethod
    def unknown(
        cls,
        *,
        observed_at: float,
        frame_id: int,
        source: str,
        reason: str,
        confidence: float = 0.0,
        geometry: Mapping[str, Any] | None = None,
    ) -> "Evidence[T]":
        return cls(
            state=EvidenceState.UNKNOWN,
            value=None,
            confidence=confidence,
            observed_at=observed_at,
            frame_id=frame_id,
            source=source,
            geometry=geometry or {},
            reason=reason,
        )

    def age(self, now: float) -> float:
        return max(0.0, float(now) - self.observed_at)

    def is_fresh(self, *, now: float, max_age_seconds: float) -> bool:
        return self.age(now) <= max(0.0, float(max_age_seconds))

    def present_value(
        self,
        *,
        now: float,
        max_age_seconds: float,
        min_confidence: float = 0.0,
    ) -> T | None:
        if self.state is not EvidenceState.PRESENT:
            return None
        if self.confidence < float(min_confidence):
            return None
        if not self.is_fresh(now=now, max_age_seconds=max_age_seconds):
            return None
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "value": self.value,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "frame_id": self.frame_id,
            "source": self.source,
            "geometry": dict(self.geometry),
            "reason": self.reason,
        }
