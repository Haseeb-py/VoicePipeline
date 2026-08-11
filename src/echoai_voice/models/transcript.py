"""Transcript and event models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """Normalized transcript event emitted to downstream modules."""

    call_id: str
    turn_id: int
    speaker: str
    text: str
    is_final: bool
    confidence: float
    timestamp_start: datetime
    timestamp_end: datetime

    def model_dump(self) -> dict[str, object]:
        """Return a JSON-ready event payload matching the module schema."""

        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat(),
        }
