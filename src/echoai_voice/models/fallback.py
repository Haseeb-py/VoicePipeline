"""Low-confidence transcription fallback models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class WhisperFallbackResult:
    """Diagnostic Whisper comparison that never re-enters the response path."""

    call_id: str
    turn_id: int
    primary_text: str
    fallback_text: str
    primary_confidence: float
    timestamp: datetime
