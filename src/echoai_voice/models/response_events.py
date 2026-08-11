"""Response events emitted by the temporary response generation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class GeneratedResponseEvent:
    """Structured AI response event for UI and later TTS consumption."""

    call_id: str
    turn_id: int
    persona_id: str
    text: str
    timestamp: datetime

    def model_dump(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "persona_id": self.persona_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class GeneratedResponseChunkEvent:
    """Incremental response text event consumed by streaming TTS."""

    call_id: str
    turn_id: int
    persona_id: str
    text: str
    sequence: int
    is_final: bool
    timestamp: datetime
