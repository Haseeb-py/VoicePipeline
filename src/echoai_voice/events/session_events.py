"""Session lifecycle events for Phase 2 and later UI wiring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from echoai_voice.models.session import ConnectionState, ListeningState, SpeakingState


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """Generic timestamped state transition event."""

    name: str
    timestamp: datetime
    connection: ConnectionState | None = None
    listening: ListeningState | None = None
    speaking: SpeakingState | None = None
    detail: str | None = None

    @classmethod
    def now(cls, name: str, **kwargs: object) -> "SessionEvent":
        return cls(name=name, timestamp=datetime.now(timezone.utc), **kwargs)
