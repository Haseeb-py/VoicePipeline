"""Runtime session models for the live voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConnectionState(StrEnum):
    """Connection lifecycle states exposed to the UI."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


class ListeningState(StrEnum):
    """Whether the pipeline is currently listening for customer speech."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


class SpeakingState(StrEnum):
    """Whether the AI is currently speaking."""

    SILENT = "silent"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"


@dataclass(slots=True)
class SessionStatus:
    """Live session status reported to the UI."""

    connection: ConnectionState = ConnectionState.DISCONNECTED
    listening: ListeningState = ListeningState.IDLE
    speaking: SpeakingState = SpeakingState.SILENT
    call_id: str | None = None
    room_name: str | None = None
