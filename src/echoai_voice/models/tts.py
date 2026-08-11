"""Text-to-speech domain models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TtsAudioChunk:
    """PCM audio chunk produced by streaming TTS synthesis."""

    pcm16: bytes
    sample_rate: int = 16000
    channels: int = 1
    is_final: bool = False
    is_interruption: bool = False
    call_id: str | None = None
    turn_id: int | None = None


@dataclass(frozen=True, slots=True)
class TtsRequest:
    """Request contract for text synthesis."""

    text: str
    voice: str
    speech_rate: float = 1.0
