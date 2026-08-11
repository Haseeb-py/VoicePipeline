"""Monotonic latency checkpoint models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LatencyCheckpointName(StrEnum):
    SPEECH_INPUT_DETECTED = "speech_input_detected"
    STT_INTERIM_RECEIVED = "stt_interim_received"
    STT_FINAL_RECEIVED = "stt_final_received"
    RESPONSE_GENERATION_STARTED = "response_generation_started"
    FIRST_RESPONSE_CHUNK = "first_response_chunk"
    RESPONSE_TEXT_COMPLETED = "response_text_completed"
    TTS_GENERATION_STARTED = "tts_generation_started"
    FIRST_AUDIO_CHUNK_AVAILABLE = "first_audio_chunk_available"
    PLAYBACK_STARTED = "playback_started"
    TTS_GENERATION_COMPLETED = "tts_generation_completed"
    RESPONSE_COMPLETED = "response_completed"
    PLAYBACK_COMPLETED = "playback_completed"


@dataclass(frozen=True, slots=True)
class LatencyCheckpoint:
    call_id: str
    turn_id: int
    name: LatencyCheckpointName
    monotonic_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)
