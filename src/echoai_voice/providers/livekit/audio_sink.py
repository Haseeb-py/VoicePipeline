"""Audio egress abstraction for synthesized TTS output."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from echoai_voice.models.tts import TtsAudioChunk

logger = logging.getLogger(__name__)

TtsAudioSubscriber = Callable[[TtsAudioChunk], None]


@dataclass(slots=True)
class AudioSink:
    """Pipeline-facing audio output boundary.

    Synthesized TTS audio chunks are pushed to this boundary, which dispatches
    them to LiveKit audio tracks and development listeners.
    """

    _subscribers: list[TtsAudioSubscriber] = field(default_factory=list)

    def subscribe(self, callback: TtsAudioSubscriber) -> None:
        """Register a subscriber for outbound TTS audio chunks."""

        self._subscribers.append(callback)

    def unsubscribe(self, callback: TtsAudioSubscriber) -> None:
        """Remove a previously registered outbound audio callback."""

        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, chunk: TtsAudioChunk) -> None:
        """Emit a synthesized audio chunk to all subscribers."""

        for callback in tuple(self._subscribers):
            try:
                callback(chunk)
            except Exception:
                logger.exception("Error in AudioSink subscriber callback")

    def clear(self) -> None:
        """Clear any buffered audio frames and signal interruption to subscribers."""

        logger.info("AudioSink cleared on barge-in interruption.")
        interrupted_chunk = TtsAudioChunk(
            pcm16=b"",
            sample_rate=16000,
            channels=1,
            is_final=True,
            is_interruption=True,
        )
        self.emit(interrupted_chunk)
