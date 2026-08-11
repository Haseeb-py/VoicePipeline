"""Audio ingress abstraction for the LiveKit microphone stream."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioFrameChunk:
    """Raw audio chunk passed into the voice pipeline."""

    pcm16: bytes
    sample_rate: int
    channels: int = 1


class AudioSource:
    """Pipeline-facing audio source boundary.

    The real LiveKit RTC track subscription is adapted into this interface so
    later STT and VAD stages do not depend directly on transport details.
    """

    def __init__(self) -> None:
        self._subscribers: list[callable[[AudioFrameChunk], None]] = []

    def subscribe(self, callback: callable[[AudioFrameChunk], None]) -> None:
        """Register a callback for inbound microphone frames."""

        self._subscribers.append(callback)

    def unsubscribe(self, callback: callable[[AudioFrameChunk], None]) -> None:
        """Remove a previously registered inbound audio callback."""

        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, chunk: AudioFrameChunk) -> None:
        """Emit an audio chunk to all subscribers."""

        for callback in tuple(self._subscribers):
            callback(chunk)
