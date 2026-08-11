"""Phase 2 application wiring for the LiveKit transport layer."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from echoai_voice.core.application import VoiceApplication, build_application
from echoai_voice.models.session import SessionStatus
from echoai_voice.providers.livekit.audio_source import AudioFrameChunk, AudioSource
from echoai_voice.providers.livekit.session_manager import LiveKitSessionManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveKitVoiceApplication:
    """Composition root for the Phase 2 LiveKit transport slice."""

    base: VoiceApplication
    livekit: LiveKitSessionManager
    audio_source: AudioSource

    @property
    def status(self) -> SessionStatus:
        return self.livekit.status


def build_livekit_application() -> LiveKitVoiceApplication:
    """Build the app and the persistent LiveKit session manager."""

    base = build_application()
    return LiveKitVoiceApplication(
        base=base,
        livekit=LiveKitSessionManager(base.settings),
        audio_source=AudioSource(),
    )


async def start_call(app: LiveKitVoiceApplication) -> SessionStatus:
    """Start a persistent LiveKit session for the duration of a call."""

    status = await app.livekit.connect()
    logger.info("LiveKit call started: %s", status)
    app.livekit.mark_listening()
    return status


async def end_call(app: LiveKitVoiceApplication) -> SessionStatus:
    """End the persistent LiveKit session."""

    status = await app.livekit.disconnect()
    logger.info("LiveKit call ended: %s", status)
    return status


def route_microphone_audio(app: LiveKitVoiceApplication, pcm16: bytes, sample_rate: int) -> None:
    """Route inbound microphone audio into the pipeline boundary."""

    app.audio_source.emit(AudioFrameChunk(pcm16=pcm16, sample_rate=sample_rate))
