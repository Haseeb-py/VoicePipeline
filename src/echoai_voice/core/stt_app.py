"""Phase 3 application wiring for Deepgram streaming STT."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from echoai_voice.core.livekit_app import LiveKitVoiceApplication, build_livekit_application
from echoai_voice.providers.deepgram.stt_provider import DeepgramStreamingSttProvider
from echoai_voice.providers.groq.whisper_provider import GroqWhisperFallbackProvider
from echoai_voice.services.transcript_event_bus import TranscriptEventBus
from echoai_voice.services.whisper_fallback import WhisperFallbackService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SttVoiceApplication:
    """Composition root for the LiveKit audio to Deepgram transcript pipeline."""

    base: LiveKitVoiceApplication
    transcript_bus: TranscriptEventBus
    stt_provider: DeepgramStreamingSttProvider
    fallback_service: WhisperFallbackService


def build_stt_application() -> SttVoiceApplication:
    """Build the LiveKit + Deepgram STT slice for Phase 3 validation."""

    base = build_livekit_application()
    bus = TranscriptEventBus()
    fallback_service = WhisperFallbackService(
        GroqWhisperFallbackProvider(
            api_key=base.base.settings.groq_api_key,
            model=base.base.settings.whisper_model,
        )
    )
    stt_provider = DeepgramStreamingSttProvider(
        settings=base.base.settings,
        transcript_bus=bus,
        fallback_service=fallback_service,
        latency_tracker=base.base.latency_tracker,
    )
    stt_provider.bind_call_id(base.livekit.status.call_id)
    return SttVoiceApplication(
        base=base,
        transcript_bus=bus,
        stt_provider=stt_provider,
        fallback_service=fallback_service,
    )


async def start_stt_call(app: SttVoiceApplication) -> None:
    """Start the LiveKit session and attach the Deepgram STT provider."""

    await app.base.livekit.connect()
    app.stt_provider.bind_call_id(app.base.livekit.status.call_id)
    await app.stt_provider.start()
    app.base.livekit.mark_listening()
    app.base.audio_source.subscribe(lambda chunk: _schedule_audio_push(app, chunk))
    logger.info("Phase 3 STT call started")


async def stop_stt_call(app: SttVoiceApplication) -> None:
    """Stop Deepgram STT and close the LiveKit session."""

    await app.stt_provider.stop()
    await app.fallback_service.cancel_all()
    await app.base.livekit.disconnect()


def _schedule_audio_push(app: SttVoiceApplication, chunk) -> None:
    import asyncio

    asyncio.create_task(app.stt_provider.push_audio(chunk))
