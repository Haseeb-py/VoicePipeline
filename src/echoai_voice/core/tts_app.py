"""Phase 5 application wiring for Deepgram Aura streaming TTS."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from echoai_voice.core.response_app import ResponseVoiceApplication, build_response_application
from echoai_voice.providers.deepgram.tts_provider import DeepgramStreamingTtsProvider
from echoai_voice.providers.livekit.audio_sink import AudioSink
from echoai_voice.services.tts_service import TtsService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TtsVoiceApplication:
    """Composition root for Response Generation plus Streaming TTS."""

    base: ResponseVoiceApplication
    tts_provider: DeepgramStreamingTtsProvider
    audio_sink: AudioSink
    tts_service: TtsService


def build_tts_application() -> TtsVoiceApplication:
    """Build the Phase 5 streaming TTS composition root."""

    base = build_response_application()
    persona_manager = base.base.base.base.persona_manager
    settings = base.base.base.base.settings
    session_manager = base.base.base.livekit

    tts_provider = DeepgramStreamingTtsProvider(settings=settings, persona_manager=persona_manager)
    audio_sink = AudioSink()
    tts_service = TtsService(
        tts_provider=tts_provider,
        audio_sink=audio_sink,
        session_manager=session_manager,
        latency_tracker=base.base.base.base.latency_tracker,
    )

    # Stream response tokens to one persistent TTS synthesis stream.
    base.response_service.subscribe_stream(tts_service.handle_response_chunk)

    return TtsVoiceApplication(
        base=base,
        tts_provider=tts_provider,
        audio_sink=audio_sink,
        tts_service=tts_service,
    )


async def start_tts_call(app: TtsVoiceApplication) -> None:
    """Start the STT + LLM + TTS streaming pipeline call."""

    await app.tts_provider.start()
    await app.base.base.stt_provider.start()
    app.base.response_service.bind_call_id(app.base.base.base.livekit.status.call_id)
    app.base.response_service.reset()
    logger.info("Phase 5 TTS voice call started")


async def stop_tts_call(app: TtsVoiceApplication) -> None:
    """Stop the streaming TTS pipeline call cleanly."""

    await app.base.base.stt_provider.stop()
    await app.tts_provider.stop()
    await app.base.base.base.livekit.disconnect()
    logger.info("Phase 5 TTS voice call stopped")
