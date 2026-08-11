"""Phase 6 application wiring for VAD & Barge-in interruption handling."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from echoai_voice.core.stt_app import start_stt_call, stop_stt_call
from echoai_voice.core.tts_app import TtsVoiceApplication, build_tts_application
from echoai_voice.providers.livekit.audio_transport import LiveKitAudioTransport
from echoai_voice.services.barge_in import BargeInController

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VadVoiceApplication:
    """Composition root for Phase 6 VAD + Barge-in Interruption pipeline."""

    base: TtsVoiceApplication
    barge_in: BargeInController
    audio_transport: LiveKitAudioTransport


def build_vad_application() -> VadVoiceApplication:
    """Build the Phase 6 VAD and barge-in interruption composition root."""

    base = build_tts_application()
    stt_provider = base.base.base.stt_provider
    session_manager = base.base.base.base.livekit

    barge_in = BargeInController(
        tts_service=base.tts_service,
        response_service=base.base.response_service,
        session_manager=session_manager,
    )

    # Raw VAD can fire on noise. Cancel output only after Deepgram supplies
    # sufficiently reliable transcript evidence for customer speech.
    stt_provider.subscribe_confirmed_speech(barge_in.handle_speech_started)
    audio_transport = LiveKitAudioTransport(
        session_manager=session_manager,
        input_source=base.base.base.base.audio_source,
        output_sink=base.audio_sink,
        latency_tracker=base.base.base.base.base.latency_tracker,
    )

    return VadVoiceApplication(
        base=base,
        barge_in=barge_in,
        audio_transport=audio_transport,
    )


async def start_vad_call(app: VadVoiceApplication) -> None:
    """Start the VAD + STT + LLM + TTS pipeline call."""

    await app.base.tts_provider.start()
    await start_stt_call(app.base.base.base)
    await app.audio_transport.start()
    app.base.base.response_service.bind_call_id(app.base.base.base.base.livekit.status.call_id)
    app.base.base.response_service.reset()
    logger.info("Phase 6 VAD & Barge-in call started")


async def stop_vad_call(app: VadVoiceApplication) -> None:
    """Stop the VAD pipeline call cleanly."""

    await app.base.base.response_service.cancel_current_response()
    await app.base.base.response_service.wait_until_idle()
    await app.base.tts_service.cancel_current_speech(emit_interruption=False)
    await app.base.tts_provider.stop()
    await app.audio_transport.stop()
    await stop_stt_call(app.base.base.base)
    logger.info("Phase 6 VAD & Barge-in call stopped")
