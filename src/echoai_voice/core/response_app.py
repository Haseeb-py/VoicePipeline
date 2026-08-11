"""Phase 4 application wiring for response generation."""

from __future__ import annotations

from dataclasses import dataclass

from echoai_voice.core.stt_app import SttVoiceApplication, build_stt_application
from echoai_voice.models.response_events import GeneratedResponseEvent
from echoai_voice.services.response_generation_service import ResponseGenerationService


@dataclass(slots=True)
class ResponseVoiceApplication:
    """Composition root for STT plus response generation."""

    base: SttVoiceApplication
    response_service: ResponseGenerationService


def build_response_application() -> ResponseVoiceApplication:
    """Build the Phase 4 response-generation slice."""

    base = build_stt_application()
    response_service = ResponseGenerationService(
        persona_manager=base.base.base.persona_manager,
        response_generator=base.base.base.response_generator,
        latency_tracker=base.base.base.latency_tracker,
    )
    response_service.bind_call_id(base.base.livekit.status.call_id)
    # Scheduling keeps Deepgram's receive loop free to process VAD events while
    # response generation and TTS are still active.
    base.transcript_bus.subscribe(response_service.submit_transcript_event)
    return ResponseVoiceApplication(base=base, response_service=response_service)


async def start_response_call(app: ResponseVoiceApplication) -> None:
    """Start the STT pipeline and response generation listener."""

    await app.base.stt_provider.start()
    app.response_service.bind_call_id(app.base.base.livekit.status.call_id)
    app.response_service.reset()


async def stop_response_call(app: ResponseVoiceApplication) -> None:
    """Stop the response-generation pipeline."""

    await app.base.stt_provider.stop()
    await app.base.base.livekit.disconnect()
