"""Phase 5 verification script for Deepgram Aura Streaming TTS."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from echoai_voice.core.tts_app import build_tts_application
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.models.tts import TtsAudioChunk


async def main() -> None:
    app = build_tts_application()
    received_chunks: list[TtsAudioChunk] = []

    # Subscribe to outbound audio sink
    app.audio_sink.subscribe(lambda chunk: received_chunks.append(chunk))

    persona_mgr = app.base.base.base.base.persona_manager
    initial_voice = app.tts_provider.current_voice
    initial_rate = app.tts_provider.current_speech_rate
    print(
        f"Initial Persona: {persona_mgr.get_active_persona().id}, "
        f"Voice: {initial_voice}, Rate: {initial_rate}"
    )

    # Trigger customer speech event
    customer_event_1 = TranscriptEvent(
        call_id="call-55",
        turn_id=1,
        speaker="customer",
        text="Can you explain your pricing options?",
        is_final=True,
        confidence=0.98,
        timestamp_start=datetime.now(timezone.utc),
        timestamp_end=datetime.now(timezone.utc),
    )

    await app.base.response_service.handle_transcript_event(customer_event_1)

    first_turn_chunks = len(received_chunks)
    print(f"Turn 1 produced {first_turn_chunks} audio chunks (streaming playback started).")

    # Switch persona dynamically to sales
    switched_persona = persona_mgr.select_persona("sales_representative")
    updated_voice = app.tts_provider.current_voice
    updated_rate = app.tts_provider.current_speech_rate
    print(
        f"Switched Persona: {switched_persona.id}, "
        f"Updated Voice: {updated_voice}, Rate: {updated_rate}"
    )

    # Trigger second customer speech event
    customer_event_2 = TranscriptEvent(
        call_id="call-55",
        turn_id=2,
        speaker="customer",
        text="What discounts can you offer today?",
        is_final=True,
        confidence=0.97,
        timestamp_start=datetime.now(timezone.utc),
        timestamp_end=datetime.now(timezone.utc),
    )

    await app.base.response_service.handle_transcript_event(customer_event_2)

    second_turn_chunks = len(received_chunks) - first_turn_chunks
    print(f"Turn 2 produced {second_turn_chunks} audio chunks after voice update.")

    verification_result = {
        "streaming_chunks_generated": len(received_chunks) > 0,
        "voice_changed_on_persona_switch": initial_voice != updated_voice,
        "speech_rate_changed_on_persona_switch": initial_rate != updated_rate,
        "speech_rate_matches_persona": updated_rate == switched_persona.tts.speech_rate,
        "speaking_state_reset": app.base.base.base.livekit.status.speaking == "silent",
    }
    print("Phase 5 Verification Results:", verification_result)


if __name__ == "__main__":
    asyncio.run(main())
