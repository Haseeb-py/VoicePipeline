"""Phase 4 verification helper for persona-aware response generation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from echoai_voice.core.response_app import build_response_application
from echoai_voice.models.response_events import GeneratedResponseEvent
from echoai_voice.models.transcript import TranscriptEvent


async def main() -> None:
    app = build_response_application()
    generated: list[GeneratedResponseEvent] = []
    app.response_service.subscribe(lambda event: generated.append(event))
    groq_client = app.base.base.base.response_generator.client

    await app.response_service.handle_transcript_event(
        TranscriptEvent(
            call_id="call-1",
            turn_id=1,
            speaker="customer",
            text="I need help with my account",
            is_final=False,
            confidence=0.89,
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
        )
    )
    interim_count = len(generated)

    await app.response_service.handle_transcript_event(
        TranscriptEvent(
            call_id="call-1",
            turn_id=1,
            speaker="customer",
            text="I need help with my account",
            is_final=True,
            confidence=0.96,
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
        )
    )
    first_prompt = groq_client.last_prompt
    first_response = generated[-1].text
    first_messages = groq_client.last_messages

    app.base.base.base.persona_manager.select_persona("friendly_support")

    await app.response_service.handle_transcript_event(
        TranscriptEvent(
            call_id="call-1",
            turn_id=2,
            speaker="customer",
            text="I also want a warmer greeting",
            is_final=True,
            confidence=0.95,
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
        )
    )

    for event in generated:
        print(event.model_dump())
    print({
        "interim_ignored": interim_count == 0,
        "persona_switched": first_prompt != groq_client.last_prompt,
        "response_changed": first_response != generated[-1].text,
        "system_prompt_included": first_messages[0]["role"] == "system",
        "latest_transcript_included": first_messages[-1] == {
            "role": "user",
            "content": "I need help with my account",
        },
        "history_included_on_second_turn": len(groq_client.last_messages) >= 4,
    })


if __name__ == "__main__":
    asyncio.run(main())
