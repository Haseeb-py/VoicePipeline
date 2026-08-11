"""Targeted Phase 7 UI/event-adapter verification."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os

os.environ["DEEPGRAM_API_KEY"] = "test-key"
os.environ["GROQ_API_KEY"] = "test-key"

import httpx

from echoai_voice.models.latency import LatencyCheckpointName
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.ui.controller import VoiceUiController


async def _check_event_adapter() -> dict[str, bool]:
    controller = VoiceUiController()
    queue = controller.subscribe_events()
    personas = controller.list_personas()
    controller.select_persona(personas[1]["id"])

    now = datetime.now(timezone.utc)
    for is_final, text, confidence in (
        (False, "I need", 0.74),
        (True, "I need help with my account", 0.93),
    ):
        await controller.transcript_bus.publish(
            TranscriptEvent(
                call_id="ui-check",
                turn_id=1,
                speaker="customer",
                text=text,
                is_final=is_final,
                confidence=confidence,
                timestamp_start=now,
                timestamp_end=now,
            )
        )

    await controller.response_service.wait_until_idle()
    controller.latency_tracker.record(
        LatencyCheckpointName.PLAYBACK_STARTED, "ui-check", 1
    )
    controller.app.base.audio_sink.clear()

    events: list[dict[str, object]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    controller.unsubscribe_events(queue)
    event_types = [str(event["type"]) for event in events]
    transcript_events = [event for event in events if event["type"] == "transcript"]
    return {
        "personas_exposed": len(personas) >= 2,
        "persona_event_exposed": "persona" in event_types,
        "interim_and_final_transcripts": len(transcript_events) == 2,
        "streamed_response_exposed": "response_chunk" in event_types,
        "final_response_exposed": event_types.count("response") == 1,
        "tts_state_exposed": "tts" in event_types,
        "barge_in_exposed": "barge_in" in event_types,
        "latency_exposed": "latency" in event_types,
        "session_snapshot_exposed": "session" in event_types,
    }


async def _check_http_surface() -> dict[str, bool]:
    from echoai_voice.ui.server import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://phase7-check"
    ) as client:
        index = await client.get("/")
        styles = await client.get("/assets/styles.css")
        script = await client.get("/assets/app.js")
        state = await client.get("/api/state")
        personas = await client.get("/api/personas")
        credentials = await client.post("/api/calls/credentials")
    return {
        "ui_served": (
            index.status_code == 200
            and "start-call" in index.text
            and "microphone-select" in index.text
            and "mic-meter-bar" in index.text
        ),
        "assets_served": styles.status_code == 200 and script.status_code == 200,
        "state_endpoint": state.status_code == 200 and "connection" in state.json(),
        "persona_endpoint": personas.status_code == 200 and len(personas.json()) >= 2,
        "browser_credentials_endpoint": (
            credentials.status_code == 200
            and bool(credentials.json().get("participant_token"))
            and bool(credentials.json().get("livekit_url"))
        ),
    }


async def _run() -> dict[str, bool]:
    return {**await _check_event_adapter(), **await _check_http_surface()}


def main() -> None:
    results = asyncio.run(_run())
    print(f"Phase 7 Verification Results: {results}")
    if not all(results.values()):
        raise SystemExit("Phase 7 verification failed")


if __name__ == "__main__":
    main()
