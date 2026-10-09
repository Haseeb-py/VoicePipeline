"""Phase 7 UI adapter over the existing real voice pipeline."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import timedelta
from typing import Any
from uuid import uuid4

from echoai_voice.core.vad_app import (
    VadVoiceApplication,
    build_vad_application,
    start_vad_call,
    stop_vad_call,
)
from echoai_voice.models.latency import LatencyCheckpoint
from echoai_voice.models.response_events import (
    GeneratedResponseChunkEvent,
    GeneratedResponseEvent,
)
from echoai_voice.models.session import ConnectionState
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.models.tts import TtsAudioChunk
from echoai_voice.services.persona_manager import PersonaUpdate


class VoiceUiController:
    """Expose Phase 1-6 state and events without changing their ownership."""

    def __init__(self) -> None:
        self._app = build_vad_application()
        self._selected_persona_id = self.persona_manager.get_active_persona().id
        self._running = False
        self._has_started = False
        self._lock = asyncio.Lock()
        self._status_task: asyncio.Task[None] | None = None
        self._event_queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._tts_started_turns: set[tuple[str | None, int | None]] = set()
        self._browser_identity: str | None = None
        self._browser_credentials_payload: dict[str, str] | None = None
        self._wire_events()

    @property
    def app(self) -> VadVoiceApplication:
        return self._app

    @property
    def persona_manager(self):
        return self._app.base.base.base.base.base.persona_manager

    @property
    def session_manager(self):
        return self._app.base.base.base.base.livekit

    @property
    def transcript_bus(self):
        return self._app.base.base.base.transcript_bus

    @property
    def response_service(self):
        return self._app.base.base.response_service

    @property
    def latency_tracker(self):
        return self._app.base.base.base.base.base.latency_tracker

    @property
    def settings(self):
        return self._app.base.base.base.base.base.settings

    @property
    def is_running(self) -> bool:
        return self._running

    def _call_is_healthy(self) -> bool:
        """Return whether every component needed for a browser call is alive."""

        return (
            self._running
            and self.session_manager.status.connection == ConnectionState.CONNECTED
            and self._app.base.base.base.stt_provider.is_running
            and self._app.audio_transport.is_running
        )

    def list_personas(self) -> list[dict[str, Any]]:
        return [
            {
                "id": persona.id,
                "name": persona.display_name,
                "description": persona.description,
                "voice": persona.tts.voice,
                "speech_rate": persona.tts.speech_rate,
            }
            for persona in self.persona_manager.list_personas()
        ]

    def snapshot(self) -> dict[str, Any]:
        status = self.session_manager.status
        persona = self.persona_manager.get_active_persona()
        running = self._call_is_healthy()
        listening = status.listening.value if running else "idle"
        speaking = status.speaking.value if running else "silent"
        return {
            "running": running,
            "connection": status.connection.value,
            "listening": listening,
            "speaking": speaking,
            "call_id": status.call_id,
            "room_name": status.room_name,
            "persona": {
                "id": persona.id,
                "name": persona.display_name,
                "voice": persona.tts.voice,
                "speech_rate": persona.tts.speech_rate,
            },
        }

    async def start_call(self) -> dict[str, Any]:
        async with self._lock:
            if self._running:
                if self._call_is_healthy():
                    return {
                        **self.snapshot(),
                        **(self._browser_credentials_payload or self.browser_credentials()),
                    }
                # A provider or room can close independently of the browser.
                # Discard that stale call graph and let this button press start
                # a fresh session instead of permanently remaining disabled.
                await self._recover_stale_call()

            if self._has_started:
                await self._recover_stale_call()

            credentials = self._browser_credentials_payload or self.browser_credentials()
            self._app.audio_transport.input_participant_identity = self._browser_identity
            self._publish("lifecycle", {"state": "starting"})
            self._publish_status()
            try:
                await start_vad_call(self._app)
            except Exception:
                with suppress(Exception):
                    await stop_vad_call(self._app)
                self._publish("error", {"message": "Voice pipeline failed to start"})
                self._app = build_vad_application()
                self.persona_manager.select_persona(self._selected_persona_id)
                self._wire_events()
                self._publish_status()
                raise

            self._running = True
            self._has_started = True
            self._publish("lifecycle", {"state": "started"})
            self._publish_status()
            self._status_task = asyncio.create_task(
                self._watch_status(), name="phase7-status-watch"
            )
            return {**self.snapshot(), **credentials}

    async def _recover_stale_call(self) -> None:
        """Release a dead call and rebuild the per-call pipeline components."""

        status_task = self._status_task
        self._status_task = None
        if status_task is not None:
            status_task.cancel()
            with suppress(asyncio.CancelledError):
                await status_task
        with suppress(Exception):
            await stop_vad_call(self._app)
        self._running = False
        self._app = build_vad_application()
        self.persona_manager.select_persona(self._selected_persona_id)
        self._tts_started_turns.clear()
        self._wire_events()
        self._publish("lifecycle", {"state": "recovered"})
        self._publish_status()

    async def end_call(self) -> dict[str, Any]:
        async with self._lock:
            if not self._running:
                return self.snapshot()

            self._publish("lifecycle", {"state": "ending"})
            status_task = self._status_task
            self._status_task = None
            if status_task is not None:
                status_task.cancel()
                with suppress(asyncio.CancelledError):
                    await status_task
            try:
                await stop_vad_call(self._app)
                # Deepgram may deliver a final result while its close stream is
                # draining. Run one final existing-service cancellation pass so
                # End Call cannot leave a late Groq/TTS task behind.
                await self.response_service.cancel_current_response()
                await self.response_service.wait_until_idle()
                await self._app.base.tts_service.cancel_current_speech(
                    emit_interruption=False
                )
                await self._app.base.tts_provider.stop()
            finally:
                self._running = False
                self._publish("lifecycle", {"state": "ended"})
                self._publish_status()
            return self.snapshot()

    def select_persona(self, persona_id: str) -> dict[str, Any]:
        persona = self.persona_manager.select_persona(persona_id)
        self._selected_persona_id = persona.id
        return self.snapshot()["persona"]

    def browser_credentials(self) -> dict[str, str]:
        """Issue short-lived credentials for the real browser participant."""

        return self._browser_credentials()

    def subscribe_events(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._event_queues.add(queue)
        self._put_event(queue, {"type": "session", "data": self.snapshot()})
        return queue

    def unsubscribe_events(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._event_queues.discard(queue)

    def _wire_events(self) -> None:
        self.transcript_bus.subscribe(self._on_transcript)
        self.response_service.subscribe(self._on_response)
        self.response_service.subscribe_stream(self._on_response_chunk)
        self.persona_manager.subscribe(self._on_persona)
        self.latency_tracker.subscribe(self._on_latency)
        self._app.base.audio_sink.subscribe(self._on_tts_audio)

    def _on_transcript(self, event: TranscriptEvent) -> None:
        self._publish("transcript", event.model_dump())

    def _on_response(self, event: GeneratedResponseEvent) -> None:
        self._publish("response", event.model_dump())

    def _on_response_chunk(self, event: GeneratedResponseChunkEvent) -> None:
        self._publish(
            "response_chunk",
            {
                "call_id": event.call_id,
                "turn_id": event.turn_id,
                "persona_id": event.persona_id,
                "text": event.text,
                "sequence": event.sequence,
                "is_final": event.is_final,
                "timestamp": event.timestamp.isoformat(),
            },
        )

    def _on_persona(self, update: PersonaUpdate) -> None:
        persona = update.persona
        self._publish(
            "persona",
            {
                "id": persona.id,
                "name": persona.display_name,
                "voice": persona.tts.voice,
                "speech_rate": persona.tts.speech_rate,
            },
        )

    def _on_latency(self, checkpoint: LatencyCheckpoint) -> None:
        metrics = self.latency_tracker.turn_metrics(
            checkpoint.call_id, checkpoint.turn_id
        )
        self._publish(
            "latency",
            {
                "call_id": checkpoint.call_id,
                "turn_id": checkpoint.turn_id,
                "checkpoint": checkpoint.name.value,
                "elapsed_ms": round(metrics.get(checkpoint.name.value, 0.0) * 1000, 1),
                "metrics_ms": {
                    name: round(seconds * 1000, 1)
                    for name, seconds in metrics.items()
                },
                "metadata": checkpoint.metadata,
            },
        )

    def _on_tts_audio(self, chunk: TtsAudioChunk) -> None:
        turn_key = (chunk.call_id, chunk.turn_id)
        if chunk.is_interruption:
            self._publish("barge_in", {"state": "interrupted"})
            self._publish_status()
            return
        if chunk.pcm16 and turn_key not in self._tts_started_turns:
            self._tts_started_turns.add(turn_key)
            self._publish(
                "tts",
                {"state": "audio_started", "call_id": chunk.call_id, "turn_id": chunk.turn_id},
            )
            self._publish_status()
        if chunk.is_final:
            self._publish(
                "tts",
                {"state": "audio_completed", "call_id": chunk.call_id, "turn_id": chunk.turn_id},
            )

    async def _watch_status(self) -> None:
        previous: tuple[str, str, str] | None = None
        while self._running:
            if not self._call_is_healthy():
                self._running = False
                self._publish(
                    "error",
                    {"message": "Voice connection ended. Start a new call to reconnect."},
                )
                self._publish_status()
                return
            status = self.session_manager.status
            current = (
                status.connection.value,
                status.listening.value,
                status.speaking.value,
            )
            if current != previous:
                previous = current
                self._publish_status()
            await asyncio.sleep(0.1)

    def _publish_status(self) -> None:
        self._publish("session", self.snapshot())

    def _publish(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "data": data}
        for queue in tuple(self._event_queues):
            self._put_event(queue, event)

    @staticmethod
    def _put_event(
        queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]
    ) -> None:
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(event)

    def _browser_credentials(self) -> dict[str, str]:
        from livekit import api

        settings = self.settings
        identity = f"echoai-ui-{uuid4().hex[:12]}"
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(identity)
            .with_name("EchoAI Development UI")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.session_manager.status.room_name or "echoai-test-room",
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=False,
                )
            )
            .with_ttl(timedelta(minutes=15))
            .to_jwt()
        )
        credentials = {
            "livekit_url": settings.livekit_url,
            "participant_token": token,
        }
        self._browser_identity = identity
        self._browser_credentials_payload = credentials
        return credentials
