"""Transcript-driven response generation for the temporary Groq integration."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from echoai_voice.models.response_events import GeneratedResponseChunkEvent, GeneratedResponseEvent
from echoai_voice.models.responses import ResponseRequest, ResponseGenerator
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.services.conversation_history import ConversationHistoryStore
from echoai_voice.services.persona_manager import PersonaManager
from echoai_voice.models.latency import LatencyCheckpointName
from echoai_voice.services.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)

ResponseSubscriber = Callable[[GeneratedResponseEvent], Awaitable[None] | None]
ResponseChunkSubscriber = Callable[[GeneratedResponseChunkEvent], Awaitable[None] | None]


@dataclass(slots=True)
class ResponseGenerationService:
    """Consume final customer transcript events and generate AI responses."""

    persona_manager: PersonaManager
    response_generator: ResponseGenerator
    history_store: ConversationHistoryStore = field(default_factory=ConversationHistoryStore)
    latency_tracker: LatencyTracker | None = None
    _subscribers: list[ResponseSubscriber] = field(default_factory=list)
    _stream_subscribers: list[ResponseChunkSubscriber] = field(default_factory=list)
    _call_id: str = "pending-call"
    _next_turn_id: int = 1
    # Deepgram can finalize short phrase fragments before the speaker has
    # completed a thought. Let the final transcript settle briefly so a later
    # fragment replaces it before expensive LLM/TTS work begins.
    finalization_delay_seconds: float = 0.35
    _generation: int = field(default=0, init=False)
    _active_response_task: asyncio.Task[None] | None = field(default=None, init=False)

    def bind_call_id(self, call_id: str | None) -> None:
        """Bind the active call identifier to generated response events."""

        if call_id:
            self._call_id = call_id

    def subscribe(self, callback: ResponseSubscriber) -> None:
        """Register a subscriber for generated responses."""

        self._subscribers.append(callback)

    def subscribe_stream(self, callback: ResponseChunkSubscriber) -> None:
        """Register a subscriber for incremental response text."""

        self._stream_subscribers.append(callback)

    def reset(self) -> None:
        """Reset conversation history for a new call."""

        self._generation += 1
        if self._active_response_task is not None:
            self._active_response_task.cancel()
            self._active_response_task = None
        self.history_store.clear()
        self._next_turn_id = 1

    @property
    def has_active_response(self) -> bool:
        """Return whether response generation or one of its subscribers is active."""

        task = self._active_response_task
        return task is not None and not task.done()

    def submit_transcript_event(self, event: TranscriptEvent) -> None:
        """Schedule a final customer turn without blocking the STT receive loop."""

        if event.speaker != "customer" or not event.is_final or not event.text.strip():
            return

        previous_task = self._active_response_task
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()

        self._generation += 1
        generation = self._generation
        task = asyncio.create_task(
            self._generate_after_finalization(event, generation),
            name=f"response-generation-turn-{event.turn_id}",
        )
        self._active_response_task = task
        task.add_done_callback(self._response_task_done)

    async def _generate_after_finalization(
        self, event: TranscriptEvent, generation: int
    ) -> None:
        """Wait briefly for a trailing final transcript before answering."""

        await asyncio.sleep(self.finalization_delay_seconds)
        if generation != self._generation:
            return
        await self._generate_for_event(event, generation)

    async def handle_transcript_event(self, event: TranscriptEvent) -> None:
        """Handle transcript events and generate responses only for final customer turns."""

        if event.speaker != "customer" or not event.is_final or not event.text.strip():
            return

        self._generation += 1
        await self._generate_for_event(event, self._generation)

    async def cancel_current_response(self) -> bool:
        """Invalidate and cancel response work so interrupted output cannot become stale audio."""

        self._generation += 1
        task = self._active_response_task
        if task is None or task.done():
            return False

        logger.info("Cancelling active response work after customer speech started.")
        task.cancel()
        # Give cancellation one event-loop turn without allowing a slow provider
        # that suppresses CancelledError to delay barge-in handling. The generation
        # token above still prevents any late result from being emitted.
        await asyncio.sleep(0)
        return True

    async def wait_until_idle(self) -> None:
        """Wait for currently scheduled response work to finish."""

        task = self._active_response_task
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    async def _generate_for_event(self, event: TranscriptEvent, generation: int) -> None:
        """Generate and emit one response if it has not been invalidated."""

        self.bind_call_id(event.call_id)
        persona = self.persona_manager.get_active_persona()
        prior_history = self.history_store.snapshot()
        self.history_store.append_customer(event.text.strip())
        request = ResponseRequest(
            persona_id=persona.id,
            system_prompt=persona.system_prompt,
            latest_customer_transcript=event.text.strip(),
            temperature=persona.llm.temperature,
            conversation_history=prior_history,
        )
        turn_id = event.turn_id
        self._next_turn_id = max(self._next_turn_id, turn_id + 1)
        self._record_latency(LatencyCheckpointName.RESPONSE_GENERATION_STARTED, event.call_id, turn_id)
        logger.info("Generating response for persona=%s turn=%s", persona.id, turn_id)
        parts: list[str] = []
        sequence = 0
        async for text_chunk in self.response_generator.stream(request):
            if generation != self._generation:
                logger.info("Discarding stale generated response after barge-in.")
                return
            if not text_chunk:
                continue
            parts.append(text_chunk)
            if sequence == 0:
                self._record_latency(LatencyCheckpointName.FIRST_RESPONSE_CHUNK, event.call_id, turn_id)
            await self._emit_stream(
                GeneratedResponseChunkEvent(
                    call_id=self._call_id,
                    turn_id=turn_id,
                    persona_id=persona.id,
                    text=text_chunk,
                    sequence=sequence,
                    is_final=False,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            sequence += 1

        if generation != self._generation:
            return
        await self._emit_stream(
            GeneratedResponseChunkEvent(
                call_id=self._call_id,
                turn_id=turn_id,
                persona_id=persona.id,
                text="",
                sequence=sequence,
                is_final=True,
                timestamp=datetime.now(timezone.utc),
            )
        )
        generated_text = "".join(parts).strip()
        if not generated_text or generation != self._generation:
            return
        self._record_latency(LatencyCheckpointName.RESPONSE_TEXT_COMPLETED, event.call_id, turn_id)
        response_event = GeneratedResponseEvent(
            call_id=self._call_id,
            turn_id=turn_id,
            persona_id=persona.id,
            text=generated_text,
            timestamp=datetime.now(timezone.utc),
        )
        await self._emit(response_event)
        if generation == self._generation:
            self.history_store.append_ai(generated_text)

    def _record_latency(
        self,
        name: LatencyCheckpointName,
        call_id: str,
        turn_id: int,
    ) -> None:
        if self.latency_tracker is not None:
            self.latency_tracker.record(name, call_id, turn_id)

    async def _emit_stream(self, event: GeneratedResponseChunkEvent) -> None:
        tasks: list[asyncio.Task[object]] = []
        for subscriber in tuple(self._stream_subscribers):
            result = subscriber(event)
            if asyncio.iscoroutine(result):
                tasks.append(asyncio.create_task(result))
        if tasks:
            await asyncio.gather(*tasks)

    def _response_task_done(self, task: asyncio.Task[None]) -> None:
        if self._active_response_task is task:
            self._active_response_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Scheduled response generation failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _emit(self, event: GeneratedResponseEvent) -> None:
        tasks: list[asyncio.Task[object]] = []
        for subscriber in tuple(self._subscribers):
            result = subscriber(event)
            if asyncio.iscoroutine(result):
                tasks.append(asyncio.create_task(result))
        if tasks:
            await asyncio.gather(*tasks)
