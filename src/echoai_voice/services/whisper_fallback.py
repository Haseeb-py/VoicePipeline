"""Non-blocking low-confidence Whisper fallback orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Protocol

from echoai_voice.models.fallback import WhisperFallbackResult
from echoai_voice.models.transcript import TranscriptEvent

logger = logging.getLogger(__name__)


class WhisperTranscriber(Protocol):
    async def transcribe(self, pcm16: bytes, sample_rate: int, channels: int) -> str: ...


FallbackSubscriber = Callable[[WhisperFallbackResult], Awaitable[None] | None]


@dataclass(slots=True)
class WhisperFallbackService:
    """Run diagnostic Whisper checks outside the primary transcript/response path."""

    provider: WhisperTranscriber
    _subscribers: list[FallbackSubscriber] = field(default_factory=list)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)

    def subscribe(self, callback: FallbackSubscriber) -> None:
        self._subscribers.append(callback)

    @property
    def active_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    def schedule(self, event: TranscriptEvent, pcm16: bytes, sample_rate: int, channels: int) -> None:
        """Schedule fallback and return immediately without gating the primary path."""

        task = asyncio.create_task(
            self._run(event, pcm16, sample_rate, channels),
            name=f"whisper-fallback-turn-{event.turn_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    async def wait_for_idle(self) -> None:
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def _run(
        self,
        event: TranscriptEvent,
        pcm16: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        try:
            fallback_text = await self.provider.transcribe(pcm16, sample_rate, channels)
        except Exception:
            logger.exception("Whisper fallback failed for turn_id=%s", event.turn_id)
            return
        result = WhisperFallbackResult(
            call_id=event.call_id,
            turn_id=event.turn_id,
            primary_text=event.text,
            fallback_text=fallback_text,
            primary_confidence=event.confidence,
            timestamp=datetime.now(timezone.utc),
        )
        for subscriber in tuple(self._subscribers):
            callback_result = subscriber(result)
            if asyncio.iscoroutine(callback_result):
                await callback_result

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Whisper fallback task failed: %s", type(error).__name__)
