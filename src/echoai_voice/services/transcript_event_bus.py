"""Async event bus for normalized transcript events."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from echoai_voice.models.transcript import TranscriptEvent

TranscriptSubscriber = Callable[[TranscriptEvent], Awaitable[None] | None]


@dataclass(slots=True)
class TranscriptEventBus:
    """Fan out transcript events to UI listeners and downstream consumers."""

    _subscribers: list[TranscriptSubscriber] = field(default_factory=list)
    _queue: asyncio.Queue[TranscriptEvent] = field(default_factory=asyncio.Queue)

    def subscribe(self, callback: TranscriptSubscriber) -> None:
        """Register a subscriber for every transcript event."""

        self._subscribers.append(callback)

    async def publish(self, event: TranscriptEvent) -> None:
        """Deliver an event to the internal queue and all subscribers."""

        await self._queue.put(event)
        await self._dispatch(event)

    async def next_event(self) -> TranscriptEvent:
        """Wait for the next transcript event."""

        return await self._queue.get()

    async def _dispatch(self, event: TranscriptEvent) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for callback in tuple(self._subscribers):
            result = callback(event)
            if asyncio.iscoroutine(result):
                tasks.append(asyncio.create_task(result))
        if tasks:
            await asyncio.gather(*tasks)
