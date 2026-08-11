"""Response generation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Single conversation turn used for prompt assembly."""

    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class ResponseRequest:
    """Structured request passed to a response generator."""

    persona_id: str
    system_prompt: str
    latest_customer_transcript: str
    temperature: float = 0.4
    conversation_history: tuple[ConversationTurn, ...] = field(default_factory=tuple)


class ResponseGenerator(Protocol):
    """Async abstraction for future replacement by Module 5."""

    async def generate(self, request: ResponseRequest) -> str:
        """Generate a response for the supplied conversation context."""

    def stream(self, request: ResponseRequest) -> AsyncIterator[str]:
        """Stream response text increments when supported by the implementation."""
