"""Conversation history storage for temporary response generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from echoai_voice.models.responses import ConversationTurn


@dataclass(slots=True)
class ConversationHistoryStore:
    """Append-only conversation history for a single call session."""

    _turns: list[ConversationTurn] = field(default_factory=list)

    def append_customer(self, text: str) -> None:
        """Record a customer turn."""

        self._turns.append(ConversationTurn(speaker="customer", text=text))

    def append_ai(self, text: str) -> None:
        """Record an AI turn."""

        self._turns.append(ConversationTurn(speaker="ai", text=text))

    def snapshot(self) -> tuple[ConversationTurn, ...]:
        """Return an immutable conversation history snapshot."""

        return tuple(self._turns)

    def clear(self) -> None:
        """Reset the history for a fresh call."""

        self._turns.clear()
