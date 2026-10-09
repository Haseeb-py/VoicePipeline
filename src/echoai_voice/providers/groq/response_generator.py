"""Temporary Groq-backed response generator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from echoai_voice.models.responses import ResponseRequest
from echoai_voice.providers.groq.client import GroqClient


@dataclass(slots=True)
class GroqResponseGenerator:
    """Groq implementation of the replaceable ResponseGenerator contract."""

    client: GroqClient

    async def generate(self, request: ResponseRequest) -> str:
        """Generate complete text while consuming Groq's streaming response."""

        return await self.client.generate_text(
            self._messages(request),
            temperature=request.temperature,
        )

    async def stream(self, request: ResponseRequest) -> AsyncIterator[str]:
        """Yield Groq response increments for streaming downstream consumers."""

        async for content in self.client.stream_chat(
            self._messages(request),
            temperature=request.temperature,
        ):
            yield content

    @staticmethod
    def _messages(request: ResponseRequest) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{request.system_prompt}\n\n"
                    "This is a live voice conversation. Give a direct, natural "
                    "spoken answer in one or two concise sentences, then ask at "
                    "most one necessary follow-up question."
                ),
            }
        ]
        for turn in request.conversation_history:
            role = "assistant" if turn.speaker == "ai" else "user"
            messages.append({"role": role, "content": turn.text})
        messages.append({"role": "user", "content": request.latest_customer_transcript})
        return messages
