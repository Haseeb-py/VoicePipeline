"""Asynchronous Groq API client isolated behind the response provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GroqClient:
    """Stream real Groq chat completions without exposing provider details upstream."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model
        self.last_prompt: str = ""
        self.last_messages: tuple[dict[str, str], ...] = ()
        self.generate_count: int = 0

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield incremental content from Groq's async streaming API."""

        self.last_messages = tuple(dict(message) for message in messages)
        self.last_prompt = "\n\n".join(
            f"{message['role'].title()}: {message['content']}" for message in messages
        )
        self.generate_count += 1

        if self._api_key in {"mock-key", "test-key"}:
            yield f"[offline-groq-response] {self.last_prompt[:120]}"
            return
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is missing from the environment")

        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self._api_key)
            stream = await client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=temperature,
                max_completion_tokens=256,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content
            await client.close()
        except Exception as exc:
            # Provider exceptions may contain request metadata; expose only type.
            logger.error("Groq streaming request failed: %s", type(exc).__name__)
            raise RuntimeError("Groq response generation failed") from None

    async def generate_text(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        """Collect a streamed response for callers requiring a complete string."""

        parts = [part async for part in self.stream_chat(messages, temperature=temperature)]
        return "".join(parts).strip()
