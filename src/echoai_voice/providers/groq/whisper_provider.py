"""Groq-hosted OpenAI Whisper fallback provider."""

from __future__ import annotations

import io
import logging
import wave

logger = logging.getLogger(__name__)


class GroqWhisperFallbackProvider:
    """Transcribe bounded PCM utterances with Whisper Large V3 Turbo."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo") -> None:
        self._api_key = api_key
        self._model = model
        self.transcription_count = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int, channels: int) -> str:
        self.transcription_count += 1
        if self._api_key in {"mock-key", "test-key"}:
            return "[offline-whisper-transcript]"
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is missing for Whisper fallback")

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16)

        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self._api_key)
            result = await client.audio.transcriptions.create(
                file=("utterance.wav", wav_buffer.getvalue()),
                model=self._model,
                language="en",
                response_format="json",
                temperature=0.0,
            )
            await client.close()
            return str(result.text).strip()
        except Exception as exc:
            logger.error("Whisper fallback request failed: %s", type(exc).__name__)
            raise RuntimeError("Whisper fallback request failed") from None
