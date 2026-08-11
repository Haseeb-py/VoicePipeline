"""Verification for non-blocking, non-duplicating Whisper fallback behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from echoai_voice.core.response_app import build_response_application
from echoai_voice.models.fallback import WhisperFallbackResult
from echoai_voice.models.response_events import GeneratedResponseEvent
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.providers.livekit.audio_source import AudioFrameChunk
from echoai_voice.services.whisper_fallback import WhisperFallbackService


class DelayedWhisperProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int, channels: int) -> str:
        self.calls += 1
        await asyncio.sleep(0.1)
        return "fallback transcription"


def _result(text: str, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="Results",
        is_final=True,
        speech_final=True,
        duration=0.5,
        channel=SimpleNamespace(
            alternatives=[SimpleNamespace(transcript=text, confidence=confidence)]
        ),
    )


async def main() -> None:
    app = build_response_application()
    provider = DelayedWhisperProvider()
    fallback_service = WhisperFallbackService(provider)
    app.base.stt_provider.fallback_service = fallback_service
    fallback_results: list[WhisperFallbackResult] = []
    transcript_events: list[TranscriptEvent] = []
    responses: list[GeneratedResponseEvent] = []
    fallback_service.subscribe(fallback_results.append)
    app.base.transcript_bus.subscribe(transcript_events.append)
    app.response_service.subscribe(responses.append)

    audio = AudioFrameChunk(b"\x01\x00" * 8000, sample_rate=16000)
    await app.base.stt_provider.push_audio(audio)
    await app.base.stt_provider.handle_deepgram_message(SimpleNamespace(type="SpeechStarted"))
    loop = asyncio.get_running_loop()
    started = loop.time()
    await app.base.stt_provider.handle_deepgram_message(_result("low confidence turn", 0.31))
    primary_return_seconds = loop.time() - started
    fallback_was_backgrounded = fallback_service.active_count == 1
    await app.response_service.wait_until_idle()
    await fallback_service.wait_for_idle()

    await app.base.stt_provider.push_audio(audio)
    await app.base.stt_provider.handle_deepgram_message(SimpleNamespace(type="SpeechStarted"))
    await app.base.stt_provider.handle_deepgram_message(_result("high confidence turn", 0.96))
    await app.response_service.wait_until_idle()

    final_transcripts = [event for event in transcript_events if event.is_final]
    result = {
        "primary_path_not_blocked": primary_return_seconds < 0.05,
        "fallback_ran_in_background": fallback_was_backgrounded,
        "fallback_only_on_low_confidence": provider.calls == 1,
        "separate_fallback_result_emitted": len(fallback_results) == 1,
        "primary_final_events_unchanged": len(final_transcripts) == 2,
        "no_duplicate_responses": len(responses) == 2,
    }
    print("Whisper Fallback Verification Results:", result)
    if not all(result.values()):
        raise AssertionError(result)


if __name__ == "__main__":
    asyncio.run(main())
