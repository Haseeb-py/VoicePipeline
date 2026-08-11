"""Phase 6 verification for VAD, cancellation, and stale-audio prevention."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from echoai_voice.core.vad_app import build_vad_application
from echoai_voice.models.tts import TtsAudioChunk


def _final_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="Results",
        is_final=True,
        speech_final=True,
        duration=1.0,
        channel=SimpleNamespace(
            alternatives=[SimpleNamespace(transcript=text, confidence=0.98)]
        ),
    )


def _interim_result(text: str, confidence: float = 0.98) -> SimpleNamespace:
    return SimpleNamespace(
        type="Results",
        is_final=False,
        speech_final=False,
        duration=0.4,
        channel=SimpleNamespace(
            alternatives=[SimpleNamespace(transcript=text, confidence=confidence)]
        ),
    )


async def main() -> None:
    app = build_vad_application()
    stt_provider = app.base.base.base.stt_provider
    response_service = app.base.base.response_service
    status = app.base.base.base.base.livekit.status
    received_chunks: list[TtsAudioChunk] = []
    app.base.audio_sink.subscribe(received_chunks.append)

    # Raw VAD/noise must not cancel pending response work. The first reliable
    # transcript hypothesis confirms a genuine interruption and cancels it.
    await stt_provider.handle_deepgram_message(
        _final_result("Please prepare a detailed explanation of my account issue.")
    )
    pending_work_was_scheduled = response_service.has_active_response
    await stt_provider.handle_deepgram_message(SimpleNamespace(type="SpeechStarted"))
    await stt_provider.handle_deepgram_message(_interim_result("noise", confidence=0.1))
    noise_did_not_cancel_pending = response_service.has_active_response
    await stt_provider.handle_deepgram_message(_interim_result("excuse me"))
    stale_audio_before_synthesis = any(chunk.pcm16 for chunk in received_chunks)

    # Let a fresh response begin speaking, then interrupt it mid-stream.
    await stt_provider.handle_deepgram_message(
        _final_result(
            "Please give me a thorough answer with enough detail to keep the audio stream active."
        )
    )
    await asyncio.sleep(0.035)
    audio_chunks_before_barge_in = len(received_chunks)
    speaking_before_barge_in = str(status.speaking)

    await stt_provider.handle_deepgram_message(SimpleNamespace(type="SpeechStarted"))
    await stt_provider.handle_deepgram_message(_interim_result("background", confidence=0.1))
    await asyncio.sleep(0.015)
    noise_did_not_interrupt_tts = not any(
        chunk.is_interruption for chunk in received_chunks[audio_chunks_before_barge_in:]
    )
    await stt_provider.handle_deepgram_message(_interim_result("stop for a moment"))
    interruption_index = next(
        (index for index, chunk in enumerate(received_chunks) if chunk.is_interruption),
        None,
    )
    count_after_interruption = len(received_chunks)
    await asyncio.sleep(0.05)
    no_stale_audio_after_flush = len(received_chunks) == count_after_interruption

    # The next finalized customer utterance must still run through the normal
    # response and TTS path after interruption.
    await stt_provider.handle_deepgram_message(
        _final_result("Now continue with the new question after my interruption.")
    )
    await response_service.wait_until_idle()
    new_audio_after_interruption = len(received_chunks) > count_after_interruption

    verification_result = {
        "response_work_does_not_block_stt": pending_work_was_scheduled,
        "background_noise_does_not_cancel_pending": noise_did_not_cancel_pending,
        "pending_response_cancelled_before_tts": not stale_audio_before_synthesis,
        "background_noise_does_not_interrupt_tts": noise_did_not_interrupt_tts,
        "mid_stream_interruption_triggered": (
            audio_chunks_before_barge_in > 0 and speaking_before_barge_in == "speaking"
        ),
        "explicit_audio_flush_emitted": interruption_index is not None,
        "no_stale_audio_after_flush": no_stale_audio_after_flush,
        "speaking_reset_to_silent": str(status.speaking) == "silent",
        "listening_reset_to_listening": str(status.listening) == "listening",
        "next_customer_turn_processed": new_audio_after_interruption,
    }
    print("Phase 6 Verification Results:", verification_result)

    if not all(verification_result.values()):
        raise AssertionError(f"Phase 6 verification failed: {verification_result}")


if __name__ == "__main__":
    asyncio.run(main())
