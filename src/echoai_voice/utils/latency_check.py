"""End-to-end monotonic latency checkpoint verification."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from echoai_voice.core.vad_app import build_vad_application
from echoai_voice.models.latency import LatencyCheckpointName


class FakeRtcAudioSource:
    def __init__(self) -> None:
        self.frames = 0
        self.playout_finished = asyncio.Event()

    async def capture_frame(self, frame) -> None:
        self.frames += 1

    async def wait_for_playout(self) -> None:
        self.playout_finished.set()

    def clear_queue(self) -> None:
        return None


def _result(text: str, *, final: bool) -> SimpleNamespace:
    return SimpleNamespace(
        type="Results",
        is_final=final,
        speech_final=final,
        duration=0.5,
        channel=SimpleNamespace(
            alternatives=[SimpleNamespace(transcript=text, confidence=0.98)]
        ),
    )


async def main() -> None:
    app = build_vad_application()
    transport = app.audio_transport
    fake_rtc_source = FakeRtcAudioSource()
    transport._rtc_output_source = fake_rtc_source
    transport.output_sink.subscribe(transport._on_output_chunk)
    transport._output_task = asyncio.create_task(transport._play_output())

    try:
        stt = app.base.base.base.stt_provider
        await stt.handle_deepgram_message(SimpleNamespace(type="SpeechStarted"))
        await stt.handle_deepgram_message(_result("latency interim", final=False))
        await stt.handle_deepgram_message(_result("latency final", final=True))
        await app.base.base.response_service.wait_until_idle()
        await asyncio.wait_for(fake_rtc_source.playout_finished.wait(), timeout=3)

        checkpoints = app.base.base.base.base.base.latency_tracker.snapshot("pending-call", 1)
        names = {checkpoint.name for checkpoint in checkpoints}
        required = {
            LatencyCheckpointName.SPEECH_INPUT_DETECTED,
            LatencyCheckpointName.STT_INTERIM_RECEIVED,
            LatencyCheckpointName.STT_FINAL_RECEIVED,
            LatencyCheckpointName.RESPONSE_GENERATION_STARTED,
            LatencyCheckpointName.FIRST_RESPONSE_CHUNK,
            LatencyCheckpointName.TTS_GENERATION_STARTED,
            LatencyCheckpointName.FIRST_AUDIO_CHUNK_AVAILABLE,
            LatencyCheckpointName.PLAYBACK_STARTED,
            LatencyCheckpointName.RESPONSE_COMPLETED,
        }
        monotonic_values = [checkpoint.monotonic_seconds for checkpoint in checkpoints]
        metrics = app.base.base.base.base.base.latency_tracker.turn_metrics("pending-call", 1)
        result = {
            "all_required_checkpoints": required.issubset(names),
            "monotonic_order": monotonic_values == sorted(monotonic_values),
            "playback_received_audio": fake_rtc_source.frames > 0,
            "metrics_exposed": all(name.value in metrics for name in required),
            "nonnegative_deltas": all(value >= 0 for value in metrics.values()),
        }
        print("Latency Verification Results:", result)
        if not all(result.values()):
            raise AssertionError(result)
    finally:
        transport.output_sink.unsubscribe(transport._on_output_chunk)
        await transport._output_queue.put(None)
        if transport._output_task is not None:
            await transport._output_task


if __name__ == "__main__":
    asyncio.run(main())
