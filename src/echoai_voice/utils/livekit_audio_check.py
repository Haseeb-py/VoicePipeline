"""Real LiveKit RTC ingress/egress verification using two room participants."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from livekit import api, rtc

from echoai_voice.config.settings import get_settings
from echoai_voice.models.tts import TtsAudioChunk
from echoai_voice.providers.livekit.audio_sink import AudioSink
from echoai_voice.providers.livekit.audio_source import AudioFrameChunk, AudioSource
from echoai_voice.providers.livekit.audio_transport import LiveKitAudioTransport
from echoai_voice.providers.livekit.session_manager import LiveKitSessionManager
from echoai_voice.models.latency import LatencyCheckpointName
from echoai_voice.services.latency_tracker import LatencyTracker


def _token(identity: str) -> str:
    settings = get_settings()
    grants = api.VideoGrants(room_join=True, room="echoai-test-room")
    return (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .to_jwt()
    )


async def main() -> None:
    settings = get_settings()
    server_session = LiveKitSessionManager(settings)
    pipeline_input = AudioSource()
    pipeline_output = AudioSink()
    latency_tracker = LatencyTracker()
    transport = LiveKitAudioTransport(
        server_session,
        pipeline_input,
        pipeline_output,
        latency_tracker=latency_tracker,
    )
    client_room = rtc.Room()
    client_source: rtc.AudioSource | None = None
    inbound_received = asyncio.Event()
    outbound_received = asyncio.Event()
    playback_completed = asyncio.Event()
    client_streams: list[rtc.AudioStream] = []
    client_reader_tasks: list[asyncio.Task[None]] = []

    def on_pipeline_input(chunk: AudioFrameChunk) -> None:
        if chunk.pcm16 and chunk.sample_rate == 16000 and chunk.channels == 1:
            inbound_received.set()

    async def read_server_audio(stream: rtc.AudioStream) -> None:
        async for event in stream:
            if bytes(event.frame.data):
                outbound_received.set()
                return

    def on_client_track(track, publication, participant) -> None:
        if not isinstance(track, rtc.RemoteAudioTrack):
            return
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=20)
        client_streams.append(stream)
        client_reader_tasks.append(asyncio.create_task(read_server_audio(stream)))

    pipeline_input.subscribe(on_pipeline_input)
    latency_tracker.subscribe(
        lambda checkpoint: playback_completed.set()
        if checkpoint.name == LatencyCheckpointName.PLAYBACK_COMPLETED
        else None
    )
    client_room.on("track_subscribed", on_client_track)

    try:
        await server_session.connect()
        await transport.start()
        await client_room.connect(settings.livekit_url, _token("echoai-audio-check-client"))

        client_source = rtc.AudioSource(16000, 1, queue_size_ms=200)
        client_track = rtc.LocalAudioTrack.create_audio_track("test-microphone", client_source)
        await client_room.local_participant.publish_track(
            client_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        pcm = b"\x01\x00" * 1600
        await client_source.capture_frame(rtc.AudioFrame(pcm, 16000, 1, 1600))
        await asyncio.wait_for(inbound_received.wait(), timeout=10)

        pipeline_output.emit(
            TtsAudioChunk(
                pcm16=pcm,
                sample_rate=16000,
                channels=1,
                call_id="livekit-check",
                turn_id=1,
            )
        )
        pipeline_output.emit(
            TtsAudioChunk(
                pcm16=b"",
                sample_rate=16000,
                channels=1,
                is_final=True,
                call_id="livekit-check",
                turn_id=1,
            )
        )
        await asyncio.wait_for(outbound_received.wait(), timeout=10)
        await asyncio.wait_for(playback_completed.wait(), timeout=10)

        playback_names = {
            checkpoint.name
            for checkpoint in latency_tracker.snapshot("livekit-check", 1)
        }

        result = {
            "real_livekit_inbound_audio": inbound_received.is_set(),
            "real_livekit_outbound_audio": outbound_received.is_set(),
            "outbound_track_published": transport._output_publication is not None,
            "real_playback_latency_checkpoints": {
                LatencyCheckpointName.PLAYBACK_STARTED,
                LatencyCheckpointName.PLAYBACK_COMPLETED,
            }.issubset(playback_names),
        }
        print("LiveKit Audio Verification Results:", result)
        if not all(result.values()):
            raise AssertionError(result)
    finally:
        client_room.off("track_subscribed", on_client_track)
        for task in client_reader_tasks:
            task.cancel()
        for task in client_reader_tasks:
            with suppress(asyncio.CancelledError):
                await task
        for stream in client_streams:
            with suppress(Exception):
                await stream.aclose()
        if client_source is not None:
            with suppress(Exception):
                await client_source.aclose()
        with suppress(Exception):
            await client_room.disconnect()
        with suppress(Exception):
            await transport.stop()
        with suppress(Exception):
            await server_session.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
