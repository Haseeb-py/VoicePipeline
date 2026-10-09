"""Real LiveKit RTC audio ingress and egress transport."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import logging
from typing import Any

from echoai_voice.models.tts import TtsAudioChunk
from echoai_voice.providers.livekit.audio_sink import AudioSink
from echoai_voice.providers.livekit.audio_source import AudioFrameChunk, AudioSource
from echoai_voice.providers.livekit.session_manager import LiveKitSessionManager
from echoai_voice.models.latency import LatencyCheckpointName
from echoai_voice.services.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveKitAudioTransport:
    """Bridge real LiveKit microphone and playback tracks to pipeline boundaries."""

    session_manager: LiveKitSessionManager
    input_source: AudioSource
    output_sink: AudioSink
    latency_tracker: LatencyTracker | None = None
    input_participant_identity: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    _room: Any | None = field(default=None, init=False)
    _rtc_output_source: Any | None = field(default=None, init=False)
    _output_publication: Any | None = field(default=None, init=False)
    _output_queue: asyncio.Queue[tuple[int, TtsAudioChunk] | None] = field(init=False)
    _output_generation: int = field(default=0, init=False)
    _output_task: asyncio.Task[None] | None = field(default=None, init=False)
    _input_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict, init=False)
    _input_streams: dict[str, Any] = field(default_factory=dict, init=False)
    _playback_started_turns: set[tuple[str, int]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._output_queue = asyncio.Queue()

    @property
    def is_running(self) -> bool:
        return self._room is not None

    async def start(self) -> None:
        """Publish the AI audio track and subscribe to remote microphone tracks."""

        if self.is_running:
            return

        room = self.session_manager.room
        if room is None:
            raise RuntimeError("LiveKit must be connected before starting audio transport")

        self._room = room
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.output_sink.subscribe(self._on_output_chunk)
        from livekit import rtc

        self._rtc_output_source = rtc.AudioSource(
            self.sample_rate,
            self.channels,
            queue_size_ms=200,
        )
        output_track = rtc.LocalAudioTrack.create_audio_track(
            "echoai-response",
            self._rtc_output_source,
        )
        publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._output_publication = await room.local_participant.publish_track(
            output_track,
            publish_options,
        )
        self._output_task = asyncio.create_task(
            self._play_output(),
            name="livekit-audio-output",
        )

        for participant in tuple(room.remote_participants.values()):
            for publication in tuple(participant.track_publications.values()):
                if publication.track is not None:
                    self._on_track_subscribed(publication.track, publication, participant)

        logger.info("LiveKit audio ingress/egress transport started")

    async def stop(self) -> None:
        """Stop track readers, playback, and release the published audio source."""

        room = self._room
        if room is None:
            return

        room.off("track_subscribed", self._on_track_subscribed)
        room.off("track_unsubscribed", self._on_track_unsubscribed)
        self.output_sink.unsubscribe(self._on_output_chunk)
        self._clear_output()

        for task in tuple(self._input_tasks.values()):
            task.cancel()
        for task in tuple(self._input_tasks.values()):
            with suppress(asyncio.CancelledError):
                await task
        self._input_tasks.clear()

        for stream in tuple(self._input_streams.values()):
            with suppress(Exception):
                await stream.aclose()
        self._input_streams.clear()

        if self._output_task is not None:
            await self._output_queue.put(None)
            with suppress(asyncio.CancelledError):
                await self._output_task
            self._output_task = None

        publication = self._output_publication
        self._output_publication = None
        if publication is not None:
            with suppress(Exception):
                await room.local_participant.unpublish_track(publication.sid)

        if self._rtc_output_source is not None:
            with suppress(Exception):
                await self._rtc_output_source.aclose()
            self._rtc_output_source = None

        self._room = None
        logger.info("LiveKit audio ingress/egress transport stopped")

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        from livekit import rtc

        participant_identity = str(getattr(participant, "identity", ""))
        if (
            self.input_participant_identity is not None
            and participant_identity != self.input_participant_identity
        ):
            logger.info(
                "Ignoring microphone track from non-active UI participant=%s",
                participant_identity,
            )
            return
        if not isinstance(track, rtc.RemoteAudioTrack):
            return
        if publication.source not in {
            rtc.TrackSource.SOURCE_MICROPHONE,
            rtc.TrackSource.SOURCE_UNKNOWN,
        }:
            return
        if track.sid in self._input_tasks:
            return

        stream = rtc.AudioStream(
            track=track,
            sample_rate=self.sample_rate,
            num_channels=self.channels,
            frame_size_ms=20,
        )
        self._input_streams[track.sid] = stream
        task = asyncio.create_task(
            self._consume_input(track.sid, stream),
            name=f"livekit-audio-input-{participant.identity}",
        )
        self._input_tasks[track.sid] = task
        task.add_done_callback(lambda completed, sid=track.sid: self._input_done(sid, completed))
        logger.info("Subscribed to LiveKit microphone track participant=%s", participant.identity)

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        sid = getattr(track, "sid", None) or getattr(publication, "sid", None)
        if sid is None:
            return
        task = self._input_tasks.pop(sid, None)
        if task is not None:
            task.cancel()
        stream = self._input_streams.pop(sid, None)
        if stream is not None:
            asyncio.create_task(stream.aclose())

    async def _consume_input(self, track_sid: str, stream: Any) -> None:
        async for event in stream:
            frame = event.frame
            self.input_source.emit(
                AudioFrameChunk(
                    pcm16=bytes(frame.data),
                    sample_rate=frame.sample_rate,
                    channels=frame.num_channels,
                )
            )
        logger.debug("LiveKit input stream ended track_sid=%s", track_sid)

    def _input_done(self, track_sid: str, task: asyncio.Task[None]) -> None:
        if self._input_tasks.get(track_sid) is task:
            self._input_tasks.pop(track_sid, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "LiveKit microphone stream failed track_sid=%s",
                track_sid,
                exc_info=(type(error), error, error.__traceback__),
            )
        stream = self._input_streams.pop(track_sid, None)
        if stream is not None:
            asyncio.create_task(stream.aclose())

    def _on_output_chunk(self, chunk: TtsAudioChunk) -> None:
        if chunk.is_interruption:
            self._clear_output()
            return
        if not chunk.pcm16 and not chunk.is_final:
            return
        self._output_queue.put_nowait((self._output_generation, chunk))

    def _clear_output(self) -> None:
        self._output_generation += 1
        while True:
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._rtc_output_source is not None:
            self._rtc_output_source.clear_queue()

    async def _play_output(self) -> None:
        from livekit import rtc

        while True:
            queued = await self._output_queue.get()
            if queued is None:
                return
            generation, chunk = queued
            if generation != self._output_generation:
                continue
            if chunk.is_final and not chunk.pcm16:
                if self._rtc_output_source is not None:
                    await self._rtc_output_source.wait_for_playout()
                self._record_latency(LatencyCheckpointName.PLAYBACK_COMPLETED, chunk)
                continue
            bytes_per_sample = 2 * chunk.channels
            if len(chunk.pcm16) % bytes_per_sample:
                logger.warning("Dropping malformed PCM chunk with %d bytes", len(chunk.pcm16))
                continue
            frame = rtc.AudioFrame(
                data=chunk.pcm16,
                sample_rate=chunk.sample_rate,
                num_channels=chunk.channels,
                samples_per_channel=len(chunk.pcm16) // bytes_per_sample,
            )
            if generation == self._output_generation and self._rtc_output_source is not None:
                turn_key = (chunk.call_id or "pending-call", chunk.turn_id or 0)
                if turn_key not in self._playback_started_turns:
                    self._playback_started_turns.add(turn_key)
                    self._record_latency(LatencyCheckpointName.PLAYBACK_STARTED, chunk)
                await self._rtc_output_source.capture_frame(frame)
                if generation != self._output_generation:
                    # An interruption may race an in-flight SDK capture. Clear a
                    # second time so the just-completed stale frame cannot play.
                    self._rtc_output_source.clear_queue()

    def _record_latency(self, name: LatencyCheckpointName, chunk: TtsAudioChunk) -> None:
        if self.latency_tracker is not None:
            self.latency_tracker.record(
                name,
                chunk.call_id or "pending-call",
                chunk.turn_id or 0,
            )
