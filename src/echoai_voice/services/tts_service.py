"""TTS orchestration service connecting generated AI responses to speech synthesis."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from dataclasses import dataclass, field
from dataclasses import replace

from echoai_voice.models.response_events import GeneratedResponseChunkEvent, GeneratedResponseEvent
from echoai_voice.providers.deepgram.tts_provider import DeepgramStreamingTtsProvider
from echoai_voice.providers.livekit.audio_sink import AudioSink
from echoai_voice.providers.livekit.session_manager import LiveKitSessionManager
from echoai_voice.models.latency import LatencyCheckpointName
from echoai_voice.services.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TtsService:
    """Consume generated AI response events, synthesize speech, and route audio to sink."""

    tts_provider: DeepgramStreamingTtsProvider
    audio_sink: AudioSink
    session_manager: LiveKitSessionManager
    latency_tracker: LatencyTracker | None = None
    _active_synthesis_task: asyncio.Task[None] | None = field(default=None, init=False)
    _stream_generation: int = field(default=0, init=False)
    _active_text_queue: asyncio.Queue[str | None] | None = field(default=None, init=False)

    async def handle_response_event(self, event: GeneratedResponseEvent) -> None:
        """Process an AI response event and stream synthesized audio chunks."""

        if not event.text.strip():
            return

        # A new response supersedes any prior stream, but a normal first stream
        # must not emit a false interruption marker.
        await self.cancel_current_speech(emit_interruption=False)
        self._stream_generation += 1
        stream_generation = self._stream_generation

        logger.info(
            "Synthesizing response for call_id=%s turn_id=%s text_len=%d",
            event.call_id,
            event.turn_id,
            len(event.text),
        )

        self.session_manager.mark_speaking()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_synthesis_task = current_task

        try:
            async for chunk in self.tts_provider.stream_speech(event.text):
                if stream_generation != self._stream_generation:
                    break
                self.audio_sink.emit(
                    replace(chunk, call_id=event.call_id, turn_id=event.turn_id)
                )
        except asyncio.CancelledError:
            logger.info("TTS synthesis cancelled mid-stream (barge-in)")
            raise
        finally:
            if self._active_synthesis_task is current_task:
                self._active_synthesis_task = None
                self.session_manager.mark_silent()
            logger.debug("Finished streaming TTS for turn_id=%s", event.turn_id)

    async def handle_response_chunk(self, event: GeneratedResponseChunkEvent) -> None:
        """Feed incremental LLM text into one active persistent TTS stream."""

        if event.sequence == 0 and not event.is_final:
            await self.cancel_current_speech(emit_interruption=False)
            self._stream_generation += 1
            generation = self._stream_generation
            text_queue: asyncio.Queue[str | None] = asyncio.Queue()
            self._active_text_queue = text_queue
            task = asyncio.create_task(
                self._run_text_stream(event.call_id, event.turn_id, generation, text_queue),
                name=f"tts-stream-turn-{event.turn_id}",
            )
            self._active_synthesis_task = task

        text_queue = self._active_text_queue
        if text_queue is None:
            return
        if event.text:
            await text_queue.put(event.text)
        if event.is_final:
            await text_queue.put(None)
            task = self._active_synthesis_task
            if task is not None:
                await task

    async def _run_text_stream(
        self,
        call_id: str,
        turn_id: int,
        generation: int,
        text_queue: asyncio.Queue[str | None],
    ) -> None:
        """Stream queued response text through Deepgram and emit audio immediately."""

        current_task = asyncio.current_task()
        self.session_manager.mark_speaking()
        self._record_latency(LatencyCheckpointName.TTS_GENERATION_STARTED, call_id, turn_id)
        first_audio = True
        completed = False

        async def text_chunks():
            while True:
                text = await text_queue.get()
                if text is None:
                    return
                yield text

        try:
            async for chunk in self.tts_provider.stream_text(text_chunks()):
                if generation != self._stream_generation:
                    break
                if first_audio and chunk.pcm16:
                    first_audio = False
                    self._record_latency(
                        LatencyCheckpointName.FIRST_AUDIO_CHUNK_AVAILABLE,
                        call_id,
                        turn_id,
                    )
                self.audio_sink.emit(replace(chunk, call_id=call_id, turn_id=turn_id))
            completed = generation == self._stream_generation
        except asyncio.CancelledError:
            logger.info("Streaming TTS text/audio cancelled mid-response")
            raise
        finally:
            if self._active_synthesis_task is current_task:
                self._active_synthesis_task = None
                self._active_text_queue = None
                self.session_manager.mark_silent()
            if completed:
                self._record_latency(LatencyCheckpointName.TTS_GENERATION_COMPLETED, call_id, turn_id)
                self._record_latency(LatencyCheckpointName.RESPONSE_COMPLETED, call_id, turn_id)
            logger.debug("Finished incremental TTS stream for turn_id=%s", turn_id)

    def _record_latency(
        self,
        name: LatencyCheckpointName,
        call_id: str,
        turn_id: int,
    ) -> None:
        if self.latency_tracker is not None:
            self.latency_tracker.record(name, call_id, turn_id)

    async def cancel_current_speech(self, *, emit_interruption: bool = True) -> bool:
        """Cancel ongoing speech synthesis immediately on customer barge-in."""

        self._stream_generation += 1
        task = self._active_synthesis_task
        current_task = asyncio.current_task()
        cancelled_active_stream = (
            task is not None and not task.done() and task is not current_task
        )
        # Flush LiveKit synchronously before awaiting provider/network cleanup.
        # The generation token above rejects any late chunks from the old task.
        if cancelled_active_stream or emit_interruption:
            self.audio_sink.clear()
        if cancelled_active_stream:
            logger.info("Cancelling active TTS task for barge-in interruption.")
            self._active_synthesis_task = None
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        text_queue = self._active_text_queue
        self._active_text_queue = None
        if text_queue is not None:
            while not text_queue.empty():
                with suppress(asyncio.QueueEmpty):
                    text_queue.get_nowait()
            await text_queue.put(None)
        await self.tts_provider.cancel_current_synthesis()

        self.session_manager.mark_silent()
        return cancelled_active_stream
