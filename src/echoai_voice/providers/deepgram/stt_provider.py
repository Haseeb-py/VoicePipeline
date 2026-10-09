"""Deepgram streaming STT adapter for the voice pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from contextlib import suppress
import logging
from typing import Any
from collections.abc import Awaitable, Callable

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.listen.v1.types.listen_v1results import ListenV1Results
    from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted
    from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd
except ModuleNotFoundError:
    AsyncDeepgramClient = None

    class ListenV1Results:
        pass

    class ListenV1SpeechStarted:
        pass

    class ListenV1UtteranceEnd:
        pass

from echoai_voice.config.settings import AppSettings
from echoai_voice.models.fallback import WhisperFallbackResult
from echoai_voice.models.transcript import TranscriptEvent
from echoai_voice.providers.livekit.audio_source import AudioFrameChunk
from echoai_voice.services.transcript_event_bus import TranscriptEventBus
from echoai_voice.services.whisper_fallback import WhisperFallbackService
from echoai_voice.services.latency_tracker import LatencyTracker
from echoai_voice.models.latency import LatencyCheckpointName

logger = logging.getLogger(__name__)

_BARGE_IN_COMMANDS = ("stop", "wait", "hold on", "excuse me", "cancel")


@dataclass(frozen=True, slots=True)
class DeepgramSttConfig:
    """Streaming STT settings for Deepgram."""

    model: str = "nova-3"
    language: str = "en-US"
    sample_rate_hz: int = 16000
    channels: int = 1
    # 250 ms cuts ordinary phrases at brief natural pauses. 500 ms keeps a
    # question together while delivering the final transcript promptly.
    endpointing_ms: int = 500
    utterance_end_ms: str = "1000"


@dataclass(slots=True)
class DeepgramStreamingSttProvider:
    """Persistent async Deepgram STT session with interim/final transcript support."""

    settings: AppSettings
    transcript_bus: TranscriptEventBus
    fallback_service: WhisperFallbackService | None = None
    latency_tracker: LatencyTracker | None = None
    config: DeepgramSttConfig = field(default_factory=DeepgramSttConfig)
    _audio_queue: asyncio.Queue[AudioFrameChunk | None] = field(init=False)
    _run_task: asyncio.Task[None] | None = field(default=None, init=False)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _startup_error: str | None = field(default=None, init=False)
    _turn_id: int = field(default=1, init=False)
    _turn_active: bool = field(default=False, init=False)
    _last_text: str = field(default="", init=False)
    _last_confidence: float = field(default=0.0, init=False)
    _call_id: str = field(default="pending-call", init=False)
    _speech_started_subscribers: list[Callable[[], Awaitable[None] | None]] = field(
        default_factory=list, init=False
    )
    _confirmed_speech_subscribers: list[Callable[[], Awaitable[None] | None]] = field(
        default_factory=list, init=False
    )
    _confirmed_speech_notified: bool = field(default=False, init=False)
    _pre_roll_audio: bytearray = field(default_factory=bytearray, init=False)
    _turn_audio: bytearray = field(default_factory=bytearray, init=False)

    def __post_init__(self) -> None:
        self._audio_queue = asyncio.Queue()
        self._speech_started_subscribers = []
        self._confirmed_speech_subscribers = []
        if self.fallback_service is not None:
            self.fallback_service.subscribe(self._handle_fallback_result)

    def subscribe_speech_started(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        """Register a callback invoked immediately when customer speech start is detected."""

        self._speech_started_subscribers.append(callback)

    def subscribe_confirmed_speech(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        """Register for transcript-confirmed speech suitable for barge-in."""

        self._confirmed_speech_subscribers.append(callback)

    @property
    def is_running(self) -> bool:
        """Return whether the background Deepgram session has started."""

        return self._run_task is not None and not self._run_task.done()

    def bind_call_id(self, call_id: str | None) -> None:
        """Bind the current LiveKit call identifier to transcript events."""

        if call_id:
            self._call_id = call_id

    async def start(self) -> None:
        """Start the persistent Deepgram websocket session once per call."""

        if self.is_running:
            return
        if not self.settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is missing from the environment")

        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._run_task = asyncio.create_task(self._run(), name="deepgram-stt-session")
        await self._ready_event.wait()
        if self._startup_error is not None:
            if self._run_task is not None:
                await self._run_task
                self._run_task = None
            raise RuntimeError(self._startup_error)

    async def stop(self) -> None:
        """Stop the persistent Deepgram websocket session."""

        self._stop_event.set()
        await self._audio_queue.put(None)
        if self._run_task is not None:
            try:
                await self._run_task
            finally:
                self._run_task = None

    async def push_audio(self, chunk: AudioFrameChunk) -> None:
        """Queue inbound microphone audio for streaming transcription."""

        self._capture_fallback_audio(chunk)
        if not self._stop_event.is_set():
            await self._audio_queue.put(chunk)

    async def handle_deepgram_message(self, message: Any) -> None:
        """Process a Deepgram websocket message and emit normalized transcript events."""

        message_type = self._message_type(message)
        if isinstance(message, ListenV1SpeechStarted) or message_type == "SpeechStarted":
            self._begin_turn()
            self._record_latency(LatencyCheckpointName.SPEECH_INPUT_DETECTED)
            await self._notify_speech_started()
            return
        if isinstance(message, ListenV1UtteranceEnd) or message_type == "UtteranceEnd":
            await self._emit_final_from_buffer(event_time=datetime.now(timezone.utc))
            return
        if isinstance(message, ListenV1Results) or message_type == "Results":
            await self._handle_results(message)

    async def _notify_speech_started(self) -> None:
        await self._notify_subscribers(self._speech_started_subscribers)

    @staticmethod
    async def _notify_subscribers(
        subscribers: list[Callable[[], Awaitable[None] | None]],
    ) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for subscriber in tuple(subscribers):
            res = subscriber()
            if asyncio.iscoroutine(res):
                tasks.append(asyncio.create_task(res))
        if tasks:
            await asyncio.gather(*tasks)

    async def _run(self) -> None:
        try:
            client = AsyncDeepgramClient(api_key=self.settings.deepgram_api_key)
            logger.info("Starting Deepgram STT session with model=%s", self.config.model)
            async with client.listen.v1.connect(
                model=self.config.model,
                language=self.config.language,
                channels=self.config.channels,
                encoding="linear16",
                sample_rate=self.config.sample_rate_hz,
                interim_results=True,
                punctuate=True,
                smart_format=True,
                endpointing=self.config.endpointing_ms,
                utterance_end_ms=self.config.utterance_end_ms,
                vad_events=True,
            ) as socket:
                self._ready_event.set()
                sender = asyncio.create_task(self._send_audio(socket), name="deepgram-stt-sender")
                receiver = asyncio.create_task(
                    self._receive_messages(socket),
                    name="deepgram-stt-receiver",
                )
                done, pending = await asyncio.wait(
                    {sender, receiver},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for task in done:
                    task.result()
                for task in pending:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        except Exception as exc:
            # SDK errors may include credential-bearing headers in repr/tracebacks.
            logger.error("Deepgram STT session failed: %s", type(exc).__name__)
            self._startup_error = "Deepgram STT session failed"
        finally:
            self._ready_event.set()

    async def _send_audio(self, socket: Any) -> None:
        while not self._stop_event.is_set():
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            await socket.send_media(chunk.pcm16)
        with suppress(Exception):
            await socket.send_close_stream()

    async def _receive_messages(self, socket: Any) -> None:
        async for message in socket:
            await self.handle_deepgram_message(message)

    async def _handle_results(self, result: ListenV1Results) -> None:
        alternative = self._first_alternative(result)
        if alternative is None:
            return

        transcript = (alternative.transcript or "").strip()
        confidence = float(getattr(alternative, "confidence", 0.0) or 0.0)
        event_time = datetime.now(timezone.utc)
        timestamp_start = event_time - timedelta(seconds=max(float(result.duration or 0.0), 0.0))
        is_final = bool(result.is_final or result.speech_final)
        event: TranscriptEvent | None = None

        if transcript:
            self._begin_turn()
            await self._confirm_reliable_speech(transcript, confidence, is_final)
            self._last_text = transcript
            self._last_confidence = confidence
            event = TranscriptEvent(
                    call_id=self._call_id,
                    turn_id=self._turn_id,
                    speaker="customer",
                    text=transcript,
                    is_final=is_final,
                    confidence=confidence,
                    timestamp_start=timestamp_start,
                    timestamp_end=event_time,
                )
            low_confidence_final = (
                is_final
                and confidence < self.settings.low_confidence_threshold
                and self.fallback_service is not None
            )
            self._record_latency(
                LatencyCheckpointName.STT_FINAL_RECEIVED
                if is_final
                else LatencyCheckpointName.STT_INTERIM_RECEIVED
            )
            # Show interim hypotheses immediately, but never send a doubtful
            # final transcript to the LLM. Whisper will publish a verified final
            # event for that turn without blocking the live audio receive loop.
            if not low_confidence_final:
                await self.transcript_bus.publish(event)

        if is_final and event is not None:
            self._schedule_fallback_if_needed(event)
        if is_final:
            self._complete_turn()

    async def _emit_final_from_buffer(self, event_time: datetime) -> None:
        if not self._turn_active or not self._last_text:
            return

        event = TranscriptEvent(
                call_id=self._call_id,
                turn_id=self._turn_id,
                speaker="customer",
                text=self._last_text,
                is_final=True,
                confidence=self._last_confidence,
                timestamp_start=event_time,
                timestamp_end=event_time,
            )
        if (
            self._last_confidence < self.settings.low_confidence_threshold
            and self.fallback_service is not None
        ):
            self._schedule_fallback_if_needed(event)
        else:
            await self.transcript_bus.publish(event)
        self._complete_turn()

    def _complete_turn(self) -> None:
        self._turn_active = False
        self._last_text = ""
        self._last_confidence = 0.0
        self._turn_audio.clear()
        self._pre_roll_audio.clear()
        self._confirmed_speech_notified = False
        self._turn_id += 1

    async def _confirm_reliable_speech(
        self, transcript: str, confidence: float, is_final: bool
    ) -> None:
        if self._confirmed_speech_notified:
            return
        # Treat explicit interruption commands as reliable immediately. This
        # arrives on an interim result and stops playback before the customer
        # has to finish saying their whole replacement question.
        normalized = transcript.casefold().lstrip(" .,!?-—")
        if any(normalized.startswith(command) for command in _BARGE_IN_COMMANDS):
            self._confirmed_speech_notified = True
            await self._notify_subscribers(self._confirmed_speech_subscribers)
            return
        # Background sounds can produce plausible interim words. Reserve normal
        # barge-in for a final transcript; explicit commands above remain fast.
        if not is_final:
            return
        letter_count = sum(character.isalpha() for character in transcript)
        if letter_count < 3 or confidence < self.settings.low_confidence_threshold:
            return
        self._confirmed_speech_notified = True
        await self._notify_subscribers(self._confirmed_speech_subscribers)

    async def _handle_fallback_result(self, result: WhisperFallbackResult) -> None:
        """Publish Whisper's corrected text as the sole final event for a noisy turn."""

        text = result.fallback_text.strip()
        if not text:
            logger.warning("Whisper returned an empty fallback transcript for turn=%s", result.turn_id)
            return
        now = datetime.now(timezone.utc)
        await self.transcript_bus.publish(
            TranscriptEvent(
                call_id=result.call_id,
                turn_id=result.turn_id,
                speaker="customer",
                text=text,
                is_final=True,
                confidence=max(
                    result.primary_confidence,
                    self.settings.low_confidence_threshold,
                ),
                timestamp_start=now,
                timestamp_end=now,
            )
        )

    def _begin_turn(self) -> None:
        if not self._turn_active:
            self._turn_active = True
            self._turn_audio = bytearray(self._pre_roll_audio)

    def _capture_fallback_audio(self, chunk: AudioFrameChunk) -> None:
        if chunk.sample_rate != self.config.sample_rate_hz or chunk.channels != self.config.channels:
            return
        if self._turn_active:
            self._turn_audio.extend(chunk.pcm16)
            return
        self._pre_roll_audio.extend(chunk.pcm16)
        max_bytes = self.config.sample_rate_hz * self.config.channels * 2
        if len(self._pre_roll_audio) > max_bytes:
            del self._pre_roll_audio[:-max_bytes]

    def _schedule_fallback_if_needed(self, event: TranscriptEvent) -> None:
        if (
            self.fallback_service is None
            or event.confidence >= self.settings.low_confidence_threshold
            or not self._turn_audio
        ):
            return
        self.fallback_service.schedule(
            event,
            bytes(self._turn_audio),
            self.config.sample_rate_hz,
            self.config.channels,
        )

    def _record_latency(self, name: LatencyCheckpointName) -> None:
        if self.latency_tracker is not None:
            self.latency_tracker.record(name, self._call_id, self._turn_id)

    @staticmethod
    def _message_type(message: Any) -> str:
        return str(getattr(message, "type", message.__class__.__name__))

    @staticmethod
    def _first_alternative(result: ListenV1Results) -> Any | None:
        alternatives = getattr(result.channel, "alternatives", None) or []
        if not alternatives:
            return None
        return alternatives[0]
