"""Persistent Deepgram Aura WebSocket TTS adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
import logging
import re
from typing import Any

from echoai_voice.config.settings import AppSettings
from echoai_voice.models.tts import TtsAudioChunk
from echoai_voice.services.persona_manager import PersonaManager, PersonaUpdate

logger = logging.getLogger(__name__)

_FLUSHED = object()
_FAILED = object()


@dataclass(frozen=True, slots=True)
class _SenderDone:
    flush_count: int


@dataclass(slots=True)
class DeepgramStreamingTtsProvider:
    """Stream LLM text through one Deepgram Speak WebSocket per call/configuration."""

    settings: AppSettings
    persona_manager: PersonaManager
    _current_voice: str = field(default="aura-2-asteria-en", init=False)
    _current_speech_rate: float = field(default=1.0, init=False)
    _client: Any | None = field(default=None, init=False)
    _connection_context: Any | None = field(default=None, init=False)
    _socket: Any | None = field(default=None, init=False)
    _receiver_task: asyncio.Task[None] | None = field(default=None, init=False)
    _connection_config: tuple[str, float] | None = field(default=None, init=False)
    _synthesis_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _active_audio_queue: asyncio.Queue[object] | None = field(default=None, init=False)
    _clear_ack: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _connection_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.persona_manager.subscribe(self._on_persona_changed)
        self._update_tts_config()

    @property
    def current_voice(self) -> str:
        return self._current_voice

    @property
    def current_speech_rate(self) -> float:
        return self._current_speech_rate

    @property
    def is_connected(self) -> bool:
        if self._is_offline:
            return True
        return self._socket is not None and self._receiver_task is not None

    @property
    def connection_count(self) -> int:
        """Expose connection creation count for persistence verification."""

        return self._connection_count

    @property
    def connection_config(self) -> tuple[str, float] | None:
        return self._connection_config

    @property
    def _is_offline(self) -> bool:
        return self.settings.deepgram_api_key in {"mock-key", "test-key"}

    def _on_persona_changed(self, update: PersonaUpdate) -> None:
        old_config = (self._current_voice, self._current_speech_rate)
        self._update_tts_config()
        logger.info(
            "TTS persona configuration updated voice=%s rate=%.2f (was voice=%s rate=%.2f)",
            self._current_voice,
            self._current_speech_rate,
            old_config[0],
            old_config[1],
        )

    def _update_tts_config(self) -> None:
        tts_cfg = self.persona_manager.resolve_tts_config()
        self._current_voice = str(tts_cfg.get("voice", self.settings.default_tts_voice))
        self._current_speech_rate = float(tts_cfg.get("speech_rate", 1.0))

    async def start(self) -> None:
        """Open the call-scoped Speak WebSocket if live credentials are configured."""

        if self._is_offline:
            self._connection_config = (self._current_voice, self._current_speech_rate)
            return
        if not self.settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is missing from the environment")
        await self._ensure_connection()

    async def stop(self) -> None:
        """Close the persistent Speak WebSocket and receiver task."""

        socket = self._socket
        receiver = self._receiver_task
        context = self._connection_context
        self._socket = None
        self._receiver_task = None
        self._connection_context = None
        self._connection_config = None
        self._active_audio_queue = None

        if socket is not None:
            with suppress(Exception):
                await socket.send_close()
        if receiver is not None:
            try:
                await asyncio.wait_for(receiver, timeout=2)
            except (TimeoutError, Exception):
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
        if context is not None:
            with suppress(Exception):
                await context.__aexit__(None, None, None)

    async def stream_speech(self, text: str) -> AsyncIterator[TtsAudioChunk]:
        """Compatibility wrapper for a single complete text response."""

        async def one_chunk() -> AsyncIterator[str]:
            if text.strip():
                yield text

        async for audio in self.stream_text(one_chunk()):
            yield audio

    async def stream_text(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[TtsAudioChunk]:
        """Send incremental LLM text and yield audio as soon as Deepgram produces it."""

        if self._is_offline:
            async for audio in self._generate_simulated_stream(text_chunks):
                yield audio
            return

        async with self._synthesis_lock:
            await self._ensure_connection()
            if self._socket is None:
                raise RuntimeError("Deepgram TTS socket is not connected")

            queue: asyncio.Queue[object] = asyncio.Queue()
            self._active_audio_queue = queue
            sender = asyncio.create_task(self._send_text_stream(text_chunks, queue))
            sender_done = False
            expected_flushes = 0
            received_flushes = 0
            try:
                while not sender_done or received_flushes < expected_flushes:
                    item = await queue.get()
                    if isinstance(item, bytes):
                        yield TtsAudioChunk(item, sample_rate=16000, channels=1)
                    elif item is _FLUSHED:
                        received_flushes += 1
                    elif isinstance(item, _SenderDone):
                        sender_done = True
                        expected_flushes = item.flush_count
                    elif item is _FAILED:
                        raise RuntimeError("Deepgram TTS stream failed")
                yield TtsAudioChunk(b"", sample_rate=16000, channels=1, is_final=True)
            except (asyncio.CancelledError, GeneratorExit):
                sender.cancel()
                with suppress(asyncio.CancelledError):
                    await sender
                await self._clear_current_synthesis()
                raise
            finally:
                if not sender.done():
                    sender.cancel()
                    with suppress(asyncio.CancelledError):
                        await sender
                if self._active_audio_queue is queue:
                    self._active_audio_queue = None

    async def cancel_current_synthesis(self) -> None:
        """Clear Deepgram's server-side text/audio buffer for barge-in."""

        if not self._is_offline and self._active_audio_queue is not None:
            await self._clear_current_synthesis()

    async def _ensure_connection(self) -> None:
        desired = (self._current_voice, self._current_speech_rate)
        if self._socket is not None and self._connection_config == desired:
            return
        if self._socket is not None:
            await self.stop()

        from deepgram import AsyncDeepgramClient

        try:
            self._client = AsyncDeepgramClient(api_key=self.settings.deepgram_api_key)
            context = self._client.speak.v1.connect(
                model=self._current_voice,
                encoding="linear16",
                sample_rate=16000,
                speed=self._current_speech_rate,
            )
            socket = await context.__aenter__()
        except Exception as exc:
            logger.error("Deepgram TTS connection failed: %s", type(exc).__name__)
            raise RuntimeError("Deepgram TTS connection failed") from None

        self._connection_context = context
        self._socket = socket
        self._connection_config = desired
        self._connection_count += 1
        self._receiver_task = asyncio.create_task(self._receive_audio(), name="deepgram-tts-receiver")
        logger.info("Deepgram TTS WebSocket connected voice=%s rate=%.2f", *desired)

    async def _receive_audio(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            async for message in socket:
                queue = self._active_audio_queue
                if isinstance(message, bytes):
                    if queue is not None:
                        await queue.put(message)
                    continue
                message_type = str(getattr(message, "type", ""))
                if message_type == "Flushed" and queue is not None:
                    await queue.put(_FLUSHED)
                elif message_type == "Cleared":
                    self._clear_ack.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Deepgram TTS receiver failed: %s", type(exc).__name__)
            if self._active_audio_queue is not None:
                await self._active_audio_queue.put(_FAILED)

    async def _send_text_stream(
        self,
        text_chunks: AsyncIterator[str],
        queue: asyncio.Queue[object],
    ) -> None:
        from deepgram.speak.v1.types import SpeakV1Text

        socket = self._socket
        if socket is None:
            await queue.put(_FAILED)
            return
        buffer = ""
        flush_count = 0
        async for text in text_chunks:
            buffer += text
            if len(buffer) >= 180 or re.search(r"[.!?][\"']?\s$", buffer):
                await socket.send_text(SpeakV1Text(text=buffer))
                await socket.send_flush()
                flush_count += 1
                buffer = ""
        if buffer.strip():
            await socket.send_text(SpeakV1Text(text=buffer))
            await socket.send_flush()
            flush_count += 1
        await queue.put(_SenderDone(flush_count))

    async def _clear_current_synthesis(self) -> None:
        socket = self._socket
        if socket is None:
            return
        self._clear_ack.clear()
        with suppress(Exception):
            await socket.send_clear()
            await asyncio.wait_for(self._clear_ack.wait(), timeout=1)

    async def _generate_simulated_stream(
        self,
        text_chunks: AsyncIterator[str],
    ) -> AsyncIterator[TtsAudioChunk]:
        emitted = False
        async for text in text_chunks:
            for _word in text.split():
                emitted = True
                yield TtsAudioChunk(b"\x00\x00" * 1600, sample_rate=16000, channels=1)
                await asyncio.sleep(0.01)
        yield TtsAudioChunk(b"", sample_rate=16000, channels=1, is_final=True)
