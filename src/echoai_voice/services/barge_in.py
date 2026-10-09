"""Barge-in controller for Voice Activity Detection (VAD) interruption handling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from echoai_voice.models.session import SpeakingState
from echoai_voice.providers.livekit.session_manager import LiveKitSessionManager
from echoai_voice.services.response_generation_service import ResponseGenerationService
from echoai_voice.services.tts_service import TtsService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BargeInController:
    """Orchestrate immediate TTS cancellation when customer speech is detected mid-AI response."""

    tts_service: TtsService
    response_service: ResponseGenerationService
    session_manager: LiveKitSessionManager
    _interruption_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def handle_speech_started(self) -> None:
        """Invoked when VAD detects customer speech start."""

        # Deepgram may emit a small burst of speech-start events around a turn.
        # Serialize cancellation so competing flushes cannot leave the session in
        # an interrupting state or tear down the next response's audio stream.
        async with self._interruption_lock:
            status = self.session_manager.status
            was_speaking = status.speaking == SpeakingState.SPEAKING
            had_response_work = self.response_service.has_active_response

            if was_speaking:
                logger.info("Customer speech detected mid-response (barge-in); cancelling TTS immediately.")
                self.session_manager.mark_interrupting()
            else:
                logger.info(
                    "Customer speech detected; cancelling response work and flushing queued audio."
                )

            await asyncio.gather(
                self.response_service.cancel_current_response(),
                # LiveKit may still be playing frames after the TTS websocket
                # has finished and the local speaking status became silent.
                # Always emit a flush on confirmed customer speech so queued
                # playback cannot continue over the customer's next question.
                self.tts_service.cancel_current_speech(emit_interruption=True),
            )
            self.session_manager.mark_listening()
