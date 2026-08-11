"""Barge-in controller for Voice Activity Detection (VAD) interruption handling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

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

    async def handle_speech_started(self) -> None:
        """Invoked when VAD detects customer speech start."""

        status = self.session_manager.status
        was_speaking = status.speaking == SpeakingState.SPEAKING
        had_response_work = self.response_service.has_active_response

        if was_speaking:
            logger.info("Customer speech detected mid-response (barge-in); cancelling TTS immediately.")
            self.session_manager.mark_interrupting()
        elif had_response_work:
            logger.info("Customer speech detected while response generation is pending; cancelling it.")
        else:
            logger.debug("Speech started event received while AI is silent.")

        await asyncio.gather(
            self.response_service.cancel_current_response(),
            self.tts_service.cancel_current_speech(emit_interruption=was_speaking),
        )
        self.session_manager.mark_listening()
