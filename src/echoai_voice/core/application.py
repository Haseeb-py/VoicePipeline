"""Application wiring for the Phase 1 foundation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from echoai_voice.config.settings import AppSettings, get_settings
from echoai_voice.providers.groq.client import GroqClient
from echoai_voice.providers.groq.response_generator import GroqResponseGenerator
from echoai_voice.services.persona_manager import PersonaManager
from echoai_voice.services.latency_tracker import LatencyTracker


@dataclass(slots=True)
class VoiceApplication:
    """Own the foundation-level dependencies for Module 3A."""

    settings: AppSettings
    persona_manager: PersonaManager
    response_generator: GroqResponseGenerator
    latency_tracker: LatencyTracker


def configure_logging(level: str) -> None:
    """Configure structured logging for local development."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_application() -> VoiceApplication:
    """Build the foundation wiring for persona-aware response generation."""

    settings = get_settings()
    configure_logging(settings.log_level)
    persona_manager = PersonaManager(settings.persona_dir)
    persona_manager.load_personas()
    client = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
    response_generator = GroqResponseGenerator(client=client)
    latency_tracker = LatencyTracker()
    return VoiceApplication(
        settings=settings,
        persona_manager=persona_manager,
        response_generator=response_generator,
        latency_tracker=latency_tracker,
    )


def persona_prompt_snapshot(app: VoiceApplication) -> dict[str, object]:
    """Return the active persona prompt and TTS settings for verification."""

    persona = app.persona_manager.get_active_persona()
    return {
        "persona_id": persona.id,
        "system_prompt": persona.system_prompt,
        "tts": app.persona_manager.resolve_tts_config(),
    }
