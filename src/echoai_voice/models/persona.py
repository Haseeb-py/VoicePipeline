"""Persona domain models used by the voice pipeline.

The persona contract is intentionally externalized so Module 2 can later
replace the local JSON-backed source without changing the rest of the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TtsConfig:
    """Text-to-speech settings resolved from the active persona."""

    provider: str
    voice: str
    speech_rate: float


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """LLM settings resolved from the active persona."""

    temperature: float


@dataclass(frozen=True, slots=True)
class PersonaConfig:
    """Immutable persona definition loaded from external configuration."""

    id: str
    display_name: str
    description: str
    system_prompt: str
    tts: TtsConfig
    llm: LlmConfig
    greeting_style: str = "standard"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonaConfig":
        """Create a validated persona config from a JSON-compatible mapping."""

        tts_payload = payload.get("tts", {})
        llm_payload = payload.get("llm", {})
        speech_rate = float(tts_payload.get("speech_rate", 1.0))
        if not 0.7 <= speech_rate <= 1.5:
            raise ValueError("Persona TTS speech_rate must be between 0.7 and 1.5")
        return cls(
            id=str(payload["id"]),
            display_name=str(payload["display_name"]),
            description=str(payload.get("description", "")),
            system_prompt=str(payload["system_prompt"]),
            tts=TtsConfig(
                provider=str(tts_payload["provider"]),
                voice=str(tts_payload["voice"]),
                speech_rate=speech_rate,
            ),
            llm=LlmConfig(
                temperature=float(llm_payload.get("temperature", 0.4)),
            ),
            greeting_style=str(payload.get("greeting_style", "standard")),
            metadata=payload.get("metadata"),
        )

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the persona."""

        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tts": {
                "provider": self.tts.provider,
                "voice": self.tts.voice,
                "speech_rate": self.tts.speech_rate,
            },
            "llm": {
                "temperature": self.llm.temperature,
            },
            "greeting_style": self.greeting_style,
            "metadata": self.metadata,
        }
