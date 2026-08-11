"""Persona management for standalone development.

The manager loads personas from disk today and can later be swapped to a
Module 2-backed provider without changing the rest of the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from echoai_voice.models.persona import PersonaConfig


@dataclass(frozen=True, slots=True)
class PersonaUpdate:
    """Snapshot emitted when the active persona changes."""

    persona: PersonaConfig


class PersonaManager:
    """Load and serve the active persona configuration."""

    def __init__(self, persona_dir: Path) -> None:
        self._persona_dir = persona_dir
        self._personas: dict[str, PersonaConfig] = {}
        self._active_persona_id: str | None = None
        self._subscribers: list[Callable[[PersonaUpdate], None]] = []

    def load_personas(self) -> None:
        """Load all persona JSON files from the configured directory."""

        self._personas.clear()
        if not self._persona_dir.exists():
            raise FileNotFoundError(f"Persona directory does not exist: {self._persona_dir}")

        for file_path in sorted(self._persona_dir.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            persona = PersonaConfig.from_dict(payload)
            self._personas[persona.id] = persona

        if not self._personas:
            raise ValueError(f"No persona files found in {self._persona_dir}")

        if self._active_persona_id not in self._personas:
            self._active_persona_id = next(iter(self._personas))
            self._notify_current_persona()

    def list_personas(self) -> tuple[PersonaConfig, ...]:
        """Return all loaded personas."""

        return tuple(self._personas.values())

    def get_active_persona(self) -> PersonaConfig:
        """Return the current active persona."""

        if self._active_persona_id is None:
            raise RuntimeError("PersonaManager has not loaded any personas")
        return self._personas[self._active_persona_id]

    def select_persona(self, persona_id: str) -> PersonaConfig:
        """Switch the active persona and notify subscribers immediately."""

        if persona_id not in self._personas:
            raise KeyError(f"Unknown persona: {persona_id}")

        self._active_persona_id = persona_id
        self._notify_current_persona()
        return self._personas[persona_id]

    def subscribe(self, callback: Callable[[PersonaUpdate], None]) -> None:
        """Register a synchronous callback for persona change events."""

        self._subscribers.append(callback)

    def resolve_tts_config(self) -> dict[str, object]:
        """Expose the active TTS settings without leaking file access."""

        persona = self.get_active_persona()
        return {
            "provider": persona.tts.provider,
            "voice": persona.tts.voice,
            "speech_rate": persona.tts.speech_rate,
        }

    def _notify_current_persona(self) -> None:
        persona = self.get_active_persona()
        update = PersonaUpdate(persona=persona)
        for callback in tuple(self._subscribers):
            callback(update)
