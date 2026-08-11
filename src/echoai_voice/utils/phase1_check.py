"""Small verification entrypoint for Phase 1."""

from __future__ import annotations

from echoai_voice.core.application import build_application, persona_prompt_snapshot


def main() -> None:
    app = build_application()
    first = persona_prompt_snapshot(app)
    print(first)
    personas = app.persona_manager.list_personas()
    if len(personas) > 1:
        switched = app.persona_manager.select_persona(personas[1].id)
        print({"switched_to": switched.id, "tts": app.persona_manager.resolve_tts_config()})


if __name__ == "__main__":
    main()
