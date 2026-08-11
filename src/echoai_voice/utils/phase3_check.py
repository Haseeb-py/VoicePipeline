"""Phase 3 verification helper for transcript event mapping."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from echoai_voice.core.stt_app import build_stt_application
from echoai_voice.models.transcript import TranscriptEvent


async def main() -> None:
    app = build_stt_application()
    events: list[TranscriptEvent] = []
    app.transcript_bus.subscribe(lambda event: events.append(event))

    await app.stt_provider.handle_deepgram_message(
        SimpleNamespace(
            type="Results",
            is_final=False,
            speech_final=False,
            duration=1.2,
            channel=SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="hello there", confidence=0.87)]
            ),
        )
    )
    await app.stt_provider.handle_deepgram_message(
        SimpleNamespace(
            type="Results",
            is_final=True,
            speech_final=True,
            duration=1.8,
            channel=SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="hello there how are you", confidence=0.91)]
            ),
        )
    )

    for event in events:
        print(event.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
