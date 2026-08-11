"""Small verification entrypoint for the LiveKit transport slice."""

from __future__ import annotations

import asyncio

from echoai_voice.core.livekit_app import build_livekit_application, route_microphone_audio


async def main() -> None:
    app = build_livekit_application()
    print(app.status.connection, app.status.listening, app.status.speaking)
    captured: list[bytes] = []

    def capture(chunk) -> None:
        captured.append(chunk.pcm16)

    app.audio_source.subscribe(capture)
    route_microphone_audio(app, b"fake-audio", 16000)
    print(len(captured))


if __name__ == "__main__":
    asyncio.run(main())
