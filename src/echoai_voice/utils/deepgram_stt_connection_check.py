"""Credential-safe live connection check for Deepgram streaming STT."""

from __future__ import annotations

import asyncio

from echoai_voice.core.stt_app import build_stt_application


async def _run() -> None:
    app = build_stt_application()
    await asyncio.wait_for(app.stt_provider.start(), timeout=20)
    assert app.stt_provider.is_running
    await asyncio.wait_for(app.stt_provider.stop(), timeout=10)
    print("deepgram_stt_connection: PASS")


def main() -> None:
    """Run the live handshake check without logging credential-bearing errors."""

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise SystemExit(
            f"deepgram_stt_connection: FAIL ({type(exc).__name__})"
        ) from None


if __name__ == "__main__":
    main()
