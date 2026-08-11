"""Credential-safe live smoke check for the full Phase 6 composition root."""

from __future__ import annotations

import asyncio

from echoai_voice.core.vad_app import build_vad_application, start_vad_call, stop_vad_call


async def _run() -> None:
    app = build_vad_application()
    started = False
    try:
        await asyncio.wait_for(start_vad_call(app), timeout=30)
        started = True
        assert app.audio_transport.is_running
        assert app.base.tts_provider.is_connected
        assert app.base.base.base.stt_provider.is_running
        print("full_stack_connection: PASS")
    finally:
        if started:
            await asyncio.wait_for(stop_vad_call(app), timeout=20)


def main() -> None:
    """Run the live integration check while suppressing SDK exception details."""

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise SystemExit(
            f"full_stack_connection: FAIL ({type(exc).__name__})"
        ) from None


if __name__ == "__main__":
    main()
