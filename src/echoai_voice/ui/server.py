"""Standalone Phase 7 development UI server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from echoai_voice.ui.controller import VoiceUiController

STATIC_DIR = Path(__file__).with_name("static")
controller = VoiceUiController()


class PersonaSelection(BaseModel):
    persona_id: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if controller.is_running:
        await controller.end_call()


app = FastAPI(
    title="EchoAI Module 3A Development UI",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def state() -> dict[str, object]:
    return controller.snapshot()


@app.get("/api/personas")
async def personas() -> list[dict[str, object]]:
    return controller.list_personas()


@app.put("/api/persona")
async def select_persona(selection: PersonaSelection) -> dict[str, object]:
    try:
        return controller.select_persona(selection.persona_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown persona") from None


@app.post("/api/calls/start")
async def start_call() -> dict[str, object]:
    try:
        return await controller.start_call()
    except Exception:
        raise HTTPException(
            status_code=503, detail="Unable to start the real voice pipeline"
        ) from None


@app.post("/api/calls/credentials")
async def call_credentials() -> dict[str, str]:
    """Issue browser credentials before the slower server pipeline startup."""

    try:
        return controller.browser_credentials()
    except Exception:
        raise HTTPException(
            status_code=503, detail="Unable to prepare the LiveKit browser session"
        ) from None


@app.post("/api/calls/end")
async def end_call() -> dict[str, object]:
    try:
        return await controller.end_call()
    except Exception:
        raise HTTPException(
            status_code=500, detail="Unable to stop the voice pipeline cleanly"
        ) from None


@app.get("/api/events")
async def events() -> StreamingResponse:
    queue = controller.subscribe_events()

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                payload = json.dumps(event["data"], separators=(",", ":"))
                yield f"event: {event['type']}\ndata: {payload}\n\n"
        finally:
            controller.unsubscribe_events(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    """Run the standalone local development server."""

    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("ECHOAI_UI_HOST", "127.0.0.1"),
        port=int(os.getenv("ECHOAI_UI_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    main()
