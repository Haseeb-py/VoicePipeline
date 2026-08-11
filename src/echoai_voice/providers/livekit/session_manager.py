"""Persistent LiveKit room/session management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from echoai_voice.config.settings import AppSettings
from echoai_voice.models.session import ConnectionState, ListeningState, SessionStatus, SpeakingState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveKitSessionConfig:
    """Connection parameters for a single persistent LiveKit session."""

    url: str
    api_key: str
    api_secret: str
    room_name: str
    identity: str


class LiveKitSessionManager:
    """Own one LiveKit room connection for the lifetime of a call."""

    def __init__(self, settings: AppSettings, room_name: str = "echoai-test-room") -> None:
        self._settings = settings
        self._config = LiveKitSessionConfig(
            url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            room_name=room_name,
            identity="echoai-server",
        )
        self._room: object | None = None
        self._status = SessionStatus(room_name=room_name)
        self._lock = asyncio.Lock()

    @property
    def status(self) -> SessionStatus:
        """Return the current session status snapshot."""

        return self._status

    @property
    def room(self) -> object | None:
        """Return the connected SDK room for transport adapters."""

        return self._room

    async def connect(self) -> SessionStatus:
        """Connect once and keep the room alive until the call ends."""

        async with self._lock:
            if self._room is not None:
                return self._status

            self._status.connection = ConnectionState.CONNECTING
            logger.info("Connecting to LiveKit room %s", self._config.room_name)
            try:
                self._room = await self._connect_room()
            except Exception:
                self._status.connection = ConnectionState.FAILED
                raise

            self._status.connection = ConnectionState.CONNECTED
            self._status.call_id = self._config.identity
            return self._status

    async def disconnect(self) -> SessionStatus:
        """Close the persistent session cleanly."""

        async with self._lock:
            if self._room is None:
                self._status.connection = ConnectionState.DISCONNECTED
                return self._status

            room = self._room
            self._room = None
            await self._disconnect_room(room)
            self._status.connection = ConnectionState.DISCONNECTED
            return self._status

    def mark_listening(self) -> None:
        """Update the local status to listening mode."""

        self._status.listening = ListeningState.LISTENING

    def mark_processing(self) -> None:
        """Update the local status to transcript processing mode."""

        self._status.listening = ListeningState.PROCESSING

    def mark_speaking(self) -> None:
        """Update the local status to speaking mode."""

        self._status.speaking = SpeakingState.SPEAKING

    def mark_interrupting(self) -> None:
        """Update the local status while active speech is being interrupted."""

        self._status.speaking = SpeakingState.INTERRUPTING

    def mark_silent(self) -> None:
        """Update the local status to silent mode."""

        self._status.speaking = SpeakingState.SILENT

    async def _connect_room(self) -> object:
        """Connect to LiveKit using the installed SDK when available."""

        try:
            from livekit import api
            from livekit import rtc
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("livekit package is not installed in the active environment") from exc

        room = rtc.Room()
        token = self._build_token(api=api)
        await room.connect(self._config.url, token)
        return room

    async def _disconnect_room(self, room: object) -> None:
        """Disconnect from LiveKit using the installed SDK when available."""

        disconnect = getattr(room, "disconnect", None)
        if disconnect is not None:
            result = disconnect()
            if asyncio.iscoroutine(result):
                await result

    def _build_token(self, api: object) -> str:
        """Build a LiveKit access token for the persistent room session."""

        if not self._config.api_key or not self._config.api_secret:
            logger.warning("LiveKit API credentials are missing; connection will fail until configured")

        grants = api.VideoGrants(room_join=True, room=self._config.room_name, agent=True)
        token = (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(self._config.identity)
            .with_name(self._config.identity)
            .with_grants(grants)
        )
        return token.to_jwt()
