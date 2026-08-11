"""Observable monotonic latency checkpoint collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

from echoai_voice.models.latency import LatencyCheckpoint, LatencyCheckpointName

LatencySubscriber = Callable[[LatencyCheckpoint], None]


@dataclass(slots=True)
class LatencyTracker:
    """Collect and expose per-turn checkpoints for logging and the future UI."""

    _checkpoints: list[LatencyCheckpoint] = field(default_factory=list)
    _subscribers: list[LatencySubscriber] = field(default_factory=list)

    def subscribe(self, callback: LatencySubscriber) -> None:
        self._subscribers.append(callback)

    def record(
        self,
        name: LatencyCheckpointName,
        call_id: str,
        turn_id: int,
        **metadata: object,
    ) -> LatencyCheckpoint:
        checkpoint = LatencyCheckpoint(
            call_id=call_id,
            turn_id=turn_id,
            name=name,
            monotonic_seconds=time.monotonic(),
            metadata=dict(metadata),
        )
        self._checkpoints.append(checkpoint)
        for callback in tuple(self._subscribers):
            callback(checkpoint)
        return checkpoint

    def snapshot(
        self,
        call_id: str | None = None,
        turn_id: int | None = None,
    ) -> tuple[LatencyCheckpoint, ...]:
        return tuple(
            checkpoint
            for checkpoint in self._checkpoints
            if (call_id is None or checkpoint.call_id == call_id)
            and (turn_id is None or checkpoint.turn_id == turn_id)
        )

    def turn_metrics(self, call_id: str, turn_id: int) -> dict[str, float]:
        checkpoints = self.snapshot(call_id, turn_id)
        if not checkpoints:
            return {}
        started = checkpoints[0].monotonic_seconds
        return {
            checkpoint.name.value: checkpoint.monotonic_seconds - started
            for checkpoint in checkpoints
        }

    def clear(self) -> None:
        self._checkpoints.clear()
