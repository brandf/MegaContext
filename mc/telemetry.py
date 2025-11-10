from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class TelemetryEvent:
    session_id: str
    event_type: str
    payload: Dict[str, Any]


class TelemetryProvider(Protocol):
    def log_event(self, event: TelemetryEvent) -> None: ...


class NoOpTelemetryProvider:
    def log_event(self, event: TelemetryEvent) -> None:  # pragma: no cover - intentional no-op
        return
