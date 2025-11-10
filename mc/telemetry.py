from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


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


class OpenTelemetryProvider:
    """
    Telemetry provider that emits events as OTEL spans so downstream systems
    (Tempo/Grafana, Honeycomb, etc.) can visualize MegaContext sessions.
    """

    def __init__(
        self,
        service_name: str = "megacontext",
        endpoint: Optional[str] = None,
        insecure: bool = False,
        resource_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise ImportError(
                "opentelemetry-sdk and opentelemetry-exporter-otlp are required for OpenTelemetryProvider"
            ) from exc

        resource = Resource.create(
            {
                "service.name": service_name,
                **(resource_attributes or {}),
            }
        )
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)

    def log_event(self, event: TelemetryEvent) -> None:
        attributes = {
            "mc.session_id": event.session_id,
            "mc.event_type": event.event_type,
        }
        for key, value in event.payload.items():
            attributes[f"mc.{key}"] = self._sanitize_value(value)
        with self._tracer.start_as_current_span(event.event_type) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple, set)):
            return json.dumps(value, default=str)
        return str(value)
