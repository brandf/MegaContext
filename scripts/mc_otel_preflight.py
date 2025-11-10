"""
Preflight for OpenTelemetry export. This sends a single MegaContext test span
to the OTLP endpoint specified by MC_OTEL_ENDPOINT.

Usage:
  MC_OTEL_ENDPOINT=http://localhost:4318 MC_OTEL_INSECURE=1 \
  uv run python -m scripts.mc_otel_preflight
"""
from __future__ import annotations

import os
from typing import Any, Dict

from mc.telemetry import OpenTelemetryProvider, TelemetryEvent


def main() -> None:
    endpoint = os.getenv("MC_OTEL_ENDPOINT")
    insecure = os.getenv("MC_OTEL_INSECURE", "0") == "1"
    if not endpoint:
        print("MC_OTEL_ENDPOINT is not set; skipping export (nothing to verify)")
        return
    provider = OpenTelemetryProvider(
        service_name="megacontext-preflight",
        endpoint=endpoint,
        insecure=insecure,
        resource_attributes={"preflight": True},
    )
    event = TelemetryEvent(
        session_id="preflight",
        event_type="mc_preflight",
        payload={"message": "hello from MegaContext", "ok": True},
    )
    provider.log_event(event)
    print(f"Sent OpenTelemetry preflight span to {endpoint}")


if __name__ == "__main__":
    main()

