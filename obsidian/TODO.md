# MegaContext TODO

Single source of truth for current work items across the documentation plan and nanochat migration. Update this file as tasks land.

## Documentation & PRD Polish

- [x] Document migration validation flow (see [[Training & Operations#Post-run validation]] and [[Lifecycle]]).
- [x] Add “Status: POR” badges to every PRD and tag the remaining legacy phase docs as historical reference (e.g., KV caching, PRD tracker, migration plan).
- [x] Cross-link future-looking pages (e.g., `obsidian/plans/Future Plan.md`, `obsidian/vision/Realtime Scenarios.md`) back to `obsidian/plans/PRDs/index.md` so readers can see how speculative ideas relate to the active roadmap.
- [x] Verify each documented command/config exists (or is explicitly marked planned) and add troubleshooting snippets where needed—focus on nanochat scripts, chat eval commands, and telemetry hooks.

## Phase 1 Close-out

- [x] Pick a production telemetry backend (standardize on OpenTelemetry → Tempo/Grafana) and implement the OTLP `TelemetryProvider` wiring (see `MC_OTEL_*` env vars).
- [x] Document the “fair comparison” methodology for MC vs. vanilla nanochat runs (tokens/FLOPs/time axes) and add dashboard guidance to [[Training & Operations]].
- [ ] Build dashboards (W&B + chosen telemetry backend) that visualize MC metrics: tree snapshots, WC edits, focus scores, and horizon trigger rates.
- [ ] Finalize inference-session UX: CLI flag docs, telemetry examples, and troubleshooting for long-lived MC-enabled chat serving.

## Nanochat Migration Phases (from [[Migration Plan - Nanochat Integration]])

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 0 | Repo audit & branch strategy | ✅ | Forks created, upstream parity verified. |
| 1 | Skeleton alignment with vanilla nanochat | ✅ | Baseline runs reproduced; hook points documented. |
| 2 | Legacy snapshot & archival | ✅ | Notebooks and phase docs moved to `_old/`; README/index now point to nanochat flow. |
| 3 | Core MegaContext scaffolding | ✅ | MegaContext Tree + Working Context wiring merged; scripts (`run10.sh`, `speedrun.sh`, `run1000.sh`) run end-to-end. |
| 4 | PRD feature layering (E2E, MegaAttention, MegaPrediction) | 🔄 | MegaAttention + MegaPrediction telemetry hooked in; Cognitive-Core + KV caching still in-flight. |
