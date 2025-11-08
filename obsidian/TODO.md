# MegaContext TODO

Single source of truth for current work items across the documentation plan and nanochat migration. Update this file as tasks land.

## Documentation & PRD Polish

- [x] Document migration validation flow (see [[Training & Operations#Post-run validation]] and [[Lifecycle]]).
- [ ] Add “Status: POR” badges to every PRD and tag the remaining legacy phase docs as historical reference (e.g., KV caching, PRD tracker, migration plan). *Need a quick pass on any stragglers under `obsidian/plans/PRDs/` + legacy phase notes.*
- [x] Cross-link future-looking pages (e.g., `obsidian/plans/Future Plan.md`, `obsidian/vision/Realtime Scenarios.md`) back to `obsidian/plans/PRDs/index.md` so readers can see how speculative ideas relate to the active roadmap.
- [ ] Verify each documented command/config exists (or is explicitly marked planned) and add troubleshooting snippets where needed—focus on nanochat scripts, chat eval commands, and telemetry hooks.
- [ ] (Deferred) Add “Last updated” metadata to top-level ops/docs once the nanochat workflow stabilizes.

## Validation, Testing & Telemetry

- [ ] Implement automated smoke tests that run `bash run10.sh --gpu 5090` in a dry-run or truncated mode plus `bash speedrun.sh` on an 8×H100 node to ensure configs stay runnable.
- [ ] Mirror WANDB dashboards (ΔNLL@H, swap rate, residency, MFU) for the nanochat scripts and attach them to the documentation before removing the remaining notebook artifacts.

## Nanochat Migration Phases (from [[Migration Plan - Nanochat Integration]])

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 0 | Repo audit & branch strategy | ✅ | Forks created, upstream parity verified. |
| 1 | Skeleton alignment with vanilla nanochat | ✅ | Baseline runs reproduced; hook points documented. |
| 2 | Legacy snapshot & archival | ✅ | Notebooks and phase docs moved to `_old/`; README/index now point to nanochat flow. |
| 3 | Core MegaContext scaffolding | ✅ | MegaContext Tree + Working Context wiring merged; scripts (`run10.sh`, `speedrun.sh`, `run1000.sh`) run end-to-end. |
| 4 | PRD feature layering (E2E, MegaAttention, MegaPrediction) | 🔄 | MegaAttention + MegaPrediction telemetry hooked in; Cognitive-Core + KV caching still in-flight. |
| 5 | Migration & validation | 🔄 | Need deterministic smoke tests, WANDB parity dashboards, and documentation sign-off. |
| 6 | Upstream parity & maintenance | ⏳ | Define rebase cadence & CI guardrails once Phases 4–5 stabilize. |
