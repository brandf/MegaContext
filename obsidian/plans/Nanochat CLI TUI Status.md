---
tags:
  - status
summary: Execution tracker for the Textual nanochat-cli TUI against PRD/TDD items.
---
# Nanochat CLI TUI Status

## Tracks
- [x] Repo wiring: package scaffold, entrypoint, add Textual/PyYAML deps.
- [x] Config: prefab load/save/override, diff skeleton.
- [x] Checkpoints: scan + basic compatibility heuristics.
- [x] Train/Eval orchestration: subprocess launch + log streaming.
- [ ] Setup flow: dependency detection + guided install inside TUI (no reliance on mc_setup/run*.sh).
- [ ] Dataset tab: dataset download/prep orchestration.
- [ ] Visualization panels: rich screens (loss, GPU, timers, MC LOD) with refresh cadences.
- [ ] Command palette + keyboard UX for core actions.
- [ ] Live screen editing + persistence.
- [ ] Plugin/extension API: config schema/panels/telemetry hooks for MC and other projects (initial plugin registry landed).
- [ ] Telemetry ingestion: WANDB streaming, OTEL spans, log tail parsing.
- [ ] Mid-run config patching + checkpoint-now signaling.
- [ ] Robust compatibility checks (checkpoint metadata, manual override).
- [ ] UI polish: alerts/toasts, status bar details, color/contrast audit.
- [ ] Resilience: retries/backoff, crash recovery of UI state.
- [ ] Packaging/bootstrap: minimal install path on fresh boxes.
- [ ] Feedback round 1 (current):
  - [ ] Config UX: dropdown selector, categorized tabs with inline editors (no raw JSON/YAML), inline dirty tracking, bold unsaved fields, save/save-as flows with reload warning.
  - [ ] Rename “prefabs” → “configs”; display values (not raw text); auto-apply edits, save writes files.
  - [ ] Checkpoints: refresh as contextual action; move logic into checkpoints component (app.py slim).
  - [ ] Train: contextual start, no config display leakage.
  - [ ] Eval: rename to Eval; drop chat mode; simple continuation prompt.
  - [ ] Dataset tab before Train; detect/download/prep dataset; train blocks when missing with modal redirect.
  - [ ] Setup first tab; no mc_setup dependency; self-managed setup; “Run” action; re-check via action not button.
  - [ ] Tab order: Setup → Config → Dataset → Checkpoints → Train → Eval.

## Notes
- Current build is a functional skeleton; prioritizing setup/dataset flows and panel/telemetry next.
