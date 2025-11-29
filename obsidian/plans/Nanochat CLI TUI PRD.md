---
tags:
  - prd
summary: Product requirements for a Textual-based nanochat-cli TUI that cleanly orchestrates nanochat/MC workflows without shell-script sprawl.
---
# nanochat-cli TUI — Product Requirements

## Context
- Current UX: `run10.sh`, `speedrun.sh`, `run1000.sh`, `mc_run.sh`, `mc_setup` have duplicated flags/env plumbing and poor discoverability.
- Goal: one beautiful, extensible Textual TUI that works with stock nanochat by default and layers MegaContext features via extension points.

## Goals
- Zero-guess workflow for train/eval/chat using nanochat checkpoints.
- First-class config management (prefabs + overrides + save-as) with clear categorization.
- Live visualization panels for training with per-screen layouts and adjustable refresh rates.
- Checkpoint audit/load constrained to config compatibility.
- Extensibility hooks so MC (and future) features slot in without breaking vanilla nanochat.

## Non-Goals
- Replacing nanochat Python training code; the TUI orchestrates it.
- Building a browser UI (stay TUI-first).
- Implementing new training algorithms beyond surfaced knobs.

## Personas
- **Researcher**: iterates on configs/MC knobs, watches live telemetry, checkpoints mid-run.
- **Engineer**: integrates into infra/CI, needs reproducible configs and auth hygiene.
- **Operator**: monitors long runs, tweaks screens/panels without restarting.

## User Journeys
- Bootstrap: pick prefab (`$10/$100/$1000`), adjust a few knobs, connect WANDB, start train, tab through screens, save modified config as new prefab.
- Resume: load checkpoint compatible with current config, continue training, trigger eval/chat on that checkpoint.
- Evaluate: load checkpoint, run prompt continuation or chat, view results inline.
- Customize: add a new visualization panel or screen while training, adjust refresh cadence.

## Functional Requirements
### Config
- Load YAML prefab bundles (core nanochat profiles like `$10/$100/$1000`).
- Categorize fields (core nanochat, MegaContext, data, telemetry, auth).
- Inline overrides with validation; ability to save as new prefab.
- WANDB auth entry + validation (persist securely in `.mc_env`-style store).
- Diff view between prefab and current overrides.

### Checkpoints
- List checkpoints filtered by compatibility with active config (depth, vocab, block size, MC on/off, etc.).
- Load/select checkpoint as the run baseline; warn on incompatibility with remediation tips.

### Eval
- Uses loaded checkpoint; run prompt continuation or chat; show latency/token stats and save transcripts.
- Allow quick-edit of eval prompts and persistence of presets.

### Train
- Start training from loaded checkpoint or fresh init; set run name (propagates to WANDB + checkpoint path).
- Live visualization screens (configurable layouts, refresh rates). Built-ins:
  - Progress (steps/tokens/ETA), loss (aggregate + components), validation snapshots.
  - Eval samples.
  - GPU metrics (MFU, VRAM, util).
  - Performance timers (controller, dataloader, step breakdown).
  - MC visualizations (LOD ASCII, swap/residency, allocator actions) via extensions.
- Actions during training:
  - Trigger checkpoint save now.
  - Patch subset of config (validated) mid-run.
  - Switch screens via tabs/shortcuts.
  - Edit screens/panels live: add/remove, tweak sources/thresholds/layout.

### Extensibility
- Plugin-style registry for:
  - Config schemas/categories (vanilla + MC modules).
  - Visualization panels and screen presets.
  - Telemetry sources (WANDB, OTEL, local logs).
- Upstream nanochat remains default; MC features load via optional plugins without hard dependencies.

## UX / UI Guidelines
- Textual-native: keyboard-first, mouse-friendly; panes, tabbed screens, modal dialogs for config/edit/save.
- Clear color palette, consistent typography, responsive to terminal size.
- Status bar with mode, run status, active screen, GPU summary, WANDB status.
- Toasts/alerts for errors (auth, validation), long-running ops, checkpoints completed.
- Command palette for quick actions (save prefab, kick checkpoint, toggle panel).

## Architecture
- Core services: Config Manager (YAML prefabs + overrides), Run Orchestrator (wraps nanochat scripts/modules), Telemetry Ingest (WANDB/OTEL/local), Panel Engine (render + refresh scheduling), Plugin Loader.
- Processes: prefer in-process `uv run`/module calls with structured stdout/stderr capture; fallback to subprocess for existing scripts with robust log tailing.
- Persistence: prefabs and overrides under `.nanochat-cli/` (or user-specified), checkpoint registry cached from `NANOCHAT_BASE_DIR`.
- Compatibility checker: compares config → checkpoint metadata (model size, tokenizer hash, MC flag, block size).

## Integrations
- WANDB: auth test, run name propagation, link surfacing.
- OTEL: optional endpoint/envs to stream spans; surface connectivity status.
- Nanochat CLI/Python: import modules when available; otherwise shell out to existing scripts.

## Telemetry & Observability
- Panel data sources: WANDB streaming API, log tail parser, OTEL spans, local JSON emitters.
- Health indicators: controller latency, swap/residency (MC), GPU headroom, WANDB heartbeat.
- Logging: structured app logs with debug level toggle; crash dumps to file.

## Performance & Resilience
- Target <150ms UI input latency under steady streaming.
- Backpressure: decimate updates when data firehose is high; keep UI responsive.
- Graceful degradation when WANDB/OTEL unavailable; retry with jitter.
- Resume UI state after crash/restart (last screen, loaded config).

## Security & Safety
- Never echo secrets; store WANDB/HF tokens in restricted perms.
- Confirm destructive actions (overwrite prefab, delete screen, stop run).
- Sandbox panel plugins (validate imports/config).

## Migration / Compatibility
- Must work with vanilla nanochat out of the box; all MC features are optional modules.
- Provide compatibility matrix per prefab (expected GPU, depth, tokens) and guardrails against misconfigured runs.

## Milestones (high level)
1) Foundation: config manager, prefab loading/saving, WANDB auth, checkpoint listing.
2) Eval/chat flows on loaded checkpoint; basic train launch with progress + loss panels.
3) Visualization screens + live editing; compatibility checker; command palette.
4) Plugin API for MC panels/config categories; MC telemetry panels.
5) Hardening: resilience, logs, packaging, doc.

## Risks / Open Questions
- How to keep compatibility checks accurate across custom forks of nanochat?
- WANDB streaming rate limits; may need local buffer for high-frequency panels.
- Panel plugin sandboxing in TUI process vs. isolated worker.
- Balancing Textual animation richness with low terminal CPU usage.
