---
tags:
  - tdd
summary: Test design for the Textual nanochat-cli TUI covering config, orchestration, visualization, extensibility, and resilience.
---
# nanochat-cli TUI — Test Design

## Scope & Strategy
- Cover core workflows: config prefabs/overrides, checkpoint compatibility, eval/chat, training orchestration, live visualization, and extensibility hooks (MC-specific modules included via plugins).
- Blend automated tests (unit, integration, golden snapshots) with manual acceptance for Textual UX flows.
- Keep vanilla nanochat the baseline; MC behaviors ride through extension points to ensure no regressions when MC modules are absent.

## Test Inventory
### Config & Prefabs
- **Prefab load/save round-trip**: YAML → model → YAML; preserves categorization metadata.
- **Override application**: apply CLI/TUI edits; verify effective config and diff view.
- **Validation**: invalid types/ranges rejected with user-facing errors; required auth surfaced.
- **Save-as**: create new prefab, ensure only deltas are stored when requested.
- **Categorization**: fields tagged correctly (core, MC, data, telemetry, auth) and drive UI grouping.

### Compatibility Checker
- **Happy path**: checkpoint metadata matches config (depth, tokenizer hash, MC flag, block size).
- **Mismatch cases**: each incompatibility yields actionable message (e.g., vocab mismatch, MC on/off conflict, device batch too large for saved profile).
- **Fallback**: when metadata missing, prompt for manual confirm and log warning.

### WANDB/Auth
- **Token capture**: secrets never echoed; stored with restricted perms.
- **Connectivity check**: valid token passes; invalid token fails with clear message.
- **Run-name propagation**: run name set in TUI appears in WANDB init call and checkpoint path.

### Run Orchestration
- **Train launch**: starts process/module with effective config; captures stdout/err; exits non-zero on failure.
- **Resume**: loads checkpoint and continues training; ensures optimizer state loaded when available.
- **Eval/chat**: uses selected checkpoint; returns tokens, latency stats; fails cleanly when checkpoint missing.
- **Config patch mid-run**: supported subset applies without restart; unsupported fields rejected with message.
- **Checkpoint-now action**: triggers save and surfaces path; verifies file existence and size grows.

### Visualization Engine
- **Panel registry**: built-ins register at startup; MC panels only when plugin loaded.
- **Refresh cadence**: fast panels throttle when update rate exceeds limit; UI remains responsive.
- **Data sources**: WANDB streaming, OTEL spans, and log tail parsers feed panels; simulated streams covered by fixtures.
- **Screen editing**: add/remove/reorder panels at runtime; layouts persist across restart when saved.
- **Rendering**: snapshot tests for key panels (progress, loss, GPU, timers, LOD ASCII) with sample data.

### Plugin / Extensibility
- **Plugin load/unload**: MC plugin can be enabled/disabled without impacting vanilla flows.
- **Schema extension**: plugin-added config fields render under their category and validate.
- **Panel extension**: third-party panel registers and renders with fixture data.
- **Error isolation**: faulty plugin raises controlled error and is quarantined without crashing core UI.

### Resilience & Recovery
- **Network loss**: WANDB/OTEL outages degrade gracefully with retries/backoff; panels show stale/paused state.
- **Log stream stall**: tailers detect EOF and resume on file growth.
- **Crash recovery**: restart restores last screen and loaded config; unsaved edits prompt warning.
- **Terminal resize**: layout recomputes without rendering errors.

### Performance
- **Input latency**: under synthetic high-frequency telemetry, keypress-to-action latency stays <150ms.
- **CPU/mem budget**: watchdog alerts if panel update loop exceeds thresholds.

### Accessibility / UX Checks (Manual)
- Keyboard-only navigation for all primary actions (load config, start train, switch screens, save prefab).
- Color contrast legible in light/dark terminals; status bar and alerts readable.
- Command palette shortcuts documented and functional.

## Test Fixtures & Tooling
- Fixture configs: prefab YAMLs for `$10/$100/$1000`, MC-on/off variants, malformed configs.
- Fixture checkpoints: tiny fake checkpoints with metadata to exercise compatibility logic.
- Telemetry fixtures: WANDB mock server, OTEL mock receiver, synthetic log emitters.
- Textual snapshot harness for rendering tests.

## Environments
- CI: headless Textual + mocked nanochat entrypoints; no GPU required.
- Integration: optional GPU box to verify GPU panels and real nanochat train/eval stubs.

## Exit Criteria
- All unit/integration suites green in CI with coverage on core modules and panel registry.
- Manual UX checklist signed off (navigation, alerts, screen editing, plugin toggle).
- Vanilla nanochat flows pass without MC plugin; MC plugin tests pass when enabled.
