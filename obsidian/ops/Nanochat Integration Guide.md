---
tags:
  - ops
summary: Onboarding guide for running MegaContext on the nanochat fork, covering environment setup, branching, commands, and CI expectations.
---
# Nanochat Integration Guide

Use this note when cloning, configuring, and contributing to the nanochat-based implementation tracked in [[Migration Plan - Nanochat Integration]]. It complements the PRDs by explaining how to run the new stack locally and how to keep parity with upstream nanochat releases.

> **Status:** Planning only. The nanochat fork has not been imported into this repository yet, so the commands/configs below are illustrative targets. Use the research notebook flow until the migration work in the PRDs lands.

## Branching & repositories

- **Fork strategy:** keep two branches per the migration plan — `main` mirrors upstream `karpathy/nanochat`, while `megacontext` hosts our changes (MegaAttention, MegaPrediction, telemetry hooks).
- **Sync cadence:** rebase `megacontext` on top of nanochat `main` every time you bump dependencies or land a major PRD feature; conflicts must be resolved before pushing new CI builds.
- **Submodules:** none. Vendor MegaContext code directly in the fork under `megacontext/`-prefixed modules so OSS users can diff the changes easily.

## Environment setup

1. Install [`uv`](https://docs.astral.sh/uv/) and create an environment next to the fork: `uv venv` then `uv sync`.
2. Enable editable installs for MegaContext modules: `uv pip install -e .` so local edits propagate into the nanochat CLI.
3. Export telemetry tokens (`WANDB_API_KEY`, `HF_TOKEN`) before running; scripts respect the same env vars described in [[Training & Operations]] and [[Telemetry]].
4. Configure dataset and artifact roots exactly as in the notebooks by setting `MEGACONTEXT_DATA_ROOT` and `MEGACONTEXT_ARTIFACT_ROOT`.

## Key commands

| Purpose | Command | Notes |
|---------|---------|-------|
| Vanilla nanochat sanity run (planned) | `uv run python -m nanochat.train --compile=False --out_dir artifacts/nanochat_baseline` | Use once the fork is pulled in and baseline configs exist. |
| MegaContext end-to-end run (planned) | `uv run python -m nanochat.train --config configs/megacontext_e2e.yaml --context_config configs/wc/prd.yaml` | Tracks the [[MegaContext End-to-End Training]] PRD; configs will land with the migration. |
| MegaPrediction decode demo (planned) | `uv run python -m nanochat.chat --config configs/megaprediction_demo.yaml` | Exercises the shared wLOD readout defined in [[MegaPrediction Training]] once implemented. |
| Legacy smoke test (current) | `uv run python -m tools.decode_demo --config configs/SampleText_TinyGPT2.yaml` | Continue using this command until the nanochat CLI is available. |

## CI & telemetry expectations

- **Tests:** run `uv run pytest --maxfail=1 --disable-warnings` before pushing. Add targeted tests when touching the new nanochat modules (especially KV cache or MegaAttention code).
- **Lint/format:** `uv run ruff check src tests` and `uv run black src tests` — matches upstream nanochat style.
- **Telemetry:** all nanochat runs must emit ΔNLL@H, swap rate, budget utilization, gist regression loss, and latency streams via the hooks described in [[Telemetry]]. Use `MEGACONTEXT_ENABLE_WANDB=1` to enforce uploads during CI trials.
- **Artifacts:** keep checkpoints under `artifacts/` with the `megacontext_*` prefix so they don't clash with nanochat's baseline runs.

## Open questions / TODOs

- Expose MegaAttention mask options as CLI flags (link to [[MegaAttention Training]]).
- Document the migration-ready dataset builder once the Arrow → nanochat binarization bridge is committed.
- Add screenshots/logs of the preferred W&B dashboard for the PRD stack so new contributors can benchmark their runs quickly.
