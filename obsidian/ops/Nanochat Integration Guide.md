---
tags:
  - ops
summary: Onboarding guide for running MegaContext on the nanochat fork, covering environment setup, branching, commands, and CI expectations.
---
# Nanochat Integration Guide

Use this note when cloning, configuring, and contributing to the nanochat-based implementation tracked in [[Migration Plan - Nanochat Integration]]. It complements the PRDs by explaining how to run the new stack locally and how to keep parity with upstream nanochat releases. For day-to-day runbooks (env prep, script matrix, telemetry targets) defer to [[Training & Operations]] and treat that doc as the canonical reference.

## Branching & repositories

- **Fork strategy:** keep two branches per the migration plan — `main` mirrors upstream `karpathy/nanochat`, while `megacontext` hosts our changes (MegaAttention, MegaPrediction, telemetry hooks).
- **Sync cadence:** rebase `megacontext` on top of nanochat `main` every time you bump dependencies or land a major PRD feature; conflicts must be resolved before pushing new CI builds.
- **Submodules:** none. Vendor MegaContext code directly in the fork under `megacontext/`-prefixed modules so OSS users can diff the changes easily.

## Environment setup

1. Install [`uv`](https://docs.astral.sh/uv/) and create an environment next to the fork: `uv venv` then `uv sync --extra gpu`.
2. Activate the venv (`source .venv/bin/activate`) so `python`/`torchrun` resolve inside the repo.
3. Export telemetry tokens (`WANDB_API_KEY`, `HF_TOKEN`) plus `NANOCHAT_BASE_DIR` (defaults to `~/.cache/nanochat`) before running; scripts respect the same env vars described in [[Training & Operations]] and [[Telemetry]].
4. Keep the repo itself editable via `uv pip install -e .` if you rely on local modules outside the `nanochat` package.

## Key commands

| Purpose | Command | Notes |
|---------|---------|-------|
| Single-GPU validation | `bash run10.sh --gpu 5090` | Downloads tokenizer/data, runs base/mid/SFT on BF16 consumer GPUs (~3 B tokens). |
| Single-GPU (H100) | `bash run10.sh --gpu h100` | Doubles device batch size, halves iteration count for the same token budget. |
| $100 tier | `bash speedrun.sh` | 8×H100, depth‑20 run mirroring upstream nanochat speedrun. |
| $1000 tier | `bash run1000.sh` | 8×H100 depth‑32 training with tuned device batch + accumulation. |
| Chat / eval | `python -m scripts.chat_cli -p "Hello world"`<br>`python -m scripts.chat_eval -- -i sft` | Use after any training script to sanity-check checkpoints from `~/.cache/nanochat`. |

## CI & telemetry expectations

- **Tests:** run `uv run pytest --maxfail=1 --disable-warnings` before pushing. Add targeted tests when touching the new nanochat modules (especially KV cache or MegaAttention code).
- **Lint/format:** `uv run ruff check src tests` and `uv run black src tests` — matches upstream nanochat style.
- **Telemetry:** all nanochat runs must emit ΔNLL@H, swap rate, budget utilization, gist regression loss, and latency streams via the hooks described in [[Telemetry]]. Use `MEGACONTEXT_ENABLE_WANDB=1` to enforce uploads during CI trials.
- **Artifacts:** keep checkpoints under `artifacts/` with the `megacontext_*` prefix so they don't clash with nanochat's baseline runs.

## Open questions / TODOs

- Expose MegaAttention mask options as CLI flags (link to [[MegaAttention Training]]).
- Document the migration-ready dataset builder once the Arrow → nanochat binarization bridge is committed (currently handled by `python -m nanochat.dataset` inside the scripts).
- Add screenshots/logs of the preferred W&B dashboard for the PRD stack so new contributors can benchmark their runs quickly.
