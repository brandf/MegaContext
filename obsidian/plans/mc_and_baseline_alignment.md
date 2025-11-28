---
tags:
  - plan
summary: Restore apples-to-apples comparisons between MegaContext runs and vanilla nanochat baselines.
---
# MC vs Baseline Alignment Plan

## Problem
- MC runs shrink the per-device batch (to fit variants) but crank up `num_iterations` so total tokens stay constant. Baseline runs keep the large batch and finish in ~38K steps, MC runs take ~192K steps, making curves impossible to compare.
- WANDB logging is keyed on `step`, so LR schedules/validation cadence misalign.
- CLI defaults were tweaked for MC (e.g., device batch 88) and break the vanilla “run once” experience.
- The auto-batch code rewrites `device_batch_size`/`total_batch_size` before we even instantiate the dataloader (`scripts/base_train.py:191-213`), so the sequential loader in `nanochat/dataloader.py:8-39` now slices different token windows per “step.” Even if total tokens match in aggregate, step *N* in MC is consuming a different portion of the corpus than step *N* in baseline, which breaks any apples-to-apples overlay.

## Goals
1. One command per mode (`run10.sh` vs `run10.sh --mc`) produces logs that line up on “step” and total tokens.
2. Baseline checkpoints remain untouched when running MC experiments.
3. WANDB comparisons require zero post-processing (tokens/step metadata baked in).

## Action Items
1. **Auto-batcher revamp**
   - ✅ Treat the run10 baseline micro-batch (`DEVICE_BATCH_SIZE × 2048`) as the canonical `total_batch_size`/step. `scripts/base_train.py` now keeps `total_batch_size` untouched, shrinks the per-rank micro-batch via `choose_micro_batch_divisor`, and bumps `grad_accum_steps` so each optimizer step still ingests the exact same token budget as the baseline. The new helper lives in `mc/auto_batch.py` with unit tests (`tests/test_mc_auto_batch.py`).
   - ✅ When `--mc` is on, scale down the *micro* batch just enough to fit variants **and** increase `grad_accum_steps = per_rank_budget / new_device_batch_size` so each optimizer step still processes ~114,688 tokens. `tokenizing_distributed_data_loader` therefore advances through the identical token ranges as the vanilla run, only sliced across multiple accumulation micro-steps.
   - ✅ Log `train/tokens_per_step`, `train/grad_accum_steps`, and `mc/grad_accum_steps` whenever we emit training metrics so WANDB plots can normalize without post-processing.
   - ✅ Guard against `total_batch_size` silently shrinking—the script now validates divisibility and raises when the MC auto-batcher can’t reproduce the baseline grad-accum math.

2. **Scheduler/validation alignment**
   - ✅ LR/optimizer schedules still key off `total_batch_size`, but all cadence gates (`eval_every`, `core_metric_every`, `sample_every`, checkpoints, and logging) now run on token intervals. Because MC preserves the same tokens/step, both modes emit validation/checkpoint/log entries on identical token counts even if grad accumulation differs.

3. **CLI hygiene**
   - ✅ Introduced `--profile baseline|mc` in `run10.sh`; it toggles `MC_ENABLED` without relying on env vars, but the legacy `--mc` flag still works for ad-hoc overrides.
   - (done previously) `MODEL_TAG`/`WANDB_RUN` continue to respect user overrides.

4. **CI sanity checks**
   - Added `tests/test_mc_auto_batch.py` to cover the variant-multiplier math and verify we keep the per-rank token budget constant. A future follow-up can still wire the full script harness, but the math is now unit-tested.

5. **Documentation**
   - ✅ README + `obsidian/plans/MegaContext Performance Review.md` now describe the matched-run workflow, the `--profile` flag, and the new telemetry (`train/tokens_per_step`, `train/grad_accum_steps`, `mc/grad_accum_steps`). `obsidian/plans/lensnet_perf_investigation.md` also references these counters so controller timings can be compared fairly.
   - TODO: provide a short WANDB panel template (tokens on x-axis) so the blog write-up can reference it directly. (Draft notes are at the bottom of this plan until we settle on the shared dashboard.)

6. **Logging cadence**
   - ✅ `log_interval` is now token-based (100 baseline steps worth of tokens). MC and vanilla emit progress points at the same cadence, so dashboards line up automatically.

## Exit Criteria
- Running `bash run10.sh` and `bash run10.sh --mc` produces WANDB runs with the same step count (within rounding error) and identical LR schedules.
- No manual flags needed to compare results; tokens/step metadata is logged automatically.
- Baseline run finishes without tinkering with batch size, and MC run still hits high VRAM utilization.***

Off the top of my head, these are the other “rocks” worth turning over:

- ✅ Validation cadence now uses token thresholds, so MC and baseline emit the same number of validation points for a given token budget.
- ✅ Sample/core metric cadence shares the same token-based triggers, so the dashboards stay aligned even when grad accumulation differs.
- ✅ Checkpoint cadence now keys on tokens as well, so each run writes the same number of checkpoints per token budget.
- WANDB naming/tagging: fixed now, but `.mc_env` still injects `WANDB_RUN=mc-run10` unless you override; easy to forget.
- Auto_batch math: fixed as described above—the per-step token budget, `tok/sec`, and MFU denominators now remain identical across MC and baseline.
- Everything else (learning rate schedule, optimizer, compile settings) is shared between the modes, so there aren’t hidden flags beyond those. If you want, I can sweep the trainer for more `if mc_enabled` branches and either delete them or make them token-based so you’re not surprised again.

## WANDB panel draft
Use the “Tokens (step-aligned)” axis for every panel so MC/baseline overlays match:
- **Panel 1 — Training loss vs tokens:** line plot of `train/loss` with `XAxis`=`train/tokens_per_step` (enable “Normalize by tokens_per_step”).
- **Panel 2 — Validation bpb vs tokens:** line plot of `val/bpb`, same axis.
- **Panel 3 — Grad accumulation:** multi-line chart for `train/grad_accum_steps` and `mc/grad_accum_steps` so reviewers can confirm they match.
- **Panel 4 — Throughput:** plot `train/tok_per_sec` to ensure both runs maintain similar throughput after the auto-batch change.
Save the layout as `MC vs Baseline (Tokens)` in WANDB so future runs can be attached to the same dashboard.
