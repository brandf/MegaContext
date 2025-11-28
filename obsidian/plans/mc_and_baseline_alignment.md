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

## Goals
1. One command per mode (`run10.sh` vs `run10.sh --mc`) produces logs that line up on “step” and total tokens.
2. Baseline checkpoints remain untouched when running MC experiments.
3. WANDB comparisons require zero post-processing (tokens/step metadata baked in).

## Action Items
1. **Auto-batcher revamp**
   - Keep the *baseline* device batch (`56 × 2048 tokens`) for both modes.
   - When `--mc` is on, scale down `device_batch_size` to fit variants **and** increase `grad_accum_steps` so each optimizer step still processes ~114,688 tokens.
   - Log `mc/grad_accum_steps` and `tokens_per_step` so WANDB plots can normalize if desired.

2. **Scheduler/validation alignment**
   - Key LR schedule and validation cadence on `total_training_tokens` instead of raw step count.
   - Ensure `eval_every` and `checkpoint_every` are computed in tokens (or use the same step count in both modes).

3. **CLI hygiene**
   - Introduce `--profile baseline|mc` to swap defaults (device batch, auto-batch flags) without touching env vars.
   - Ensure `MODEL_TAG`/`WANDB_RUN` defaults never overwrite user args (already started).

4. **CI sanity checks**
   - Add a tiny “compare” test that runs `scripts/base_train.py` for a few steps with and without MC and asserts:
     - Logged `total_training_tokens` match.
     - `grad_accum_steps` adjustment kicks in.
     - Checkpoint directories remain isolated.

5. **Documentation**
   - Update README + `obsidian/plans/MegaContext Performance Review.md` with the new workflow and explain how to run matched comparisons.
   - Provide a short WANDB panel template (tokens on x-axis) so the blog write-up can reference it directly.

## Exit Criteria
- Running `bash run10.sh` and `bash run10.sh --mc` produces WANDB runs with the same step count (within rounding error) and identical LR schedules.
- No manual flags needed to compare results; tokens/step metadata is logged automatically.
- Baseline run finishes without tinkering with batch size, and MC run still hits high VRAM utilization.***

Off the top of my head, these are the other “rocks” worth turning over:

Validation cadence: scripts/base_train.py forces eval_every = 25 whenever mc_enabled=1, while vanilla run10 keeps the nanochat default (250). So MC runs log 10× more validation points. We should either make that configurable or normalize both.

Checkpoint cadence: the code currently checkpoints every 1,000 steps for both, but because MC has ~5× more steps, it writes checkpoints far more often (and larger total size). If you want exactly one checkpoint per token budget, we should switch the condition to tokens instead of steps.

WANDB naming/tagging: fixed now, but .mc_env still injects WANDB_RUN=mc-run10 unless you override; easy to forget.

auto_batch math: right now MC shrinks device_batch_size and increases step count to keep tokens constant. Baseline uses the raw batch. That’s the root of the “step mismatch” frustration we just discussed; fixing it will require the alignment plan we wrote down.

Everything else (learning rate schedule, optimizer, compile settings) is shared between the modes, so there aren’t any hidden flags beyond those. If you want, I can sweep the trainer for more if mc_enabled branches and kill them or make them token-based so you’re not surprised again.
