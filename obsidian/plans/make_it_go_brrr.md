---
tags:
  - plan
summary: Step-by-step plan to make MegaContext training throughput competitive by profiling, eliminating duplicated compute, and rethinking horizon usage.
---
# Make It Go BRRR — MegaContext Throughput Recovery

> Goal: bring the `--mc` pipeline back into the same order-of-magnitude throughput as vanilla nanochat while preserving end-to-end supervision. The plan below moves from "measure"  "simplify"  "re-architect".

---

## Phase 0 — Instrument Everything

1. **Fine-grained timers**  
   - Add `torch.cuda.Event` / `time.perf_counter` spans around every major task inside `MCController.process_batch` and the main `scripts/base_train` loop: tree builds, variant sampling, horizon forward, lens loss, base forward, optimizer step.  
   - Export per-step timings to WANDB (`train/time/*`) and `mc_timing` telemetry so we get historical traces, not just stdout.
2. **Variant counters**  
   - Log how many WCs survive each stage (`N_recency`, `N_allocated`, `N_siblings`, `N_deduped`).  
   - Emit “horizon tokens processed per step” so we can normalize ΔNLL vs compute.
3. **Memory snapshot hooks**  
   - When `MC_MEMORY_DEBUG` is set, dump `torch.cuda.memory_summary()` after each phase, tied to the timing logs.  
   - Store summaries in `report/` so we have persistent evidence when tuning knobs.

> Deliverable: WANDB dashboard that shows `build_ms`, `variants`, `horizon_ms`, `lens_ms`, `base_forward_ms`, `optimizer_ms`, `tokens/sec`.

---

## Phase 1 — Make Horizons Optional & Cheap

1. **Config gate**  
   - Add `--mc_enable_horizon` (default `1` for parity, but allow disabling).  
   - When disabled, still build all WC variants and run them through the base model for full next-token loss; simply skip the extra horizon-specific ΔNLL bookkeeping so “pure focus allocator” mode is just variants + regular NLL.
2. **Packed batching** *(DONE: controller groups equal-length variants into shared forwards; further padding optimizations remain optional)*  
   - When horizons are enabled, batch all variants for the teacher-forced window into shared forwards instead of serialized loops.  
   - Use padding or length-based grouping so the GPU sees large batches rather than many tiny ones.
3. **Teacher-forced superset** *(DONE: base forward now iterates over all variants and logs LOD0-only loss separately)*  
   - Compute the full next-token loss for **every** variant (LOD0 and LOD1+) so the training signal is a superset of the vanilla objective.  
   - Keep track of which variant is “pure LOD0 recency” so we can still report a vanilla-compatible loss metric even if we drop the separate baseline forward.

> Exit: Able to flip horizons off to validate speed, and when enabled they add ≤20% overhead because they’re batched.

---

## Phase 2 — Eliminate Duplicate Base Forwards

1. **LOD0 guarantee** *(DONE)*  
   - Controller always tags the recency-baseline WC (`lod_hint == 0`) so `train/loss_lod0` can be logged even when the main forward consumes every variant.
2. **Variant-forward replacement** *(DONE)*  
   - `MCController.process_batch` now returns the sampled variants and their embeddings; the main training loop iterates over them and accumulates the per-variant next-token losses instead of running a duplicate “baseline” forward.
3. **Grad weighting** *(Deferred until Phase 3)*  
   - Current implementation averages per-variant losses (fixed count), so gradient magnitude stays stable. If we add dynamic variant counts later, revisit weighting to keep LR comparable.

> Result: every WC variant participates in the core loss, horizons are optional, and we no longer pay for duplicate forwards. Remaining work (weighting) can wait until variant counts become dynamic.

---

## Phase 3 — Smarter Variant Scheduling

1. **Dynamic budget**  
   - Adapt `max_counterfactuals` per sequence based on available compute headroom (tokens/sec target).  
   - When GPU is saturated, reduce variant count automatically; when idle, explore more siblings.
2. **Priority sampling**  
   - Prefer variants that are likely to change the loss (e.g., high lens-score variance, large ΔNLL from previous steps).  
   - Avoid wasting slots on gists that already look perfect.
3. **Asynchronous controller**  
   - Offload tree builds and LensNet scoring to a background CUDA stream or CPU worker so the main training step overlaps with controller prep for the next batch.

---

## Phase 4 — Verify, Compare, Iterate

1. **Tokens-per-second parity check**  
   - Run the full `run10.sh --mc` pipeline with horizons off/on and record `tokens/sec`, `mc/*` losses, WANDB traces.  
   - Compare against the vanilla run using normalized axes (tokens processed, FLOPs, wall-clock).
2. **A/B gating**  
   - Keep the old “baseline forward” path behind a flag until the new superset training proves stable.  
   - Document the CLI recipe for “fast MC” vs “full MC” modes.
3. **Update PRD + README**  
   - Reflect the new training strategy in [[MegaContext End-to-End Training]] and `README.md` so future work doesn’t resurrect the duplicated compute pattern.

---

## Open Questions

- How small can we make `H` before ΔNLL supervision becomes noisy?  
- Do we need separate LoRA adapters for horizon vs full-sequence loss once the base forward covers both?  
- Can we cache WC embeddings across batches to avoid re-embedding LOD0 each time?

---

**Next Action:** implement Phase 0 instrumentation so we have real measurements before chopping features. Once the logs show where the GPU time goes, proceed with the Phase 1/2 refactors to collapse duplicate forwards and make horizons optional. Only then re-run the WANDB comparison to confirm we’re getting “more BRRR” per dollar.***
