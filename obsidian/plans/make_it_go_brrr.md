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

> Deliverable: WANDB dashboard that shows `build_ms`, `variants`, `lens_ms`, `base_forward_ms`, `optimizer_ms`, `tokens/sec`.

---

## Phase 1 — Remove Horizon Dependency

1. **Single-source supervision** *(DONE)*  
   - Controller no longer evaluates teacher-forced horizons; all WC variants train directly on the next-token objective, and LensNet targets are derived from those losses.
2. **Packed batching** *(DONE: controller groups equal-length variants into shared forwards; further padding optimizations remain optional)*  
   - Variant forwards share batches so we avoid serialized micro-runs.
3. **Teacher-forced superset** *(DONE)*  
   - Every variant contributes to the main loss, and the recency LOD0 loss is logged separately for vanilla comparisons.

> Exit: One forward path per batch (`process_batch`) replaces “baseline + horizon”; the trainer only consumes the aggregated loss.

---

## Phase 2 — Eliminate Duplicate Base Forwards

1. **LOD0 guarantee** *(DONE)*  
   - Controller always tags the recency-baseline WC (`lod_hint == 0`) so `train/loss_lod0` can be logged even when the main forward consumes every variant.
2. **Variant-forward replacement** *(DONE)*  
   - `MCController.process_batch` now runs the variant forwards itself and returns the aggregated loss, so the main training loop no longer touches MC-specific embeddings.
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

## Phase 5 — Dtype Safety Net

1. **Central casting helper**  
   - Add a `_to_model_dtype(...)` helper to `MCController` so every tensor that touches the GPT (embeddings, pos encodings, alibi) automatically matches the model’s dtype/device.
2. **Autocast alignment**  
   - Run `process_batch` under the same autocast context as the main training forward so controller forwards inherit the trainer’s dtype decisions.
3. **Regression tests**  
   - Add unit tests that build bf16/ fp16 working contexts against fp32 models (and vice versa) to ensure `process_batch` succeeds without dtype mismatches.

> Exit: No more manual dtype patches—controller forwards inherit the right dtype automatically and tests catch regressions immediately.

---

## Phase 6 — Effective Batch Normalization

1. **Variant-aware batch math** *(DONE: script now divides `device_batch_size` by `mc_max_counterfactuals` and rescales iterations when `--mc_auto_batch=1`)*  
   - Derive an “effective batch size” so MC runs scale the base batch size automatically as variant counts change.
2. **Auto-downscale & compensate** *(DONE)*  
   - When `--mc` is enabled, reduce `device_batch_size` by the variant multiplier and increase `num_iterations` proportionally so token throughput matches the vanilla run without manual tuning.
3. **Telemetry feedback** *(TODO)*  
   - Log `mc/effective_batch_size` and `mc/variants_per_sample` to WANDB so OOMs can be tied back to variant choices and we can refine the heuristic (e.g., include LensNet/GistNet overhead).

> Exit: flipping `--mc` on no longer surprises you with OOMs; the script automatically keeps the total work per step in the same ballpark as the vanilla recipe.

---

## Open Questions

- How small can we make `H` before ΔNLL supervision becomes noisy?  
- Do we need separate LoRA adapters for horizon vs full-sequence loss once the base forward covers both?  
- Can we cache WC embeddings across batches to avoid re-embedding LOD0 each time?
- Can we make dtype handling automatic so controller/model always agree (helper + tests)?
- Can we make dtype handling automatic so controller/model always agree (helper + tests)?

---

**Next Action:** implement Phase 0 instrumentation so we have real measurements before chopping features. Once the logs show where the GPU time goes, proceed with the Phase 1/2 refactors to collapse duplicate forwards and make horizons optional. Only then re-run the WANDB comparison to confirm we’re getting “more BRRR” per dollar.***
