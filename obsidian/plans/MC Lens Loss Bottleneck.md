---
tags:
  - investigation
  - performance
summary: LensNet loss is the dominant runtime cost (seconds per step). This doc maps the hot spots and outlines fixes to drive it toward zero.
---
# MC Lens Loss Bottleneck — What’s Slow and How to Fix It

## Problem (today)
- `mc/lens_loss_ms` is ~2.9 s per micro-step even though `mc/lens_forward_ms` is ~2 ms. The controller, not the base model, dominates step time.
- At your current settings (device micro-batch=13 after auto-batch, `mc_num_random_variants=2`, grad_accum=5), we process ~195 working contexts per optimizer step:
  - 13 samples × (baseline + 2 variants) = 39 WCs per micro-step
  - 5 grad-accum micro-steps → ~195 WCs total
- The loss code is Python-heavy: building pairwise targets, masks, and stats per variant/pair, then launching many tiny ops. That’s where the ~3 s comes from.

## Where the time is spent (code pointers)
- `mc/runtime.py:_compute_lens_losses` (around line 2128):
  - Flattens all variants, runs `global_score_cache = self._batch_variant_scores(all_variants)` once (fast, ~2 ms).
  - For each sample: `_build_preference_pairs`, `_build_lod_lookup`, `_build_pairwise_targets`, then `_batched_pref_losses` and `_batched_aux_losses`.
  - The heavy part is the per-pair/variant Python work (mask/target construction, scatter into lists) and multiple small tensor ops per pair.
- `mc/runtime.py:_build_pairwise_targets` and `_build_lod_lookup`:
  - Python loops over positions/lods with tensor `.item()`/masking; lots of host-side work and GPU syncs.
- `mc/runtime.py:_batched_aux_losses`:
  - Iterates over `pair_entries`, does elementwise ops per pair; not fused or vectorized.

## Constraints / goal
- Keep the training objective intact (pairwise preference + budget/coverage terms) but make `lens_loss_ms` negligible compared to the base forward. Target: tens of ms → ideally sub-ms.
- Maintain the same variant semantics: `(1 + mc_num_random_variants)` per sample; grad-accum stays, no reuse across micro-steps.

## Fix options (stack-ranked by impact)
1) **Batch the entire preference loss on GPU (eliminate Python loops):**
   - Build flat tensors for all pairs across the micro-batch:
     - `scores_flat`, `targets_flat`, `weights_flat`, `scale_flat`, `span_tokens_flat`.
   - Compute the Bradley–Terry (or hinge) loss in one fused kernel or a small number of vectorized ops. Use torch ops only; no per-pair Python.
   - Compute budget/coverage penalties from precomputed histograms via `torch.bincount`/`scatter_add` on GPU.
   - Expected impact: remove ~90% of `lens_loss_ms` (turn seconds into single-digit ms).

2) **Precompute / cache pairwise targets on CPU without `.item()` syncs:**
   - Move `positions`, `lods` to CPU once per WC; build masks/targets with numpy/torch CPU ops; avoid per-element `.item()` on CUDA tensors.
   - Cache LOD lookups per WC to reuse across pairs.
   - Expected impact: large cut to Python overhead even before full batching.

3) **Reduce pair count aggressively (sampling):**
   - Sample a fixed small number of preference pairs per variant (e.g., top-K spans or K random spans) instead of all spans.
   - Downsample span tokens before loss (e.g., stride >1) to shrink targets/masks.
   - Expected impact: linear reduction in loss compute proportional to pair count.

4) **Gate the loss frequency:**
   - Compute lens loss every N steps (or every M micro-steps), accumulate/average, and set it to zero otherwise.
   - Expected impact: amortize cost without changing math when it runs.

5) **Toggle off auxiliary terms that add overhead:**
   - Temporarily disable budget/coverage penalties (`lens_budget_weight`, `lens_margin`, etc.) to isolate the pairwise loss cost. Re-enable after batching.
   - Expected impact: simplifies the loss graph; less Python/tensor churn.

6) **AOT/fused custom kernel:**
   - Write a custom CUDA/torch.compile fused op for the Bradley–Terry loss over flattened pairs. This bypasses Python and leverages GPU fully.
   - Expected impact: minimal overhead; stretch goal if vectorized torch ops aren’t enough.

7) **Variant count sanity:**
   - Verify we are only doing `(micro_batch_size × (1 + mc_num_random_variants))` variants per micro-step and that grad-accum is the only multiplier. (Current telemetry shows 39 per micro-step for micro-batch=13, which is correct.)
   - No change expected; just ensure we don’t inadvertently inflate pair counts.

## Recommended path (pragmatic + fast to implement)
1. Implement a vectorized preference loss:
   - Build per-variant score cache once (`global_score_cache` is already batched).
   - Flatten all pair payloads into contiguous tensors and compute the Bradley–Terry/hinge loss in a single torch op block.
   - Compute budget/coverage penalties with `torch.bincount`/`scatter_add`.
   - Remove per-pair loops and `.item()` calls.
2. If still >50 ms, add pair sampling (e.g., cap pairs per variant) and/or loss gating (every N steps).
3. Only then consider a fused kernel if torch ops aren’t sufficient.

## Stretch ideas (if we need near-zero cost)
- **Two-stage training:** train LensNet offline or in a separate stage with cached WCs, then freeze during main training (lens loss = 0 during main loop).
- **Proxy loss:** replace pairwise comparisons with a cheap L2/hinge on a small subset of spans or a distilled target (e.g., match a slower teacher’s scores offline).
- **Asynchronous loss:** compute LensNet loss on a background stream / secondary device and update weights less frequently (requires careful optimizer handling).

## Debug checkpoints
- After vectorization, log `mc/lens_loss_ms`, `mc/lens_forward_ms`, `mc/variants_total` at step 1 to confirm the drop.
- Track peak memory; batching should also reduce per-step allocations by avoiding many tiny tensors.

If we execute the vectorized loss and optional pair sampling, `lens_loss_ms` should fall from ~2.9 s to the low-ms range, bringing `tok/sec` back toward the expected ~3× slowdown relative to baseline.***
