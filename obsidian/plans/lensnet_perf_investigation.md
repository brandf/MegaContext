---
tags:
  - investigation
  - lensnet
summary: Why LensNet timings dwarf the base model and how to fix it.
---
# LensNet Performance Investigation

## Observations
- Training logs with `mc_log_timers=1` show `base_forward≈0.5 ms` vs `lens_forward≈1344 ms`. This contradicts the FLOP math: even with five working-context variants per sequence, the GPT trunk should dominate compute (~3.6 e15 FLOPs forward) while LensNet should sit in the low e13 range (≈160× smaller).
- The `lens_forward` timer is derived from `t_lens1 - t_lens0` in `MCController._process_batch_impl` (`mc/runtime.py:510-529`). That block wraps the **entire** `_compute_lens_losses` routine (preference-pair selection, per-span mask construction, KL terms, telemetry) rather than the raw LensNet batched call. So the numbers currently conflate model inference with extensive Python/CPU bookkeeping.
- Instrumentation shows `_batch_variant_scores` runs exactly once per micro-step and processes all variants in a single batched LensNet forward, so we are not repeatedly invoking LensNet itself.

## Root-Cause Suspicions
1. **Timer scope mismatch.** `lens_ms` measures `_compute_lens_losses` in full, not just the LensNet matmul. This includes preference-pair enumeration, histogram/coverage recomputation, KL updates, and multiple `.log_event` calls, so the metric is dominated by CPU code rather than GPU inference.
2. **Massive CPU/GPU synchronization in `_build_pairwise_targets`.**  
   - Every span iterates over `positions` and `lods` tensors (`mc/runtime.py:2233-2270`) that live on the GPU. The code calls `.item()` inside nested Python loops, forcing thousands of host/device syncs per variant and per preference pair.
   - With ~1.5k positions per working context, ~8 preference pairs per sample, and 160 variants per batch, these syncs alone can consume hundreds of milliseconds.
3. **Inefficient padding/copying.** `_stack_working_contexts`/`_batch_variant_scores` zero-initialize `[num_variants, max_len, embed_dim]` tensors (and matching cos/sin/alibi buffers) before every LensNet call. For 160 variants and `max_len≈1536`, that’s ~1.6 GB of data cleared and copied per step just to pack sequences, dwarfing the actual LensNet matmul cost.
4. **Additional Python-side histograms and coverage scans** (e.g., `_lod_histogram`, `_wc_token_coverage`) also loop over GPU tensors with `.item()`, compounding the sync overhead logged under `lens_ms`.

## Action Plan
1. **Split timers & telemetry** ✅
   - `mc/runtime.py` now records both `lens_forward_ms` (pure LensNet inference) and `lens_loss_ms` (end-to-end loss assembly) so we can spot whether GPU math or bookkeeping dominates. The legacy `lens_ms` key remains as an alias for compatibility.
2. **Remove `.item()` sync loops** ✅
   - All preference/mask helpers now move positional tensors to CPU once, eliminating thousands of `.item()` GPU syncs per step.
3. **Optimize variant packing** ✅
   - `_stack_working_contexts` avoids zero-filling gigabyte-sized buffers by using `torch.empty` plus per-variant slicing/zeroing, so we only touch the padding actually needed each step.
4. **Validate actual LensNet invocation count** ✅
   - Added automated tests (see `tests/test_mc_components.py::test_lensnet_timers_and_usage`) plus per-step telemetry to ensure we still execute a single batched LensNet call in training, and that `lens_loss_ms ≥ lens_forward_ms`.
5. **Revisit LensNet torch.compile**
   - Once the CPU-side overhead is under control, debug the compile failures separately (run `scripts/mc_compile_harness.py --enable-compile`). But that won’t fix the current 1300 ms issue until we address the CPU-side overhead above.

## Latest Telemetry (Nov 27 2025, evening)
- `lens_forward_ms` is **≈1.6–1.8 ms** (single batched call).
- `lens_loss_ms` dropped to **≈160–180 ms** after vectorizing the pairwise-target builder and removing `.item()` loops. Controller overhead is now within a few hundred microseconds of the GPU forward instead of dwarfing it.
- `mc_smoke_train.py` covers the full MCController path with torch.compile (cudagraphs disabled for the aux nets) so we can reproduce regressions without a full `run10.sh`.
- Per-variant metadata (positions/lods/spans) is cached on the CPU, so preference targets reuse those tensors instead of re-copying from GPU each time.
- Bradley–Terry losses are now computed in a single batched kernel: per-pair masked scores get concatenated and reduced via scatter-add on the GPU, cutting another ~40 ms from `lens_loss_ms`.
- Added a ΔNLL-aware GistNet penalty (`mc_gist_delta_weight`) so any variant whose loss exceeds the baseline gets an explicit gradient push toward parity; WANDB now exposes `mc/pref_span_corr`, the Pearson correlation between span-level LensNet scores and their target ΔNLL signs, so we can verify LensNet favors spans that hurt less.

### Next optimizations
1. **Batch Bradley–Terry loss on GPU:** targets/masks are now dense tensors; push ΔNLL arrays + weights through a batched GPU loss kernel so the remaining ~160 ms shrinks toward the 10–20 ms band.
2. **Histogram/coverage vectorization:** `_lod_histogram` and `_wc_token_coverage` still rebuild dictionaries via Python loops—rewrite these using tensor ops (`torch.unique`, `scatter_add`) to shave the remaining tens of milliseconds from controller bookkeeping.
3. **Re-enable cudagraphs for aux nets (stretch):** once the CPU work is sub‑20 ms and the loss path is batched, revisit torch.compile with cudagraph capture so GistNet/LensNet match the base model’s mode.
