---
tags:
  - lensnet
  - plan
summary: Simplify LensNet training by sampling random working-context variants and supervising LensNet with pairwise ranking losses.
---

# LensNet Simplification Plan

## Goal
Replace the existing LensNet training loop (allocator-driven variants + absolute targets vs. a “best” WC) with a simpler, more debuggable pipeline:

1. **Variant sampling:** For every training sequence, build one pure LOD0 baseline WC plus `N` *random* focused variants by compressing the MegaContext tree down to a fixed `train_wc_length`. No LensNet involvement—variants are purely stochastic augmentations.
2. **Next-token measurement:** Run the standard next-token loss for every WC variant (already needed for GistNet/backbone training).
3. **Pairwise ranking loss:** For each pair of variants from the same sequence, compare their losses and teach LensNet to prefer the lower-loss WC by aligning its per-entry scores with the relative focus decisions (expand vs. collapse) between the two variants.
4. **Telemetry & tests:** Keep the ASCII WC visualization and LensNet debug logs, but update them to reflect the new sampling + ranking pipeline. Achieve 10/10 test coverage for the new behavior.

This eliminates the allocator/LensNet feedback loop, reduces implicit bias in the sampling procedure, and grounds the supervision directly in observed Δloss values.

## Prioritized Checklist

1. [x] **Config + CLI plumbing** — surfaced `train_wc_length`, `num_random_variants`, `random_variant_iterations`, `max_lens_pairs`, and `lens_collapse_weight` through `MCConfig`, `scripts/base_train.py`, and `run10.sh`.
2. [x] **Random WC sampler** — `_build_random_variant_set` now clones the LOD0 baseline and stochastically collapses it to the requested training length with deterministic RNG seeds; coverage/tail invariants are enforced via `_ensure_wc_full_coverage`.
3. [x] **Random-only `_build_sample_context`** — when `num_random_variants > 0`, training batches use `[lod_0_baseline + N random variants]` and skip the legacy allocator/LensNet refinement.
4. [x] **Pairwise LensNet supervision** — `_build_preference_pairs` produces deterministic pairings, `_build_pairwise_targets` consumes raw Δloss, and `_compute_lens_losses` applies per-entry regression + optional rank/budget penalties while caching policy scores for telemetry.
5. [x] **Regression tests** — added unit tests covering the random sampler (length/purity/determinism) and a LensNet pairwise-loss gradient smoke test.
6. [ ] **Telemetry + CLI polish** — keep ASCII/LensDebug logging wired to the new sampler, expose the remaining knobs in `run10.sh`/`mc_run.sh`, and stream pairwise stats (`mc/preference_corr_*`, `mc/adv_delta_*`) to WandB.
7. [ ] **Documentation refresh** — rewrite the LensNet PRD + architecture notes plus the LensNet Review checklist to describe the random-variant + pairwise-ranking pipeline and its invariants.
8. [ ] **Integration validation** — run targeted train slices (with `--mc_log_lens_debug 1`) to confirm `mc/preference_corr_mean` stays negative once LensNet converges and capture updated ASCII artifacts for the doc set.

## Execution Notes
- Keep backward compatibility toggles until the new pipeline is stable (e.g., a `--mc_use_random_variants` flag).
- While developing, default to the new path when `num_random_variants > 0`; fall back otherwise.
- Remove obsolete tests/telemetry once the new path is fully adopted.
