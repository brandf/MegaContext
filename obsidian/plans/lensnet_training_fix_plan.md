# LensNet Training Alignment Plan

## Current Reality
* Training mode still routes through the allocator’s `update_focus`, which calls LensNet to score candidate edits. Even though we try to normalize lengths deterministically afterwards, the allocator intrinsically requests scores multiple times per batch.
* Random-variant normalization also calls `_collapse_wc_randomly`, which instantiates an allocator and therefore re-enters LensNet.
* The batched preference scoring path works as intended (single LensNet batch), but the extra allocator invocations violate the “no LensNet during training” requirement and trigger the cudagraph crashes we keep chasing.

## Target Behavior
1. **Baseline/variant construction**: Focus allocator must not use LensNet while training. Instead we follow the deterministic “gist substitution” procedure you described: repeatedly collapse random eligible spans (LOD2 with 10 % chance, else LOD1) until the working-context length hits `train_wc_length`.
2. **LensNet usage**: During training we only run a single LensNet forward per micro-step to derive policy scores for preference learning. Inference mode continues to use the existing LensNet-driven allocator.
3. **Assertions/tests**: Enforce the invariant with unit tests so regressions are caught immediately.

## Action Plan
1. **Split allocator paths**
   * Add a dedicated `TrainingWCVariationAllocator` (name TBD) that implements the deterministic random-collapse procedure: repeatedly collapse eligible spans (LOD2 with configurable probability when the target permits, else LOD1) until the working-context length exactly matches `train_wc_length`. This helper owns all training-time WC adjustments and never references LensNet.
   * Remove every training-time call to `FocusAllocator`. `_collapse_wc_randomly` and related helpers rely exclusively on the new training allocator. The legacy LensNet-driven `FocusAllocator` becomes inference-only and is instantiated solely when `context == "inference"`.
2. **Refactor `_collapse_wc_randomly`**
   * Remove allocator construction; directly apply the deterministic substitutions you specified (LOD2 with configurable probability, else LOD1).
   * Ensure coverage invariants/padding remain unchanged.
3. **Guard LensNet usage**
   * Track whether we are in training vs. inference in `MCController`, and assert that LensNet is never called through allocator helpers while in training context.
4. **Add regression tests**
   * Extend `tests/test_mc_components.py` (or add a new suite) to:
     - Build sample variants in training mode and assert LensNet wasn’t invoked (mock or counter).
     - Verify deterministic collapse hits the exact target length without LensNet.
   * Add a test ensuring inference mode still runs LensNet via allocator.
5. **Cleanup**
   * Remove `torch.compiler` workarounds once LensNet is no longer re-entered unpredictably during training.
   * Document the invariant in `LensNet Pairwise Training.md` so future contributors know LensNet is inference-only outside preference scoring.

## Milestones
1. Implement `TrainingWCVariationAllocator` + training guard (ETA: same day).
2. Update unit tests to cover the new invariant (ETA: same day after code change).
3. Re-enable LensNet compile without cudagraph hacks once tests confirm no extra invocations (ETA: immediately after step 2).
4. Run full training smoke test to validate single batched LensNet usage (ETA: following day or after GPU time slot).
