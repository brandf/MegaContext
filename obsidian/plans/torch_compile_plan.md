---
tags:
  - torch.compile
  - lensnet
  - gistnet
summary: Structured plan to make torch.compile reliable for LensNet/GistNet without disabling cudagraph capture.
---

# Torch.compile Stabilization Plan

## Goals
- Keep `torch.compile(mode="reduce-overhead")` enabled for LensNet (and eventually GistNet) without turning off Inductor’s cudagraph capture.
- Ensure compiled modules run reliably during training and inference by auditing tensor lifetimes and entry points.
- Add instrumentation and tests so future regressions are caught immediately.

## Plan

### 1. Instrument LensNet/GistNet usage
- Add lightweight counters/logging in `MCController` to record:
  - How many times LensNet/GistNet run per micro-step (training vs inference).
  - Shapes of the inputs.
  - Whether we re-enter LensNet within the same backward pass (e.g., inference allocator).
- Expose these metrics via debug telemetry so we can validate behavior quickly.

### 2. Audit tensor lifetimes
- Trace every place where LensNet outputs are stored beyond the compiled call:
  - `WorkingContextVariant.policy_scores`
  - Focus allocator cached scores
  - Preference/policy telemetry
- For each path, decide whether to:
  - Consume the data immediately inside the compiled graph.
  - Clone the tensor at the user boundary (single helper, no ad-hoc `.clone()` sprinkled everywhere).
  - Re-run LensNet explicitly so we never hold onto compiled outputs.

### 3. Restructure buffer management
- Introduce helper functions:
  - `_run_lensnet_batched(inputs) -> scores` that performs the cudagraph step mark + clone once.
  - `_lensnet_allocator_scores(wc)` for inference allocator.
- Ensure these helpers manage clones and mark-step boundaries consistently so compiled graphs aren’t reused incorrectly.

### 4. Build a torch.compile harness
- Create a standalone script/test that:
  - Compiles LensNet with the actual config.
  - Runs the same batched call as training.
  - Runs the allocator-style repeated calls.
  - Checks for cudagraph errors.
- Run this harness before enabling compile in `mc_run`.
- ✅ Implemented in `scripts/mc_compile_harness.py` (`python scripts/mc_compile_harness.py --device cuda --enable-compile`) which exercises GistNet + both LensNet call paths and surfaces cudagraph issues.

### 5. Re-enable compile progressively
- Once the harness passes, re-enable compile for training (single batch) and gate inference allocator behind a config flag so we can roll out gradually.
- Update `mc_run.sh` to expose a `--mc_compile_lensnet_inference` flag if needed.

### 6. Documentation & tests
- Update `obsidian/reference/LensNet Pairwise Training.md` with the new invariants (“LensNet outputs must be consumed or cloned via the helper”).
- Add unit tests that:
  - Mock LensNet to ensure `_run_lensnet_batched` is used everywhere.
  - Simulate repeated calls (allocator) to confirm we call `cudagraph_mark_step_begin`.
- Verify `tests/test_mc_components.py` covers these cases.

## Execution checklist
1. [x] Add instrumentation/logging for LensNet/GistNet invocation counts.
2. [x] Implement `_run_lensnet_batched` and `_lensnet_allocator_scores` helpers with mark-step + clone.
3. [x] Refactor runtime/allocator to use the helpers exclusively.
4. [x] Build/run the torch.compile harness.
5. [ ] Update docs and unit tests.
6. [ ] Verify full pytest suite + harness + end-to-end training smoke test (or as close as feasible without full GPU run).
