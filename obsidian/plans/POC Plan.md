# POC Plan

Source: `planning/POC_PLAN.md`. This note summarizes the minimum hot-path milestone that proves MegaContext end-to-end on a single frozen base model.

## Phase 0 — Repository readiness
- [ ] Task 0.1 — Verify `uv` bootstrap covers environment creation and canonical commands.
- [ ] Task 0.2 — Refresh `README.md` with prerequisites (Python 3.11, GPU) and demo outline.
- [ ] Task 0.3 — Ensure smoke tests for dataset tooling and base model stubs run in CI.

## Phase 1 — Base runtime skeleton
- [ ] Task 1.1 — Finalize `tools/prepare_dataset.py` to emit deterministic 32-token blocks.
- [ ] Task 1.2 — Implement `src/runtime/base_model.py` targeting `HuggingFaceTB/SmolLM3-3B` (bf16).
- [ ] Task 1.3 — Provide `src/runtime/working_context.py` (token pass-through) plus unit tests for masks and budgets.
- [ ] Task 1.4 — Add CLI demo `tools.decode_demo` to confirm logits generation.
- **Exit criteria:** Dataset prep works, base model forwards succeed, smoke tests pass.

## Phase 2 — Minimal gist compression
- [ ] Task 2.1 — Implement `src/gistnet/blocks.py` and `src/gistnet/model.py` with RoPE attention and residual MLPs.
- [ ] Task 2.2 — Extend dataset tooling for `(tokens, gist_tokens)` batches with cached teacher embeddings.
- [ ] Task 2.3 — Build `tools/train_gistnet.py` with masked-attention curriculum and W&B ΔNLL logging.
- [ ] Task 2.4 — Add determinism + smoke eval tests (≤5 % ΔNLL degradation).
- [ ] Task 2.5 — Document gist training flow in `notebooks/gistnet.ipynb`.
- [ ] Task 2.6 — Revise dataset prep to emit 4k-token context slices with teacher hidden states and metadata.
- [ ] Task 2.7 — Introduce `MegaContext`/`WorkingContext` tensor wrappers with iterators for legal windows.
- [ ] Task 2.8 — Upgrade training loop to batch working contexts, replay teacher caches, and optimize ΔNLL/logit agreement with shrinking budgets.
- **Progress note:** Core GistNet modules, dataset tooling with teacher caches, trainer scaffold, and notebook docs exist; curriculum training, ΔNLL smoke evals, and logging remain.
- **Exit criteria:** Gist checkpoints hit ΔNLL targets, deterministic tests pass, compression pipeline documented.

## Phase 3 — LensNet, focus allocator, runtime loop
- [ ] Task 3.1 — Implement `src/megacontext/memory/tree.py` with ingest/update APIs and persistence.
- [ ] Task 3.2 — Finalize `src/runtime/working_context.py` to tile L0/L1 entries with token costs.
- [ ] Task 3.3 — Build `src/lensnet/model.py` plus `src/lensnet/dataloader.py`.
- [ ] Task 3.4 — Implement `src/runtime/focus_allocator.py` with greedy expand/collapse and competition.
- [ ] Task 3.5 — Assemble `src/runtime/engine.py` knitting ingest, focus, and decode.
- [ ] Task 3.6 — Provide unit + integration tests (tree ingest, allocator edge cases, synthetic stream).
- [ ] Task 3.7 — Ship `tools.run_poc_loop` demo showing expansion/collapse within budget.
- **Exit criteria:** End-to-end loop runs on demo corpus with logged focus actions and budget invariants.

## Phase 4 — Proof demo & documentation
- [ ] Task 4.1 — Benchmark baseline LLM vs MegaContext (loss, swap rate, latency).
- [ ] Task 4.2 — Capture walkthrough of focus reallocations.
- [ ] Task 4.3 — Update `README.md` with POC summary, commands, troubleshooting.
- [ ] Task 4.4 — Populate `docs/poc_results.md` with metrics and lessons.
- **Exit criteria:** Demo artifacts show retained context beyond working window, enabling the paper milestone.

## Links
- [[Home]] — project entry point.
- [[Core Components]] — module references for implementation phases.
- [[Runtime Loop]] — runtime flow tied to Phases 3–4.
