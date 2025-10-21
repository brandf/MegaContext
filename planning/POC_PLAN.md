# MegaContext Proof of Concept (POC) Plan

This milestone isolates the minimum “hot path” required to demonstrate MegaContext end-to-end on a single base model. The goal is to show that hierarchical gists plus dynamic focus can keep a frozen LLM within a fixed working window while retaining task-relevant history. We aim for clarity, determinism, and reproducibility rather than exhaustive optimization.

## Phase 0 — Repository Readiness
**Goal:** Ensure contributors can reproduce the prototype quickly.
- Task 0.1: Verify `uv` bootstrap script provisions the environment, installs dependencies, and documents canonical commands (`uv venv`, `uv run pytest`, `uv run python -m …`).
- Task 0.2: Refresh `README.md` to reference the POC plan, describe prerequisites (Python 3.11, GPU requirements), and outline the expected proof demo.
- Task 0.3: Confirm smoke tests for dataset tooling and base model stubs run in CI.

## Phase 1 — Base Runtime Skeleton
**Goal:** Stand up the frozen base-model runtime and data pipelines used later in the loop.
- Task 1.1: Finalize `tools/prepare_dataset.py` and associated configs to produce deterministic 32-token blocks from a toy corpus (e.g., project docs).
- Task 1.2: Implement `src/runtime/base_model.py` targeting `HuggingFaceTB/SmolLM3-3B` (bf16) with forward helpers used across the prototype.
- Task 1.3: Provide `src/runtime/working_context.py` (token pass-through version) and unit tests covering tensor shapes, legality masks, and budget calculations.
- Task 1.4: Add a CLI demo (`uv run python -m tools.decode_demo --config configs/runs/base_llm.yaml`) that streams a short prompt to confirm logits generation.

**Exit criteria:** Dataset prep works on a sample corpus, base model forwards succeed, smoke tests pass under CI.

## Phase 2 — Minimal Gist Compression
**Goal:** Train and validate a single-level (32→1) gist model sufficient to replace segments without catastrophic degradation.
- **Design directive:** keep gist-side components tensor-first. Prefer thin Python wrappers around PyTorch modules and persist MegaContext structures as contiguous L0/L1/L2 tensors that mirror on-disk layouts instead of dense Python object graphs.
- Task 2.1: Implement `src/gistnet/blocks.py` and `src/gistnet/model.py` with RoPE-enabled self-attention, shared slot queries, and residual MLPs outputting the base embedding dimension.
- Task 2.2: Extend dataset tooling to emit paired `(tokens, gist_tokens)` batches over horizon `H=64`; cache teacher embeddings for repeatability.
- Task 2.3: Build `tools/train_gistnet.py` with a masked-attention curriculum (per Gist Token paper) and W&B logging of ΔNLL@H.
- Task 2.4: Add unit tests for determinism (seeded RNG) plus a smoke eval comparing base vs gist-replaced loss with ≤5 % degradation on the toy corpus.
- Task 2.5: Document gist training steps and expected metrics in `notebooks/gistnet.ipynb` (Colab-ready guide).
- Task 2.6: Revise dataset preparation to output 4k-token context slices with cached teacher hidden states and future horizons, along with stride/layout metadata described in the schema sketch (gists/logits generated later during training).
- Task 2.7: Introduce `MegaContext`/`WorkingContext` tensor wrappers that hydrate from dataset slices plus the current GistNet, manage contiguous L0/L1/L2 buffers and offsets, and provide iterators for enumerating all legal `W_l` windows and gist substitution patterns.
- Task 2.8: Upgrade the training loop to batch working-context windows, run prediction through the base model (replaying cached teacher hidden states/logits where applicable), and optimize ΔNLL/logit agreement with a curriculum that progressively shrinks `W_l`.

*Progress (current): core GistNet modules, dataset tooling with teacher caches, trainer scaffold, and notebook documentation are implemented; outstanding work covers curriculum training, ΔNLL smoke evals, and logging.*

**Upcoming extensions under evaluation:**
- Treat the current pooled hidden-state regression as a stepping stone. Long-term we want to phase it out in favour of training directly on prediction fidelity.
- Expand dataset prep to emit 4k-token contexts plus 32–64 token horizons, caching the teacher’s final hidden states so later stages can reconstruct rollouts and ΔNLL without re-running the model.
- Proposed schema sketch: store `context_tokens` (4k L0 ids), `context_mask`, cached teacher hidden stacks (one per block), and `future_tokens` for the horizon, alongside metadata that captures tokenizer, stride, and window offsets. Training will materialize MegaContext hierarchy on the fly with the latest GistNet checkpoints, so no gists/logits are persisted in the dataset.
- Generate hierarchical L1/L2 gists for each slice at training time so experiments always reflect the current compressor.
- Construct batched working-context windows of width `W_l` by sliding across each prepared MegaContext (optionally ordered for KV-cache reuse), run the base model forward with those contexts, and compare the horizon rollout in latent/logit space to the teacher outputs to maximize gist substitutability.
- Prototype a curriculum that gradually shrinks `W_l` during training to push the model toward using higher-level, lossy summaries.
- Introduce lightweight `MegaContext`/`WorkingContext` tensor wrappers that own contiguous L0/L1/L2 buffers, expose combinator utilities (e.g., enumerate all legal `W_l`-sized windows, replace spans with specific gist levels), and surface batching hooks the trainer can call without hand-rolling slicing logic. The `MegaContext` helper should encapsulate offsets/parent pointers and emit iterator handles (or precomputed index tensors) that the trainer can batch together; `WorkingContext` should provide views for token embeddings vs gist embeddings, plus utilities to materialize KV-cache keys for a chosen slice.

**Exit criteria:** Gist checkpoints reproduce ΔNLL targets, deterministic tests pass, and documentation explains the compression pipeline.

## Phase 3 — LensNet, Focus Allocator, and Runtime Loop
**Goal:** Integrate MegaContext storage with dynamic focus so a streaming run respects a fixed working budget.
- Task 3.1: Implement `src/megacontext/memory/tree.py` with ingest/update APIs for 32-token blocks, node metadata (span id, level, offsets), and persistence to `{L0,L1,L2}.ctx`; include round-trip tests.
- Task 3.2: Finish `src/runtime/working_context.py` to tile L0 tokens and L1 gists contiguously, reporting token-equivalent costs.
- Task 3.3: Build `src/lensnet/model.py` with dual cross-attention (working entries ↔ gist cache) and a signed focus score head; create `src/lensnet/dataloader.py` to replay working-context snapshots.
- Task 3.4: Implement `src/runtime/focus_allocator.py` with greedy expand/collapse respecting contiguity, cooldowns, and budget hysteresis. Include Slot-Attention-style normalised competition to keep allocation bounded.
- Task 3.5: Assemble `src/runtime/engine.py` that ingests streams, updates the gist tree, queries LensNet, applies focus adjustments, and decodes via the base model.
- Task 3.6: Provide unit tests across components (tree ingest, allocator edge cases, LensNet mask handling) plus an integration test with a deterministic synthetic stream.
- Task 3.7: Ship `uv run python -m tools.run_poc_loop --config configs/runs/poc_smollm3.yaml` demonstrating a short session that expands/collapses spans while respecting the budget.

**Exit criteria:** End-to-end loop runs on the demo corpus, logs focus actions, maintains budget invariants, and tests cover core behavior.

## Phase 4 — Proof Demo & Documentation
**Goal:** Capture evidence that MegaContext works and explain the prototype clearly.
- Task 4.1: Create a minimal benchmark script comparing (a) baseline LLM, (b) MegaContext-enabled run on a single synthetic task; report loss, swap rate, and latency.
- Task 4.2: Record a short walkthrough (structured log or screenshots) showing focus reallocations in the demo run.
- Task 4.3: Update `README.md` with a POC summary: architecture sketch, command sequence, expected outcomes, troubleshooting.
- Task 4.4: Ensure `docs/poc_results.md` captures metrics, figures, and lessons learned leading into the next milestone.

**Exit criteria:** Demo artifacts prove that MegaContext can retain context beyond the working window with manageable complexity, unlocking the PAPER milestone.
