---
tags:
  - review
summary: Actionable “hard” tasks extracted from the 2025-11-10 MC review; sized for codex-high with detailed implementation plans and tests.
---

# MegaContext — Hard Action Items

These are deeper changes with meaningful behavior or API implications. Each item includes design notes and acceptance tests.

## A) Apply MC RoPE for batched training (B > 1) — Implemented
- Problem: `scripts/base_train.py` only uses `mc_result.positional_cache` when `B == 1`. For `B > 1` the model uses default rotary embeddings, losing global MC positions — likely degrading MC training.
- Goal: Build per-sample `(cos, sin, alibi)` tensors for the whole batch and pass them as overrides to the model, keeping a single fused forward.
- Design:
  - `MCController.process_batch(...)` already returns `positional_caches: Dict[str, Tuple[cos, sin, alibi]]` keyed by `session_id = f"train_step_{step}_sample_{idx}"`.
  - In `scripts/base_train.py`, reconstruct batch-ordered caches by iterating `idx in range(B)` and building the expected session ids. Stack each `[1, T, 1, H/2]` cos/sin into `[B, T, 1, H/2]`. If ALiBi exists, stack `[1, H, T, T]` into `[B, H, T, T]`.
  - Pass `(cos, sin)` as `cos_sin_override` and stacked `alibi` as `alibi_override` to the model forward.
- Implemented in `scripts/base_train.py`:
  - Assembles per-sample caches from `mc_result.positional_caches` into batched `(cos, sin, alibi)` when `positional_cache` is `None`.
  - Fills missing ALiBi per sample with zeros of shape `[1, H, T, T]` before concatenation.
  - Falls back to default RoPE if assembly fails, to avoid interrupting training.
- Tests:
  - Added `tests/test_mc_batched_rope.py` to verify shapes and a successful forward with overrides and cached embeddings.
- Acceptance: Batched training now applies MC-conditioned RoPE for all samples.
- Tests:
  - New `tests/test_mc_batched_rope.py`:
    - Build a dummy model as in existing tests.
    - Create a controller, craft a batch `tokens = torch.randint(..., (2, 8))`.
    - Call `process_batch`; assemble the batched `(cos, sin, alibi)` as above and run a forward with `inputs_embeds=mc_result.cached_embeddings` and overrides.
    - Assert no exception and that cos/sin have batch dimension 2 and correct shapes.
  - Optional regression: monkeypatch `nanochat.gpt.apply_rotary_emb` to record the max absolute difference between rotating with overrides vs. default cos/sin; verify they differ when positions differ per-sample.
- Acceptance:
  - Batch training applies MC RoPE for B>1.
  - Single-sample behavior unchanged; tests pass on CPU.

## B) Padding-aware TransformerGistNet (masked attention over padded tokens) — Implemented
- Problem: In `MegaContextTree._build_higher_levels`, we zero out padded tokens but TransformerGistNet has no key padding mask; attention can still attend to padded positions.
- Goal: Thread a key padding mask into GistNet and convert it to a bias mask compatible with nanochat’s Block/attention.
- Design:
  - Update `TransformerGistNet.forward(self, blocks: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None)`.
  - Convert `key_padding_mask` of shape `[B, T]` to an additive bias of shape `[B, 1, T, T]` where masked key positions receive `-inf` and pass as `alibi` to `Block(...)` (sum with any future bias if needed).
  - Keep PoolingOnlyGistNet unchanged.
- Implemented:
  - `mc/gistnet.py`: `TransformerGistNet.forward(..., key_padding_mask=None)` builds an additive bias masking padded keys across all queries; supports optional CLS.
  - `mc/mega_context.py`: pass per-block masks when building higher LODs and when recomputing single nodes.
- Tests:
  - Added `tests/test_gistnet_padding.py`: verifies masked outputs are closer to a reference that ignores padded rows than unmasked outputs.
- Acceptance: GistNet respects padding; mean-only head remains unchanged.
- Tests:
  - New `tests/test_gistnet_padding.py`:
    - Create an embeddings tensor with 3 valid rows and 1 padded row in a 4-token block.
    - Build tree with `block_size=4`, `TransformerGistNet` of 1 layer. Call internal `_reshape_for_pool` to get block + mask and feed GistNet directly with and without mask; assert that with mask, the output is closer (cosine similarity) to the summary that ignores the padded row.
- Acceptance:
  - GistNet accepts mask; no regressions for mean-only head.
  - Test demonstrates padding doesn’t influence attention when masked.

## C) MC LOD0 cache policy (configurable memory tradeoff) — Implemented
- Problem: LOD0 cache can grow unbounded; need an opt-out or auto-release policy.
- Goal: Provide a simple knob to disable caching and a future hook for auto-release.
- Design:
  - Add `cache_lod0: bool = True` to `MCConfig`.
  - After building the tree (both train and inference paths), apply policy: if `False`, call `release_lod0_cache(disable_future_cache=True)`.
- Implemented:
  - `mc/config.py`: added `cache_lod0: bool = True`.
  - `mc/runtime.py`: call `tree.release_lod0_cache(True)` after tree build when disabled (train and inference).
- Tests:
  - Added `tests/test_mc_cache_policy.py`: asserts `_lod0_cache is None` after `begin_inference_session` under `cache_lod0=False`.
- Acceptance: flag works in both paths with default behavior preserved.
- Tests:
  - Extend `tests/test_mc_append_cache.py` to initialize controller with `cache_lod0=False` and assert `tree._lod0_cache is None` post-build and after appends.
- Acceptance:
  - Flag works for both training and inference paths.

## D) MC timing instrumentation — Implemented
- Logging:
  - `scripts/base_train.py` now tracks `mc/time_controller_ms` (average over the last 100 steps) and clears the samples each logging window.
- Controller telemetry:
  - `mc/runtime.py` emits `mc_timing` spans with `build_ms`, `positional_ms`, `horizon_ms`, `lens_ms`, plus finer-grained `horizon_forward_ms`/`horizon_loss_ms` accumulators.
- Docs:
  - README and [[Training & Operations]] mention the new WANDB gauge and OTEL fields so operators know where to look.

## E) Optional: Add a “stochastic greedy” allocator variant for ablations — Implemented
- Goal: Introduce a minor policy variant to compare stability/quality (sample among top-k |scores| instead of pure argmax).
- Implemented:
  - `mc/focus_allocator.py`: added `StochasticGreedyFocusAllocator` and `allocator_type == "stochastic_greedy"` in the factory. Added optional `sample_top_k`, `sample_temperature` to `FocusAllocatorConfig` with defaults.
  - Preserves protections, thresholds, and soft-length behavior while sampling among top-|score| candidates.
- Tests:
  - Added `tests/test_allocator_stochastic.py`: builds the allocator and ensures update runs with supplied scores.
- Acceptance: Policy selectable via `--allocator stochastic_greedy`.

---
Owner handoff: codex-high should implement A and B first (they’re highest impact for training quality), then C and D. E is optional for ablation breadth.
