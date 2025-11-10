---
tags:
  - review
summary: Actionable “hard” tasks extracted from the 2025-11-10 MC review; sized for codex-high with detailed implementation plans and tests.
---

# MegaContext — Hard Action Items

These are deeper changes with meaningful behavior or API implications. Each item includes design notes and acceptance tests.

## A) Apply MC RoPE for batched training (B > 1)
- Problem: `scripts/base_train.py` only uses `mc_result.positional_cache` when `B == 1`. For `B > 1` the model uses default rotary embeddings, losing global MC positions — likely degrading MC training.
- Goal: Build per-sample `(cos, sin, alibi)` tensors for the whole batch and pass them as overrides to the model, keeping a single fused forward.
- Design:
  - `MCController.process_batch(...)` already returns `positional_caches: Dict[str, Tuple[cos, sin, alibi]]` keyed by `session_id = f"train_step_{step}_sample_{idx}"`.
  - In `scripts/base_train.py`, reconstruct batch-ordered caches by iterating `idx in range(B)` and building the expected session ids. Stack each `[1, T, 1, H/2]` cos/sin into `[B, T, 1, H/2]`. If ALiBi exists, stack `[1, H, T, T]` into `[B, H, T, T]`.
  - Pass `(cos, sin)` as `cos_sin_override` and stacked `alibi` as `alibi_override` to the model forward.
- Implementation steps:
  1) Edit `scripts/base_train.py:352`–`scripts/base_train.py:416` micro-step block.
  2) After `mc_result` is not `None`, compute:
     - `B = x.size(0); T = x.size(1)`
     - Build lists `cos_list, sin_list, alibi_list` from `mc_result.positional_caches[f"train_step_{step}_sample_{idx}"]` for `idx in range(B)`.
     - Stack to `cos = torch.cat([c.to(device) for c in cos_list], dim=0)` (shape `[B, T, 1, H/2]`), same for `sin`.
     - If any `alibi` is not `None`, set missing ones to zeros of shape `[1, num_heads, T, T]` and stack to `[B, num_heads, T, T]`.
  3) Use these in the existing forward path:
     - `cos_sin_override = (cos, sin)`
     - `alibi_override = alibi if built else None`
  4) Keep single-sample fast-path unchanged.
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

## B) Padding-aware TransformerGistNet (masked attention over padded tokens)
- Problem: In `MegaContextTree._build_higher_levels`, we zero out padded tokens but TransformerGistNet has no key padding mask; attention can still attend to padded positions.
- Goal: Thread a key padding mask into GistNet and convert it to a bias mask compatible with nanochat’s Block/attention.
- Design:
  - Update `TransformerGistNet.forward(self, blocks: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None)`.
  - Convert `key_padding_mask` of shape `[B, T]` to an additive bias of shape `[B, 1, T, T]` where masked key positions receive `-inf` and pass as `alibi` to `Block(...)` (sum with any future bias if needed).
  - Keep PoolingOnlyGistNet unchanged.
- Implementation steps:
  1) In `mc/gistnet.py`, change `TransformerGistNet.forward` signature to accept `key_padding_mask`.
  2) Build `bias = None`. If mask is provided, create `float_mask = (~mask).float() * -1e9` and broadcast to `[B, 1, T, T]` (keys masked across all queries).
  3) Pass `alibi=bias` into `block(x, cos_sin, kv_cache=None, alibi=bias)`.
  4) In `mc/mega_context.py:92`–`mc/mega_context.py:99`, when `gistnet is not None`, compute the per-block mask from `mask.view(-1, block_size, 1).squeeze(-1)` and pass to `gistnet(flat, key_padding_mask=mask_2d)`.
  5) Preserve runtime compatibility by keeping default `key_padding_mask=None` in all callsites outside the tree build.
- Tests:
  - New `tests/test_gistnet_padding.py`:
    - Create an embeddings tensor with 3 valid rows and 1 padded row in a 4-token block.
    - Build tree with `block_size=4`, `TransformerGistNet` of 1 layer. Call internal `_reshape_for_pool` to get block + mask and feed GistNet directly with and without mask; assert that with mask, the output is closer (cosine similarity) to the summary that ignores the padded row.
- Acceptance:
  - GistNet accepts mask; no regressions for mean-only head.
  - Test demonstrates padding doesn’t influence attention when masked.

## C) MC LOD0 cache policy (configurable memory tradeoff)
- Problem: LOD0 cache can grow unbounded; need an opt-out or auto-release policy.
- Goal: Provide a simple knob to disable caching and a future hook for auto-release.
- Design:
  - Add `cache_lod0: bool = True` to `MCConfig`.
  - After building the tree (both train and inference paths), apply policy: if `False`, call `release_lod0_cache(disable_future_cache=True)`.
- Implementation steps:
  1) Update `mc/config.py` (dataclass and `__post_init__` no change required besides storing the flag).
  2) In `mc/runtime.py:203` (train sample) and `mc/runtime.py:1034` (inference init), apply the policy immediately after `build_mega_context`.
  3) Keep defaults to True to preserve current behavior.
- Tests:
  - Extend `tests/test_mc_append_cache.py` to initialize controller with `cache_lod0=False` and assert `tree._lod0_cache is None` post-build and after appends.
- Acceptance:
  - Flag works for both training and inference paths.

## D) Minimal instrumentation of MC time/memory in training loop
- Goal: Add coarse timings for MC phases to `nanochat.report` for visibility.
- Implementation steps:
  1) In `scripts/base_train.py`, around calls to `mc_controller.process_batch`, capture wall-clock times for:
     - tree/variants build, horizon loss pass, and lens loss.
  2) Aggregate simple moving averages and log every 100 steps via existing `wandb_run.log` dictionary (prefix `mc/time_*`).
- Tests: none (timing only); ensure fields appear in logs with sane values on a quick CPU run.
- Acceptance:
  - Report includes `mc/time_build`, `mc/time_horizon`, `mc/time_lens`.

## E) Optional: Add a “stochastic greedy” allocator variant for ablations
- Goal: Introduce a minor policy variant to compare stability/quality (sample among top-k |scores| instead of pure argmax).
- Implementation steps:
  1) Add `allocator_type == "stochastic_greedy"` case in `build_focus_allocator`.
  2) Subclass `GreedyFocusAllocator` with an overridden `_select_edits` that samples up to `k` candidates using a temperature over |scores|.
  3) Thread `k` and `temperature` via `FocusAllocatorConfig` (optional; give defaults).
- Tests:
  - Unit test ensures that with non-degenerate scores, selected indices vary across seeds.
- Acceptance:
  - Variant selectable via `scripts/base_train.py --allocator stochastic_greedy`.

---
Owner handoff: codex-high should implement A and B first (they’re highest impact for training quality), then C and D. E is optional for ablation breadth.

