---
tags:
  - review
summary: Comprehensive code and docs review of MegaContext (MC) and its integration with nanochat, with prioritized fixes, risks, and a plan to reach a 10/10 research-quality project.
---

# MegaContext x nanochat — Review and Recommendations (2025-11-10)

Scope: MC core (tree, working context, gist/lens, allocator, positional), the MC runtime/controller, and integration points in training/inference scripts. Perspective: a new AI researcher onboarding to run e2e training and ablations locally (CPU/mocked) and on GPUs.

## Summary
- The MC implementation is clean, modular, and test-backed. The integration path into nanochat is pragmatic and minimally invasive: MC is a sidecar that produces positional caches, auxiliary losses (LOD1/LOD2/gist regression, lens supervision), and cached embeddings for reuse.
- The training script integration is mature for single-batch positional override and loss mixing, with thoughtful telemetry hooks (OpenTelemetry) and reproducibility knobs.
- A few correctness/performance edge cases and polish opportunities remain, especially around positional variants, GistNet padding semantics, and making ablation workflows smoother for tomorrow’s runs.

## High‑Priority Issues (fix before long runs)
- Positional “LOD 2D” compatibility
  - Risk: `gaussian_lod2d*` doubles cos/sin channels in `GaussianRoPE` but `nanochat.gpt.apply_rotary_emb` expects cos/sin with head_dim/2. Using 2D variants will mis-shape rotary embedding and likely error or silently mis-apply.
  - Where: `mc/gaussian_rope.py:64`–`mc/gaussian_rope.py:70` and registry keys `gaussian_lod2d*` (`mc/gaussian_rope.py:78`).
  - Recommendation: Either (a) gate 2D modes with a clear ValueError in `MCController` unless model supports a 2D rotary path, or (b) implement/model-side support for 2D RoPE. For tomorrow: disable 2D modes; document that only `simple|gaussian|gaussian_alibi` are compatible with current GPT rotary.

- GistNet and padding semantics
  - Risk: When building higher LODs, padded tokens are zeroed but Transformer-based GistNet still attends to them (no attention mask), potentially biasing summaries in short final blocks.
  - Where: masked blocks in `MegaContextTree._build_higher_levels` (`mc/mega_context.py:92`–`mc/mega_context.py:99`), GistNet forward lacks key_padding_mask.
  - Recommendation: Extend `TransformerGistNet` to accept a key padding mask (per-block) and thread it into `Block(..., attn_mask=...)` as a boolean mask → float bias. For tomorrow: keep current mean/MLP head ablations on short context; note padding caveat in ablation notes.

- Multi-sample positional caches not used by base_train
  - Observation: `MCController.process_batch` returns `positional_caches` per session, but `scripts/base_train.py` only uses `positional_cache` when batch size == 1. Multi-sample training falls back to vanilla model RoPE, not MC-conditioned RoPE.
  - Where: `mc/runtime.py:182`–`mc/runtime.py:194` and `scripts/base_train.py:352`–`scripts/base_train.py:416`.
  - Recommendation: This is reasonable for now (shared caches can’t represent per-sample positions/LODs). Call this out in docs. If we want per-sample RoPE during training, we’d have to loop per-sample or restructure model forward.

## Correctness and API Review
- Tree and incremental updates
  - Hierarchy build and incremental recompute are coherent; single-node recompute makes gist updates cheap after appends (`mc/mega_context.py:173`–`mc/mega_context.py:181`).
  - LOD0 cache is used for slicing; there’s a manual `release_lod0_cache` path. Consider adding a config toggle to disable LOD0 caching for very long sequences to bound memory.

- WorkingContext operations
  - Append/replace maintain positions and LODs correctly and invalidate positional caches (`mc/working_context.py`). Position stride handling for collapse/expand looks consistent.
  - Residency tracking in allocator is well integrated; after replace, residency array is rebuilt with zeros for the injected span (`mc/focus_allocator.py:129`–`mc/focus_allocator.py:147`).

- Focus allocator policy (Greedy)
  - Expand/collapse checks protect recent tokens, enforce block alignment, and verify positional continuity (`mc/focus_allocator.py:211`–`mc/focus_allocator.py:254`). Thresholds and “prefer_collapse” when over budget make behavior predictable.

- Losses and projections
  - LOD1/LOD2 losses compute gist regression on teacher-forced horizons; top‑k probability projection is renormalized and covered by tests (`tests/test_projection_topk_norm.py`).
  - Token horizon loss uses `ignore_index=-1` padding; logits are from a single forward over `[WC + horizon]` via `inputs_embeds`, with MC RoPE stitched for the horizon (`mc/runtime.py:640`–`mc/runtime.py:720`).

- Positional encoding and ALiBi
  - Gaussian decay by LOD, optional ALiBi slopes; ALiBi bias is derived from WC positions (square bias tensor) and shaped as expected by nanochat attention (`mc/runtime.py:258`–`mc/runtime.py:268`, `nanochat/gpt.py:91`–`nanochat/gpt.py:106`).

- Telemetry
  - OTLP provider is optional and safe; event schemas are structured and include allocator utilization + residency stats (`mc/runtime.py:851`–`mc/runtime.py:880`). Preflight script is handy (`scripts/mc_otel_preflight.py`).

## Performance Review
- Sequential per-sample build in training
  - Intentional for determinism and to avoid GPU stream contention. Acceptable for now; consider a bounded worker pool later (config has `build_workers` but it isn’t used yet).

- Recompute footprint
  - Hierarchical recompute after appends is O(1) per LOD; good. GistNet compute dominates; padding-aware attention would improve quality without extra FLOPs.

- Memory
  - LOD0 cache can grow large; provide a config to disable or auto-release past thresholds (e.g., after horizon evaluation) to bound memory in long contexts.

## Testing Review (CPU/mocked focus)
- Strengths
  - Unit tests cover tree shape invariants, allocator edits, positional selection, telemetry fields, projection renorm, and inference facade.
  - Smoke tests ensure MC controller returns cached embeddings and positional artifacts and does not regress on interface.

- Gaps to consider
  - Positional variants: add tests that assert 2D LOD modes raise a clear error with the current GPT rotary path.
  - GistNet padding: a small test verifying that padding rows don’t influence pooled summaries once masked attention is implemented.
  - LOD0 cache lifecycle: tests for `release_lod0_cache()` effect on slices and recompute.
  - Long horizon LOD2 path: a CPU test that fabricates enough tokens to exercise LOD2 trimming logic (already partly covered, but an explicit unit would help).

## Documentation and Consistency
- The README and Obsidian notes are strong. Add:
  - A warning on `gaussian_lod2d*` positional modes’ incompatibility with current GPT rotary.
  - A note that per-sample positional caches are returned but not applied in batch training (design limitation), with suggested approaches if needed.
  - Brief mention of LOD0 cache memory tradeoff and how to disable it when memory is scarce.

## Minor Cleanups
- Remove or wire up unused imports/knobs:
  - `ThreadPoolExecutor` in `mc/runtime.py:4` and `build_workers` in `MCConfig` aren’t used.
  - Consider asserting `embed_dim % num_heads == 0` once in `MCController.__init__` for symmetry with LensNet/GistNet assertions.

## Feature Suggestions (next iterations)
- Padding-aware TransformerGistNet
  - Thread a key padding mask to `Block` inside `TransformerGistNet` or create a minimal attention helper that accepts boolean masks → bias.

- Configurable LOD0 caching policy
  - `MCConfig`: `cache_lod0: bool = True` and optionally `cache_lod0_max_tokens` to auto-release.

- Optional “Mean-only” gist path during training
  - For long contexts or small GPUs, enable cheap pooling (mean/linear) for LOD build; swap to transformer head in evaluation.

- Richer allocator policies
  - Add a “stochastic greedy” or “bandit” variant for ablations; seed-controlled for reproducibility.

- Better ablation ergonomics
  - Promote `run10.sh` flags into a small YAML config (or keep CLI but document a matrix for standard ablations: block_size, gist head/pooling, lens depth, thresholds).

## Plan to 10/10 Project Quality
1) Guardrails and docs (today)
   - Assert/disable `gaussian_lod2d*` in `MCController` with a clear message.
   - README/Obsidian: call out per-sample positional limitation and LOD0 cache memory tradeoff.

2) Quality of signal (week)
   - Implement padding-aware GistNet attention; add CPU test that verifies invariance to padded trailing tokens.
   - Add an option to use mean-only GistNet during training; compare against transformer head on validation LOD losses.

3) Reliability and profiling (week)
   - Add tests for `release_lod0_cache()` and long-horizon LOD2 trimming.
   - Basic timing/peak memory logging for MC paths (build tree, lens pass, allocator) into the existing `get_report()` log.

4) Experiment workflow (week)
   - Provide ablation presets in `run10.sh` (or a simple yaml) for: block_size ∈ {16, 32, 64}, gist head ∈ {linear, mlp}, gist pooling ∈ {mean, cls, query}, lens layers ∈ {2, 4}, thresholds ∈ {0.05, 0.1, 0.2}.
   - Add an “MC summary” to the final report: ΔNLL@H, swap rate, utilization, gist loss, and wall-clock.

## Integration Notes (nanochat)
- Training
  - Inputs: `MCController.process_batch()` expects token tensors `[B, T]` and model exposing `.transformer.wte` for embeddings.
  - Outputs: `cached_embeddings`, single-sample `positional_cache`, and per-sample `positional_caches` map for advanced usage.
  - Limitations: Batch-size>1 doesn’t apply MC RoPE by default; this is by design given a single shared `(cos, sin)` in model fwd.

- Inference
  - Facade: `begin_inference_session` + `inference_step` mirror training sampling, with allocator rebuild + updates and telemetry snapshots.

## Reference Pointers
- Gist build with zeroed padding but no attention mask: `mc/mega_context.py:92`–`mc/mega_context.py:99`.
- 2D positional channel doubling: `mc/gaussian_rope.py:64`–`mc/gaussian_rope.py:70`.
- Per-sample positional cache map creation: `mc/runtime.py:239`–`mc/runtime.py:256`.
- LensNet/Loss projection to embeddings: `mc/runtime.py:781`–`mc/runtime.py:818`.
- Greedy allocator edit checks: `mc/focus_allocator.py:211`–`mc/focus_allocator.py:254`.

## TL;DR for Tomorrow’s Runs
- Use positional `gaussian` or `gaussian_alibi` only.
- Keep `gistnet_pooling=mean` for cheap ablations; note padding caveat.
- Start with `block_size=32`, `allocator_recent_tokens=128`, `expand/collapse=0.1`, `allocator_iterations=2`.
- Track `mc/*` metrics and the OTEL spans; compare curves normalized by tokens/time.

If you want, I can implement the guardrail for 2D positional modes and add a brief README note before you run.

