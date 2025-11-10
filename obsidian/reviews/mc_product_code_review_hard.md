# MegaContext Review — Deep Work

## Recommended Pre-run Checks (High Difficulty)
- [x] Patch the LensNet loss path so gradients actually reach `lensnet` before running any MC-enabled sweeps (`mc/runtime.py:474-804`), then rerun `uv run pytest tests/test_mc_components.py`.

## Critical / Blocking
1. **LensNet never receives gradients.** `_refine_variants` stores `scores.detach()` and `_compute_lens_losses` only consumes the detached tensors (`mc/runtime.py:474-804`). `lens_loss` therefore cannot update `lensnet`, making any focus ablation meaningless. We need to keep the live tensor (or recompute scores) so `F.mse_loss` backpropagates.

## High-Impact Engineering Tasks
- [x] **Top-k logit projection skews gist losses.** `_compute_lod_losses` multiplies truncated probabilities with embedding vectors but never renormalizes (`mc/runtime.py:758-778`). When `top_k << vocab`, the sum of weights is <1, so the predicted gist energy is biased low. Normalize `top_vals` before the weighted sum.
- [x] **MegaContext tree append doubles embedding cost.** `MegaContextTree.append()` re-embeds tokens even though the caller already produced embeddings, and `_gather_child_block()` re-embeds spans via `_embed_tokens_range()` instead of slicing `_lod0_cache` (`mc/mega_context.py:131-205`). Dedupe these passes so incremental updates stay within the intended compute budget.
- [x] **Telemetry promises (partial).** Added swap-rate, token-budget utilization, and residency (mean/p95) to focus telemetry plus a `delta_nll` event comparing each variant to the recency baseline.
- [x] **Positional caches favor the first variant, not the best one.** `_build_session_positional()` now prefers the recency-baseline variant when available, otherwise falls back to the first variant (`mc/runtime.py`).
- [x] **Thread-pool tree building fights the GPU.** Collapsed `process_batch()` to a sequential build to avoid GPU stream contention and nondeterminism. `build_workers` is effectively ignored for now.
- [ ] **Working-context edits remain `O(n)` concatenations.** `WorkingContext.append/replace` repeatedly `torch.cat`s full tensors, which can scale poorly. Consider a ring buffer/block allocator for O(1) edits.
- [x] **Random span sampling syncs on every `.item()`.** `_sample_random_span_starts()` builds GPU tensors and converts each element back to Python via `.item()` (`mc/runtime.py:417-438`), forcing a sync per start. Generate on CPU (or call `.tolist()` once) to avoid the perf cliff.

## Documentation & Alignment (Deep Work)
- After the instrumentation lands, bring `obsidian/ops/Telemetry.md` up to date with the implemented metrics and how they surface in WANDB/Grafana.
- Explain the current focus allocator behavior (greedy-only, expand/collapse thresholds, lack of cooldown) somewhere in `obsidian/ops/Training & Operations.md` so future architectural changes have a baseline to compare against.
- [x] Add a design note outlining a chunked (rope-like) WorkingContext backend: `obsidian/reviews/Chunked WorkingContext Design.md`.

## Testing & Validation Gaps
- [x] Add regression tests covering the horizon-loss projection (`mc/runtime.py:758-778` via `_project_logits_to_embeddings`) to assert top-k renormalization.
- [x] Added tests for LOD0 append cache (`tests/test_mc_append_cache.py`) and allocator telemetry structure (`tests/test_focus_telemetry.py`).
- [x] Added tests for positional cache selection (`tests/test_positional_selection.py`) and runtime focus telemetry with residency (`tests/test_runtime_telemetry.py`).
- Extend `tests/test_mc_components.py` (or a new suite) to cover the ThreadPool vs. sequential build path and WorkingContext edit performance (CPU mocks are fine, but we need coverage).
- Build a small harness that exercises the telemetry provider with fake swap-rate/residency stats, ensuring the new metrics serialize cleanly before wiring them into OTEL/WANDB.

## Path to 10/10 (Deep Work Edition)
1. **Close the observability gap.** Implement the remaining telemetry (residency), and surface all metrics in WANDB/Grafana.
2. **Optimize the tree/allocator hot paths.** Deduplicate embedding work, move random-span sampling off the GPU, and reassess the ThreadPool strategy so MC overhead stays within budget.
3. **Strengthen the data structures.** Replace the O(n) `WorkingContext` edits with a buffer-friendly structure and ensure positional caches select the best variant, not just the first.
4. **Broaden regression coverage.** Add CPU-mode integration tests for horizon losses, telemetry emission, and the optimized append path so high-risk logic is protected ahead of future ablations.
