# MegaContext Review — Deep Work

## Recommended Pre-run Checks (High Difficulty)
- [ ] Patch the LensNet loss path so gradients actually reach `lensnet` before running any MC-enabled sweeps (`mc/runtime.py:474-804`), then rerun `uv run pytest tests/test_mc_components.py`.

## Critical / Blocking
1. **LensNet never receives gradients.** `_refine_variants` stores `scores.detach()` and `_compute_lens_losses` only consumes the detached tensors (`mc/runtime.py:474-804`). `lens_loss` therefore cannot update `lensnet`, making any focus ablation meaningless. We need to keep the live tensor (or recompute scores) so `F.mse_loss` backpropagates.

## High-Impact Engineering Tasks
- **Top-k logit projection skews gist losses.** `_compute_lod_losses` multiplies truncated probabilities with embedding vectors but never renormalizes (`mc/runtime.py:758-778`). When `top_k << vocab`, the sum of weights is <1, so the predicted gist energy is biased low. Normalize `top_vals` before the weighted sum.
- **MegaContext tree append doubles embedding cost.** `MegaContextTree.append()` re-embeds tokens even though the caller already produced embeddings, and `_gather_child_block()` re-embeds spans via `_embed_tokens_range()` instead of slicing `_lod0_cache` (`mc/mega_context.py:131-205`). Dedupe these passes so incremental updates stay within the intended compute budget.
- **Telemetry promises are unmet.** The ops doc highlights ΔNLL, swap rate, residency, and token-budget utilization (`obsidian/ops/Telemetry.md:24-139`), but `MCController` only emits structural snapshots and coarse counters (`mc/runtime.py:807-835`). Implement the missing metrics (or revise the plan) so tomorrow’s ablations aren’t flying blind.
- **Positional caches favor the first variant, not the best one.** `_build_session_positional()` always picks variant `0` per sample (`mc/runtime.py:240-251`), even if a sibling has the lowest loss. Feed the best variant (or at least the primary variant selected elsewhere) so LensNet/GPT consume the right positional context.
- **Thread-pool tree building fights the GPU.** `process_batch()` fans out to a `ThreadPoolExecutor`, but every worker still drives the same CUDA stream (`mc/runtime.py:154-208`). This adds contention without speedup. Either use per-sample CUDA streams or collapse to the sequential builder.
- **Working-context edits remain `O(n)` concatenations.** `WorkingContext.append/replace` repeatedly `torch.cat`s full tensors (`mc/working_context.py:67-156`), making allocator experiments quadratic beyond a few thousand tokens. Consider a ring buffer or block allocator so edits stay O(1).
- **Random span sampling syncs on every `.item()`.** `_sample_random_span_starts()` builds GPU tensors and converts each element back to Python via `.item()` (`mc/runtime.py:417-438`), forcing a sync per start. Generate on CPU (or call `.tolist()` once) to avoid the perf cliff.

## Documentation & Alignment (Deep Work)
- After the instrumentation lands, bring `obsidian/ops/Telemetry.md` up to date with the implemented metrics and how they surface in WANDB/Grafana.
- Explain the current focus allocator behavior (greedy-only, expand/collapse thresholds, lack of cooldown) somewhere in `obsidian/ops/Training & Operations.md` so future architectural changes have a baseline to compare against.

## Testing & Validation Gaps
- Add regression tests covering the horizon-loss projection (`mc/runtime.py:758-778`) so future probability normalization changes don’t regress silently.
- Extend `tests/test_mc_components.py` (or a new suite) to cover the ThreadPool vs. sequential build path and WorkingContext edit performance (CPU mocks are fine, but we need coverage).
- Build a small harness that exercises the telemetry provider with fake swap-rate/residency stats, ensuring the new metrics serialize cleanly before wiring them into OTEL/WANDB.

## Path to 10/10 (Deep Work Edition)
1. **Close the observability gap.** Implement the promised telemetry metrics (ΔNLL@H, swap rate, residency, token budgets) and surface them in WANDB/Grafana before running wide sweeps.
2. **Optimize the tree/allocator hot paths.** Deduplicate embedding work, move random-span sampling off the GPU, and reassess the ThreadPool strategy so MC overhead stays within budget.
3. **Strengthen the data structures.** Replace the O(n) `WorkingContext` edits with a buffer-friendly structure and ensure positional caches select the best variant, not just the first.
4. **Broaden regression coverage.** Add CPU-mode integration tests for horizon losses, telemetry emission, and the optimized append path so high-risk logic is protected ahead of future ablations.
