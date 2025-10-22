# Focus Allocator — Block-Aligned Actions

LensNet supplies signed focus scores; the allocator turns those scores into concrete expand/collapse actions while preserving contiguity, budget, and level-of-detail (LOD) constraints. It is the practical enforcer of the [[Architecture Overview#Key terms & invariants|contiguity invariant]] inside the working context.

## POC constraints & terminology

- **Block alignment:** GistNet currently compresses 32-token blocks. In the POC, every working-context entry must cover exactly one full block at a single LOD (either 32 raw tokens or their 32→1 gist). Higher-level gists (e.g., L2) cover 32 contiguous L1 blocks.
- **Score granularity:** LensNet may emit per-entry scores, but the allocator aggregates them per block so that siblings share a single action score. A future LensNet variant can predict directly per block to avoid this aggregation.
- **Action budget:** Apply at most `N_diff` expand/collapse operations per iteration (default 4). This keeps the system near equilibrium and prevents thrashing.
- **Positional alignment:** When swapping L0/L1 entries, reuse the original absolute token indices for RoPE; gists occupy the central token index of their covered span so the base LLM receives consistent phase information.

## Greedy loop (POC)

1. **Collect candidates.** Partition focus scores by block and compute one score per expandable or collapsible unit:
   - Positive scores (`> τ_expand`, default 0.2) become expand candidates (e.g., replace an L1 gist with its 32 L0 tokens or expand an L2 gist into 32 L1 children).
   - Negative scores (`< -τ_collapse`, default 0.2) become collapse candidates (e.g., replace 32 L0 tokens with their L1 gist).
   - Ignore candidates that would violate block alignment (mixed LODs) or budget limits.
2. **Rank.** Maintain two priority queues: descending for expands, ascending for collapses. Tie-break by recency or distance to the cursor.
3. **Apply diff-limited updates.** Pop from the queues alternately (largest expand, largest collapse) until:
   - You have applied `N_diff` actions,
   - One queue empties, or
   - Applying the next action would break the `W_max` budget.
   Collapses refund token budget; expands consume it. If the net cost drifts away from `W_max`, bias the next iteration toward the side that restores balance.
4. **Re-run LensNet if needed.** Because changing LODs alters the scores, optionally iterate LensNet → allocator until either (a) no legal actions remain above thresholds or (b) you reach a maximum number of refinement steps (default 2–3).

## Hysteresis & guardrails

- **Action cooldown:** Track the last action applied per block and dampen (or mask out) the opposite action for `cooldown_steps = 2` iterations. This prevents jitter where the allocator repeatedly expands and collapses the same span.
- **Legality masks:** Blocks at minimum LOD (L0) cannot expand; blocks at maximum LOD (current root level) cannot collapse. These masks should be enforced both in LensNet’s output (runtime masking) and inside the allocator.
- **Consistency checks:** After every iteration, verify that working-context entries still tile the timeline without overlap and that every node’s children share the same LOD.

## Recommended runtime defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `τ_expand` | 0.20 | Minimum signed score magnitude before expanding an entry. |
| `τ_collapse` | 0.20 | Symmetric collapse threshold; keep equal to `τ_expand` until adaptive tuning is available. |
| `N_diff` | 4 | Maximum expand/collapse actions per iteration to cap churn. |
| `cooldown_steps` | 2 | Minimum iterations before a block can flip actions. |
| `lens_update_interval` | 32 tokens (`K`) | LensNet runs once per block and consumes cached tail gists. |
| `tail_gist_window` | 5 L1 nodes + current L2 | Conditioning set passed to LensNet. |

These defaults keep the working context near equilibrium while allowing meaningful detail movement; they are the baseline values for automated tests and ablations.

## Future directions

- Smarter action selection (e.g., matching total expand/collapse mass, soft assignments, or small linear programs) to balance budget and latency.
- Learning a differentiable surrogate (“focus router”) that could eventually replace the greedy loop.
- Adaptive thresholds (`τ_expand`, `τ_collapse`) based on recent utilization to keep the loop stable.

For now, the greedy, block-aligned allocator keeps the POC simple while leaving room for more sophisticated controllers later.
