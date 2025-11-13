---
tags:
  - lensnet
  - review
summary: Assessment of the current LensNet implementation/training vs. the intended design, plus risks, debug hooks, and next-step recommendations.
---

# LensNet Review (2023-11-12)

## High-level goals for an “ideal” LensNet

1. **Policy quality**
   - Emits signed per-entry utilities whose sign correlates with the counterfactual ΔNLL benefit of either (a) expanding that entry, or (b) collapsing the *block* (stride = block_size) that contains it. Today the focus allocator collapses aligned spans—so LensNet must aggregate per-entry collapse scores into block-level decisions. The easy path is to average (or max) over the entries that share a parent and feed that to the allocator; the better path is to let LensNet learn the mapping by supervising the parent block’s ΔNLL on every constituent entry (i.e., copy the collapse label onto all entries that will be collapsed together). Either way, policy quality means the allocator can hand LensNet per-entry scores and still take block-level actions deterministically.
   - Keeps scores calibrated so the focus allocator can trade expansions vs. collapses without thrashing.
2. **Training loop**
   - Uses explicit ΔNLL labels (or high-fidelity proxies) collected under the same distribution as deployment.
   - Enforces legality masks (LOD0 cannot expand, LODmax cannot collapse) during supervision to avoid teaching LensNet impossible actions.
   - Mixes regression + ranking + budget regularization, with clear weighting.
3. **Runtime performance**
   - Runs in <3 ms / focus step on 8 k-token WCs.
   - Supports bf16 where available, fp32 fallback for stability.
   - Provides telemetry for score histograms + allocator edits to diagnose drift.
4. **Debugability + tests**
   - Unit tests that cover target construction, loss polarity, legality masking, and replay buffers.
   - Toggleable debug logs that correlate scores with realized ΔNLL (just added via `--mc_log_lens_debug`).
   - CI tests ensuring LensNet forward pass is deterministic given a fixed WC snapshot.

## Current implementation (code + docs)

| Area | What the code does today | Observations / gaps |
|------|-------------------------|---------------------|
| Architecture (`mc/lensnet.py`) | 2-layer nanochat block stack with Gaussian RoPE + tanh head.| Matches doc simplification (“phase 1 ignores tail gists”). No auxiliary features (span width, cursor distance) wired yet. |
| Training target (`_build_lens_targets`, `mc/runtime.py:1675`) | Compares each variant to the “best” (lowest-token-loss) variant and sets target = `+1` if best uses more detail (expand), `-1` if best uses less detail (collapse). | This is *relative to the best variant’s LOD map*, not ΔNLL per position. If the best variant keeps a gist collapsed, but a different variant actually improves ΔNLL in that region, we never see the label. On the flip side, piggybacking on the best variant is still desirable because it costs zero extra ΔNLL evaluations (we already paid the base-model forward for GistNet training). Any improved approach must stay compute efficient—e.g., reuse the per-variant losses we already have, or cache ΔNLL deltas from those runs, instead of running new counterfactual passes. |
| Loss (`_compute_lens_losses`, `mc/runtime.py:1409`) | Bradley–Terry (logistic) preference loss weighted by ΔNLL magnitude + temperature, plus optional rank/budget penalties. | Ranking/budget hooks now exist, but we still need to tune temperature/weighting and add legality-aware masking in future work. |
| Legality masking | At inference the allocator clamps illegal actions, but training targets include every WC entry regardless of LOD. | LensNet is penalized for not collapsing the root (LOD2) even though collapse is illegal, confusing gradients. |
| ΔNLL usage | We only use ΔNLL to pick the “best variant”. No per-span ΔNLL is logged. | The doc’s “counterfactual utilities” are unimplemented—hence the dangling TODO references in `obsidian/architecture/components/LensNet Training.md`. |
| Telemetry | Prior to this review the only signals were `mc/lens_loss`, `mc/adv_delta_mean/p95`. | New `--mc_log_lens_debug` instrumentation now surfaces score stats + correlations, revealing the polarity bug (scores anti-correlate with ΔNLL). |
| Tests | Unit tests rely on dummy `ZeroLensNet` to avoid running the real model. No coverage for target generation, legality, or loss weighting. | Means regressions (like flipped sign) go unnoticed. |

## Likely issues + proposed fixes

1. **Label polarity mismatch**
   - Evidence: `corr_mean ≈ +0.95` between average score and ΔNLL delta ⇒ high scores associate with worse variants.
   - Cause: `_build_lens_targets` derives desired LOD from the *best variant* instead of actual ΔNLL improvements per entry. If the best WC already expands a span, every other variant gets target `+1`, even if expanding makes things worse.
   - Fix: Use true ΔNLL-derived utilities (as specified in docs) or at least compute per-entry deltas between the variant and LOD-map of the best variant. Also ensure the weighting `weight = 1 + clamp(Δloss, 0)` actually boosts positive examples.

2. **Missing legality masking during training**
   - LensNet is punished for not “expanding” LOD0 tokens or “collapsing” LOD2 roots. That pushes scores negative/positive indiscriminately.
   - Fix: Mask out non-actionable positions (level==0 for expand targets, level==max for collapse) when computing both targets and loss.

3. **No ranking/budget loss**
   - Without ranking, LensNet chases absolute ±1 targets regardless of magnitude differences, encouraging saturated tanh outputs that don’t correlate with utility magnitude.
   - Budget loss from the spec is absent, so LensNet learns to push everything toward expand (or collapse) depending on noise, leading to unstable allocator behavior.
   - Fix: Implement the documented `L_rank` (pairwise hinge) + `L_budget` terms; clamp gradients to keep tanh outputs in a usable range.

4. **Variant weighting penalizes good samples**
   - Weight = 1 for best variants, >1 for worse ones, meaning bad WCs dominate the loss. We need the opposite: highlight edits that improved ΔNLL.
   - Fix: Weight by `1 + clamp(best_loss - variant_loss, 0)` (i.e., improvements) or normalize weights.

5. **Score normalization / dtype drift**
   - LensNet runs in `_target_dtype` (bf16 on H100). Without LayerNorm/scale, scores cluster near zero or saturate. Observed score std ≈ 0.3 suggests we might be under-using dynamic range.
   - Fix: Add LayerNorm before the head, or reintroduce auxiliary features (LOD level, span width, cursor distance) to give the net a meaningful prior.

## Open questions / areas needing deeper investigation

- **Counterfactual data pipeline**: docs reference trace logs + replay buffers, but the current training loop generates targets on-the-fly per batch. Is the logging pipeline (ΔNLL per action) still planned? If yes, we need to align with that spec.
- **Tail-gist conditioning**: the architecture doc calls for tail gists + cross-attention; current code doesn’t. Decide whether to keep the simplified transformer or bring back the two-stage perceiver.
- **Allocator siblings**: do we ever apply multiple collapse edits per variant per iteration? If not, LensNet never sees examples where chaining collapses is optimal.
- **Loss scaling**: `mc_lens_loss_weight` defaults to 0.1; with current magnitudes this might be too low compared to the core token loss.

## Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **LensNet diverges silently** (scores anti-correlate with ΔNLL) | Focus allocator churns budget, degrading WC quality. | Keep `--mc_log_lens_debug` on for dev runs; add WandB panel for score/delta correlation; alert if |corr| < threshold. |
| **Legality violations** teach impossible actions | Allocator clamps them, but gradients push LensNet toward zero, wasting capacity. | Mask training targets + losses for illegal positions; add unit tests that assert mask coverage. |
| **Replay drift** (if we add buffers later) | LensNet trains on stale distributions. | Version trace logs alongside base-model checkpoints; include timestamp in dataset metadata. |
| **Performance regressions** (LensNet forward >3 ms) | Slows inference. | Keep architecture small (2 layers), monitor `[MC Eval Timers] controller_internal`. Add perf tests that assert <X ms on CI. |

## Debugability checklist

- [x] ASCII WC logger (`--mc_log_lod_ascii_*`)
- [x] LensNet score/ΔNLL correlation logger (`--mc_log_lens_debug`)
- [ ] Dump per-action ΔNLL and resulting allocator edits (needs counterfactual evaluator hook).
- [ ] WandB panels for score histograms, corr_mean, corr_max, corr_min.
- [ ] CLI flag to run LensNet in eval mode only (no gradient) for ablations.

## Testing gaps / towards “10/10” LensNet tests

1. **Target generation unit tests**
   - Feed synthetic WC + “best” variant map; assert `_build_lens_targets` outputs +1/‑1 only for actionable positions.
2. **Loss polarity tests**
   - Mock variants with known Δloss; ensure `_compute_lens_losses` pushes scores in the correct direction (e.g., gradient sign check).
3. **Legality masking tests**
   - Verify LOD0 tokens never receive positive targets; LOD2 gists never receive negative ones.
4. **Budget regularizer tests** (once implemented)
   - Ensure `L_budget` penalizes unbalanced scores as specified.
5. **End-to-end smoke**
   - Run a tiny training step with synthetic ΔNLL labels and assert corr_mean < 0 (scores anti-correlate with delta). This becomes part of the regression suite.

## Prioritized action plan

- [ ] **(P0) Fix target polarity / legality masking**
  - Targets now take the absolute Δloss magnitude but force the sign to match the best variant’s desired action (expand → +, collapse → −). Collapse broadcasts still respect block alignment.
- [ ] **(P1) Collapse weighting**
  - Added `--mc_lens_collapse_weight` knob (defaults to 1.0) that scales the MSE penalty on collapse targets so we can make LensNet more collapse-sensitive if needed.
  - Mask illegal entries (LOD0 expand, LODmax collapse) during both target construction and loss.
- [ ] **(P0) Add budget & ranking losses**
  - Implement `L_budget` and `L_rank` from the spec so LensNet learns ordering + net-zero token flow.
- [ ] **(P1) Score/ΔNLL telemetry in WandB**
  - Log corr_mean/corr_max/corr_min and score histograms so we can monitor polarity without scrolling logs.
- [ ] **(P1) Reinstate auxiliary features** (LOD level, span width, cursor distance) ahead of the scoring head to help calibration.
- [ ] **(P1) Update tests**
  - Add unit tests for target masks, polarity, and loss composition; add smoke test for negative correlation.
- [ ] **(P2) Explore per-block collapse supervision**
  - Aggregate collapse utilities per block and broadcast to entries so per-entry scores align with allocator behavior without extra ΔNLL calls.
- [ ] **(P2) Replay / tail-gist conditioning**
  - Decide whether to reintroduce tail gists or a small replay buffer once the core loop is stable.

We’ll start executing from the top of the list (P0 items first), validating each change with `--mc_log_lens_debug 1` before moving down the stack.
