---
tags:
  - analysis
summary: Review of the current MegaContext run10 telemetry (steps 0–170) and recommended follow‑ups.
---
# MegaContext Performance Review — run10 (steps 0‑170)

## Training Throughput
- **Tok/sec:** stabilized around **3.9k tokens/s** after the most recent controller optimizations (variant pack reverted to on-device tensors). The earlier dip (~3.1k) was due to the pinned CPU staging experiment; reverting restored throughput without lowering batch size.
- **Per-step timings (latest logs):**
  - `build_ms≈260`, `variant_ms≈220`, `lens_ms≈235`, `variant_pack_ms≈4`, `other_ms≈2.4k`.
  - The controller components now sum to ~720 ms; the remaining ~2.4 s are the base GPT forward/backward/optimizer, which matches expectations for the 56×2048 batch.
- **MFU:** ~0.41, unchanged from the prior good run; no signs of GPU under-utilization once the regression was fixed.

## Controller Metrics
- **LOD layout:** Baseline windows stay at 1,831 tokens with only ~7 gists per WC—LensNet aggressively preserves LOD0 tail/head while sampling gisted blocks mid-span (as seen in the ASCII dumps). Focus iterations remain zero in these snapshots because the deterministic training allocator is handling WC normalization.
- **LensNet performance:**
  - `mc/lens_loss` plateaued at ~0.18–0.20 after step 70 and is gently trending downward; supervision is stable.
  - `mc/policy_score_abs_mean` climbs to ~0.7 while `mc/policy_score_std_mean` settles around 0.2, indicating LensNet is outputting stronger, better-differentiated scores as training progresses.
  - `mc/preference_corr_mean` remains noisy (0.1–0.4) but positive; `corr_max` stays >0.9, suggesting at least some preference pairs are strongly aligned with the ΔNLL targets.
- **Variant quality:**
  - `mc/adv_delta_mean` and `p95` hover between 5 and 6 nats → variants are consistently worse than the recency baseline (expected at early steps). The spread (`mc/adv_delta_std ≈ 0.8–1.0`) is wide, giving LensNet good signal.
  - `mc/lod_loss/0` drops to ~5.5 while `mc/lod_loss/1` sticks around 6.5–6.8: gists are still ~1 nat worse than raw tokens but trending slowly down.
  - **New metric `mc/lod_delta/1`:** rising to ~5 nats confirms that replacing a block with its gist imposes a consistent penalty. This is the clearest indicator we have that LOD1 compressions still cost significantly more than we’d like.

## Findings
1. **Throughput recovered.** Our instrumentation confirmed the regression source (host-staging) and `other_ms` now exposes the true base-model cost per step.
2. **Gists remain expensive.** `lod_loss_1` > `lod_loss_0` by ~1 nat and `lod_delta/1` trending upward shows we haven’t yet made LOD1 spans competitive with raw tokens.
3. **LensNet scoring is healthy.** Policy scores separate (abs mean and std diverge), correlation stays positive, and the lens loss decreases steadily.
4. **Variants provide strong supervision.** ΔNLL statistics (mean/p95/std) continue rising, so preference pairs remain informative at this learning rate.
5. **Token-aligned telemetry is live.** The base trainer now logs `train/tokens_per_step`, `train/grad_accum_steps`, and `mc/grad_accum_steps`, so WANDB overlays immediately show that baseline and MC runs stayed in lock-step. All cadence triggers (eval/core/ckpt/log) are keyed off the same token counter, which eliminates the step-count mismatches called out in the previous review.

## Next Steps
1. **Improve gist quality**
   - Add a ΔNLL-aware auxiliary loss to GistNet (e.g., distill the LOD0 logits via KL) so gists explicitly minimize the `mc/lod_delta/1` penalty.
   - Experiment with deeper or multi-head GistNet configs to reduce the 5 nat gap (current runs still use 2-layer transformer).
2. **Correlate LensNet scores with ΔNLL**
   - Log `mc/pref_score_corr` vs `mc/lod_delta` per span to ensure LensNet actually favors gists that hurt less.
3. **Monitor controller overhead**
   - Keep `variant_pack_ms` telemetry and confirm it stays <10 ms after future changes. If it rises, re-evaluate packing strategies before touching batch size.
4. **Document the new metrics**
   - ✅ `lensnet_perf_investigation.md` now references the `lod_delta` telemetry; additionally, the README + alignment plan describe how to read the `train/tokens_per_step` and grad-accum logs when comparing runs.
5. **Optional:** Add WANDB panels for `mc/other_ms`, `mc/variant_pack_ms`, `mc/lod_delta/1`, and the new token-per-step series so regressions are visible without diving into logs. Template draft lives in `obsidian/plans/mc_and_baseline_alignment.md`.

_Update (Nov 28): the first two bullets are now wired up—`mc_gist_delta_weight` penalizes variants whose ΔNLL exceeds the baseline, and WANDB exposes `mc/pref_span_corr` so we can see span-level LensNet alignment without spelunking raw logs._

With these follow-ups, we can focus on lowering the LOD1 penalty while maintaining the restored throughput.
