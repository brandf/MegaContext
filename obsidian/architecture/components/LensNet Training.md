---
tags:
  - components
summary: Current LensNet training recipe based on random working-context variants and pairwise preference supervision.
---
# LensNet Training (Phase 1)

LensNet is a shallow transformer (2/4/8 layers) that operates directly on the [[Working Context]] embeddings. Phase 1 drops the historical tail-gist extras and trains the controller via **random variant sampling + pairwise preference comparisons**. This page documents the data pipeline, loss, and telemetry that now exist in code (`mc/runtime.py`, `scripts/base_train.py`).

## High-Level Loop

1. **Build variants.** For each training sequence we construct:
   - `lod_0_baseline`: pure LOD0 window that preserves the recent tail.
   - `num_random_variants` stochastic compressions sampled by running the focus allocator with random scores (see `_build_random_variant_set`).
2. **Score variants.** Run the base model on every variant to obtain next-token losses.
3. **Compute advantages.** For each variant compute `adv_delta = loss_variant − loss_baseline`. Negative numbers mean “better than baseline”.
4. **Form preference pairs.** For every `(better, worse)` pair (based on `adv_delta`) emit a tuple `(better, worse, strength)` where `strength = |adv_delta_better − adv_delta_worse|`.
5. **Optimize LensNet.** Feed the *worse* WC through LensNet to obtain policy scores and apply the Bradley–Terry loss + optional rank/budget penalties.

This replaces the legacy “best WC LOD map” regression and trace-log replay buffer. All supervision is local to the current batch and amortizes the base-model forward pass we already perform for GistNet training.

## Data Specification

| Field | Shape | Source |
| --- | --- | --- |
| `baseline_variant` | `WorkingContextVariant` | `_build_lod0_baseline_variant` |
| `random_variants` | List of `WorkingContextVariant` | `_build_random_variant_set` |
| `adv_delta` | scalar per variant | `_compute_variant_losses` |
| `preference_pairs` | List of `(better, worse, strength)` | `_build_preference_pairs` |

Notes:
- Variants always respect coverage + tail invariants before entering the loss.
- `strength` is `tanh(|Δloss|)` in code to squash outliers before weighting the loss.

## Preference Loss

LensNet outputs signed **policy scores** `s_i` per WC entry (tanh-clamped to ±1). For each `(better, worse)` pair we compute:

1. Align entries via the best WC’s LOD map (`_build_pairwise_targets`), resulting in a per-entry target `t_j ∈ {−strength, +strength}` and mask `m_j`.
2. Apply a Bradley–Terry / logistic preference loss with temperature `τ = mc_lens_temperature`:

$$
L_{\text{pref}} = \frac{1}{|M|} \sum_{j \in M} \text{softplus}\!\left(-\frac{t_j}{|t_j|} \cdot \frac{s_j}{\tau}\right)\,|t_j|
$$

Implementation detail (`mc/runtime.py::_compute_lens_losses`):
- `t_j > 0` ⇒ pushing scores positive (expand).
- `t_j < 0` ⇒ pushing scores negative (collapse).
- `collapse_weight` optionally reweights collapse targets to balance expand-heavy batches.

### Rank & Budget Penalties (Optional)

We retain the legacy hooks:

- **Rank loss** (`lens_rank_weight`): hinge loss that forces the mean score over positive targets to exceed the mean over negative targets by `lens_margin`.
- **Budget loss** (`lens_budget_weight`): squared difference between collapse/expand mass weighted by span sizes to discourage “expand everything”.

Phase 1 keeps these weights low (0.5 / 0.1) so the preference loss dominates.

## Temperature & Hyperparameters

All CLI knobs surface through `run10.sh` and `MCConfig`:

| Flag | Description |
| --- | --- |
| `--mc_num_random_variants` | Number of random WCs per sequence. |
| `--mc_train_wc_length` | Target length for random variants. |
| `--mc_max_lens_pairs` | Upper bound on `(better, worse)` pairs per sample. |
| `--mc_lens_temperature` | Bradley–Terry temperature (default 1.0). |
| `--mc_lens_rank_weight`, `--mc_lens_budget_weight`, `--mc_lens_margin`, `--mc_lens_collapse_weight` | Legacy regularizer knobs that still work. |

Lowering the temperature sharpens comparisons (steeper gradients for a given Δloss). Raising it smooths updates when the random variants are noisy.

## Telemetry

We log the following metrics to W&B (`scripts/base_train.py`):

| Metric | Meaning |
| --- | --- |
| `mc/adv_delta_mean`, `mc/adv_delta_p95` | Average/p95 Δloss relative to baseline (want ≤ 0). |
| `mc/preference_corr_mean` | Correlation between policy scores and `adv_delta` (want negative). |
| `mc/lens_loss` | Mean preference loss value. |
| `mc/variants_total`, `mc/variants_mean` | How many WCs were evaluated per batch. |

`--mc_log_lens_debug` prints per-variant stats (“PrefDebug”) so we can inspect score distributions and correlations during training.

## Stability Tricks

| Mechanism | Knobs | Purpose |
| --- | --- | --- |
| Advantage normalization | `lens_adv_norm_beta` | Maintain an EMA of `adv_delta` mean/variance so normalized advantages (`norm_adv_delta`) drive the preference strength. |
| Policy KL regularization | `lens_kl_weight` | Keeps LensNet from thrashing by penalizing divergence from the previous policy scores per working context. |
| Budget smoothing | `lens_budget_smooth_weight`, `lens_budget_smooth_beta` | Tracks an EMA of net expand/collapse mass and penalizes deviations to keep scores budget-neutral despite random variants. |

All three reuse the WC variants already generated for base LLM + GistNet training; no extra model passes are required.

## Future Work (Phase 2 Ideas)

- Reintroduce tail-gist cross conditioning once preference training is stable.
- Log per-entry legality masks and re-enable a soft illegality penalty if we observe the allocator fighting LensNet.
- Explore replay buffers / curriculum sampling so LensNet sees more diverse focus plans than pure random variants.

For implementation details see `mc/runtime.py` (`_build_random_variant_set`, `_compute_variant_losses`, `_compute_lens_losses`) and `scripts/base_train.py` (W&B logging, CLI plumbing).
