---
tags:
  - lensnet
  - theory
summary: Relating the LensNet random-variant training recipe to reinforcement learning and contrastive learning paradigms.
---

# LensNet Pairwise Training — Relation to RL & Contrastive Learning

## Setup Recap
During MC training we now:

1. Build a *baseline* working context (pure LOD0 tail-preserving window) trimmed to the current curriculum target length so it is directly comparable to every random variant.
2. Sample `N` *random variants* by stochastically collapsing/expanding the baseline down to a fixed `train_wc_length`.
3. Run next-token loss for every variant, producing per-variant NLLs and the natural “preference” ordering between them.
4. Supervise LensNet by comparing variant pairs (our `preference_pairs`): the lower-loss WC is the “better” policy action, and we regress LensNet’s per-entry focus scores so that it prefers the edits that differentiate better vs. worse variants.

## Connection to Reinforcement Learning

| RL Concept | LensNet Analogue | Notes |
| --- | --- | --- |
| **Policy** | LensNet focus scores over WC entries | Scores translate to “collapse vs. expand” actions for each span. |
| **Environment** | MegaContext tree compression | Applying collapse/expand changes the WC “state” and produces the new observation (variant). |
| **Action** | Editing a span (collapse or expand) | Random variants simulate trajectories sampled from a stochastic policy. |
| **Reward** | Negative next-token loss (`-ΔNLL`) | Lower loss ⇒ higher reward; pairwise deltas give advantage estimates. |
| **Policy improvement** | Pairwise ranking loss | Similar to preference-based RL: we only know *which* variant is better, not its absolute reward. |

Key similarities:

- *Implicit advantage estimation*: Δloss between two variants is analogous to an advantage signal (A(s,a) ≈ reward difference when choosing different actions in the same state).
- *Off-policy data*: We generate variants via random focus edits instead of LensNet’s current policy, then use them to update LensNet—akin to off-policy RL with importance weights ≈ 1 (because we deliberately ignore the behavior policy).
- *Preference-based RL*: Rather than regressing a scalar reward, we enforce that LensNet ranks higher-quality variants above lower-quality ones, which mirrors preference-based policy optimization (e.g., RLHF without the KL regularizer).

Differences from classic RL:

- No temporal credit assignment: all actions happen “at once” per WC; there’s no multi-step trajectory or discounting.
- Deterministic reward signal: next-token loss is supervised, so the reward has no sampling noise besides stochastic regularization.
- No exploration loop: LensNet does not interact with the environment to gather new rollouts; randomness is injected procedurally via the allocator.

## Connection to Contrastive Learning

The pairwise supervision also maps cleanly to contrastive objectives:

- Each sample provides *positive* (better WC) and *negative* (worse WC) views of the **same underlying sequence**.
- The regression/ranking loss can be seen as aligning LensNet scores with the *relative* quality of each entry, akin to InfoNCE where we want embeddings to move closer to positives and away from negatives.
- Because the two variants share the same tail and differ only in focus decisions, the training signal is inherently contrastive: we only care about *differences* between variants.

**Instrumentation**  
To mirror RLHF dashboards we now emit:

- `mc/adv_delta_mean` / `mc/adv_delta_p95` — running statistics of the per-variant advantage (ΔNLL relative to the baseline WC).
- `mc/preference_corr_{mean,max,min}` — correlation between policy scores and the observed advantages (now reported as `n/a` instead of `NaN` when variance is zero).
- `mc/preference_agreement` — share of preference pairs where LensNet’s signed scores pick the same winner as the measured Δloss.
- `mc/policy_score_abs_mean`, `mc/policy_score_std_mean` — how much of the tanh range LensNet is actually using.

These WandB traces serve as the “reward model agreement” + “advantage histogram” analogs from standard RLHF setups, with additional visibility into policy calibration.

Contrastive parallels:

- *Shared context*: both variants originate from the same base tree, just like two augmentations of the same image/text snippet in contrastive pretraining.
- *Temperature / margin*: the LensNet margin parameter fills the same role as the temperature in InfoNCE, controlling how strongly we separate positives vs. negatives.
- *Batch negatives*: every variant pair within a sample acts as a mini contrastive pair, and we can sample many such pairs per batch without cross-sample alignment issues.

Where it differs:

- Instead of embedding similarity, we regress per-entry *policy scores*. The contrastive structure lives in token-level targets, not global embeddings.
- The objective is asymmetric (prefer expand vs. collapse) rather than symmetric alignment of two augmentations.

## Terminology Suggestions

| Old Term | Proposed Mapping | Rationale |
| --- | --- | --- |
| “Best variant” | “Preferred policy action” | Emphasizes RL-flavored preference data. |
| “Random variant” | “Stochastic rollout” | Highlights that we sample from a behavior policy. |
| “Δloss supervision” | “Preference delta / advantage” | Matches RLHF and preference-learning papers. |
| “Pairwise targets” | “Contrastive policy targets” | Signals that we only care about pairwise ordering. |

Adopting this vocabulary clarifies to readers that LensNet is trained with **preference-based, contrastive supervision**: we treat WC edits as actions, next-token losses as rewards, and learn a policy (LensNet) that ranks higher-reward edits above lower-reward ones.

## Which Paradigm Fits Best?

From an implementation standpoint we are **closer to preference-based RL**:

- The supervision signal is a *relative reward* (ΔNLL) collected from variants of the same underlying state.
- LensNet acts as a *policy* whose logits should increase or decrease the probability of editing particular spans.
- We can reuse RLHF terminology (policy, reward, advantage, preference pair) without distortion.

However, the *mechanics* of our loss—pairwise comparisons over different “views” of the same sequence—feel contrastive. We can borrow the temperature/margin ideas from contrastive learning to control how sharply we separate positives/negatives, even while describing alignment in RL language.

**Recommendation:** use RL-centric vocabulary (policy, rollout, reward, preference pair, advantage) when discussing LensNet training, and draw contrastive analogies when explaining the loss geometry.

## Learning from RL / Contrastive Research

| Inspiration | Technique | Applicability to LensNet |
| --- | --- | --- |
| Preference-based RL (e.g., RLHF) | Bradley–Terry or logistic preference loss; advantage normalization; reward-model regularization | Replace our MSE targets with logistic preference losses, track running statistics for Δloss to stabilize gradients, optionally introduce per-span baselines. |
| Off-policy policy gradients | Importance sampling, KL regularization, trust regions | Weight pairings by how far the sampled variant distribution drifts from the current LensNet policy; add KL penalties between “old” and “new” focus scores to avoid thrashing. |
| Contrastive learning (InfoNCE, SimCLR) | Temperature scaling, hard negative mining, multi-positive batches | Treat large Δloss pairs as “hard negatives,” schedule a temperature parameter that sharpens targets when Δloss is big, and group pairs across the batch to improve sample efficiency. |
| Curriculum / self-play | Progressive difficulty, adversarial perturbations | Start with mild compressions (small train_wc_length drop) and gradually introduce harsher edits so LensNet learns a spectrum of focus decisions. |

### Execution Checklist

1. [x] **Terminology alignment**
   - Update docs + metrics to use RLHF-style names (`preference_pairs`, `policy_scores`, `adv_delta`).
   - Add WandB charts mirroring RLHF dashboards (reward mean, advantage histogram).

2. [x] **Loss upgrades**
   - Replaced the per-entry MSE with a Bradley–Terry (logistic) preference loss scaled by ΔNLL magnitude and a tunable temperature.
   - Added `mc_lens_temperature` CLI/config knob so we can sweep how sharp the preference comparisons are.

3. [x] **Stability enhancements**
   - **Advantage normalization:** track an EMA of `adv_delta` mean/variance (`lens_adv_norm_beta`) and z-score deltas before feeding them into the Bradley–Terry loss so noisy batches don’t explode gradients. Falls back to raw Δloss if the EMA hasn’t been initialized.
   - **Policy KL regularization:** keep a cache of the previous policy scores per variant and add a symmetric KL penalty (`lens_kl_weight`) so LensNet can’t thrash its logits between iterations.
   - **Budget smoothing:** maintain an EMA of expand-minus-collapse mass (`lens_budget_smooth_beta`) and penalize deviations (`lens_budget_smooth_weight`) so random variants don’t skew the controller toward reckless expansion or collapse.

### Stability Enhancements in Practice

| Technique | Why | How |
| --- | --- | --- |
| Advantage normalization | ΔNLL magnitudes can vary wildly between samples, producing unstable gradients. | Maintain an EMA (`lens_adv_norm_beta`) of the mean/variance of `adv_delta`, compute normalized advantages per variant, and derive preference strength from that z-score. |
| Policy KL regularization | Prevents LensNet from flipping sign every batch, which destabilizes the allocator. | Cache the previous policy scores per WC and add a symmetric KL term (`lens_kl_weight`) when computing `_compute_lens_losses`. |
| Budget smoothing | Random variants sometimes bias a batch toward expand-only or collapse-only plans. | Track an EMA of net expand mass (`lens_budget_smooth_beta`) and penalize deviations via `lens_budget_smooth_weight`. |

4. [x] **Curriculum + hard-negative mining**
   - Random variant target lengths now anneal linearly from 80 % of `max_seq_len` down to `mc_train_wc_length` (default `0.75 × max_seq_len`), keeping the trimmed baseline and every variant at the same length for fair comparisons.
   - Every non-baseline WC is paired with the best-performing variant before we sort remaining pairs by raw Δloss and keep the top `mc_lens_hard_negative_ratio` fraction, guaranteeing that each supervision example includes a “real” hard negative.

5. [ ] **Evaluation + ablations**
   - LensDebug + WandB now expose `mc/preference_agreement` and policy-score range metrics; remaining work is to script ablations that sweep the stability knobs and report their impact on `mc/preference_corr_mean`, swap rate, and downstream validation loss.

Executing this roadmap lets us systematically inject proven RLHF/contrastive tricks into LensNet while keeping the mental model firmly rooted in preference-based policy learning.
