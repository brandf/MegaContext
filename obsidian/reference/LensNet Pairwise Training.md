---
tags:
  - lensnet
  - theory
summary: Relating the LensNet random-variant training recipe to reinforcement learning and contrastive learning paradigms.
---

# LensNet Pairwise Training — Relation to RL & Contrastive Learning

## Setup Recap
During MC training we now:

1. Build a *baseline* working context (pure LOD0 tail-preserving window).
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
- `mc/preference_corr_{mean,max,min}` — correlation between policy scores and the observed advantages.

These WandB traces serve as the “reward model agreement” + “advantage histogram” analogs from standard RLHF setups.

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

3. [ ] **Stability enhancements**
   - Track running mean/variance of ΔNLL to normalize advantages before feeding them into the loss (akin to advantage normalization in PPO).
   - Add optional KL regularization between consecutive LensNet policies (`KL(old_scores || new_scores)`) to keep updates smooth.

4. [ ] **Curriculum + hard-negative mining**
   - Bucket variant pairs by Δloss magnitude; oversample “hard” comparisons to accelerate learning, similar to contrastive hard-negative mining.
   - Gradually shrink `train_wc_length` or increase `num_random_variants` during training to expose LensNet to more challenging edits over time.

5. [ ] **Evaluation + ablations**
   - Extend LensDebug to log reward-model style metrics (agreement rate, normalized advantage) and contrastive ones (temperature-scaled loss, InfoNCE analog).
   - Run controlled ablations to measure the impact of each addition on `mc/preference_corr_mean`, swap rate, and downstream validation loss.

Executing this roadmap lets us systematically inject proven RLHF/contrastive tricks into LensNet while keeping the mental model firmly rooted in preference-based policy learning.
