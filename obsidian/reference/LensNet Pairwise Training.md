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
4. Supervise LensNet by comparing variant pairs: the lower-loss WC is the “better” policy action, and we regress LensNet’s per-entry focus scores so that it prefers the edits that differentiate better vs. worse variants.

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
