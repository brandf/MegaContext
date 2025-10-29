---
tags:
  - vision
summary: Explores speculative planning on the MegaContext gist tree using latent reasoning and hierarchical refinement.
---
MegaPrediction adds a speculative region to the gist tree so the system can plan future context, refine it, and commit when ready.

---

- **Present cursor:** separates committed history from speculative spans.
- **Latent planners:** operate in gist space before reifying tokens.
- **Hierarchical refinement:** progressively decode from L2 gists to L0 tokens.
- **[[LensNet]] role:** scores speculative spans to decide refinement order.
- **Training:** reuse ΔNLL losses, explore RL for compute/latency trade-offs.

---
## Details

MegaPrediction complements [[MegaContext Tree]] by forecasting what the model will need next, operating over the same gist hierarchy but inside a speculative "future" region of the tree that sits beyond a movable present cursor.

### Present cursor & speculative spans

- Predicted gists and draft L0 tokens occupy normal tree nodes marked as speculative; the cursor keeps runtime code from confusing unverified futures with committed history.
- The system can iterate—refine, de-gist, or diffuse—within that future partition without disturbing the past, then atomically advance the cursor to commit once outputs are final.
- This layout lets [[LensNet]], the [[Focus Allocator]], and telemetry reuse existing data structures while remaining aware of which spans are tentative.

### Techniques

#### Latent chain-of-thought planners
- Run coarse reasoning entirely in gist space, pruning L0 details to keep planning cheap.
- Optional decoding of these gists back to tokens supports debugging, but normal operation leaves them latent.
- [[LensNet]] focus scores can decide which latent hypotheses justify refinement toward user-visible text.

#### Hierarchical refinement diffusion
- Seed a draft with an L2 gist sequence, pass it through a [[DeGistNet]]-style decoder to reach L1, and apply diffusion-style refinement to squeeze entropy before landing on L0 tokens.
- Hierarchical passes preserve long-range coherence while keeping compute proportional to the level of detail.

#### Hybrid [[LensNet]]-guided refinement
- Combine latent planning and hierarchical output: start in L2+, let [[LensNet]] score speculative spans, and only refine the segments that matter.
- Iteratively alternate scoring and refinement until the tree stabilizes at the desired level of detail, then commit.

### Training & evaluation signals

- Reuse ΔNLL-style losses by comparing refined L0 outputs against ground-truth continuations, avoiding the need to score intermediate gists directly.
- Future RL fine-tuning can blend task rewards with latency or compute costs accrued during speculative planning, pushing MegaPrediction toward efficient, high-quality forecasts.
