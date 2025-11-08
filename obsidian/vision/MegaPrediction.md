---
tags:
  - vision
summary: Defines speculative planning atop the wLOD working-context tree, leveraging MegaAttention's multi-scale readouts for unified next-token and gist forecasts.
---
> **Reference:** Conceptual backdrop for the [[MegaPrediction Training]] PRD. Use that PRD for implementation; this page highlights ideas that reach beyond the current POR.

MegaPrediction adds a speculative region to the [[MegaContext Tree]] so the system can plan future context, refine it, and commit when ready—now leveraging the full **wLOD working-context tree** surfaced by [[MegaAttention Training|MegaAttention]]. Instead of bolting on standalone heads, we reuse the final hidden state from the most recent node at each wLODn to forecast tokens, block gists, and ultra-long summaries in one pass.

---

- **Present cursor:** separates committed history from speculative spans.
- **MegaAttention readouts:** reuse the shared LM head on the most recent wLODn hidden states (see [[MegaPrediction Training]]).
- **Latent planners:** operate in gist space before reifying tokens.
- **Hierarchical refinement:** progressively decode from LOD2 gists to LOD0 tokens.
- **[[LensNet]] role:** scores speculative spans to decide refinement order.
- **Training:** reuse ΔNLL losses, explore RL for compute/latency trade-offs.

### MegaAttention-Driven Readouts

- The mixed-LOD batch fed into [[MegaAttention Training|MegaAttention]] already carries wLOD0→wLOD2 nodes. MegaPrediction simply taps the **latest node per level**, applies the shared LM head, and interprets outputs at the appropriate time scale (token, 32-token gist, 1024-token summary).
- No extra projection heads are required; supervision flows through [[MegaPrediction Training]] by pairing each readout with ground-truth gists produced by [[GistNet]] over the teacher-forced future horizon.
- Consistency across iterative edits relies on [[Hierarchical KV Caching Strategy]], so speculative spans never attend to stale summaries.

---
## Beyond the PRD: ideas to incubate

The PRD focuses on wiring speculative readouts into the runtime. Once that lands, we can push further in a few directions:

1. **Speculative workspaces:** treat the “future” partition as a collaborative scratchpad with fork/merge semantics so agents can iterate safely before committing to history.
2. **Richer DeGistNet decoders:** experiment with diffusion-style refinement (LOD2 → LOD1 → LOD0) so we can explore multiple drafts cheaply before materializing tokens.
3. **Hybrid planners:** blend latent planning with selective refinement—LensNet scores every speculative span, but only the high-utility regions get de-gisted.
4. **Policy learning:** go beyond greedy editing with RL/bandit objectives (accuracy × latency × compute), turning MegaPrediction into a tunable control policy rather than a fixed loop.
5. **Telemetry feedback:** log speculative-span outcomes (accepted vs. discarded) and reuse that signal to teach LensNet/Focus Allocator which edits are worth exploring.

Capture new concepts here first; once they graduate into deliverables, open or extend a PRD so the POR remains the single source of truth.
