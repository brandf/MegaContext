# README Critique — MegaContext

## Perspective 1: LLM Researcher

### Clarity & Flow
- The TL;DR and early mechanics read well, but the jump from high-level flow to detailed dataclasses (in `POC architecture & interfaces`) is abrupt; a short narrative bridge or schematic would help transition from concept to implementation.
- The **Grand vision** section is inspiring yet long and currently interrupts the technical deep dive; consider trimming or moving it after the POC sections to maintain momentum for readers hunting for mechanics first.
- Several key invariants (contiguity, block size, signed utilities) are defined once and reused later—adding inline reminders or cross-links when they reappear in GistNet/LensNet sections would reduce backtracking.
- A diagram of the runtime loop (e.g., MegaTexture-style mip map vs. working context) referenced from `assets/` would anchor the metaphor and clarify the bidirectional focus cycle.

### Content Gaps & Accuracy
- GistNet Stage 4 introduces `G2 = G1 + ε`, but `ε` is undefined; clarify whether this is noise, learned offset, or typo.
- POC scope mentions “2 laters” instead of “layers,” and the runtime loop never states how often the lifetime tree rebalances when generated tokens arrive versus user tokens—spell out the causal ordering.
- ΔNLL@`H` is pivotal yet the doc never states typical `H` choices for different tasks (narrative vs. code) or how to normalize utilities across levels; include guidance to avoid inconsistent labeling regimes.
- The Long-term storage example is compelling but assumes token-equivalent storage for 4 096-d features; note how this maps to actual gist widths (e.g., 4 096 = base embedding dim) so readers don’t conflate arbitrary features with token embeddings.
- LensNet’s legality regularizer references `sum_{L0}` and `sum_{L2}`, but the text earlier says L2 is the top level in the POC; clarify whether higher levels exist in future plans or mask terms accordingly to avoid confusion.

### Recommendations & Open Questions
- Provide a compact summary table contrasting “What’s fixed in the POC” vs. “What’s future work,” so researchers know which assumptions they can challenge now.
- Add explicit expectations for evaluation (target ΔNLL reductions, acceptable swap rates) to make theoretical claims falsifiable.
- Open questions: How resilient is the approach when causal relevance spans multiple 32-token blocks (e.g., code diff + explanation) given the strict block alignment? How do you handle cross-span dependencies that require overlapping gist windows?
- What guarantees exist that expanding/collapsing maintains positional encoding consistency once LoRA fine-tuning adjusts the base? Outline whether RoPE phase reuse has empirical backing yet.

## Perspective 2: Coding Assistant / POC Implementer

### Clarity & Flow
- The module/table layout is valuable, but a “getting started” checklist (environment, `make setup`, expected directory tree) is missing; implementers must piece it together from scattered notes.
- Runtime pseudocode names methods like `working.patch` and `allocator.apply` without describing their signatures or return contracts; a short API contract per method would prevent guesswork.
- Training phases (B1/B2/B3) are dense; summarizing them with a timeline or state diagram would clarify which artifacts need to be regenerated between phases.

### Content Gaps & Accuracy
- The sample YAML config omits required packages, seed handling, and how datasets should be prepared (e.g., tokenized shard format). Without that, reproducing the run is ambiguous.
- Storage guidance (`{L0,L1,L2}.ctx` flat files) lacks concrete dtype/shape layouts or how to map `span_id` to byte offsets—essential for anyone implementing serialization.
- Focus allocator guidance talks about aggregating per-block scores but does not detail how to merge siblings with mixed focus directions, nor how `working` maintains contiguity after alternating expand/collapse steps.
- The ΔNLL labeling workflow describes recomputation but not batching limits, caching strategy, or how to reuse KV caches—critical for scoping GPU time.
- Mention of LoRA placement is helpful, yet there is no instruction on how to swap between pure frozen vs. LoRA-augmented checkpoints during inference or evaluation.

### Implementation Readiness & Open Questions
- Spell out exact tensor shapes for `working.to_tensor()` (ordering, padding, mask semantics) so an agent can wire it into Hugging Face models without trial-and-error.
- Clarify how generated tokens are re-ingested: do we recompute gists immediately, or lazily once 32 tokens accumulate? How are partial blocks handled mid-decoding?
- Provide expectations for `WorkingContext` telemetry—what metrics should tests assert (e.g., contiguity checks, residency cooldown counts)?
- Open questions: What are safe defaults for `N_diff`, thresholds, and cooldowns under different workloads? How should we seed randomness for deterministic tests (only via `PYTHONHASHSEED`, or additional torch/cuRAND seeds)?

### Additional Suggestions
- Include concrete CLI examples (`make ingest-data`, `make label-dnll`) and note required environment variables (W&B API key, dataset roots) to reduce setup friction.
- Add a minimal end-to-end test plan outlining which smoke tests must pass before merging (e.g., substitutability regression under `tests/test_gistnet.py` with fake data).
- Document expected failure signals (e.g., ΔNLL spikes above X, swap rates below Y) so future automation or agents can triage issues quickly.

