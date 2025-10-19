# README Refinement Roadmap

## Phase 1 — Correctness & Terminology Fixes
- Repair typos/inconsistencies (e.g., “2 laters”), clarify undefined symbols, and restate runtime invariants where missing.
- Tighten causal ordering around ingestion vs. generation, and highlight ΔNLL@`H` defaults per workload.
**Status:** Ready for your review.
### Questions
- Q1: In GistNet Stage 4, should `ε` represent learned noise, a trainable offset, or just be dropped?  
  + A1: idk what that is, and I don't like the name g_star...i think G2 should be another learned 'slot' query like Stage 2 
- Q2: When decoding, do newly generated tokens immediately trigger gist recomputation, or should the system wait until the 32-token buffer fills?
  + A2: there are two places GistNet is used: 1) to incrementally update the lifetime gist tree, we should just buffer until we have 32 L0 tokens/embeddings and/or 32 L1 gists before we update the lifetime context gists. 2) the LensNet is conditioned on gists for the recent context, but it's only updated periodically (e.g. every 256 entries).  we can either get these from the Lifetime gist tree or compute them before LensNet is run.  
- Q3: What horizons (`H`) should we document as defaults for narrative vs. code traces?
  + A3: i'm not sure.  what do you think?  just use your best judgement.

## Phase 2 — Structure & Signposting
- Reorder or trim sections (e.g., relocate **Grand vision**) and add bridges that guide readers from high-level concepts into POC mechanics.
- Introduce visual aids (diagram reference) and reminder callouts for key invariants.
**Status:** Ready for your review.
### Questions
- Q4: Are you comfortable moving the **Grand vision** section after the POC mechanics, or would you prefer it summarized instead?
  + A4: yes that's fine.  i don't want it to be a distraction, but at the same time I think it might get people interested in reading the rest of this lengthy document.  your call on what to do with it.  
- Q5: Do you want a single composite diagram (lifetime ↔ working context) or multiple smaller schematics? Any style preferences?  
  + A5: multiple smaller is probably better.  also i don't think the existing ascii diagram is very good / accurate.  If these are too hard to make, maybe you should just make a <DIAGRAM NEEDED> block where you describe what we want and I'll make it in figma or whatever.
- Q6: Should reminder callouts be inline text, sidebars, or linked footnotes?
  + A6: this is markdown, so if you're able to link within the doc that would be great, otherwise use your best judgement.
  
## Phase 3 — Implementation Interfaces & Storage Details
- Expand API contracts for runtime classes, document tensor shapes, and spell out serialization layouts (`{L0,L1,L2}.ctx` structure).
- Align tooling guidance with Python-native workflows (no `make`) and clarify dataset preparation steps.
**Status:** Blocked on Phase 2 sign-off.
### Questions
- Q7: Do you prefer standard `uv` CLI invocations or `poetry run` as the canonical example in docs?
  + A7: i'm not that familiar with either (or python ecosystem in general) so use your best judgement for whatever is appropriate / modern / popular.
- Q8: For `{L0,L1,L2}.ctx`, can we assume contiguous float16 arrays with separate index metadata, or do you favor an embedded header per node?  
  + A8: for L0 they can just be integer tokens indexing the vocabulary of the base model.  for L1 and L2 they should be whatever float format vectors that are used by the embeddings in the base model. i think they can all be packed densly with no other metadata/headers so that random access is simple, given that they are all fixed size.  the files themselves can have a FIXED SIZED header if that's useful.
- Q9: How should we recommend seeding for deterministic tests beyond `PYTHONHASHSEED` (e.g., `torch.manual_seed`, `random.seed`)?
  + A9: idk, use your best judgement.

## Phase 4 — Evaluation Criteria & Telemetry
- Document success thresholds (ΔNLL targets, swap rates, latency budgets) and failure signals so implementers know how to validate runs.
- Clarify recommended defaults for allocator thresholds/cooldowns and logging expectations.
**Status:** Blocked on Phase 3 sign-off.
### Questions
- Q10: What ΔNLL reduction and swap-rate ranges should we present as acceptable for the POC?
  + A10: I have never done any serious LLM evals or even a big LLM project like this before, so idk.  use youre best judgement.  What I would like the POC to show is that compared to the non-MegaContext base model the MegaContext version is faster (and compute is bounded) at long contexts for similar or better accuracy.
- Q11: Which allocator defaults (`τ_expand`, `τ_collapse`, `N_diff`, cooldown) should we treat as canonical until empirical tuning?  
  + A11: idk, use your best judgement.  whatever the smallest values that are likely to produce reasonable results.
- Q12: Which telemetry metrics are “must log” for every run vs. nice-to-have?
  + A12: use your best judgement. whatever an AI researcher would need to understand that it's working or broken.
