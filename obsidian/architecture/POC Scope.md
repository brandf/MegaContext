---
tags:
  - architecture
summary: Defines the guardrails that keep the proof-of-concept focused, reproducible, and verifiable.
---
The POC sticks to a frozen base model, a two-level gist tree, and synchronous updates so contributors can prove end-to-end behavior quickly.

---

- **Frozen base:** no fine-tuning during initial loop; LoRA is follow-up work.
- **Hierarchy:** two-level [[MegaContext Tree]] suffices for moderate contexts.
- **Update cadence:** synchronous ingest/refocus between autoregressive steps.
- **Linkages:** informs [[POC Plan]] deliverables and [[POC Architecture]] assumptions.

---
## Details

Guardrails that keep the proof-of-concept focused and verifiable while we build out the full [[MegaContext Tree]] loop.

- **Frozen base LLM:** start with an off-the-shelf checkpoint; no fine-tuning during the initial loop. Lightweight LoRA adapters are a follow-up once the pipeline is stable.
- **Two-level MegaContext gist tree:** limit lifetime memory to moderate sizes so two hierarchical levels (LOD0 tokens, LOD1 gists, LOD2 roots) suffice for the demo.
- **Synchronous updates:** keep the [[MegaContext Tree]] resident in RAM/GPU for the POC; ingest → focus → decode happens between autoregressive steps with no background streaming yet.

These constraints tie directly to the milestones in [[POC Plan]] and inform the performance envelope summarized in [[Performance Sketch]].
