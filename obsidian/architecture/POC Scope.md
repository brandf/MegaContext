---
title: "POC Scope & Constraints"
type: "concept"
status: "active"
tags: ["ops"]
summary: "Defines the guardrails that keep the proof-of-concept focused, reproducible, and verifiable."
links:
  - "[[POC Architecture]]"
  - "[[plans/POC Plan]]"
  - "[[MOC - Knowledge Workflow]]"
---

## Summary (L0)
- The POC sticks to a frozen base model, a two-level gist tree, and synchronous updates so contributors can prove end-to-end behavior quickly.

## Highlights (L1)
- **Frozen base:** no fine-tuning during initial loop; LoRA is follow-up work.
- **Hierarchy:** two-level MegaContext tree suffices for moderate contexts.
- **Update cadence:** synchronous ingest/refocus between autoregressive steps.
- **Linkages:** informs [[plans/POC Plan]] deliverables and [[POC Architecture]] assumptions.

## Deep Dive (L2)

Guardrails that keep the proof-of-concept focused and verifiable while we build out the full MegaContext loop.

- **Frozen base LLM:** start with an off-the-shelf checkpoint; no fine-tuning during the initial loop. Lightweight LoRA adapters are a follow-up once the pipeline is stable.
- **Two-level MegaContext gist tree:** limit lifetime memory to moderate sizes so two hierarchical levels (L0 tokens, L1 gists, L2 roots) suffice for the demo.
- **Synchronous updates:** keep the MegaContext tree resident in RAM/GPU for the POC; ingest → focus → decode happens between autoregressive steps with no background streaming yet.

These constraints tie directly to the milestones in [[plans/POC Plan]] and inform the performance envelope summarized in [[Performance Sketch]].
