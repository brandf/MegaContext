# POC Scope & Constraints

Guardrails that keep the proof-of-concept focused and verifiable while we build out the full MegaContext loop.

- **Frozen base LLM:** start with an off-the-shelf checkpoint; no fine-tuning during the initial loop. Lightweight LoRA adapters are a follow-up once the pipeline is stable.
- **Two-level MegaContext gist tree:** limit lifetime memory to moderate sizes so two hierarchical levels (L0 tokens, L1 gists, L2 roots) suffice for the demo.
- **Synchronous updates:** keep the MegaContext tree resident in RAM/GPU for the POC; ingest → focus → decode happens between autoregressive steps with no background streaming yet.

These constraints tie directly to the milestones in [[plans/POC Plan]] and inform the performance envelope summarized in [[Performance Sketch]].
