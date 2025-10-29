---
tags:
  - architecture
summary: Highlights structural notes covering MegaContext design, proof-of-concept interfaces, runtime loop, and scope constraints.
---
# Architecture

MegaContext separates context into two parts (see [[Architecture Details]] for complete explanation): [[MegaContext Tree]] (unbounded storage) and [[Working Context]] (fixed GPU window). The tree stores complete history as hierarchical [[Glossary#Gist / Gist Embedding|gists]] on disk, while the working context holds a dynamically focused token+gist mix within a fixed budget for inference.

## Architecture Documentation

- [[Architecture Details]]
    - Two-context architecture, invariants, key terms.
    - Explains [[MegaContext Tree]] vs [[Working Context]], core [[Components]], update cadence, and invariants.
- [[POC Architecture]]
    - Module responsibilities, storage layout, sample configs.
    - Module table, environment assumptions, binary formats, run configs.
- [[Runtime Loop]]
    - Ingest → focus → decode pipeline.
    - Step-by-step flow (ingest, gist updates, [[LensNet]] scoring, [[Focus Allocator]] actions, decode, telemetry).
- [[POC Scope]]
    - guardrails for the proof-of-concept milestone.
    - [[Glossary#Frozen Base Model|Frozen base model]], two-level gist tree, synchronous updates; ties directly to [[POC Plan]] exit criteria.
