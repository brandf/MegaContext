---
tags:
  - moc
  - architecture
summary: Highlights structural notes covering MegaContext design, proof-of-concept interfaces, runtime loop, and scope constraints.
---
Highlights structural notes covering MegaContext design, proof-of-concept interfaces, runtime loop, and scope constraints.

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
    - Frozen base model, two-level gist tree, synchronous updates; ties directly to [[POC Plan]] exit criteria.
