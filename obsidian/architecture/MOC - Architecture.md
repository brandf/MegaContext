---
title: "Architecture Map"
type: "moc"
status: "active"
tags: ["moc","architecture"]
summary: "Navigation map for architecture notes under obsidian/architecture/."
links:
  - "[[Architecture Overview]]"
  - "[[POC Architecture]]"
  - "[[Runtime Loop]]"
  - "[[POC Scope]]"
---

- Highlights structural notes covering MegaContext design, proof-of-concept interfaces, runtime loop, and scope constraints.

## TL;DR
- [[Architecture Overview]] — two-context architecture, invariants, key terms.
- [[POC Architecture]] — module responsibilities, storage layout, sample configs.
- [[Runtime Loop]] — ingest → focus → decode pipeline.
- [[POC Scope]] — guardrails for the proof-of-concept milestone.

## Details
- **Architecture Overview:** Explains lifetime vs working context, core components, update cadence, and invariants.
- **POC Architecture & Interfaces:** Module table, environment assumptions, binary formats, run configs.
- **Runtime Loop:** Step-by-step flow (ingest, gist updates, LensNet scoring, allocator actions, decode, telemetry).
- **POC Scope & Constraints:** Frozen base model, two-level gist tree, synchronous updates; ties directly to plan exit criteria.
