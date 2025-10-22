---
title: "MegaContext Map of Content"
type: "moc"
status: "active"
tags: ["moc"]
summary: "Entry map that orients contributors across the MegaContext knowledge graph."
links:
  - "[[Architecture Overview]]"
  - "[[MOC - Core Components]]"
  - "[[MOC - Knowledge Workflow]]"
aliases: ["MegaContext"]
---

- MegaContext combines a lifetime gist tree with a focused working window; use this MOC to jump to any subsystem, workflow, or roadmap detail.

## TL;DR
- **Orientation:** [[Architecture Overview]], [[MOC - Knowledge Workflow]]
- **Core modules:** [[MOC - Core Components]], [[GistNet]], [[LensNet]], [[Focus Allocator]]
- **Runtime:** [[Runtime Loop]], [[POC Architecture]]
- **Operations & training:** [[Training & Operations]], [[Implementation Roadmap]]
- **Research arcs:** [[Comparison - MegaContext vs RAG]], [[Grand Vision]], [[Cognitive Core]], [[MegaPrediction]], [[Pruning MegaContext]], [[Related Work]]
- **Milestones:** [[plans/POC Plan]], [[plans/Paper Plan]], [[plans/Future Plan]]

## Details

### Orientation & Architecture
- [[Architecture Overview]] — concept note outlining the two-context architecture and invariants.
- [[POC Scope]] — constraints that keep the proof-of-concept tractable.
- [[POC Architecture]] — interfaces, binary formats, and environment assumptions.

### Core Modules (Linked MOC)
- [[MOC - Core Components]] — sub-map for module deep dives.
  - [[GistNet]] — 32→1 compression.
  - [[LensNet]] — focus scoring.
  - [[Focus Allocator]] — block-aligned action loop.

### Runtime & Performance
- [[Runtime Loop]] — ingest → focus → decode walkthrough.
- [[Performance Sketch]] — compute/storage envelopes.

### Training & Workflow
- [[Training & Operations]] — alternating optimization and telemetry practice.
- [[Implementation Roadmap]] — build order tied to milestones.
- [[MOC - Knowledge Workflow]] — Capture → Process → Refine → Create pipeline for this vault.

### Research & Vision
- [[Comparison - MegaContext vs RAG]] — positioning vs retrieval pipelines.
- [[Grand Vision]] — higher-order goals and future directions.
- [[Cognitive Core]] — roadmap for small reasoning cores backed by MegaContext.
- [[MegaPrediction]] — speculative planning on the gist tree.
- [[Pruning MegaContext]] — telemetry-driven memory hygiene.
- [[Related Work]] — citations and inspiration.

### Plans & Milestones
- [[plans/POC Plan]] — execution tracker for the prototype.
- [[plans/Paper Plan]] — research paper milestone.
- [[plans/Future Plan]] — post-paper growth and adoption.

## Layer 3 · Capture & Processing Log
- 2025-10-22: README content refactored into atomic notes; MOC created to anchor progressive summarization.
- 2025-10-22: Workflow MOC introduced to support Capture → Process → Refine → Create cadence.

## Metadata & Queries
- `type:: moc` — surface in graph view to see first-order clusters.
- `status:: active` — this map is maintained alongside architectural updates.
- Use Dataview query (in Obsidian) `table summary where type="concept"` to list individual atomic notes.
