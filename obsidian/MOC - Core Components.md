---
title: "Core Components Map"
type: "moc"
status: "active"
tags: ["moc","modules","megacontext"]
summary: "Navigation hub for MegaContext runtime components and their relationships."
links:
  - "[[GistNet]]"
  - "[[LensNet]]"
  - "[[Focus Allocator]]"
  - "[[Runtime Loop]]"
aliases: ["Core Components"]
---

## Layer 0 · Capture Summary
- Use this map to traverse the runtime stack: compression ([[GistNet]]), scoring ([[LensNet]]), allocation ([[Focus Allocator]]), and orchestration ([[Runtime Loop]]).

## Layer 1 · Component Clusters
- **Compression:** [[GistNet]] → feeds [[MegaContext]] hierarchy (see [[POC Architecture]]).
- **Focus control:** [[LensNet]] ↔ [[Focus Allocator]] work in tandem each block.
- **Runtime orchestration:** [[Runtime Loop]] consumes outputs from all components.
- **Telemetry & tooling:** [[Training & Operations]], [[Implementation Roadmap]], [[Performance Sketch]].

## Layer 2 · Detailed Notes
- [[GistNet]] — 32→1 hierarchical gists aligned with the base embedding space.
- [[LensNet]] — dual cross-attention scorer providing signed utilities per working-context entry.
- [[Focus Allocator]] — greedy, hysteresis-aware application of LensNet utilities.
- [[Runtime Loop]] — streaming ingest, focus adjustment, and base LLM decode.
- [[Training & Operations]] — counterfactual labeling, alternating optimization, and telemetry.
- [[Performance Sketch]] — expected compute/storage envelopes at various scales.

## Layer 3 · Change Log
- 2025-10-22: Refactored from README summary into dedicated MOC; added dense cross-links to subsystem notes.
