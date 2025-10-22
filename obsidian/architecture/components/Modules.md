---
tags:
  - moc
  - modules
summary: Navigation map for MegaContext module notes under obsidian/modules/.
---
## TL;DR

Use this map to traverse the runtime stack: compression ([[GistNet]]), scoring ([[LensNet]]), allocation ([[Focus Allocator]]), and orchestration ([[Runtime Loop]]).

- **Compression:** [[GistNet]] → feeds [[MegaContext Tree]] hierarchy (see [[POC Architecture]]).
- **Focus control:** [[LensNet]] ↔ [[Focus Allocator]] work in tandem each block.
- **Runtime orchestration:** [[Runtime Loop]] consumes outputs from all components.
- **Telemetry & tooling:** [[Training & Operations]], [[Implementation Roadmap]], [[Performance Sketch]].

## Details
- [[GistNet]] — 32→1 hierarchical gists aligned with the base embedding space.
- [[LensNet]] — dual cross-attention scorer providing signed utilities per working-context entry.
- [[Focus Allocator]] — greedy, hysteresis-aware application of LensNet utilities.
- [[Runtime Loop]] — streaming ingest, focus adjustment, and base LLM decode.
- [[Training & Operations]] — counterfactual labeling, alternating optimization, and telemetry.
- [[Performance Sketch]] — expected compute/storage envelopes at various scales.
