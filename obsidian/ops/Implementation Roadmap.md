---
tags:
  - ops
summary: Sequenced build order for MegaContext components, aligned with milestone plans.
---
Quick reference for building the MegaContext stack—from compression to evaluation—mirroring milestone plans.

---

- **Compression first:** implement [[GistNet]].
- **Memory scaffolding:** build the MegaContext tree per [[POC Architecture]].
- **Focus stack:** launch [[LensNet]] and [[Focus Allocator]].
- **Integration:** wire the [[Runtime Loop]] and validate with telemetry.
- **Evaluation:** run loss-vs-budget, C1/C2, and stress tests before scaling.

---
## Details

1. **32→1 GistNet** — implement and train substitutability (see [[GistNet]]).
2. **MegaContext tree builder** — streaming, 2-level hierarchy in RAM (see [[POC Architecture & Interfaces]] and [[Runtime Loop]]).
3. **LensNet v1 (non-causal)** — implement query-conditioned scorer, train on offline labels (see [[LensNet]]).
4. **Focus allocator** — greedy expand/collapse with hysteresis (see [[Focus Allocator]]).
5. **End-to-end POC loop** — integrate ingest → refocus → decode with telemetry (see [[Runtime Loop]]).
6. **Evaluation** — Loss vs budget scans, C1/C2 relevance tests, and stress cases before moving to research-grade milestones.