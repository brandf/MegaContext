# Implementation Roadmap

A condensed build order that mirrors the original README outline while deferring detailed milestone tracking to [[plans/POC Plan]], [[plans/Paper Plan]], and [[plans/Future Plan]].

1. **32→1 GistNet** — implement and train substitutability (see [[GistNet]]).
2. **MegaContext tree builder** — streaming, 2-level hierarchy in RAM (see [[POC Architecture & Interfaces]] and [[Runtime Loop]]).
3. **LensNet v1 (non-causal)** — implement query-conditioned scorer, train on offline labels (see [[LensNet]]).
4. **Focus allocator** — greedy expand/collapse with hysteresis (see [[Focus Allocator]]).
5. **End-to-end POC loop** — integrate ingest → refocus → decode with telemetry (see [[Runtime Loop]]).
6. **Evaluation** — Loss vs budget scans, C1/C2 relevance tests, and stress cases before moving to research-grade milestones.
