# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory. This document is the quickstart index; detailed design notes now live in the Obsidian vault.*

**Roadmaps:** Development is tracked across three Obsidian plans:
- `obsidian/plans/POC Plan.md` — the hot-path prototype proving MegaContext end-to-end.
- `obsidian/plans/Paper Plan.md` — the research milestone targeting a publishable system.
- `obsidian/plans/Future Plan.md` — post-paper growth, adoption, and advanced research.

**Obsidian vault:** Shared project notes and canvases live in `obsidian/`; open that directory as a vault in Obsidian. The `.obsidian/` settings folder stays local by default so personal layouts do not churn the repo.

---

## TL;DR — MegaContext
MegaContext is a proposed system architecture for virtualized LLM context - inspired by a graphics concept called MegaTexture by ID software.

It separates a model’s context into a "MegaContext" (stored on disk) and a "working context" (on GPU). A learned GistNets model is used to build the MegaContext as a hierarchical gist tree. The working context compresses the MegaContext into a fixed-size mix of tokens and gists that is used for inference.

To dynamical adapt level of detail, a learned LensNet model, continuously/incrementally refocuses the MegaContext onto the working context, giving the model effectively infinite memory at constant compute with automatic context management.

---

## Obsidian Knowledge Base

**Core system**
- [MegaContext Vault Index](obsidian/MegaContext.md) — entry point linking every vault note.
- [Architecture Overview](obsidian/Architecture%20Overview.md) — conceptual walkthrough of the runtime loop and invariants.
- [Core Components](obsidian/Core%20Components.md) — high-level summary of GistNet, LensNet, allocator, and supporting modules.
- [Runtime Loop](obsidian/Runtime%20Loop.md) — step-by-step decode flow for ingest → focus → decode.

**Engineering details**
- [POC Architecture & Interfaces](obsidian/POC%20Architecture.md) — module map, environment assumptions, binary formats, and sample configs.
- [POC Scope & Constraints](obsidian/POC%20Scope.md) — guardrails for the prototype milestone.
- [GistNet](obsidian/GistNet.md), [LensNet](obsidian/LensNet.md), [Focus Allocator](obsidian/Focus%20Allocator.md) — deep dives on each subsystem.
- [Performance Sketch](obsidian/Performance%20Sketch.md) — compute/storage envelope estimates.
- [Training & Operations](obsidian/Training%20%26%20Operations.md) — alternating optimization, instrumentation, and validation checklist.
- [Implementation Roadmap](obsidian/Implementation%20Roadmap.md) — condensed build order aligned with the milestone plans.

**Research & positioning**
- [Comparison — MegaContext vs. RAG](obsidian/Comparison%20-%20MegaContext%20vs%20RAG.md) — framing against retrieval pipelines.
- [Grand Vision](obsidian/Grand%20Vision.md) — long-term motivation and future directions.
- [Cognitive Core](obsidian/Cognitive%20Core.md) — roadmap for compact reasoning models backed by MegaContext memories.
- [MegaPrediction](obsidian/MegaPrediction.md) — speculative planning over the gist hierarchy.
- [Pruning MegaContext](obsidian/Pruning%20MegaContext.md) — telemetry-driven memory maintenance.
- [Related Work](obsidian/Related%20Work.md) — references that inform the project.

---

## Development & Contribution

- Follow the conventions in `AGENTS.md` for directory layout, testing, linting, and communication.
- Bootstrap: `uv venv`, `uv sync`, then `uv run pytest --maxfail=1 --disable-warnings` for smoke tests. Lint/format via `uv run ruff check src tests` and `uv run black src tests`.
- Tooling and demos live under `tools/`; see the corresponding Obsidian notes for command examples.
- Update progress in the Obsidian plan notes before hand-off so the next contributor has full context.

---

## License & contributions

MIT License (suggested). PRs welcome—please include reproducible tests for GistNet, LensNet, the focus allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures—focusing detailed computation only where needed. It opens a path to persistent, updatable, and truly lifelong language models.*
