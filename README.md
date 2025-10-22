# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory. This document is the quickstart index; detailed design notes now live in the Obsidian vault.*

---

## TL;DR — MegaContext
MegaContext is a proposed system architecture for virtualized LLM context - inspired by a graphics concept called MegaTexture by ID software.

It separates a model’s context into a "MegaContext" (stored on disk) and a "working context" (on GPU). A learned GistNets model is used to build the MegaContext as a hierarchical gist tree. The working context compresses the MegaContext into a fixed-size mix of tokens and gists that is used for inference.

To dynamical adapt level of detail, a learned LensNet model, continuously/incrementally refocuses the MegaContext onto the working context, giving the model effectively infinite memory at constant compute with automatic context management.

---

## Documentation

- Obsidian vault lives under `obsidian/`; open it as a vault in Obsidian. The entry map is [MegaContext Map](obsidian/MOC%20-%20MegaContext.md).

### [Architecture](obsidian/architecture/MOC%20-%20Architecture.md)
- [Architecture Overview](obsidian/architecture/Architecture%20Overview.md) — lifetime vs working context, invariants, and core components.
- [POC Architecture & Interfaces](obsidian/architecture/POC%20Architecture.md) — module responsibilities, storage layout, sample configs.
- [Runtime Loop](obsidian/architecture/Runtime%20Loop.md) — ingest → gist update → focus → decode pipeline.
- [POC Scope & Constraints](obsidian/architecture/POC%20Scope.md) — guardrails for the proof-of-concept milestone.

### [Modules](obsidian/modules/MOC%20-%20Modules.md)
- [GistNet — Local Gist Extraction](obsidian/modules/GistNet.md) — 32→1 hierarchical compression.
- [LensNet — Focus Scoring](obsidian/modules/LensNet.md) — dual cross-attention controller for working-context entries.
- [Focus Allocator — Block-Aligned Actions](obsidian/modules/Focus%20Allocator.md) — greedy expand/collapse loop that enforces budgets.

### [Operations](obsidian/ops/MOC%20-%20Ops.md)
- [Training & Operations](obsidian/ops/Training%20%26%20Operations.md) — alternating optimization, telemetry, evaluation.
- [Implementation Roadmap](obsidian/ops/Implementation%20Roadmap.md) — sequenced build order for the stack.
- [Performance Sketch](obsidian/ops/Performance%20Sketch.md) — compute/storage envelopes and long-term retention costs.
- [Pruning MegaContext](obsidian/ops/Pruning%20MegaContext.md) — telemetry-driven pruning strategy.
- [Knowledge Workflow](obsidian/ops/Knowledge%20Workflow.md) — Capture → Process → Refine → Create note lifecycle.

### [Plans](obsidian/plans/MOC%20-%20Plans.md)
- [POC Plan](obsidian/plans/POC%20Plan.md) — proof-of-concept execution plan.
- [Paper Plan](obsidian/plans/Paper%20Plan.md) — research milestone plan with benchmarks and packaging.
- [Future Plan](obsidian/plans/Future%20Plan.md) — post-paper roadmap for platform, learning, applications, research, and DX.

### [Vision](obsidian/vision/MOC%20-%20Vision.md)
- [Grand Vision](obsidian/vision/Grand%20Vision.md) — long-term motivation and future directions.
- [Cognitive Core](obsidian/vision/Cognitive%20Core.md) — roadmap for compact reasoning models backed by MegaContext.
- [MegaPrediction — Forecasting Future Context](obsidian/vision/MegaPrediction.md) — speculative planning inside the gist tree.

### [Reference](obsidian/reference/MOC%20-%20Reference.md)
- [Comparison — MegaContext vs. RAG](obsidian/reference/Comparison%20-%20MegaContext%20vs%20RAG.md) — framing against retrieval pipelines.
- [Related Work](obsidian/reference/Related%20Work.md) — citations inspiring compression, focus, and memory design.

---

## Development & Contribution

- Follow the conventions in `AGENTS.md` for directory layout, testing, linting, and communication.
- Bootstrap: `uv venv`, `uv sync`, then `uv run pytest --maxfail=1 --disable-warnings` for smoke tests. Lint/format via `uv run ruff check src tests` and `uv run black src tests`.
- Tooling and demos live under `tools/`; see the corresponding Obsidian notes for command examples.
- Update progress in the Obsidian plan notes before hand-off so the next contributor has full context.

---

# License

MIT License (suggested). PRs welcome—please include reproducible tests for GistNet, LensNet, the focus allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures—focusing detailed computation only where needed. It opens a path to persistent, updatable, and truly lifelong language models.*
