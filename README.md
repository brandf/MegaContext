# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

---

## TL;DR — MegaContext
MegaContext virtualizes context by pairing a disk-backed gist tree called the MegaContext Tree with a budgeted working context governed by GistNet, LensNet, and the Focus Allocator.

It separates a model’s context into a MegaContext Tree (stored on disk) and a Working Context (on GPU). A learned GistNet model is used to build the MegaContext Tree as a hierarchy of gists. The working context compresses the MegaContext Tree into a fixed-size mix of tokens and gists that are used for inference.

To dynamically adapt level of detail, a learned LensNet model, continuously/incrementally refocuses the MegaContext Tree onto the Working Context, giving the model effectively infinite memory at constant compute with automatic context management.

---

## Documentation

Obsidian vault lives under `obsidian/`; for the best viewing experience open in Obsidian client.

### Getting Started
- [MegaTexture Analogy](obsidian/getting%20started/MegaTexture%20Analogy.md) — visualizes the MegaTexture inspiration that guides the project.
- [How MegaContext Works](obsidian/getting%20started/How%20MegaContext%20Works.md) — high-level walkthrough of the workflow and major components.
- [Glossary](obsidian/getting%20started/Glossary.md) — quick reference for recurring terms and acronyms.

### Architecture
- [Architecture Details](obsidian/architecture/Architecture%20Details.md) — lifetime vs. working context, invariants, and core components.
- [POC Architecture](obsidian/architecture/POC%20Architecture.md) — module responsibilities, storage layout, and sample configs.
- [Runtime Loop](obsidian/architecture/Runtime%20Loop.md) — ingest → gist update → focus → decode pipeline.
- [POC Scope](obsidian/architecture/POC%20Scope.md) — guardrails for the proof-of-concept milestone.

### Modules
- [GistNet](obsidian/architecture/components/GistNet.md) — 32→1 hierarchical compression that builds the gist tree.
- [LensNet](obsidian/architecture/components/LensNet.md) — dual cross-attention controller for working-context entries.
- [Focus Allocator](obsidian/architecture/components/Focus%20Allocator.md) — greedy expand/collapse loop that enforces budgets.
- [MegaContext Tree](obsidian/architecture/components/MegaContext%20Tree.md) — disk-resident hierarchy of gists and tokens.
- [Working Context](obsidian/architecture/components/Working%20Context.md) — GPU-resident mix of tokens and gists surfaced to the LLM.

### Ops
- [Training & Operations](obsidian/ops/Training%20%26%20Operations.md) — alternating optimization, telemetry, and evaluation.
- [Implementation Roadmap](obsidian/ops/Implementation%20Roadmap.md) — sequenced build order for the stack.
- [Performance Sketch](obsidian/ops/Performance%20Sketch.md) — compute/storage envelopes and long-term retention costs.

### Plans
- [POC Plan](obsidian/plans/POC%20Plan.md) — proof-of-concept execution plan.
- [Research Paper Plan](obsidian/plans/Research%20Paper%20Plan.md) — research milestone plan with benchmarks and packaging.
- [Future Plan](obsidian/plans/Future%20Plan.md) — post-paper roadmap for platform, learning, applications, research, and DX.

### Vision
- [Grand Vision](obsidian/vision/Grand%20Vision.md) — long-term motivation and future directions.
- [Cognitive Core](obsidian/vision/Cognitive%20Core.md) — roadmap for compact reasoning models backed by MegaContext.
- [MegaPrediction](obsidian/vision/MegaPrediction.md) — speculative planning inside the gist tree.
- [MegaCuration](obsidian/vision/MegaCuration.md) — learned pruning and curation anchored by focus telemetry.
- [Realtime Scenarios](obsidian/vision/Realtime%20Scenarios.md) — near-term applications unlocked by effectively infinite context.

### Reference
- [MegaContext & RAG](obsidian/reference/MegaContext%20%20%26%20RAG.md) — framing against retrieval pipelines.
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
