---
tags:
  - index
summary: MegaContext is a system architecture for virtualized LLM context - inspired by graphics tech called MegaTexture by id Software. It enables effectively infinite context at constant compute through hierarchical compression and learned dynamic focus.
---

# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

---

## What is MegaContext?

MegaContext virtualizes context by pairing a disk-backed gist tree called the **MegaContext Tree** with a budgeted **Working Context** governed by **GistNet**, **LensNet**, and the **Focus Allocator**.

It separates a model's context into two parts:
- **MegaContext Tree** (stored on disk) — the complete interaction or document history (potentially millions or billions of tokens) stored as a hierarchical gist tree
- **Working Context** (on GPU) — a fixed 8k–32k token budget mixing raw tokens with learned gists drawn from the MegaContext tree

A learned **GistNet** model builds the MegaContext Tree as a hierarchy of compressed representations (gists). The working context compresses the MegaContext Tree into a fixed-size mix of tokens and gists that are used for inference.

To dynamically adapt level of detail, a learned **LensNet** model continuously and incrementally refocuses the MegaContext Tree onto the Working Context, giving the model **effectively infinite memory at constant compute** with automatic context management.

---

## The MegaTexture Analogy

MegaContext is inspired by **MegaTexture**, a graphics technology from id Software that virtualized texture memory:

- In graphics, **MegaTexture** streams the visible portions of a vast texture mipmap into GPU memory at the appropriate resolution.
- **MegaContext** mirrors that idea for language: instead of mipmap tiles, it maintains embeddings at multiple levels of detail (token L0, gist L1, gist L2, …), yielding effectively unbounded context for a frozen LLM.

Just as MegaTexture lets game artists have unlimited texture detail by streaming the visible portions at appropriate resolutions, MegaContext lets language models have unlimited context by focusing detailed computation only where needed.

---

## Why MegaContext Matters

The core intuition motivating this work is that **long context is only useful if the model can focus on the relevant parts and ignore distractors efficiently**.

MegaContext aims to enable:

- **Persistent memory:** unbounded history at constant compute—conversations could persist for years without retraining or forgetting
- **Dynamic attention:** learned focus policies that can refocus based on relevance, not just recency. Something that wasn't previously relevant can become relevant, requiring dynamic focusing and defocusing
- **Smaller, smarter models:** shift parameter budget away from memorized facts toward reasoning, abstraction, and planning. Knowledge lives in the MegaContext memory instead of the weights
- **Core knowledge as dynamic system prompt:** a curated MegaContext becomes a massive, updatable system prompt that evolves independently of model weights
- **Agentic workflows:** richer coding/knowledge agents with persistent tasks across long-lived projects without manual summarization or brittle RAG pipelines
- **Virtually infinite memory:** the MegaContext can grow unbounded while per-step compute and GPU RAM remain constant

---

## How It Works

### Core Components

- **MegaContext gist tree** — built incrementally as text streams in (every 32 tokens → L1 gist; every 32 L1 gists → L2 gist; etc.)
- **Working context** — contiguous window over the tree; total token cost is capped by `W_max`
- **GistNet** — a lightweight network that compresses local spans (e.g., 32→1) into **gists** that act as substitutable stand-ins for their source tokens. Stacking gists-of-gists yields a hierarchical, lossy representation of the full MegaContext history
- **LensNet + focus allocator** — LensNet scores each working-context entry (token embedding or gist) for expansion or collapse; a block-aligned focus allocator applies those scores, streaming finer- or coarser-grained entries in and out while respecting contiguity and the budget

### Runtime Lifecycle

1. **Ingest & summarize.** Buffer incoming tokens in 32-token blocks, roll them into new or updated gist nodes, and persist the MegaContext tree
2. **Assemble the working context.** Lay out a contiguous-in-time sequence of tokens and gists whose combined token-equivalent cost stays within `W_max`
3. **Refocus.** LensNet reads the current working context, emits signed focus scores, and the focus allocator applies block-aligned expansions/collapses without breaking contiguity or budget
4. **Decode.** The frozen base LLM consumes the refreshed working context to predict the next token(s), feeding newly generated tokens back into step 1

---

## Project Status

We are currently in **Phase 2** of the proof-of-concept milestone, building out the GistNet compression model with teacher-student training. The POC aims to demonstrate that hierarchical gists plus dynamic focus can keep a frozen LLM within a fixed working window while retaining task-relevant history.

See [[POC Plan]] for the full roadmap.

---

## Documentation Navigation

### Getting Started
- [[Getting Started]] — start here to learn about the MegaContext project
- [[MegaTexture Analogy]] — the inspiration from graphics tech
- [[How MegaContext Works]] — overview with visual diagrams
- [[Glossary]] — key terms and definitions

### Architecture
- [[Architecture]] — system design, runtime, and POC scope
- [[Architecture Details]] — two-context architecture, invariants, key terms
- [[POC Architecture]] — module responsibilities, storage layout, sample configs
- [[Runtime Loop]] — ingest → focus → decode pipeline
- [[POC Scope]] — guardrails for the proof-of-concept milestone
- [[Components]] — deep dives into GistNet, LensNet, Focus Allocator, and more

### Plans & Operations
- [[Plans]] — milestone plans (POC, Paper, Future)
- [[POC Plan]] — stepwise guide for delivering the proof-of-concept
- [[Ops]] — training cadence, workflow, performance, pruning
- [[Base Runtime]] — operational runtime details

### Vision & Future
- [[Vision]] — long-range research and product direction
- [[Grand Vision]] — why MegaContext matters long term and future directions
- [[Cognitive Core]] — advanced cognitive architectures
- [[MegaPrediction]] — speculative planning within the gist tree
- [[MegaCuration]] — adaptive pruning and knowledge management

### Reference
- [[Reference]] — comparative analyses and literature
- [[Related Work]] — prior art and research context
- [[MegaContext  & RAG]] — comparison with RAG approaches

---

## Contributing

This is an open research project. We welcome contributions that advance the proof-of-concept and the broader vision.

- Follow conventions in `AGENTS.md` for directory layout, testing, linting, and communication
- Bootstrap: `uv venv`, `uv sync`, then `uv run pytest --maxfail=1 --disable-warnings` for smoke tests
- Lint/format via `uv run ruff check src tests` and `uv run black src tests`
- Update progress in the Obsidian plan notes before hand-off so the next contributor has full context

See the [GitHub repository](https://github.com/brandf/MegaContext) for more details.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures—focusing detailed computation only where needed. It opens a path to persistent, updatable, and truly lifelong language models.*
