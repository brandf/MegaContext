---
tags:
  - index
summary: MegaContext is a system architecture for virtualized LLM context - inspired by graphics tech called MegaTexture by id Software. It enables effectively infinite context at constant compute through hierarchical compression and learned dynamic focus.
---

# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

---

## What is MegaContext?

MegaContext virtualizes context by separating it into two parts (see [[Architecture Details]] for complete explanation): the [[MegaContext Tree]] (unbounded storage on disk) holds the complete history as a hierarchical gist tree, while the [[Working Context]] (fixed GPU window of 8k–32k tokens) holds a dynamically focused mix of raw tokens and compressed gists. [[GistNet]] compresses history into the tree, [[LensNet]] scores relevance, and the [[Focus Allocator]] streams the right level of detail into the working window, giving the model **effectively infinite memory at constant compute** with automatic context management.

---

## The MegaTexture Analogy

MegaContext is inspired by **MegaTexture**, a graphics technology from id Software that virtualized texture memory—streaming visible portions of vast textures at appropriate resolutions. MegaContext applies this same principle to language model context.

See [[MegaTexture Analogy]] for the complete explanation.

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

## [[How MegaContext Works]]

### Core [[Components]]

- **MegaContext gist tree** — built incrementally as text streams in (every 32 tokens → L1 gist; every 32 L1 gists → L2 gist; etc.)
- **[[Working Context]]** — contiguous window over the tree; total token cost is capped by `W_max`
- **[[GistNet]]** — a lightweight network that compresses local spans (e.g., 32→1) into **[[Glossary#Gist / Gist Embedding|gists]]** that act as [[Glossary#Substitutability|substitutable]] stand-ins for their source tokens. Stacking gists-of-gists yields a hierarchical, lossy representation of the full MegaContext history
- **[[LensNet]] + [[Focus Allocator]]** — [[LensNet]] scores each working-context entry (token [[Glossary#Embedding|embedding]] or gist) for [[Glossary#Expand|expansion]] or [[Glossary#Collapse|collapse]]; a block-aligned [[Focus Allocator]] applies those scores, streaming finer- or coarser-grained entries in and out while respecting contiguity and the budget

### [[Runtime Loop|Runtime Lifecycle]]

1. **Ingest & summarize.** Buffer incoming tokens in 32-token blocks, roll them into new or updated gist nodes, and persist the [[MegaContext Tree]]
2. **Assemble the [[Working Context]].** Lay out a contiguous-in-time sequence of tokens and gists whose combined token-equivalent cost stays within `W_max`
3. **Refocus.** [[LensNet]] reads the current [[Working Context]], emits signed focus scores, and the [[Focus Allocator]] applies block-aligned [[Glossary#Expand|expansions]]/[[Glossary#Collapse|collapses]] without breaking contiguity or budget
4. **Decode.** The [[Glossary#Frozen Base Model|frozen base LLM]] consumes the refreshed [[Working Context]] to predict the next token(s), feeding newly generated tokens back into step 1

---

## Project Status

We are currently in **Phase 2** of the proof-of-concept milestone, building out the [[GistNet]] compression model with teacher-student training. The POC aims to demonstrate that hierarchical gists plus dynamic focus can keep a frozen LLM within a fixed working window while retaining task-relevant history.

See [[POC Plan]] for the full roadmap.

---

## Documentation Navigation

### Core Concepts
- [[Getting Started]] — start here to learn about the MegaContext project
- [[MegaTexture Analogy]] — the inspiration from graphics tech
- [[How MegaContext Works]] — overview with visual diagrams
- **[[System Properties]]** — fundamental properties and guarantees of MegaContext
- [[Examples]] — practical examples and use cases
- [[Glossary]] — key terms and definitions

### [[Architecture]]
- [[Architecture]] — system design, runtime, and [[POC Scope|POC scope]]
- [[Architecture Details]] — two-context architecture, invariants, key terms
- [[POC Architecture]] — module responsibilities, storage layout, sample configs
- [[Invariants]] — system invariants and constraints
- [[Storage Format]] — persistent storage structure and serialization
- [[Runtime Loop]] — ingest → focus → decode pipeline
- [[POC Scope]] — guardrails for the proof-of-concept milestone

### [[Components]]
- [[Components]] — overview of all system components

#### Core Components
- [[GistNet]] — learned compression network
- [[LensNet]] — learned focus scoring network
- [[Focus Allocator]] — budget-aware context management
- [[MegaContext Tree]] — hierarchical gist tree structure
- [[Working Context]] — fixed-size attention window

#### Component Details
- [[GistNet Architecture Details]] — detailed GistNet design
- [[GistNet Training]] — teacher-student compression training
- [[LensNet Scoring]] — focus scoring mechanisms
- [[LensNet Training]] — focus learning strategies
- [[Focus Allocator Strategies]] — expansion/collapse algorithms
- [[Tree Operations]] — tree manipulation and maintenance
- [[Node Metadata]] — node attributes and tracking
- [[Working Context Assembly]] — initial context construction
- [[Working Context Refocusing]] — dynamic context adjustment

### [[Plans]] & [[Operations|Operations]]
- [[Plans]] — milestone plans (POC, Paper, Future)
- [[POC Plan]] — stepwise guide for delivering the proof-of-concept
- [[POC Implementation]] — implementation details and current status
- [[Ops]] — training cadence, workflow, performance, pruning
- [[Alternating Optimization]] — training GistNet and LensNet together
- [[Telemetry]] — metrics and monitoring
- [[Base Runtime]] — operational runtime details

### [[Vision]] & Future
- [[Vision]] — long-range research and product direction
- [[Grand Vision]] — why MegaContext matters long term and future directions
- [[Cognitive Core]] — advanced cognitive architectures
- [[MegaPrediction]] — speculative planning within the gist tree
- [[MegaCuration]] — adaptive pruning and knowledge management

### [[Reference]]
- [[Reference]] — comparative analyses and literature
- [[Related Work]] — prior art and research context
- [[Comparisons]] — detailed comparisons with other approaches
- [[MegaContext & RAG]] — comparison with RAG approaches

---

## Contributing

This is an open research project. We welcome contributions that advance the proof-of-concept and the broader [[Vision|vision]].

- Follow conventions in `AGENTS.md` for directory layout, testing, linting, and communication
- Bootstrap: `uv venv`, `uv sync`, then `uv run pytest --maxfail=1 --disable-warnings` for smoke tests
- Lint/format via `uv run ruff check src tests` and `uv run ruff format src tests`
- Update progress in the Obsidian plan notes before hand-off so the next contributor has full context

See the [GitHub repository](https://github.com/brandf/MegaContext) for more details.

---

*MegaContext virtualizes sequence memory just as [[MegaTexture Analogy|MegaTexture]] virtualized textures—focusing detailed computation only where needed. It opens a path to persistent, updatable, and truly lifelong language models.*
