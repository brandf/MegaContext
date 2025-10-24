[](Runtime%20Loop.md)---
summary: MegaContext virtualizes context by pairing a disk-backed gist tree called the MegaContext with a budgeted working context governed by GistNet, LensNet, and the Focus Allocator.
---
MegaContext virtualizes context by pairing a disk-backed gist tree called the [[MegaContext Tree]] with a budgeted working context governed by [[GistNet]], [[LensNet]], and the [[Focus Allocator]].

It separates a model’s context into a [[MegaContext Tree]] (stored on disk) and a [[Working Context]] (on GPU). A learned [[GistNet]] model is used to build the [[MegaContext Tree]] as a hierarchy of gists. The working context compresses the [[MegaContext Tree]] into a fixed-size mix of tokens and gists that are used for inference.

To dynamically adapt level of detail, a learned [[LensNet]] model, continuously/incrementally refocuses the [[MegaContext Tree]] onto the [[Working Context]], giving the model effectively infinite memory at constant compute with automatic context management.

---

- **Dual contexts:** [[MegaContext Tree]] tree vs. [[Working Context]].
- **Compression:** [[GistNet]] builds hierarchical gists aligned with base embeddings.
- **Focus/Defocus:** [[LensNet]] scores working entries; [[Focus Allocator]] adjusts detail.
- **See also:** [[Runtime Loop]] for execution, [[POC Architecture]] for interfaces.

---
## Details

### How MegaContext Works

Large language models are constrained by a fixed context window. MegaContext removes this limit by separating:

- **MegaContext** — the complete interaction or document history (potentially millions or billions of tokens) stored as a *hierarchical gist tree* on disk (RAM for the POC).
- **Working context** — a fixed 8k–32k token budget on GPU, mixing raw tokens with gists drawn from the MegaContext tree. The frozen base LLM sees only this window, which stays contiguous in “time” even as individual spans switch between token-level and gist-level representations.

### Core components

- **MegaContext gist tree** — built incrementally as text streams in (every 32 tokens → L1 gist; every 32 L1 gists → L2 gist; etc.).
- **Working context** — contiguous window over the tree; total token cost is capped by `W_max`.
- **GistNet** — a lightweight network that compresses local spans (e.g., 32→1) into **gists** that act as substitutable stand-ins for their source tokens. Stacking gists-of-gists yields a hierarchical, lossy representation of the full MegaContext history.
- **LensNet + focus allocator** — LensNet scores each working-context entry (token embedding or gist) for expansion or collapse; a block-aligned focus allocator applies those scores, streaming finer- or coarser-grained entries in and out while respecting contiguity and the budget.

### Intuitions / Motivation
The core intuition that's motivating this work is that long context is only useful if the model can focus on the relevant parts and ignore distractors efficiently.
- "Relevant parts" is inherently non-causal - something that wasn't previously relevant can become relevant, so this implies dynamic focusing/defocusing. One-way compression/summarization schemes are fundamentally flawed.
- RAG/tools without context maintainance is anti-productive for long running context, it's trading more context for the immediate task for more distractors on all future tasks.
- Exciting new future LLM scenarios will be unlocked at 100M+ context lengths, and at this scale both memory and compute requirements must be sub-linear to be practical for widespread consumer applications.

---

### Runtime lifecycle at a glance
![[ArchitectureDiagram.png]]
![[GistNetDiagram.png]]
![[LensNet Diagram.png]]
![[WorkingSetUpdateDiagram.png]]

1. **Ingest & summarize.** Buffer incoming tokens in 32-token blocks, roll them into new or updated gist nodes, and persist the MegaContext tree (disk later, RAM for the POC).
2. **Assemble the working context.** Lay out a contiguous-in-time sequence of tokens and gists whose combined token-equivalent cost stays within `W_max`. Every position represents exactly one interval of the MegaContext history at some level of detail.
3. **Refocus.** LensNet reads the current working context (plus tail gists), emits signed focus scores, and the (currently greedy) focus allocator applies block-aligned expansions/collapses without breaking contiguity or budget.
4. **Decode.** The frozen base LLM consumes the refreshed working context to predict the next token(s), feeding newly generated tokens back into step 1.

**Update cadence & buffering.**
- **MegaContext maintenance:** Both user tokens and model-generated tokens are buffered until a full 32-token block (L0) or 32 L1 children are available before rebuilding the corresponding gist nodes. This keeps gist updates block-aligned and prevents churn in the hierarchy.
- **LensNet conditioning gists:** LensNet only refreshes its conditioning set on its own schedule (e.g., every 256 working-context entries). Those gists can be read from the MegaContext tree or recomputed lazily immediately before each LensNet call; either path observes the same block-aligned buffers.

---

### Key terms & invariants

| Term | Meaning |
|------|---------|
| `MegaContext` | Full, append-only history stored as a hierarchical gist tree (disk later, RAM for the POC). |
| `Working context` (`WC`) | Fixed-size GPU window (8k–32k token budget) that the base LLM sees; built from contiguous-in-time entries. |
| Working-context entry | Either a block of raw tokens (`L0`) or a gist summarizing that block or its ancestors (`L1`, `L2`, …). Exactly one entry covers each moment in the MegaContext history. |
| `L0 / L1 / L2` | Level of detail (LOD): `L0`=tokens, `L1`=32→1 gist, `L2`=gist of gists. Higher `L` means coarser detail and lower token cost. |
| `W_max` | Token-equivalent budget for the working context (sum of entry costs ≤ `W_max`). |
| Block size `K` | Number of new tokens processed per update (POC: `K = 32`). |
| Horizon `H` | Lookahead range used when computing ΔNLL or task losses (defaults: 64 for narrative traces, 96 for mixed agent turns, 128 for code). |
| ΔNLL@`H` | Change in negative log-likelihood over horizon `H` when replacing a region with its gist; used for supervision. |

**Invariants**
- Working context entries tile the MegaContext history without gaps or overlaps; switching LOD swaps entries but preserves temporal continuity.
- GistNet outputs **gists** that reuse the base embedding dimension and can replace their source token blocks directly in the working context.
- LensNet and the focus allocator update entries between decode steps while keeping the budget and contiguity invariants intact.

These definitions appear throughout the rest of the vault; refer back here when new notation shows up later.

---

### Document roadmap

1. **POC architecture** — see [[POC Architecture]] for module responsibilities and binary formats.
2. **Module deep dives** — [[GistNet]], [[LensNet]], and [[Focus Allocator]] unpack each subsystem.
3. **POC scope & performance sketch** — [[POC Scope]] and [[Performance Sketch]] capture boundaries and envelope math.
4. **Training & operations** — [[Training & Operations]] explains alternating optimization, labeling, and instrumentation.
5. **Roadmap & vision** — revisit [[Grand Vision]] and [[Cognitive Core]] for long-term ambitions, plus [[MegaPrediction]] and [[MegaCuration]] for future extensions.

For a step-by-step execution walk-through, check [[Runtime Loop]] which ties these components together at inference time.
