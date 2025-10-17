# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.  MegaContext compresses a model’s lifetime context into a hierarchical summary tree and dynamically “focuses” relevant regions into a fixed-size working context at inference time.  This document is both a conceptual overview and a technical design spec for an initial proof-of-concept (POC).*

---

## Why MegaContext?

Large language models are constrained by a fixed context window.  
MegaContext removes this limit by separating:

- **Lifetime context** — the complete interaction or document history (potentially millions or billions of tokens) stored as a *hierarchical summary tree* on disk or in RAM.  
- **Working context** — a small, fixed-size slice of that history (e.g., 8k–32k tokens) mixed from raw tokens and learned summaries, fed to the frozen LLM for each decoding step.

### Analogy: MegaTexture → MegaContext
In graphics, **MegaTexture** streams the visible portions of a vast texture map into GPU memory at appropriate resolution.  
**MegaContext** does the same for text: only the high-resolution “tiles” (recent or relevant spans) are loaded into the model’s working memory, while distant regions remain summarized at coarse levels on disk.

---

## Grand vision: why this matters

The POC will prove the mechanism, but the broader implications are transformative:

### ♾️ Virtually infinite memory
Lifetime context can grow unbounded while per-step compute and GPU RAM remain constant.  A conversation could persist for years without retraining or forgetting.

### 🧩 Smaller, smarter models
An LLM trained end-to-end with MegaContext could shift parameter budget away from memorized facts toward reasoning, abstraction, and planning.  
Knowledge lives in the *lifetime memory* instead of the weights.

### 💻 Agentic coding & persistent tasks
Today, agents rely on brittle, lossy context management (manual summarization, sub-agents, RAG hacks).  
MegaContext treats context management as a **first-class architectural component**, allowing seamless long-term reasoning and creative iteration.

### 🌐 Core knowledge as dynamic system prompt
Shipping LLMs with a **core lifetime context** transforms in-context learning:  
the model boots with a massive “system prompt” of structured world knowledge that updates hourly from external data—without retraining weights.  
A cloud-hosted MegaContext model could refresh its understanding of the world continually, combining retrieval and reasoning in a unified pipeline.

---

## POC scope & constraints

- **Frozen base LLM** (no fine-tuning).  
- **Two-level summary tree:**  
  - Level 1: 32 tokens → 1 summary  
  - Level 2: 32 summaries → 1 summary  
  ⇒ overall **1024× compression**.  
- **Local-only summarization.** Each node sees only its own 32-token window (plus small boundary context).  
- **Synchronous updates.** Lifetime tree lives in RAM/GPU; updates happen between autoregressive steps.  
- **Non-causal Lens** (explained below) with a **deterministic Allocator** controlling focus.  
- **Ordered, position-anchored substitution.** Summaries occupy the central RoPE index of their span.

---

## System overview

```
Streaming text  ──►  Lifetime Summary Tree  ──►  Allocator ──► Working Context  ──►  Frozen LLM ──► Next Token Prediction
                               ▲                    ▲                │   │  ▲                             │
                               │                    ┕━━━━━━━Lens━━━━━┚   │  ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┚
                               ┕━━━━━━━━━━━━━━━━━━━━━━━Summarizer━━━━━━━━┚                      
                          
```


---

## Performance sketch

| Setup | Lifetime tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | context-limited |
| **MegaContext (2-lvl)** | ~33 M | 8 k | ~0.5 GB | few MB | constant compute per step |
| **Future** | billions | 8 k | same 0.5 GB | 10–50 MB/s | fits consumer SSD bandwidth |

Per-step compute ≈ base decode cost; summarization and Lens overhead < 1 %.

---

## Components

### 1️⃣ Lifetime summary tree
Built incrementally as text streams in:
- Every 32 tokens → 1 summary (Level 1).  
- Every 32 Level-1 summaries → 1 Level-2 summary.  
- Summaries are stored with metadata (token range, RoPE center, parent/child IDs) and can be serialized to disk.

### 2️⃣ Working context
A fixed-size mixture of raw tokens and summaries forming a contiguous window over the lifetime tree.  
At any step, its total token cost ≤ `W_max` (e.g., 8 k).

---

## The Lens — how focus is decided

### Why “Lens”?
Imagine the working context as the image projected through a **dynamic-shaped lens** that focuses and defocuses regions of the lifetime context.  
The Lens determines *where* to zoom in (expand summaries into details) and *where* to blur (collapse details into summaries) — keeping total compute constant.

### What it operates on
- The Lens reads the **working context** (not the lifetime tree).  
  It analyzes the embeddings currently fed into the base LLM — the only state that resides on GPU.
- It outputs one **focus score** per feature (token span or summary).

### Why non-causal is essential
The Lens must understand *future queries* to know which past facts matter.

**Example**
```
C1: "My shirt is red. My pants are green."
C2: "My shirt is red. My pants are green. What color hat would match my shirt?"
```


Because the base LLM is causal, the hidden states for “shirt” and “pants” are identical in C1 and C2; they never see the question.  
A non-causal Lens can look at the full working context (including the query) and boost focus on the “shirt” fact.

### POC implementation
Non-causality is achieved by **conditioning on the query vector** `q` (hidden state at the generation cursor):

`u_i = MLP(LN([h_i, q, h_i⊙q, |h_i−q|, φ_i]))`

where `h_i` = pooled hidden of span _i_, and `φ_i` = metadata (level, width, distance, etc.).

The Lens is tiny (< 100 k params) and runs in microseconds.

---

## The Allocator — how focus is applied

The **Allocator** receives the focus scores `{u_i}` and updates the *level-of-detail (LOD)* of the working context.

### Purpose
- Maintain the token budget (`W_max`).
- Translate Lens intent into discrete **expand/collapse** actions.
- Keep the representation adaptive: *summarization is not a one-way door*.

### Why dynamic LOD matters
Traditional compression methods summarize once and lose detail forever.  
MegaContext continually re-evaluates importance: if a previously collapsed region becomes relevant again, it can be expanded back into its children summaries or raw tokens.  
This enables the model’s effective memory to **evolve over time** as new information arrives.

### Greedy two-phase policy (POC)
1. **Collapse phase:** if over budget, greedily collapse low-focus sibling spans until within limit.  
2. **Expand phase:** if under budget, greedily expand high-focus summaries that fit remaining space.  
Hysteresis or min-residency rules can prevent thrashing.

---

## The 32→1 local summarizer (two-layer tree)

**Input:** 32 raw token embeddings `E ∈ R^{32×d}`  
**Output:** single vector `s* ∈ R^d` matching the base LLM’s embedding dimension.  
**Position:** central token index of the span (RoPE aligned).

Tiny architecture:

1. (Optional) local 1D conv or 2-layer local attention for micro-context.  
2. Single learnable query `s₀`.  
3. 2–3 rounds of `CrossAttn(Q=s, K=E, V=E)` + MLP residual.  
4. Output `s*` + LayerNorm.

**Training objective:** *substitutability* — minimize the KL/NLL gap between base LLM predictions on full vs. summarized span.  
Stacking two layers of 32→1 achieves 1024× compression for the POC.

---

## Comparison: MegaContext vs. RAG

| Aspect | RAG | MegaContext |
|---------|-----|-------------|
| **Storage** | External documents, often text chunks in a vector DB | Hierarchical learned summaries (vectors) directly aligned to the model’s lifetime |
| **Retrieval trigger** | Query-time semantic search | Continuous, learned focus from the Lens |
| **Integration** | Concatenate retrieved text to prompt | Replace/expand in working context with proper positional encoding |
| **Training** | Separate retriever / generator | Single substitutability & focus training |
| **Memory type** | Stateless look-up | Persistent evolving memory with reversible summarization |

MegaContext is *structurally* similar to RAG in that both pull relevant data into a fixed context, but differs fundamentally: it treats compression and focus as an **integrated learned process** rather than retrieval over external text.

---

## Training data & streaming behavior

- **Summarizer training:** any long-form corpus; each 32-token window provides (full vs summary) pairs.  
- **Lens training:** logged working-context snapshots from real LLM runs.  Counterfactual losses (`expand`/`collapse`) computed offline.  
- **Streaming:** as new tokens arrive, the system:  
  1. Buffers 32 tokens → creates new L1 summary.  
  2. When 32 L1s exist → create L2 summary.  
  3. Lens+Allocator decide which regions to expand/collapse before the next decode step.

Summaries can be serialized as fp16 or quantized vectors (e.g., 8-bit) with metadata JSON.

---

## Evaluation plan

- **Perplexity vs. token budget** (loss @ Horizon).  
- **Causal vs. non-causal Lens** on C1/C2-style tests.  
- **Boundary artifacts** (information split across spans).  
- **Stress test** at 1024× compression.  
- **Memory & compute traces** verifying constant per-step cost.

---

## Related work

| Concept | Reference | Relevance |
|----------|------------|------------|
| MegaTexture (id Software, 2007) | Virtualized textures | Direct analogy |
| Perceiver / Perceiver IO (DeepMind 2021-22) | Latent cross-attention | Architectural similarity |
| Slot Attention (Locatello 2020) | Object-like latent slots | Summarizer inspiration |
| Compressive Transformer (Rae 2019) | Long-term compressed memory | Temporal analog |
| Gist tokens / LLMLingua 2 (2023-24) | Prompt compression | Substitutability idea |
| RAG / Retrieval-Augmented Generation | Search-based retrieval | Conceptual cousin |
| MegaContext (this work) | — | Unified learned compression + focus over frozen LLMs |

---

## Implementation roadmap

1. **32→1 Summarizer** — implement & train substitutability.  
2. **Lifetime Tree Builder** — streaming, 2-level hierarchy in RAM.  
3. **Lens v1 (non-causal)** — implement query-conditioned scorer, train on offline labels.  
4. **Allocator** — greedy expand/collapse, hysteresis.  
5. **E2E POC** — run step-loop (score → allocate → update → decode).  
6. **Evaluate** — loss vs budget, C1/C2 relevance, stress tests.

---

## Future directions

- Async disk streaming of the lifetime tree.  
- RL-trained allocator optimizing accuracy × latency.  
- Multi-token summaries for structured data.  
- Joint training of LLM + MegaContext from scratch.  
- Shared or federated lifetime memories between agents.

---

## License & contributions

MIT License (suggested).  
PRs welcome — please include reproducible tests for summarizer, Lens, allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures — focusing detailed computation only where needed.  
It opens a path to persistent, updatable, and truly lifelong language models.*
