# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory. This document is both a conceptual overview and a technical design spec for an initial proof-of-concept (POC).*

---
## TL;DR — MegaContext
MegaContext is a proposed system architecture for virtualized LLM context - think “MegaTexture for text.”, if you're familiar with this graphics concept.

It separates a model’s context into a lifetime context (a hierarchical summary tree stored on disk) and a working context (a fixed-size mix of tokens and summaries on GPU).  A standard (even pre-trained) LLM then operates on the working context.

A lightweight learned Lens model (and streaming Allocator) continuously/incrementally refocus the full lifetime context onto the working context, giving the model effectively infinite memory at constant compute.

---

## Why MegaContext?

Large language models are constrained by a fixed context window.  
MegaContext removes this limit by separating:

- **Lifetime context** — the complete interaction or document history (potentially millions or billions of tokens) stored as a *hierarchical summary tree* on disk or in RAM.  
- **Working context** — a small, fixed-size slice of that history (e.g., 8k–32k tokens) mixed from raw tokens and learned summaries, fed to the frozen LLM for each decoding step.

### Analogy: MegaTexture → MegaContext
This is not required to understand MegaContext, but for those that are interested in learning about the inspiration [this video](https://www.youtube.com/watch?v=BiQCz2NjPR8) provides a good overview of the problems Mega Texture solves.
- In graphics, **MegaTexture** streams the visible portions of a vast texture map into GPU memory at appropriate resolution.  
- **MegaContext** does the same for text: only the high-resolution “tiles” (recent or relevant spans) are loaded into the model’s working memory, while distant regions remain summarized at coarse levels.

### Intuitions / Motivation
The core intuition that's motivating this work is that long context is only useful if the model can focus on the relevant parts and ignore distractors (efficiently).  
- "Relevant parts" is inherently non-causal (something that wasn't previously relevant can become relevant), so this implies dynamic focusing/defocusing.  One-way compression/summarization schemes are fundamentally flawed.
- Exciting new future LLM scenarios will be unlocked at 100M+ context lengths, and at this scale both memory and compute requirements must be sub-linear to be practical for widespread consumer applications.

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
the model boots with a massive “system prompt” of structured world knowledge that updates externally and without retraining weights.  
- A cloud-hosted MegaContext model could refresh its understanding of the world continually, combining retrieval and reasoning in a unified pipeline.
- An agentic coding system could provide an entire codebase as a system prompt (lifetime context), eliminating the expensive / error prone processes of reading parts of the projects code.
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
## System overview

```
Streaming text  ──►  Lifetime Summary Tree  ──►  Allocator ──► Working Context  ──►  Frozen Base LLM ──► Next Token Prediction
                               ▲                    ▲                │   │  ▲                                  │
                               │                    ┕━━━━━━━Lens━━━━━┛   │  ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                               ┕━━━━━━━━━━━━━━━━━━━━━━━Summarizer━━━━━━━━┛
                          
```


---

## Performance sketch

| Setup | Lifetime tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | context-limited |
| **MegaContext (POC)** | ~33 M | 8 k | ~0.5 GB | few MB | constant compute per step |
| **MegaContext (Future)** | billions | 32 k | ~2 GB | 10–50 MB/s | fully trained base model |

Per-step compute ≈ base decode cost; summarization and Lens overhead < 1 %.

### Long-term storage example: lifetime memory for a 24/7 robot (10 years)

**Assumptions**

- **Sampling:** 500 feature vectors / sec  
- **Feature size:** 4,096-dim; stored as fp16 (2 bytes) unless noted  
- **Duration:** 10 years ≈ 3.1536×10⁸ seconds ⇒ **N = 500 × 3.1536×10⁸ ≈ 1.5768×10¹¹** leaf vectors  
- **Tree arity:** 32→1 at each level (no overlap)  
- **Tree depth:** log₃₂(N) ≈ **8 levels** (root near level 8)  
- **Node payload:** one vector per node (same width as leaves, different precision per scenario)

#### Storage breakdown

| Scenario | Estimated storage | How to read this |
|---|---:|---|
| **Raw leaves only (fp16)** | **~1.29 PB** | N × 4096 × 2 bytes = 1.5768e11 × 8192 B |
| **Full 32-ary tree at fp16 (leaves + *all* internal levels)** | **~1.33 PB** | Geometric factor for all nodes: (1 + 1/32 + 1/32² + …) = 32/31 ≈ **1.032×** overhead over leaves |
| **Full tree, 8-bit everywhere** | **~667 TB** | Leaves 8-bit: ~646 TB; internal nodes count = N/31 ≈ 5.09e9; internal 8-bit adds ~20.8 TB; total ≈ 646 + 20.8 |
| **Pruned: keep only 1% of leaves @ 8-bit; keep *all* internal nodes @ 8-bit** | **~27 TB** | Leaves: 0.01 × 646 TB ≈ 6.46 TB; internal 8-bit ≈ 20.8 TB; total ≈ 27.3 TB |
| **Pruned + compressed: 1% leaves @ 8-bit with entropy coding (~×0.5); internal @ 8-bit with entropy coding (~×0.5)** | **~13–14 TB** | Leaves ≈ 3.2 TB + internal ≈ 10.4 TB |
| **More aggressive: 0.5% leaves @ 8-bit + entropy (~×0.5); internal @ 4-bit + entropy (~×0.5)** | **~6–8 TB** | Leaves: 0.005 × 646 TB × 0.5 ≈ **1.6 TB**; internal: (20.8 TB × 0.5 for 4-bit) × 0.5 entropy ≈ **5.2 TB**; total ~6.8 TB |

**Key takeaways**

- A **full 32-ary tree** only adds ~**3.2%** storage over leaves when stored at the **same precision** (factor 32/31), so multilevel LOD itself is cheap; **precision and pruning dominate** total footprint.  
- With **8-bit quantization** and **reasonable pruning** of raw leaves (e.g., keep only salient 0.5–1%), plus straightforward **entropy coding**, **a decade of continuous 500 Hz, 4k-dim features** compresses to **single-digit TBs**—practical for local SSD arrays.  
- This makes a **lifelong, high-bandwidth memory** feasible: raw details can be recovered where preserved; elsewhere, multilevel summaries maintain global context with the **working context** handling on-demand re-expansion.

---
## The Lens — how focus is decided

### Why “Lens”?
The Lens acts like an optical lens that dynamically **focuses** and **defocuses** regions within the lifetime context while keeping total compute constant.  
It predicts where to spend detail (expand summaries into raw tokens) and where to blur (collapse raw tokens into summaries), ensuring that the **fixed-size working context** maintains maximal relevance.

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

### Conceptual overview
- The Lens runs independently of the frozen base LLM.  
- It operates directly on the **working context embeddings** (`~8k features`), not on live LLM hidden states.  
- It conditions on a small **summary set** (`L2 + last 5 L1` summaries, total ≈ 6) taken from the end of the context, which implicitly encodes the upcoming query/task.  
- The model outputs one **signed focus score** `u_i` per feature:
  - `u_i > 0`: expand / focus (increase detail, go one level down)
  - `u_i < 0`: collapse / defocus (reduce detail, go one level up)

At runtime, the **Allocator** interprets these scores to expand and collapse spans while keeping the working context within its token budget.

### Why dynamic LOD matters
Traditional compression methods summarize once and lose detail forever.  
MegaContext continually re-evaluates importance: if a previously collapsed region becomes relevant again, it can be expanded back into its children summaries or raw tokens.  
Note that this expansion is NOT a lossy decoding of the summary latent - the lifetime context preserves the full token-level details on disk (or in RAM for the POC), so the LLM has full access to the whole lifetime context, just not all at once.
This enables the model’s effective memory to **evolve over time** as new information arrives.  Similar to how you're now thinking about your first kiss 😘

### Architecture (POC: dual cross-attention LensNet)

1. **Inputs**
   - `X ∈ R^{N×d}` — embeddings of all features in the working context (≈ 8 000 tokens or summaries).  
   - `S ∈ R^{K×d}` — six summary embeddings from the tail of the context (L2 + 5 L1).  
   - `φ_i` — per-feature metadata (level, width, distance to cursor, system/user flags, etc.).  
   - All embeddings are first **down-projected** to a compact Lens width `d_lens ≈ 512`.

2. **Stage 1 — Summaries read the context (8k → 6)**  
   Each summary slot attends across all working-context features to build a condensed, query-conditioned representation:
   \[
   \tilde S = \text{Softmax}\!\big((S W_Q)(X W_K)^\top / \sqrt d\big)(X W_V)
   \]

3. **Stage 2 — Context reads refined summaries (6 → 8k)**  
   The 8 k context features query the six updated summaries to broadcast relevance back:
   \[
   \tilde X = \text{Softmax}\!\big((X W'_Q)(\tilde S W'_K)^\top / \sqrt d\big)(\tilde S W'_V)
   \]

4. **Stage 3 — Per-feature scoring head**  
   Concatenate each updated context vector `\tilde X_i` with its metadata `φ_i` and predict a scalar score:
   \[
   u_i = \text{MLP}(\text{LN}([\tilde X_i, φ_i])) \rightarrow \mathbb{R}
   \]

5. **Stacks / refinement**  
   1–3 stacked dual-attn blocks may be used for iterative refinement; parameters `(W_Q,K,V)` are the only learned weights.

**Complexity:** O(N × K × d) per pass.  
With N = 8 000, K = 6, d = 512 ⇒ ~25 M mult-adds — trivial compared to the base model.

### Update cadence (block-wise refocus)

The Lens runs **once every K tokens** (POC: K = 32).  
During each block update:

1. Gather the latest summaries S.  
2. Run LensNet to produce signed scores `u_i`.  
3. The Allocator executes expansions/collapses subject to the working-context budget.  
4. The updated context is frozen for the next K tokens.

This matches the intended inference cadence (no per-token recompute).

### Training objectives

#### 1️⃣ Signed focus supervision
Each feature receives a **signed target utility** `y_i` derived from counterfactual NLL deltas:

- Expandable items (L1/L2 children) ⇒ positive `y_i > 0`  
- Collapsible spans ⇒ negative `y_i < 0`  
- Others ⇒ 0 / masked.

The Lens learns to regress and rank these utilities.

\[
\mathcal{L}_{\text{reg}} = \frac{1}{|M|}\sum_{i\in M}(u_i - y_i)^2,
\qquad
\mathcal{L}_{\text{rank}} = \text{softplus}(-(u_i-u_j))\text{ for ordered pairs.}
\]

#### 2️⃣ Zero-sum budget regularizer
To maintain constant working-context size:
\[
P=\sum_i c_i^{+}\,\text{ReLU}(u_i),\quad
N=\sum_i c_i^{-}\,\text{ReLU}(-u_i)
\]
\[
\mathcal{L}_{\text{budget}}=\big((P-N)/(ε+P+N)\big)^2
\]
(`c_i^+` / `c_i^-` = token cost / refund.)  
This encourages net-zero expand/defocus mass per block.

#### 3️⃣ Legality penalties
Prevent impossible actions:
\[
\mathcal{L}_{\text{illegal}}=
\alpha\!\!\sum_{\text{L0}}\!\!\text{ReLU}(u_i)
+\beta\!\!\sum_{\text{L2}}\!\!\text{ReLU}(-u_i)
\]
(\(\alpha,\beta≈0.3\)).  
At inference, invalid directions are hard-masked to 0.

#### 4️⃣ Total loss
\[
\mathcal{L}=
\mathcal{L}_{\text{reg}}
+0.5\,\mathcal{L}_{\text{rank}}
+0.1\,\mathcal{L}_{\text{budget}}
+\mathcal{L}_{\text{illegal}}
\]

### Inference procedure

1. **Mask** illegal sides (L0 can’t expand; L2 can’t collapse).  
2. **Optional rebalance**: rescale positive/negative masses to match before sending to the Allocator.  
3. **Allocator** greedily applies expand/collapse actions within the token budget, honoring hysteresis rules.

### Summary of POC parameters

| Item | Value / Notes |
|------|----------------|
| Input embeddings | 8 k features (mixed L0/L1/L2) |
| Conditioning summaries | 6 (L2 + 5 L1) |
| Down-projection width | 512 |
| Attention heads | 8 |
| Stacks | 1–3 |
| Update cadence | every 32 tokens |
| Output | signed focus score `u_i` per feature |
| Runtime | < 3 ms per update @ 8 k tokens |
| Params | ≈ 100 k – 200 k total |


**In short:**  
The Lens is a compact, non-causal controller built as a dual cross-attention network (`8k → 6 → 8k`).  
It runs once per block, predicts balanced signed focus scores for every feature, and guides the Allocator to keep the working context sharp, legal, and budget-neutral.

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
