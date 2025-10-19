# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory. This document is both a conceptual overview and a technical design spec for an initial proof-of-concept (POC).*

---
## TL;DR — MegaContext
MegaContext is a proposed system architecture for virtualized LLM context - think “MegaTexture for text.”, if you're familiar with this graphics concept.

It separates a model’s context into a lifetime context (a hierarchical gist tree stored on disk) and a working context (a fixed-size mix of tokens and gists on GPU).  A standard (even pre-trained) LLM then operates on the working context.

A lightweight learned LensNet (and streaming focus allocator) continuously/incrementally refocus the full lifetime context onto the working context, giving the model effectively infinite memory at constant compute.

The next section walks through how the runtime loop stays within a fixed working context while tracking the entire lifetime history. A later **Grand vision** section explains why the mechanism matters before the remaining sections dive into the proof-of-concept (POC) specification.

---

## How MegaContext Works

Large language models are constrained by a fixed context window.  
MegaContext removes this limit by separating:

- **Lifetime context** — the complete interaction or document history (potentially millions or billions of tokens) stored as a *hierarchical gist tree* on disk (RAM for the POC).  
- **Working context** — a fixed 8k–32k token budget on GPU, mixing raw tokens with gists drawn from the lifetime tree. The frozen base LLM sees only this window, which stays contiguous in “time” even as individual spans switch between token-level and gist-level representations.

### Core components

- **Lifetime gist tree** — built incrementally as text streams in (every 32 tokens → L1 gist; every 32 L1 gists → L2 gist; etc.).  
- **Working context** — contiguous window over the tree; total token cost is capped by `W_max`.  
- **GistNet** — a lightweight network that compresses local spans (e.g., 32→1) into **gists** that act as substitutable stand-ins for their source tokens. Stacking gists-of-gists yields a hierarchical, lossy representation of the full lifetime history.  
- **LensNet + focus allocator** — LensNet scores each working-context entry (token embedding or gist) for expansion or collapse; a block-aligned focus allocator applies those scores, streaming finer- or coarser-grained entries in and out while respecting contiguity and the budget.

### Analogy: MegaTexture → MegaContext
This is not required to understand MegaContext, but for those that are interested in learning about the inspiration [this video](https://www.youtube.com/watch?v=BiQCz2NjPR8) provides a good overview of the problems Mega Texture solves.
- In graphics, **MegaTexture** streams the visible portions of a vast texture mipmap into GPU memory at the appropriate resolution.  
- **MegaContext** mirrors that idea for language: instead of mipmap tiles, it maintains embeddings at multiple levels of detail (token L0, gist L1, gist L2, …), yielding effectively unbounded context for a frozen LLM.

### Intuitions / Motivation
The core intuition that's motivating this work is that long context is only useful if the model can focus on the relevant parts and ignore distractors (efficiently).  
- "Relevant parts" is inherently non-causal (something that wasn't previously relevant can become relevant), so this implies dynamic focusing/defocusing.  One-way compression/summarization schemes are fundamentally flawed.
- Exciting new future LLM scenarios will be unlocked at 100M+ context lengths, and at this scale both memory and compute requirements must be sub-linear to be practical for widespread consumer applications.

---

### Runtime lifecycle at a glance

```
Streaming text  ──► Lifetime Gist Tree  ──►  Focus Allocator  ──►  Working Context  ──►  Frozen Base LLM ──► Next Token Prediction
                               ▲                    ▲          
                               └───────── LensNet ──┘
```

1. **Ingest & summarize.** Buffer incoming tokens in 32-token blocks, roll them into new or updated gist nodes, and persist the lifetime tree (disk later, RAM for the POC).
2. **Assemble the working context.** Lay out a contiguous-in-time sequence of tokens and gists whose combined token-equivalent cost stays within `W_max`. Every position represents exactly one interval of the lifetime history at some level of detail.
3. **Refocus.** LensNet reads the current working context (plus tail gists), emits signed focus scores, and the (currently greedy) focus allocator applies block-aligned expansions/collapses without breaking contiguity or budget.
4. **Decode.** The frozen base LLM consumes the refreshed working context to predict the next token(s), feeding newly generated tokens back into step 1.

The next sections unpack each stage: lifetime storage, compression (GistNet), focus control (LensNet + focus allocator), and the training schedule that keeps them aligned.

---

### Key terms & invariants

| Term | Meaning |
|------|---------|
| `Lifetime context` | Full, append-only history stored as a hierarchical gist tree (disk later, RAM for the POC). |
| `Working context` (`WC`) | Fixed-size GPU window (8k–32k token budget) that the base LLM sees; built from contiguous-in-time entries. |
| Working-context entry | Either a block of raw tokens (`L0`) or a gist summarizing that block or its ancestors (`L1`, `L2`, …). Exactly one entry covers each moment in the lifetime history. |
| `L0 / L1 / L2` | Level of detail (LOD): `L0`=tokens, `L1`=32→1 gist, `L2`=gist of gists. Higher `L` means coarser detail and lower token cost. |
| `W_max` | Token-equivalent budget for the working context (sum of entry costs ≤ `W_max`). |
| Block size `K` | Number of new tokens processed per update (POC: `K = 32`). |
| Horizon `H` | Lookahead range used when computing ΔNLL or task losses (typically 32–128 tokens). |
| ΔNLL@`H` | Change in negative log-likelihood over horizon `H` when replacing a region with its gist; used for supervision. |

**Invariants**
- Working context entries tile the lifetime history without gaps or overlaps; switching LOD swaps entries but preserves temporal continuity.
- GistNet outputs **gists** that reuse the base embedding dimension and can replace their source token blocks directly in the working context.
- LensNet and the focus allocator update entries between decode steps while keeping the budget and contiguity invariants intact.

These definitions appear throughout the rest of the document; refer back here when new notation shows up later.

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

- **Frozen base LLM** no fine-tuning initially, with LoRA finetuning as a follow up  
- **Two-level Lifetime gist tree:** The POC will be limited to moderate sized contexts so only 2 laters should be sufficient   
- **Synchronous updates.** Lifetime tree lives in RAM/GPU for POC (rather than disk); updates happen between autoregressive steps.
  
---

## Performance sketch

| Setup | Lifetime tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | context-limited |
| **MegaContext (POC)** | ~1 M | 8 k | ~0.5 GB | few MB | constant compute per step |
| **MegaContext (Future)** | 1 B+ | 32 k | ~2 GB | 10–50 MB/s | fully trained base model |

Per-step compute ≈ base decode cost; gist extraction and LensNet overhead < 1 %.

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
- This makes a **lifelong, high-bandwidth memory** feasible: raw details can be recovered where preserved; elsewhere, multilevel gists maintain global context with the **working context** handling on-demand re-expansion.

---

## GistNet — local gist extraction (32→1, two-layer tree)

### Purpose
GistNet replaces short, fixed-length token sequences with compact **gist embeddings** ("gists") that can stand in for their original tokens inside the base LLM’s context.  
Each gist preserves the meaning of its 32-token span while freeing token budget for new information.  
Stacking two 32→1 layers provides **1024× compression** in the proof of concept (POC).

---

### Inputs & outputs
| Symbol | Shape | Meaning |
|---------|--------|---------|
| `E ∈ R[32, d]` | 32 raw token embeddings (no contextualization) |
| `G₀ ∈ R[K, d]` | learnable slot query (`K=1` for 32→1) |
| `g* ∈ R[d]` | single gist vector aligned with base LLM embedding dim |

---

### POC architecture (32→32→1→32→32→1)

GistNet alternates **self-attention** and **cross-attention** to gradually compress and refine each 32-token span.

#### Stage 1 — Local token self-attention (32 → 32)
- Apply 1–2 standard self-attention + MLP blocks within the 32-token window.  
- Add RoPE or sinusoidal positional encodings for local ordering.  
- Output is `E1`, a locally contextualized version of the raw embeddings.

#### Stage 2 — Compression (32 → 1)
- Introduce a single learned slot query `G₀`.  
- Perform cross-attention where the slot reads from the tokens:  

```
G1 = CrossAttn(query=G0, key=E1, value=E1)
G1 = G1 + MLP(LN(G1)) # residual + feedforward
```

- `G1` is the first gist embedding for this 32-token span.

#### Stage 3 — Expansion (1 → 32)
- Expand information back into the 32-token space for refinement:  

```
E2 = CrossAttn(query=E1, key=G1, value=G1)
E2 = E1 + MLP(LN(E2))
```

- Optionally run one self-attention block over `E2` to diffuse the gist info across tokens.

#### Stage 4 — Final compression (32 → 1)
- Run a second cross-attention with a fresh slot query derived from `G1`:  

```
G2 = G1 + ε
g_star = CrossAttn(query=G2, key=E2, value=E2)
g_star = LN(MLP(g_star))
```
- The result `g_star` is the final gist vector for the span and becomes a node in the lifetime gist tree.

#### Stage 5 — Hierarchical stacking
- Two 32→1 layers are stacked hierarchically (32² = 1024 tokens per top-level gist).  
- The lower layer runs directly on token embeddings; the upper operates on lower-layer outputs.


### Architectural properties
| Property | Description |
|-----------|--------------|
| **Causal scope** | Operates strictly within 32-token windows; no long-range attention. |
| **Parameter sharing** | Shared weights across all spans; upper and lower layers may share or specialize. |
| **Complexity** | O(32²·d) per span — negligible compared to the base LLM. |
| **Dimensionality** | Outputs match the base model’s embedding size `d`. |
| **Positioning** | Gist inserted at the central token index for RoPE alignment. |
| **Precision** | bf16 or fp16; supports later quantization for storage. |


### Training objectives

#### 1. Substitutability (primary)
Train the model so that replacing a span with its gist minimally changes the base LLM’s predictions.

For each training example:

```
Loss_subst = KL( P_base(x_{t+1:T} | E)
|| P_base(x_{t+1:T} | E_replaced) )
```
or equivalently minimize the ΔNLL between the full and gist-replaced context over a short horizon (H = 32–128 tokens).

#### 2. Reconstruction (optional)
Encourage the gist to preserve enough information to approximate its original tokens:

```
Loss_recon = || E_reconstructed - E ||^2
```

#### 3. Contrastive span separation (optional)
Discourage neighboring spans from collapsing to identical gists:

```
Loss_contrast = max(0, margin - cosine_similarity(g_i*, g_j*))
```
for adjacent spans (margin ≈ 0.2).

Total loss:
```
Loss = Loss_subst + 0.1 * Loss_recon + 0.05 * Loss_contrast
```

### Implementation details (POC)
| Item | Setting |
|------|----------|
| Window size | 32 tokens |
| Slots | 1 (single learnable query) |
| Layers per 32→1 block | 2 self + 2 cross |
| Refinement stack | 32→1→32→1 |
| Embedding dim | same as base LLM (e.g., 4096) |
| Internal hidden width | 512 |
| Attention heads | 8 |
| RoPE | applied to token positions only (slots omit it) |
| Activation | GELU |
| Norm | Pre-LayerNorm |
| Parameters | ~0.5M per layer |
| Output | single `g*` vector per span |
| Runtime | <1 ms per 32-token span on GPU |

### Training pipeline (POC)
1. **Dataset:** long-form text (4k–16k tokens), chunked into 32-token spans.  
2. **Teacher:** frozen base LLM used for ΔNLL@H computation.  
3. **Objective:** minimize ΔNLL@H between original and gist-replaced contexts.  
4. **Curriculum:** start with contiguous text, then include structured data (lists, code, tables).  
5. **Optimizer:** AdamW, lr = 1e-4, cosine decay, bf16 precision.  
6. **Output:** store 32→1 and 1024→1 gists in the lifetime gist tree for later use by LensNet and the focus allocator.

### Recap
GistNet is a **local autoencoder for token spans** that learns to produce substitutable embeddings aligned with the base model’s token space.  
It uses **self- and cross-attention refinement (32→1→32→1)** to compress meaning while remaining directly compatible with the base LLM’s embedding layer.  
Stacked hierarchically, GistNet forms the **Lifetime Gist Tree** that supports scalable, virtualized context in MegaContext.

---

## LensNet — how focus is decided

### Why “Lens”?
LensNet acts like an optical lens that dynamically **focuses** and **defocuses** regions within the lifetime context while keeping total compute constant.  
It predicts where to spend detail (expand gists into raw tokens) and where to blur (collapse raw tokens into gists), ensuring that the **fixed-size working context** maintains maximal relevance.

### What it operates on
- LensNet reads the **working context** (not the lifetime tree).  
  It analyzes the embeddings currently fed into the base LLM — the only state that resides on GPU.
- It outputs one **focus score** per entry (token embedding or gist).

### Why non-causal is essential
LensNet must understand *future queries* to know which past facts matter.

**Example**
```
C1: "My shirt is red. My pants are green."
C2: "My shirt is red. My pants are green. What color hat would match my shirt?"
```


Because the base LLM is causal, the hidden states for “shirt” and “pants” are identical in C1 and C2; they never see the question.  
A non-causal LensNet can look at the full working context (including the query) and boost focus on the “shirt” fact.

### Conceptual overview
- LensNet runs independently of the frozen base LLM.  
- It operates directly on the **working context embeddings** (≈ 8k entries), not on live LLM hidden states.  
- It conditions on a small **gist set** (`L2 + last 5 L1` gists, total ≈ 6) taken from the end of the context, which implicitly encodes the upcoming query/task.  
- The model outputs one **signed focus score** `u_i` per entry:
  - `u_i > 0`: expand / focus (increase detail, go one level down)
  - `u_i < 0`: collapse / defocus (reduce detail, go one level up)

At runtime, the **focus allocator** interprets these scores to expand and collapse spans while keeping the working context within its token budget.

### Why dynamic LOD matters
Traditional compression methods summarize once and lose detail forever.  
MegaContext continually re-evaluates importance: if a previously collapsed region becomes relevant again, it can be expanded back into its children gists or raw tokens.  
Note that this expansion is NOT a lossy decoding of the gist latent - the lifetime context preserves the full token-level details on disk (or in RAM for the POC), so the LLM has full access to the whole lifetime context, just not all at once.
This enables the model’s effective memory to **evolve over time** as new information arrives.  Similar to how you're now thinking about your first kiss 😘

### Architecture (POC: dual cross-attention LensNet)

1. **Inputs**
   - `X ∈ R^{N×d}` — embeddings of all entries in the working context (≈ 8 000 tokens or gists).  
   - `G ∈ R^{K×d}` — six gist embeddings from the tail of the context (L2 + 5 L1).  
   - `φ_i` — per-entry metadata (level, width, distance to cursor, system/user flags, etc.).  
   - All embeddings are first **down-projected** to a compact LensNet width `d_lens ≈ 512`.

2. **Stage 1 — Gists read the context (8k → 6)**  
   Each gist slot attends across all working-context entries to build a condensed, query-conditioned representation:

```
tilde_G = Softmax(((G W_Q)(X W_K)^T) / sqrt(d)) * (X W_V)
```

3. **Stage 2 — Context reads refined gists (6 → 8k)**  
   The ≈8k context entries query the six updated gists to broadcast relevance back:

```
tilde_X = Softmax(((X W'_Q)(tilde_G W'_K)^T) / sqrt(d)) * (tilde_G W'_V)
```

4. **Stage 3 — Per-entry scoring head**  
   Concatenate each updated context vector `tilde_X_i` with its metadata `φ_i` and predict a scalar score:

```
u_i = MLP(LN([tilde_X_i, φ_i])) -> R
```

5. **Stacks / refinement**  
   1–3 stacked dual-attn blocks may be used for iterative refinement; parameters `(W_Q,K,V)` are the only learned weights.

**Complexity:** O(N × K × d) per pass.  
With N = 8 000, K = 6, d = 512 ⇒ ~25 M mult-adds — trivial compared to the base model.

### Update cadence (block-wise refocus)

LensNet runs **once every K tokens** (POC: K = 32).  
During each block update:

1. Gather the latest gists `G`.  
2. Run LensNet to produce signed scores `u_i`.  
3. The focus allocator executes expansions/collapses subject to the working-context budget.  
4. The updated context is frozen for the next K tokens.

This matches the intended inference cadence (no per-token recompute).

### Training objectives

#### 1️⃣ Signed focus supervision
Each entry receives a **signed target utility** `y_i` derived from counterfactual NLL deltas:

- Expandable items (L1/L2 children) ⇒ positive `y_i > 0`  
- Collapsible spans ⇒ negative `y_i < 0`  
- Others ⇒ 0 / masked.

LensNet learns to regress and rank these utilities.

```
L_reg  = (1 / |M|) * sum_{i in M} (u_i - y_i)^2
L_rank = softplus(-(u_i - u_j))  # for ordered pairs
```

#### 2️⃣ Zero-sum budget regularizer
To maintain constant working-context size:

```
P = sum_i c_i_plus * ReLU(u_i)
N = sum_i c_i_minus * ReLU(-u_i)
L_budget = ((P - N) / (eps + P + N))^2
```
(`c_i^+` / `c_i^-` = token cost / refund.)  
This encourages net-zero expand/defocus mass per block.

#### 3️⃣ Legality penalties
Prevent impossible actions:

```
L_illegal = alpha * sum_{L0} ReLU(u_i) + beta * sum_{L2} ReLU(-u_i)
```
(alpha, beta ≈ 0.3).  
At inference, invalid directions are hard-masked to 0.

#### 4️⃣ Total loss

```
L_total = L_reg + 0.5 * L_rank + 0.1 * L_budget + L_illegal
```

### Inference procedure

1. **Mask** illegal sides (L0 can’t expand; L2 can’t collapse).  
2. **Optional rebalance**: rescale positive/negative masses to match before sending to the focus allocator.  
3. The focus allocator greedily applies expand/collapse actions within the token budget, honoring hysteresis rules.

### Summary of POC parameters

| Item | Value / Notes |
|------|----------------|
| Input embeddings | ≈8 k entries (mixed L0/L1/L2) |
| Conditioning gists | 6 (L2 + 5 L1) |
| Down-projection width | 512 |
| Attention heads | 8 |
| Stacks | 1–3 |
| Update cadence | every 32 tokens |
| Output | signed focus score `u_i` per entry |
| Runtime | < 3 ms per update @ 8 k tokens |
| Params | ≈ 100 k – 200 k total |


**In short:**  
LensNet is a compact, non-causal controller built as a dual cross-attention network (`8k → 6 → 8k`).  
It runs once per block, predicts balanced signed focus scores for every entry, and guides the focus allocator to keep the working context sharp, legal, and budget-neutral.

---

## Focus allocator — block-aligned actions

*(Name TBD: “focus allocator” is a working title until we settle on something punchier.)*

LensNet alone only supplies signed focus scores. The allocator turns those scores into concrete expand/collapse actions while preserving contiguity, budget, and level-of-detail (LOD) constraints.

### POC constraints & terminology

- **Block alignment:** GistNet currently compresses 32-token blocks. In the POC, every working-context entry must cover exactly one full block at a single LOD (either 32 raw tokens or their 32→1 gist). Higher-level gists (e.g., L2) cover 32 contiguous L1 blocks.
- **Score granularity:** LensNet may emit per-entry scores, but the allocator aggregates them per block so that siblings share a single action score. A future LensNet variant can predict directly per block to avoid this aggregation.
- **Action budget:** Apply at most `N_diff` expand/collapse operations per iteration (default 4). This keeps the system near equilibrium and prevents thrashing.

### Greedy loop (POC)

1. **Collect candidates.** Partition focus scores by block and compute one score per expandable or collapsible unit:
   - Positive scores (`> τ_expand`) become expand candidates (e.g., replace an L1 gist with its 32 L0 tokens or expand an L2 gist into 32 L1 children).
   - Negative scores (`< -τ_collapse`) become collapse candidates (e.g., replace 32 L0 tokens with their L1 gist).
   - Ignore candidates that would violate block alignment (mixed LODs) or budget limits.
2. **Rank.** Maintain two priority queues: descending for expands, ascending for collapses. Tie-break by recency or distance to the cursor.
3. **Apply diff-limited updates.** Pop from the queues alternately (largest expand, largest collapse) until:
   - You have applied `N_diff` actions,
   - One queue empties, or
   - Applying the next action would break the `W_max` budget.
   Collapses refund token budget; expands consume it. If the net cost drifts away from `W_max`, bias the next iteration toward the side that restores balance.
4. **Re-run LensNet if needed.** Because changing LODs alters the scores, optionally iterate LensNet → allocator until either (a) no legal actions remain above thresholds or (b) you reach a maximum number of refinement steps (default 2–3).

### Hysteresis & guardrails

- **Action cooldown:** Track the last action applied per block and dampen (or mask out) the opposite action for `cooldown_steps` iterations. This prevents jitter where the allocator repeatedly expands and collapses the same span.
- **Legality masks:** Blocks at minimum LOD (L0) cannot expand; blocks at maximum LOD (current root level) cannot collapse. These masks should be enforced both in LensNet’s output (runtime masking) and inside the allocator.
- **Consistency checks:** After every iteration, verify that working-context entries still tile the timeline without overlap and that every node’s children share the same LOD.

### Future directions

- Smarter action selection (e.g., matching total expand/collapse mass, soft assignments, or small linear programs) to balance budget and latency.
- Learning a differentiable surrogate (“focus router”) that could eventually replace the greedy loop.
- Adaptive thresholds (`τ_expand`, `τ_collapse`) based on recent utilization to keep the loop stable.

For now, the greedy, block-aligned allocator keeps the POC simple while leaving room for more sophisticated controllers later.

---

## Joint Training (Alternating / “EM-style”): GistNet + LensNet + Base-LoRA

**Goal:** Let all three modules co-adapt without full end-to-end backprop through the discrete focus allocator or long unrolls.  
**Method:** Short alternating phases where some modules are frozen while others learn from on-policy signals produced by the frozen parts. Repeat for a few cycles.

### What “EM-style” means here
We alternate optimization across modules:
- **E-like step:** hold policy parts fixed to produce supervision/targets (e.g., counterfactual utilities).
- **M-like step:** update another module to better fit those targets.
It’s not exact EM; it’s an **alternating optimization schedule** that stabilizes joint training.

### Modules
- **GistNet** `Gist` (32→1, two levels; substitutability objective)
- **LensNet** `LensNet` (dual cross-attn 8k→6→8k; signed focus scores)
- **Base-LoRA** `LoRA` (tiny adapters on the base LLM to improve gist compatibility)
- **Focus allocator** is always **discrete greedy** (no relaxation needed)

### Phase B1 — Update GistNet (fix LensNet + LoRA)
**Fix:** `LensNet`, `LoRA`
**Update:** `Gist`

**Procedure (on-policy):**
1. Build/refresh lifetime trees with current `Gist`.
2. For each training block (size K=32): run `LensNet` + focus allocator to pick expands/collapses; form the working context used by the base LLM.
3. Optimize **GistNet** on spans touched in this block using:
   - **Substitutability loss**: KL(full || replaced) or ΔNLL@H (H=32–128) for the gist that *was actually* inserted.
   - **Stability loss** (optional): L2 between current gist and previous checkpoint to avoid drift.
   - **Boundary aux** (optional): light reconstruction on edge tokens.

**Intuition:** With the current focusing policy fixed, make gists better drop-in replacements for *exactly the places the policy cares about*.

### Phase B2 — Update LensNet (fix GistNet + LoRA)
**Fix:** `Gist`, `LoRA`  
**Update:** `LensNet`

**Procedure:**
1. Using the fixed `Gist`, generate **counterfactual labels** on on-policy snapshots:
   - For candidate expands/collapses in the current working context, compute ΔNLL/ΔKL (batched).
   - Convert to **signed utility per token** (expand positive; collapse negative).
2. Train `LensNet` with:
   - **Signed regression + ranking** (within snapshot)
   - **Zero-sum budget** regularizer (token-cost weighted)
   - **Legality** penalties; keep runtime masking
   - **Update-every-K** cadence (Lens runs once per block)

**Intuition:** Given the current gists, learn a better focusing policy.


### Phase B3 — Update Base-LoRA (fix GistNet + LensNet)
**Fix:** `Gist`, `LensNet`  
**Update:** `LoRA` (small ranks; keep it tiny)

**Where to place LoRA (recommended):**
- Input embedding projection
- QKV/O of the **first 2 attention blocks** *or* the **last 2** (pick one set; not both)

**Losses:**
- **Task NLL@H** with the *discrete* working context produced by `LensNet` + focus allocator
- **Substitutability keep-alive** (weak): prevents gist semantics drifting away from what the base understands
- (Optional) **KL to teacher** if you have a larger teacher-with-MegaContext

**Intuition:** Slightly adapt the base to “like” gist tokens and the current WC geometry (positional anchoring, variance, etc.).


### Schedule & Hyperparameters

- **Cycle length:** B1 → B2 → B3 = **one cycle**. Repeat **3–5 cycles**.
- **Step counts per phase (per cycle):**  
  - B1 (GistNet): 2–4k steps  
  - B2 (LensNet): 2–4k steps  
  - B3 (LoRA): 1–2k steps  
- **Batching:** mixed long-context tasks; block size K=32; horizon H=64.
- **Optimizers:** AdamW (bf16), cosine LR with warmup per phase.
- **Checkpoints:** save after each phase; early-stop on validation **Loss@H vs. token-budget**.


### Data flow per cycle (pseudo)

1. **B1:**  
   - Freeze `LensNet`, `LoRA`.  
   - Decode blocks with current WC (from LensNet + focus allocator).  
   - Update `Gist` using on-policy substitutability losses on the replaced spans.

2. **B2:**  
   - Freeze `Gist`, `LoRA`.  
   - From the same blocks, compute counterfactual utilities (expand/collapse candidates).  
   - Update `LensNet` with signed utilities + budget/legality losses.

3. **B3:**  
   - Freeze `Gist`, `LensNet`.  
   - Run normal blocks (LensNet + focus allocator active) and update `LoRA` on Task NLL@H (+ weak substitutability keep-alive).


### Stability & efficiency tips

- **Warm starts:** Do a short **sequential pretrain** (GistNet then LensNet) before the first B1; it reduces early oscillations.
- **Small LoRA ranks:** r=4–16, low LR; the goal is interface alignment, not knowledge injection.
- **Hysteresis in focus allocator:** min residency steps to prevent expand/collapse thrash during B2/B3.
- **On-policy labeling:** Always regenerate ΔNLL labels *after* the last B1 so LensNet trains on current gists.
- **Curriculum:** start with narrative/doc tasks; add lists/tables/code once stable.
- **Telemetry:** track (a) Loss@H vs budget, (b) swap rate, (c) residency time, (d) non-causal C1/C2 tests.


### When to stop
- **Validation Loss@H vs budget** improves then plateaus across cycles.
- **Swap rate** stabilizes; no ping-pong.
- **Ablations:** freezing any one of {GistNet, LensNet, LoRA} now causes a measurable drop.

**Outcome:** All three modules co-learn: **GistNet** encodes what the policy needs, **LensNet** chooses expansions that actually help, and **LoRA** nudges the base LLM to be friendlier to mixed-LOD inputs—without the cost/fragility of full end-to-end training.


---

## Comparison: MegaContext vs. RAG

| Aspect | RAG | MegaContext |
|---------|-----|-------------|
| **Storage** | External documents, often text chunks in a vector DB | Hierarchical learned gists (vectors) directly aligned to the model’s lifetime |
| **Retrieval trigger** | Query-time semantic search | Continuous, learned focus from LensNet |
| **Integration** | Concatenate retrieved text to prompt | Replace/expand in working context with proper positional encoding |
| **Training** | Separate retriever / generator | Single substitutability & focus training |
| **Memory type** | Stateless look-up | Persistent evolving memory with reversible summarization |

MegaContext is *structurally* similar to RAG in that both pull relevant data into a fixed context, but differs fundamentally: it treats compression and focus as an **integrated learned process** rather than retrieval over external text.

---

## Training data & streaming behavior

- **GistNet training:** any long-form corpus; each 32-token window provides (full vs gist) pairs.  
- **LensNet training:** logged working-context snapshots from real LLM runs.  Counterfactual losses (`expand`/`collapse`) computed offline.  
- **Streaming:** as new tokens arrive, the system:  
  1. Buffers 32 tokens → creates new L1 gist.  
  2. When 32 L1s exist → create L2 gist.  
  3. LensNet + focus allocator decide which regions to expand/collapse before the next decode step.

Gists can be serialized as fp16 or quantized vectors (e.g., 8-bit) with metadata JSON.

---

## Evaluation plan

- **Perplexity vs. token budget** (loss @ Horizon).  
- **Causal vs. non-causal LensNet** on C1/C2-style tests.  
- **Boundary artifacts** (information split across spans).  
- **Stress test** at 1024× compression.  
- **Memory & compute traces** verifying constant per-step cost.

---

## Related work

| Concept | Reference | Relevance |
|----------|------------|------------|
| MegaTexture (id Software, 2007) | Virtualized textures | Direct analogy |
| Perceiver / Perceiver IO (DeepMind 2021-22) | Latent cross-attention | Architectural similarity |
| Slot Attention (Locatello 2020) | Object-like latent slots | GistNet inspiration |
| Compressive Transformer (Rae 2019) | Long-term compressed memory | Temporal analog |
| Gist tokens / LLMLingua 2 (2023-24) | Prompt compression | Substitutability idea |
| RAG / Retrieval-Augmented Generation | Search-based retrieval | Conceptual cousin |
| MegaContext (this work) | — | Unified learned compression + focus over frozen LLMs |

---

## Implementation roadmap

1. **32→1 GistNet** — implement & train substitutability.  
2. **Lifetime Tree Builder** — streaming, 2-level hierarchy in RAM.  
3. **LensNet v1 (non-causal)** — implement query-conditioned scorer, train on offline labels.  
4. **Focus allocator** — greedy expand/collapse, hysteresis.  
5. **E2E POC** — run step-loop (score → allocate → update → decode).  
6. **Evaluate** — loss vs budget, C1/C2 relevance, stress tests.

---

## Future directions

- Async disk streaming of the lifetime tree.  
- RL-trained focus allocator optimizing accuracy × latency.  
- Multi-token gists for structured data.  
- Joint training of LLM + MegaContext from scratch.  
- Shared or federated lifetime memories between agents.

---

## License & contributions

MIT License (suggested).  
PRs welcome — please include reproducible tests for GistNet, LensNet, the focus allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures — focusing detailed computation only where needed.  
It opens a path to persistent, updatable, and truly lifelong language models.*
