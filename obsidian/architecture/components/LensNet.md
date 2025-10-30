---
tags:
  - components
summary: Dual cross-attention controller that scores working-context entries for expansion or collapse.
---
![[LensNet Diagram.png]]
LensNet reads the [[Working Context|working context]] plus tail gists to emit signed utilities that tell the [[Focus Allocator]] where to zoom in or back off, keeping the window relevant at constant compute.

- **Operates on:** [[Working Context|working-context]] embeddings (≈8k entries) and a tail of gists.
- **Outputs:** signed focus scores per entry; positive to expand, negative to collapse.
- **Architecture:** dual cross-attention blocks (`context ↔ tail gists`) followed by scalar heads.
- **Cadence:** runs every `K` tokens before allocator actions. See [[POC Implementation]] for specific values.
- **Training:** counterfactual ΔNLL utilities, budget regularizers, legality penalties. See [[LensNet Training]].
- **Interfaces:** works alongside [[GistNet]] outputs and the greedy [[Focus Allocator]].

## What is LensNet?

LensNet acts like an optical lens that dynamically **focuses** and **defocuses** regions within the [[MegaContext Tree|MegaContext]] while keeping total compute constant. It predicts where to spend detail (expand gists into raw tokens) and where to blur (collapse raw tokens into gists), ensuring the **fixed-size [[Working Context|working context]]** maintains maximal relevance.

## Purpose: Relevance Prediction

LensNet solves the core problem of **adaptive resolution**: which parts of a massive context deserve full token-level detail, and which can be compressed into gists? By scoring each entry in the working context with a signed focus value, LensNet enables the [[Focus Allocator]] to:

- **Expand** high-relevance gists into their underlying raw tokens
- **Collapse** low-relevance token spans into gists
- Maintain constant GPU memory and compute usage

## Operating Assumptions

- LensNet reads the **[[Working Context|working context]]**, not the [[MegaContext Tree]]. It analyzes the embeddings currently fed into the base LLM — the only state resident on GPU.
- It outputs one **focus score** per entry (token embedding or gist).
- The [[Architecture Details#Key terms & invariants|contiguity invariant]] ensures each score maps to a single, non-overlapping lifetime span, so expand/collapse actions remain block-aligned.

### Why Non-Causal is Essential

LensNet must understand *future queries* to know which past facts matter. Because the base LLM is causal, hidden states for earlier tokens cannot "see" upcoming questions; LensNet compensates by operating on the full [[Working Context|working context]].

## High-Level Architecture

### Conceptual Flow

1. LensNet runs independently of the frozen base LLM.
2. It operates directly on the **[[Working Context|working context]] embeddings** (≈ 8k entries), not on live LLM hidden states.
3. It conditions on a small **gist set** (`L2 + last 5 L1` gists, total ≈ 6) taken from the end of the context, which implicitly encodes the upcoming query/task.
4. The model outputs one **signed focus score** `u_i` per entry:
    - `u_i > 0`: expand / focus (increase detail, go one level down)
    - `u_i < 0`: collapse / defocus (reduce detail, go one level up)

> **Diagram needed — `assets/lensnet_focus.png`:** Show LensNet reading a tail slice of gists plus the [[Working Context|working context]], then emitting signed scores that the allocator converts into expand/collapse actions.

At runtime, the **[[Focus Allocator|focus allocator]]** interprets these scores to expand and collapse spans while keeping the [[Working Context|working context]] within its token budget.

### Dual Cross-Attention Architecture

The POC implementation uses a dual cross-attention design [1, 2]:

1. **Inputs:** context embeddings (`N ≈ 8k`), tail gists (`K = 6`), metadata (levels, span widths, distances)
2. **Stage 1:** Tail gists read the context (cross-attention: gists query context) [1]
3. **Stage 2:** Context queries updated gists (cross-attention: context queries gists) [2]
4. **Stage 3:** Scoring head produces signed utilities per entry

**Complexity:** `O(N × K × d_lens)` per pass. With `N ≈ 8k`, `K = 6`, `d_lens = 512`, the update costs ~25 M multiply-adds—negligible relative to the base model decode.

See [[LensNet Scoring]] for full inference details and [[LensNet Training]] for architecture implementation.

## Key Properties

- **Block-wise updates:** Runs once every K tokens, not per token
- **Signed outputs:** Positive scores encourage expansion; negative scores encourage collapse
- **Budget-aware:** Training includes zero-sum regularizers to maintain constant working-context size
- **Legality-enforced:** Cannot expand L0 tokens or collapse L2 root; violations penalized in training and masked at inference
- **Compact:** ≈ 100k–200k parameters, < 3 ms per update @ 8k tokens

## Role in the System

LensNet serves as the **attention controller** [3, 4] in MegaContext's adaptive resolution pipeline:

```
[[GistNet]] → compresses spans into gists
LensNet → scores entries for relevance
[[Focus Allocator]] → executes expand/collapse actions
[[Working Context]] → maintained at constant size with optimal relevance
```

It bridges the gap between the static [[MegaContext Tree]] (all history) and the dynamic [[Working Context]] (what the LLM sees), enabling the system to "zoom" through context at multiple resolutions based on task relevance. This content-based addressing approach [3] allows the system to learn optimal memory routing [4] without manual heuristics.

## Update Cadence (Block-Wise Refocus)

LensNet runs **once every K tokens**. During each block update:

1. Gather the latest gists `G`.
2. Run LensNet to produce signed scores `u_i`.
3. The [[Focus Allocator|focus allocator]] executes expansions/collapses subject to the [[Working Context|working-context]] budget.
4. The updated context is frozen for the next K tokens.

This matches the intended inference cadence (no per-token recompute).

## Related Pages

**Training & Scoring:**
- [[LensNet Training]] — objectives, loss functions, supervision signals
- [[LensNet Scoring]] — inference procedure, masking, rebalancing

**Integration:**
- [[Focus Allocator]] — uses LensNet scores to execute refocus actions
- [[Working Context Refocusing]] — overall refocus workflow
- [[GistNet]] — produces the gist embeddings LensNet conditions on

**Implementation:**
- [[POC Implementation]] — parameter values, runtime characteristics
- [[Alternating Optimization]] — joint training with GistNet

---

## References

1. **Perceiver** (Jaegle et al., 2021) — [[papers/Perceiver - 2103.03206v2|Analysis]] — Latent cross-attention bottleneck architecture
2. **Perceiver IO** (Jaegle et al., 2021) — [[papers/Perceiver IO - 2107.14795v3|Analysis]] — Query-based decoding for arbitrary structured outputs
3. **Neural Turing Machines** (Graves et al., 2014) — [[papers/Neural Turing Machines|Analysis]] — Content-based addressing and memory controllers
4. **Differentiable Neural Computer** (Graves et al., 2016) — [[papers/DNC|Analysis]] — Learned memory allocation and routing

See [[Related Work]] for the complete bibliography of all research papers referenced throughout the documentation.
