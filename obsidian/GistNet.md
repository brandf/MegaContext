---
title: "GistNet — Local Gist Extraction"
type: "concept"
status: "active"
tags: ["module","compression","gistnet"]
summary: "32→1 hierarchical encoder that substitutes token spans with gists compatible with the base LLM."
links:
  - "[[MOC - Core Components]]"
  - "[[LensNet]]"
  - "[[Focus Allocator]]"
  - "[[Training & Operations]]"
---

## Layer 0 · Capture Summary
- GistNet compresses each 32-token span into a gist vector aligned with base embeddings, enabling reversible multi-level representations inside MegaContext.

## Layer 1 · Key Points
- **Purpose:** replace raw tokens with substitutable gists to free working-context budget.
- **Architecture:** self-attention + cross-attention stack (32→1→32→1) with shared slot queries.
- **Hierarchy:** two 32→1 layers deliver 1024× compression.
- **Training:** minimize ΔNLL@H and optional contrastive losses; see [[Training & Operations]].
- **Interfaces:** feeds the MegaContext tree consumed by [[LensNet]] and [[Focus Allocator]].

## Layer 2 · Detailed Notes

### Overview

GistNet replaces short, fixed-length token sequences with compact **gist embeddings** ("gists") that can stand in for their original tokens inside the base LLM’s context. Each gist preserves the meaning of its 32-token span while freeing token budget for new information. Stacking two 32→1 layers provides **1024× compression** in the proof of concept (POC).

## Inputs & outputs

| Symbol | Shape | Meaning |
|---------|--------|---------|
| `E ∈ R[32, d]` | 32 raw token embeddings (no contextualization) |
| `Q₁, Q₂ ∈ R[1, d]` | Learned slot queries for the two compression passes |
| `g_final ∈ R[d]` | Final gist vector aligned with the base LLM embedding dim |

## POC architecture (32→32→1→32→32→1)

GistNet alternates **self-attention** and **cross-attention** to gradually compress and refine each 32-token span.

### Stage 1 — Local token self-attention (32 → 32)
- Apply 1–2 standard self-attention + MLP blocks within the 32-token window.
- Add RoPE or sinusoidal positional encodings for local ordering.
- Output is `E1`, a locally contextualized version of the raw embeddings.

### Stage 2 — Compression (32 → 1)
- Introduce the first learned slot query `Q₁` (shared across spans).
- Perform cross-attention where the slot reads from the tokens:

```
G1 = CrossAttn(query=Q1, key=E1, value=E1)
G1 = G1 + MLP(LN(G1)) # residual + feedforward
```

- `G1` is the first gist embedding for this 32-token span.

### Stage 3 — Expansion (1 → 32)
- Expand information back into the 32-token space for refinement:

```
E2 = CrossAttn(query=E1, key=G1, value=G1)
E2 = E1 + MLP(LN(E2))
```

- Optionally run one self-attention block over `E2` to diffuse the gist info across tokens.

### Stage 4 — Final compression (32 → 1)
- Run a second cross-attention with the independent learned slot query `Q₂`:

```
g_final = CrossAttn(query=Q2, key=E2, value=E2)
g_final = LN(MLP(g_final))
```

- The result `g_final` is the final gist vector for the span and becomes a node in the MegaContext gist tree.

### Stage 5 — Hierarchical stacking
- Two 32→1 layers are stacked hierarchically (32² = 1024 tokens per top-level gist).
- The lower layer runs directly on token embeddings; the upper operates on lower-layer outputs.
- This per-block stacking preserves the [[Architecture Overview#Key terms & invariants|contiguity invariant]] noted in the glossary—each gist still maps to an exact, non-overlapping span in the MegaContext history.

> **Diagram needed — `assets/gist_hierarchy.png`:** Depict an L0 token block rolling up into an L1 gist and then into an L2 gist, with pointers back to the MegaContext timeline.

## Architectural properties

| Property | Description |
|-----------|--------------|
| **Limited scope** | Operates strictly within 32-token windows; no long-range attention. |
| **Parameter sharing** | Shared weights across all spans; upper and lower layers may share or specialize. |
| **Complexity** | O(32²·d) per span — negligible compared to the base LLM. |
| **Dimensionality** | Outputs match the base model’s embedding size `d`. |
| **Positioning** | Gist inserted at the central token index for RoPE alignment. |
| **Precision** | bf16 or fp16; supports later quantization for storage. |

## Training objectives

### 1. Substitutability (primary)
Train the model so that replacing a span with its gist minimally changes the base LLM’s predictions.

For each training example:

```
Loss_subst = KL( P_base(x_{t+1:T} | E)
|| P_base(x_{t+1:T} | E_replaced) )
```

or equivalently minimize the ΔNLL between the full and gist-replaced context over a short horizon (H = 32–128 tokens).

### 2. Contrastive span separation (optional)
Discourage neighboring spans from collapsing to identical gists:

```
Loss_contrast = max(0, margin - cosine_similarity(g_i_final, g_j_final))
```

for adjacent spans (margin ≈ 0.2).

Total loss:

```
Loss = Loss_subst + 0.05 * Loss_contrast
```

## Implementation details (POC)

| Item | Setting |
|------|----------|
| Window size | 32 tokens |
| Slots | 2 shared learned queries (`Q₁`, `Q₂`) |
| Layers per 32→1 block | 2 self + 2 cross |
| Refinement stack | 32→1→32→1 |
| Embedding dim | same as base LLM (e.g., 4096) |
| Internal hidden width | 512 |
| Attention heads | 8 |
| RoPE | applied to token positions only (slots omit it) |
| Activation | GELU |
| Norm | Pre-LayerNorm |
| Parameters | ~0.5M per layer |
| Output | single `g_final` vector per span |
| Runtime | <1 ms per 32-token span on GPU |

Runtime figures assume a single NVIDIA L4 running bf16 inference with `HuggingFaceTB/SmolLM3-3B`; expect faster throughput on A100-class hardware.

## Training pipeline (POC)

1. **Dataset:** long-form text (4k–16k tokens), chunked into 32-token spans.
2. **Teacher:** frozen base LLM used for ΔNLL@H computation.
3. **Objective:** minimize ΔNLL@H between original and gist-replaced contexts.
4. **Curriculum:** start with contiguous text, then include structured data (lists, code, tables).
5. **Optimizer:** AdamW, lr = 1e-4, cosine decay, bf16 precision.
6. **Output:** store 32→1 and 1024→1 gists in the MegaContext gist tree for later use by LensNet and the focus allocator.

## Recap

GistNet is a **local encoder for token spans** whose only goal is to emit substitutable gist vectors aligned with the base model’s embedding space. It uses **self- and cross-attention refinement (32→1→32→1)** to squeeze each 32-token block into a single vector without ever decoding back to tokens. Stacked hierarchically, GistNet forms the **MegaContext gist tree** that supplies the tail gists consumed by [[LensNet]] and the allocator during the focus loop.

## Layer 3 · Change Log
- 2025-10-22: Added metadata, progressive summarization layers, and links to focus stack + training workflow.
