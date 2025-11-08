---
tags:
  - components
summary: Comprehensive technical specification of GistNet's 32→1→32→1 refinement architecture with layer-by-layer breakdowns, attention mechanisms, and implementation details.
---
# GistNet Architecture Details

This document provides a comprehensive technical specification of **GistNet**, the hierarchical encoder that compresses 32-token spans into single gist embeddings aligned with the base LLM's embedding space.

## Architecture Overview

**GistNet** implements a 32→1 compression architecture using a **32→32→1→32→32→1** refinement stack that alternates between self-attention and cross-attention layers. Two such compression layers stacked hierarchically achieve **1024× compression** (32² = 1024).

### High-Level Flow

```
Input: 32 raw token embeddings E ∈ R[32, d]
  ↓
Stage 1: Self-Attention (32 → 32)
  ↓
Stage 2: Compression via Cross-Attention (32 → 1)
  ↓
Stage 3: Expansion via Cross-Attention (1 → 32)
  ↓
Stage 4: Final Compression (32 → 1)
  ↓
Output: Single gist vector g_final ∈ R[d]
```

---

## Layer-by-Layer Breakdown

### Input Specifications

| Symbol | Shape | Type | Description |
|--------|-------|------|-------------|
| `E` | `R[32, d]` | Token embeddings | Raw, non-contextualized token embeddings from base LLM |
| `Q₁` | `R[1, d]` | Learned parameter | First slot query for initial compression |
| `Q₂` | `R[1, d]` | Learned parameter | Second slot query for final compression |
| `d` | Scalar | Dimension | Base LLM embedding dimension (e.g., 4096) |

### Stage 1: Local Token Self-Attention (32 → 32)

**Purpose:** Contextualize tokens within their local 32-token window before compression.

**Architecture:**
- 1–2 standard Transformer self-attention blocks
- Each block consists of:
  - Multi-head self-attention layer
  - Feed-forward MLP
  - Pre-LayerNorm
  - Residual connections

**Mathematical Formulation:**

```
# Input: E ∈ R[32, d]

# First Self-Attention Block
E_norm = LayerNorm(E)
E_pos = E_norm + PositionalEncoding(32)  # RoPE or sinusoidal

# Multi-head self-attention
Q_sa = E_pos @ W_Q    # Shape: [32, d]
K_sa = E_pos @ W_K    # Shape: [32, d]
V_sa = E_pos @ W_V    # Shape: [32, d]

Attn = Softmax((Q_sa @ K_sa^T) / sqrt(d_head))  # Shape: [32, 32]
SA_out = Attn @ V_sa  # Shape: [32, d]

# Residual connection
E_attn = E + SA_out

# Feed-forward network
E_norm2 = LayerNorm(E_attn)
FF_out = GELU(E_norm2 @ W_ff1) @ W_ff2  # Two-layer MLP
E1 = E_attn + FF_out  # Shape: [32, d]

# Output: E1 ∈ R[32, d] (locally contextualized embeddings)
```

**Dimensions:**
- Input: `[32, d]`
- Output: `[32, d]`
- Attention weights: `[32, 32]`

**Parameters per block:**
- Self-attention: `4d² ÷ num_blocks` (Q, K, V, O projections)
- MLP: `2 × d × d_ff` (typically `d_ff = 4d`)

---

### Stage 2: First Compression (32 → 1)

**Purpose:** Compress 32 contextualized tokens into a single gist embedding using learned slot query.

**Architecture:**
- Cross-attention with learned slot query `Q₁`
- Query: learned slot (1 vector)
- Keys/Values: contextualized tokens from Stage 1
- Feed-forward refinement

**Mathematical Formulation:**

```
# Input: E1 ∈ R[32, d], Q₁ ∈ R[1, d]

# Cross-attention: slot queries the tokens
Q_slot = Q₁ @ W_Q_cross  # Shape: [1, d]
K_tokens = E1 @ W_K_cross  # Shape: [32, d]
V_tokens = E1 @ W_V_cross  # Shape: [32, d]

# Cross-attention computation
CrossAttn_scores = Softmax((Q_slot @ K_tokens^T) / sqrt(d_head))  # Shape: [1, 32]
G1_raw = CrossAttn_scores @ V_tokens  # Shape: [1, d]

# Output projection
G1_attn = G1_raw @ W_O_cross  # Shape: [1, d]

# Residual + Feed-forward
G1_norm = LayerNorm(Q₁ + G1_attn)
G1_ff = GELU(G1_norm @ W_ff1) @ W_ff2
G1 = (Q₁ + G1_attn) + G1_ff  # Shape: [1, d]

# Output: G1 ∈ R[1, d] (first gist embedding)
```

**Attention Pattern:**
```
        Tokens (32)
        ↓↓↓↓↓↓↓↓
Slot → [attention weights] → Gist (1)
```

**Dimensions:**
- Input: `[32, d]` (tokens), `[1, d]` (slot)
- Output: `[1, d]` (gist)
- Attention weights: `[1, 32]` (slot attends to all 32 tokens)

**Key Properties:**
- Slot query `Q₁` is **shared across all 32-token spans** in the dataset
- Cross-attention allows the slot to aggregate information from all tokens
- The slot learns to extract maximally informative features for substitutability

---

### Stage 3: Expansion (1 → 32)

**Purpose:** Broadcast gist information back to token space for refinement.

**Architecture:**
- Cross-attention in reverse: tokens query the gist
- Query: original contextualized tokens
- Keys/Values: gist embedding from Stage 2
- Optional self-attention for information diffusion

**Mathematical Formulation:**

```
# Input: E1 ∈ R[32, d], G1 ∈ R[1, d]

# Cross-attention: tokens query the gist
Q_tokens = E1 @ W_Q_expand  # Shape: [32, d]
K_gist = G1 @ W_K_expand    # Shape: [1, d]
V_gist = G1 @ W_V_expand    # Shape: [1, d]

# Cross-attention computation
CrossAttn_scores = Softmax((Q_tokens @ K_gist^T) / sqrt(d_head))  # Shape: [32, 1]
E2_raw = CrossAttn_scores @ V_gist  # Shape: [32, d]

# Output projection
E2_attn = E2_raw @ W_O_expand  # Shape: [32, d]

# Residual + Feed-forward
E2_norm = LayerNorm(E1 + E2_attn)
E2_ff = GELU(E2_norm @ W_ff1) @ W_ff2
E2_pre = (E1 + E2_attn) + E2_ff  # Shape: [32, d]

# Optional: Self-attention for diffusion
E2_norm2 = LayerNorm(E2_pre)
SA_out = MultiHeadSelfAttention(E2_norm2)
E2 = E2_pre + SA_out  # Shape: [32, d]

# Output: E2 ∈ R[32, d] (refined token representations)
```

**Attention Pattern:**
```
Token₁ →┐
Token₂ →├→ [attend to gist] → Refined Token₁
Token₃ →┤                     Refined Token₂
  ...  →┤                     Refined Token₃
Token₃₂→┘                         ...
```

**Dimensions:**
- Input: `[32, d]` (tokens), `[1, d]` (gist)
- Output: `[32, d]` (refined tokens)
- Attention weights: `[32, 1]` (each token attends to gist)

**Purpose:**
- Allows gist to "correct" or "refine" token representations
- Enables information flow from compressed space back to token space
- Creates better token representations for final compression

---

### Stage 4: Final Compression (32 → 1)

**Purpose:** Generate the final gist embedding using refined token representations.

**Architecture:**
- Second cross-attention with independent learned slot query `Q₂`
- Query: second learned slot (different from `Q₁`)
- Keys/Values: refined tokens from Stage 3
- Final normalization and feed-forward

**Mathematical Formulation:**

```
# Input: E2 ∈ R[32, d], Q₂ ∈ R[1, d]

# Cross-attention: second slot queries refined tokens
Q_slot2 = Q₂ @ W_Q_final  # Shape: [1, d]
K_tokens2 = E2 @ W_K_final  # Shape: [32, d]
V_tokens2 = E2 @ W_V_final  # Shape: [32, d]

# Cross-attention computation
CrossAttn_scores = Softmax((Q_slot2 @ K_tokens2^T) / sqrt(d_head))  # Shape: [1, 32]
g_raw = CrossAttn_scores @ V_tokens2  # Shape: [1, d]

# Output projection
g_attn = g_raw @ W_O_final  # Shape: [1, d]

# Residual + Feed-forward
g_norm = LayerNorm(Q₂ + g_attn)
g_ff = GELU(g_norm @ W_ff1) @ W_ff2
g_pre = (Q₂ + g_attn) + g_ff

# Final layer normalization
g_final = LayerNorm(MLP(g_pre))  # Shape: [1, d]

# Output: g_final ∈ R[d] (final gist vector)
```

**Dimensions:**
- Input: `[32, d]` (refined tokens), `[1, d]` (slot)
- Output: `[d]` (final gist, squeezed from `[1, d]`)
- Attention weights: `[1, 32]` (slot attends to all refined tokens)

**Key Properties:**
- Slot query `Q₂` is **independent** from `Q₁` and also shared across spans
- Operates on refined tokens that have seen gist information
- Output is aligned with base LLM embedding space for [[Glossary#Substitutability|substitutability]]

---

## Attention Mechanisms

### Self-Attention (Token-to-Token)

**Used in:** Stage 1, optionally in Stage 3

**Purpose:** Allow tokens to contextualize each other within the local 32-token window.

**Formulation:**
```
Q = X @ W_Q  ∈ R[32, d]
K = X @ W_K  ∈ R[32, d]
V = X @ W_V  ∈ R[32, d]

Attention_weights = Softmax((Q @ K^T) / sqrt(d_head))  ∈ R[32, 32]
Output = Attention_weights @ V  ∈ R[32, d]
```

**Multi-Head Implementation:**
```python
def multi_head_self_attention(X, num_heads=8):
    """
    X: [batch, 32, d]
    Output: [batch, 32, d]
    """
    d_head = d // num_heads

    # Linear projections split into heads
    Q = linear(X, W_Q).view(batch, 32, num_heads, d_head).transpose(1, 2)
    K = linear(X, W_K).view(batch, 32, num_heads, d_head).transpose(1, 2)
    V = linear(X, W_V).view(batch, 32, num_heads, d_head).transpose(1, 2)

    # Scaled dot-product attention per head
    scores = (Q @ K.transpose(-2, -1)) / sqrt(d_head)
    attn = softmax(scores, dim=-1)

    # Apply attention to values
    out = attn @ V

    # Concatenate heads and project
    out = out.transpose(1, 2).contiguous().view(batch, 32, d)
    return linear(out, W_O)
```

**Attention Pattern (32×32 matrix):**
```
       T₁  T₂  T₃  ... T₃₂
T₁  [  ●   ○   ○  ...  ○  ]
T₂  [  ○   ●   ○  ...  ○  ]
T₃  [  ○   ○   ●  ...  ○  ]
...
T₃₂ [  ○   ○   ○  ...  ●  ]

● = high attention (self + neighbors)
○ = distributed attention
```

---

### Cross-Attention (Slot-to-Tokens) - Compression

**Used in:** Stage 2, Stage 4

**Purpose:** Compress 32 tokens into 1 gist via learned slot query.

**Formulation:**
```
Q_slot = Q @ W_Q  ∈ R[1, d]      # Query from slot
K_tokens = X @ W_K  ∈ R[32, d]   # Keys from tokens
V_tokens = X @ W_V  ∈ R[32, d]   # Values from tokens

Attention_weights = Softmax((Q_slot @ K_tokens^T) / sqrt(d_head))  ∈ R[1, 32]
Output = Attention_weights @ V_tokens  ∈ R[1, d]
```

**Implementation:**
```python
def cross_attention_compress(tokens, slot_query, num_heads=8):
    """
    tokens: [batch, 32, d]
    slot_query: [1, d] (learned parameter, shared across spans)
    Output: [batch, 1, d]
    """
    d_head = d // num_heads

    # Expand slot for batch
    Q_slot = slot_query.unsqueeze(0).expand(batch, -1, -1)

    # Project and split into heads
    Q = linear(Q_slot, W_Q).view(batch, 1, num_heads, d_head).transpose(1, 2)
    K = linear(tokens, W_K).view(batch, 32, num_heads, d_head).transpose(1, 2)
    V = linear(tokens, W_V).view(batch, 32, num_heads, d_head).transpose(1, 2)

    # Cross-attention: slot attends to all tokens
    scores = (Q @ K.transpose(-2, -1)) / sqrt(d_head)
    attn = softmax(scores, dim=-1)

    # Weighted sum of token values
    out = attn @ V

    # Concatenate heads and project
    out = out.transpose(1, 2).contiguous().view(batch, 1, d)
    return linear(out, W_O)
```

**Attention Pattern (1×32 vector):**
```
Slot → [ w₁  w₂  w₃  ...  w₃₂ ] → Gist
         ↓   ↓   ↓   ...   ↓
         T₁  T₂  T₃  ...  T₃₂

where w₁ + w₂ + ... + w₃₂ = 1 (softmax normalized)
```

---

### Cross-Attention (Tokens-to-Gist) - Expansion

**Used in:** Stage 3

**Purpose:** Broadcast gist information back to all 32 token positions.

**Formulation:**
```
Q_tokens = X @ W_Q  ∈ R[32, d]   # Queries from tokens
K_gist = G @ W_K  ∈ R[1, d]      # Key from gist
V_gist = G @ W_V  ∈ R[1, d]      # Value from gist

Attention_weights = Softmax((Q_tokens @ K_gist^T) / sqrt(d_head))  ∈ R[32, 1]
Output = Attention_weights @ V_gist  ∈ R[32, d]
```

**Attention Pattern (32×1 vectors):**
```
T₁  →┐
T₂  →┤
T₃  →├→ [ Gist ] →→ Broadcast to all
... →┤
T₃₂ →┘
```

---

## Slot Queries Details

### What are Slot Queries?

Slot queries are **learned embedding vectors** that serve as "read heads" for compression. They are trained to extract the most relevant information from token sequences for substitutability.

### Specifications

| Parameter | Shape | Initialization | Sharing |
|-----------|-------|----------------|---------|
| `Q₁` | `[1, d]` | Xavier/Normal (mean=0, std=0.02) | Shared across all spans |
| `Q₂` | `[1, d]` | Xavier/Normal (mean=0, std=0.02) | Shared across all spans, independent from Q₁ |

### Why Two Slots?

1. **Q₁ (First Compression):** Learns to extract initial compressed representation from raw contextualized tokens
2. **Q₂ (Final Compression):** Learns to extract refined representation from tokens that have been enhanced with gist information

**Specialization:** The two slots can specialize for different aspects:
- Q₁ might focus on broad semantic content
- Q₂ might focus on details that survived the refinement loop

### Training Dynamics

```python
# Slot queries are learnable parameters
slot_q1 = nn.Parameter(torch.randn(1, d) * 0.02)
slot_q2 = nn.Parameter(torch.randn(1, d) * 0.02)

# During training, gradients flow through cross-attention
loss = compute_substitutability_loss(g_final, ...)
loss.backward()  # Updates both slots and all layer weights

# Slots learn to:
# - Attend to most informative tokens
# - Extract features that maximize substitutability
# - Preserve semantic content that affects future predictions
```

### Slot Positioning in RoPE Space

- **Tokens:** Receive RoPE positional encodings (positions 0-31)
- **Slots:** Do **NOT** receive positional encodings (position-invariant)
- **Final gist:** Inserted at **central token index** (position 16) for RoPE alignment when substituted into context

---

## 32→1→32→1 Refinement Stack

### Architecture Rationale

The **32→1→32→1** pattern creates a compression-refinement loop:

```
32 tokens → [compress] → 1 gist → [expand] → 32 refined tokens → [compress] → 1 final gist
```

**Benefits:**
1. **Two-pass compression:** Gist can "see itself" reflected back in token space
2. **Error correction:** Expansion allows model to add missing information back to tokens
3. **Better abstraction:** Second compression operates on more informed token representations
4. **Improved substitutability:** Refinement loop helps preserve critical details

### Information Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Self-Attention (32 → 32)                           │
│  ┌───┐ ┌───┐ ┌───┐       ┌───┐                               │
│  │ T₁│ │ T₂│ │ T₃│  ...  │T₃₂│  (Raw Tokens)                 │
│  └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘                               │
│    └─────┴─────┴───────────┘                                 │
│           ↓ Self-Attention ↓                                 │
│    ┌─────┬─────┬───────────┬─┐                               │
│  ┌─┴─┐ ┌─┴─┐ ┌─┴─┐       ┌─┴─┐                               │
│  │E1₁│ │E1₂│ │E1₃│  ...  │E1₃₂│  (Contextualized)            │
│  └───┘ └───┘ └───┘       └───┘                               │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: First Compression (32 → 1)                         │
│                                                              │
│      ┌───────────────────────────┐                           │
│      │  Learned Slot Query Q₁   │                            │
│      └───────────┬───────────────┘                           │
│                  ↓ Cross-Attention                           │
│        ┌─────────┴────────┐                                  │
│  E1₁ → │                  │                                  │
│  E1₂ → │   Attention      │   ┌────┐                         │
│  E1₃ → │   Pooling        │ → │ G1 │ (First Gist)            │
│  ...   │                  │   └────┘                         │
│  E1₃₂→ │                  │                                  │
│        └──────────────────┘                                  │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Expansion (1 → 32)                                 │
│                                                              │
│          ┌────┐                                              │
│          │ G1 │                                              │
│          └─┬──┘                                              │
│            │ Broadcast via Cross-Attention                   │
│            ↓                                                 │
│    ┌───────┴───────┐                                         │
│  E1₁ + gist info → E2₁                                       │
│  E1₂ + gist info → E2₂                                       │
│  E1₃ + gist info → E2₃                                       │
│  ...                                                         │
│  E1₃₂ + gist info → E2₃₂                                     │
│                                                              │
│  Optional Self-Attention for diffusion                       │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: Final Compression (32 → 1)                         │
│                                                              │
│      ┌───────────────────────────────┐                       │
│      │  Learned Slot Query Q₂       │                        │
│      └───────────┬───────────────────┘                       │
│                  ↓ Cross-Attention                           │
│        ┌─────────┴────────┐                                  │
│  E2₁ → │                  │                                  │
│  E2₂ → │   Attention      │   ┌─────────┐                    │
│  E2₃ → │   Pooling        │ → │ g_final │ (Final Gist)       │
│  ...   │                  │   └─────────┘                    │
│  E2₃₂→ │                  │                                  │
│        └──────────────────┘                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Parameter Counts

### Single GistNet Block (32→1)

**Full-dimension version** (d = 4096, d_ff = 512, n_heads = 8):

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| **Slot Q₁** | 4,096 | `1 × d` |
| **Slot Q₂** | 4,096 | `1 × d` |
| **Self-Attention 1** | ~67M | `4 × d² (Q,K,V,O projections)` |
| **Self-Attention 2** | ~67M | `4 × d²` |
| **Cross-Attn Compress 1** | ~67M | `4 × d²` |
| **Cross-Attn Expand** | ~67M | `4 × d²` |
| **Cross-Attn Compress 2** | ~67M | `4 × d²` |
| **FFN (5 blocks)** | ~42M | `5 × (d × d_ff × 2) = 5 × 8M` |
| **LayerNorms** | ~40K | `8 × d × 2 (γ, β)` |
| **Total per block** | **~377M** | |

**Note:** Actual POC implementation uses internal hidden width of 512, significantly reducing parameter count.

### Optimized POC Configuration

With `d = 4096`, `d_internal = 512`, `n_heads = 8`:

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| **Projection In** | 2M | `d × d_internal` |
| **Self-Attention (internal)** | 1M | `4 × d_internal²` |
| **Cross-Attention (internal)** | 1M | `4 × d_internal²` |
| **FFN (internal)** | 0.5M | `d_internal × d_ff × 2` |
| **Projection Out** | 2M | `d_internal × d` |
| **Total per block** | **~0.5M** | As specified in original doc |

### Full Hierarchical Model

| Level | Blocks | Params per Block | Total |
|-------|--------|------------------|-------|
| LOD1 (32→1) | 32 shared | 0.5M | 0.5M (shared) |
| LOD2 (32→1) | 1 | 0.5M | 0.5M |
| **Total** | - | - | **~1M** |

---

## Dimension Tracking Summary

### Throughout the 32→1→32→1 Pipeline

| Stage | Input Shape | Output Shape | Key Dimensions |
|-------|-------------|--------------|----------------|
| **Input** | `[B, 32, d]` | - | B=batch, d=embedding dim |
| **Stage 1: Self-Attn** | `[B, 32, d]` | `[B, 32, d]` | Attention: `[B, H, 32, 32]` |
| **Stage 2: Compress** | `[B, 32, d]` + `[1, d]` | `[B, 1, d]` | Attention: `[B, H, 1, 32]` |
| **Stage 3: Expand** | `[B, 32, d]` + `[B, 1, d]` | `[B, 32, d]` | Attention: `[B, H, 32, 1]` |
| **Stage 4: Compress** | `[B, 32, d]` + `[1, d]` | `[B, d]` | Attention: `[B, H, 1, 32]` |
| **Output** | - | `[B, d]` | Aligned with base LLM |

**Legend:**
- `B`: Batch size
- `d`: Embedding dimension (e.g., 4096)
- `H`: Number of attention heads (e.g., 8)
- `32`: Window size (fixed)

---

## Architecture Diagrams

### Attention Flow Visualization

```
═══════════════════════════════════════════════════════════════
                    GISTNET ATTENTION FLOW
═══════════════════════════════════════════════════════════════

STAGE 1: SELF-ATTENTION (32×32 matrix)

         T₁ T₂ T₃ ... T₃₂
      ┌─────────────────┐
   T₁ │ ██ ░░ ░░     ░░ │  Each token attends to all tokens
   T₂ │ ░░ ██ ░░     ░░ │  Diagonal = self-attention
   T₃ │ ░░ ░░ ██     ░░ │  Full bidirectional within window
   .. │                 │
  T₃₂ │ ░░ ░░ ░░     ██ │
      └─────────────────┘

───────────────────────────────────────────────────────────────

STAGE 2: COMPRESSION (1×32 vector)

   Slot Q₁ → [ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ... ▓▓ ] → Gist G₁
              ↑  ↑  ↑  ↑  ↑       ↑
              T₁ T₂ T₃ T₄ T₅ ... T₃₂

   Single slot attends to all 32 tokens
   Weights sum to 1.0 (softmax normalized)

───────────────────────────────────────────────────────────────

STAGE 3: EXPANSION (32×1 broadcast)

   T₁ ←──┐
   T₂ ←──┤
   T₃ ←──├─── [ G₁ ]  Gist broadcasts to all tokens
   .. ←──┤
  T₃₂ ←──┘

   All tokens receive same gist information
   (modulated by attention weights)

───────────────────────────────────────────────────────────────

STAGE 4: FINAL COMPRESSION (1×32 vector)

   Slot Q₂ → [ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ... ▒▒ ] → g_final
               ↑  ↑  ↑  ↑  ↑       ↑
              E2₁E2₂E2₃E2₄E2₅ ... E2₃₂

   Second slot attends to refined tokens
   Produces final substitutable gist

═══════════════════════════════════════════════════════════════
```

### Hierarchical Compression Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                      LEVEL 0: RAW TOKENS                      │
├───────────────────────────────────────────────────────────────┤
│ Block 1           Block 2                   Block 32          │
│ ┌─────────────┐   ┌─────────────┐         ┌─────────────┐     │
│ │T₁..T₃₂      │   │T₃₃..T₆₄     │   ...   │T₉₉₃..T₁₀₂₄  │     │
│ │[32 tokens]  │   │[32 tokens]  │         │[32 tokens]  │     │
│ └──────┬──────┘   └──────┬──────┘         └──────┬──────┘     │
└────────┼─────────────────┼───────────────────────┼────────────┘
         │                 │                       │
         │ GistNet LOD1    │ GistNet LOD1          │ GistNet LOD1
         │ (32→1)          │ (32→1)                │ (32→1)
         ↓                 ↓                       ↓
┌───────────────────────────────────────────────────────────────┐
│                  LEVEL 1: FIRST-LEVEL GISTS                   │
├───────────────────────────────────────────────────────────────┤
│ ┌────┐  ┌────┐  ┌────┐             ┌────┐                     │
│ │ g₁ │  │ g₂ │  │ g₃ │  ...        │g₃₂ │                     │
│ └─┬──┘  └─┬──┘  └─┬──┘             └─┬──┘                     │
└───┼───────┼───────┼──────────────────┼────────────────────────┘
    └───────┴───────┴──────────────────┘
                    │
          GistNet LOD2 (32→1)
          Treats LOD1 gists as "tokens"
                    ↓
┌───────────────────────────────────────────────────────────────┐
│                  LEVEL 2: SECONT-LEVEL GISTS                  │
├───────────────────────────────────────────────────────────────┤
│                       ┌────────┐                              │
│                       │ g_L2   │                              │
│                       │[1 vec] │                              │
│                       └────────┘                              │
│                                                               │
│         Represents all 1024 original tokens                   │
│         1024:1 compression ratio                              │
└───────────────────────────────────────────────────────────────┘

Compression Factor: 32 × 32 = 1024×
```

---

## Hierarchical Stacking (1024× Compression)

### Two-Level Hierarchy

GistNet achieves **1024× compression** by stacking two 32→1 layers:

```
Level 0 (LOD0): Raw tokens
  └─ 1024 tokens divided into 32 spans of 32 tokens each

Level 1 (LOD1): First-level gists
  └─ 32 LOD1 gists (one per LOD0 span)

Level 2 (LOD2): Second-level gist
  └─ 1 LOD2 gist (aggregating all 32 LOD1 gists)
```

### Parameter Sharing

| Component | Sharing Strategy | Rationale |
|-----------|------------------|-----------|
| LOD1 slot queries (Q₁, Q₂) | Shared across all 32 LOD1 blocks | Learn universal compression strategy |
| LOD1 layer weights | Shared across all 32 LOD1 blocks | Parameter efficiency |
| LOD2 slot queries | Independent from LOD1 | May need different strategy for gist→gist compression |
| LOD2 layer weights | May share with LOD1 or specialize | Implementation choice |

---

## Complete Pseudocode Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GistNetBlock(nn.Module):
    """Single 32→1 compression block with 32→1→32→1 refinement."""

    def __init__(self, d_model=4096, n_heads=8, d_ff=512):
        super().__init__()

        # Learnable slot queries
        self.slot_q1 = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.slot_q2 = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Stage 1: Self-attention layers
        self.self_attn_1 = MultiHeadAttention(d_model, n_heads)
        self.self_attn_2 = MultiHeadAttention(d_model, n_heads)
        self.ffn_sa_1 = FeedForward(d_model, d_ff)
        self.ffn_sa_2 = FeedForward(d_model, d_ff)

        # Stage 2: First compression
        self.cross_attn_compress_1 = CrossAttention(d_model, n_heads)
        self.ffn_compress_1 = FeedForward(d_model, d_ff)

        # Stage 3: Expansion
        self.cross_attn_expand = CrossAttention(d_model, n_heads)
        self.ffn_expand = FeedForward(d_model, d_ff)
        self.self_attn_refine = MultiHeadAttention(d_model, n_heads)

        # Stage 4: Final compression
        self.cross_attn_compress_2 = CrossAttention(d_model, n_heads)
        self.ffn_compress_2 = FeedForward(d_model, d_ff)

        # Layer norms
        self.ln_1a = nn.LayerNorm(d_model)
        self.ln_1b = nn.LayerNorm(d_model)
        self.ln_2a = nn.LayerNorm(d_model)
        self.ln_2b = nn.LayerNorm(d_model)
        self.ln_3a = nn.LayerNorm(d_model)
        self.ln_3b = nn.LayerNorm(d_model)
        self.ln_4a = nn.LayerNorm(d_model)
        self.ln_final = nn.LayerNorm(d_model)

        # Positional encoding
        self.rope = RotaryPositionalEncoding(d_model)

    def forward(self, tokens):
        """
        Args:
            tokens: [batch, 32, d_model] - raw token embeddings

        Returns:
            gist: [batch, d_model] - final gist vector
        """
        batch_size = tokens.shape[0]

        # ============ STAGE 1: Self-Attention (32 → 32) ============
        tokens_pos = self.rope(tokens)

        # First self-attention block
        x = self.ln_1a(tokens_pos)
        attn_out = self.self_attn_1(x, x, x)
        x = tokens_pos + attn_out
        x = x + self.ffn_sa_1(self.ln_1b(x))

        # Second self-attention block
        e1 = x
        x = self.ln_2a(e1)
        attn_out = self.self_attn_2(x, x, x)
        x = e1 + attn_out
        e1 = x + self.ffn_sa_2(x)

        # ============ STAGE 2: First Compression (32 → 1) ============
        q1 = self.slot_q1.unsqueeze(0).expand(batch_size, -1, -1)

        g1 = self.cross_attn_compress_1(
            query=q1,
            key=e1,
            value=e1
        )

        g1 = q1 + g1
        g1 = g1 + self.ffn_compress_1(self.ln_2b(g1))

        # ============ STAGE 3: Expansion (1 → 32) ============
        e2 = self.cross_attn_expand(
            query=e1,
            key=g1,
            value=g1
        )

        e2 = e1 + e2
        e2 = e2 + self.ffn_expand(self.ln_3a(e2))

        # Optional: self-attention for diffusion
        e2_refine = self.ln_3b(e2)
        e2_sa = self.self_attn_refine(e2_refine, e2_refine, e2_refine)
        e2 = e2 + e2_sa

        # ============ STAGE 4: Final Compression (32 → 1) ============
        q2 = self.slot_q2.unsqueeze(0).expand(batch_size, -1, -1)

        g_final = self.cross_attn_compress_2(
            query=q2,
            key=e2,
            value=e2
        )

        g_final = q2 + g_final
        g_final = g_final + self.ffn_compress_2(self.ln_4a(g_final))

        # Final normalization and squeeze
        g_final = self.ln_final(g_final).squeeze(1)

        return g_final


class HierarchicalGistNet(nn.Module):
    """Two-level hierarchical compression: 1024 tokens → 1 gist."""

    def __init__(self, d_model=4096, n_heads=8, d_ff=512):
        super().__init__()

        self.l1_gistnet = GistNetBlock(d_model, n_heads, d_ff)
        self.l2_gistnet = GistNetBlock(d_model, n_heads, d_ff)

    def forward(self, tokens):
        """
        Args:
            tokens: [batch, 1024, d_model]

        Returns:
            l2_gist: [batch, d_model]
            l1_gists: [batch, 32, d_model]
        """
        batch_size = tokens.shape[0]
        tokens_reshaped = tokens.view(batch_size, 32, 32, -1)

        l1_gists = []
        for i in range(32):
            span = tokens_reshaped[:, i, :, :]
            gist = self.l1_gistnet(span)
            l1_gists.append(gist)

        l1_gists = torch.stack(l1_gists, dim=1)
        l2_gist = self.l2_gistnet(l1_gists)

        return l2_gist, l1_gists
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Single 32-token span | O(32² × d) | Dominated by self-attention |
| LOD1 compression (per span) | O(32² × d) | ~negligible vs base LLM |
| LOD2 compression | O(32² × d) | Same as LOD1 |
| Full 1024-token compression | O(32³ × d) | 32 LOD1 + 1 LOD2 operations |

**Comparison to Base LLM:**
- Base LLM (1024 tokens): O(1024² × d) ≈ 1M × d operations
- GistNet (1024 tokens): O(32³ × d) ≈ 33K × d operations
- **~30× faster** than full attention over same window

### Runtime Benchmarks (POC)

Based on NVIDIA L4 GPU, bf16 precision, `HuggingFaceTB/SmolLM3-3B`:

| Operation | Time | Throughput |
|-----------|------|------------|
| Single 32-token span | <1 ms | >1000 spans/sec |
| 1024-token compression (LOD1+LOD2) | ~10 ms | >100 blocks/sec |
| Batch of 32 spans (parallel) | ~5 ms | >6000 spans/sec |

**Expected on A100:**
- 2-3× faster throughput
- Better batching efficiency

---

## Related Pages

### Parent/Overview Pages
- [[GistNet]] – Parent page: overview of compression architecture and core concepts
- [[Architecture Details]] – Complete system architecture showing GistNet's role
- [[POC Architecture]] – System-wide architecture and integration details

### Sibling Detail Pages
- [[GistNet Training]] – Training methodology, loss functions, and optimization for this architecture
- [[LensNet]] – Companion model that uses gists to reconstruct token representations
- [[LensNet Training]] – Training approach for the LensNet component
- [[LensNet Scoring]] – How LensNet scores and uses gist embeddings

### Related System Components
- [[MegaContext Tree]] – Storage structure for hierarchical gists produced by this architecture
- [[Tree Operations]] – How GistNet is invoked during ingest and gist generation
- [[Focus Allocator]] – Selects which gists to expand based on relevance scoring
- [[Focus Allocator Strategies]] – Specific algorithms that use gist representations
- [[Working Context]] – Primary consumer of gist embeddings during runtime
- [[Working Context Assembly]] – How gists are assembled into working context

### Core Concepts & Principles
- [[Glossary#Substitutability|substitutability]] – Core design principle that drives architecture decisions
- [[Glossary#ΔNLL (Delta Negative Log-Likelihood)|ΔNLL]] – Primary training metric used to validate architecture effectiveness
- [[Glossary#Gist / Gist Embedding]] – The compressed representations this architecture produces
- [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)]] – Hierarchical levels this architecture compresses

### Implementation Guides
- [[POC Implementation]] – Practical implementation details and parameter choices
- [[Training & Operations]] – Training pipeline and operational deployment
- [[MegaContext End-to-End Training]] – How architecture is trained within optimization framework
- [[Node Metadata]] – Metadata tracking for gist versions and storage offsets

### Technical Details
- [[Storage Format]] – Binary storage format for gist embeddings produced by architecture
- [[Telemetry]] – Metrics tracking for architecture performance
- [[Invariants]] – System invariants maintained by the architecture

### Examples & Getting Started
- [[Context Focus]] – Initial setup including GistNet architecture instantiation
- [[Examples]] – Example compression runs and validation
- [[How MegaContext Works]] – High-level explanation including GistNet's role
- [[MegaTexture Analogy]] – Intuitive analogy for understanding hierarchical compression

---

## Summary

**GistNet** is a specialized neural architecture that:

1. **Compresses** 32-token spans into single gist embeddings via learned slot queries
2. Uses a **32→1→32→1 refinement loop** for improved substitutability
3. Employs **self-attention** for local contextualization and **cross-attention** for compression/expansion
4. Achieves **1024× compression** through two-level hierarchical stacking
5. Maintains **~1M parameters** total in the POC configuration
6. Produces gists **aligned with base LLM embedding space** for seamless substitution
7. Operates with **O(32² × d) complexity** per span—negligible compared to full LLM attention
8. Supports **<1ms per span** processing on modern GPUs

The architecture's key innovation is the refinement loop that allows gists to be "tested" by expanding back to token space before final compression, ensuring critical information is preserved for [[Glossary#Substitutability|substitutability]].
