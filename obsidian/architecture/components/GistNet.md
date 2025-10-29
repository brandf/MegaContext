---
tags:
  - module
summary: 32→1 hierarchical encoder that substitutes token spans with gists compatible with the base LLM.
---
![[GistNetDiagram.png]]

GistNet compresses each N-token/gist block (32 in [[POC Implementation]]) into a single latent space gist aligned with base embeddings, enabling substitutable multi-level representations inside MegaContext.

## What is GistNet?

GistNet is a **local encoder for token spans** that replaces short, fixed-length token sequences with compact **gist embeddings** ("gists"). Each gist preserves the meaning of its original tokens while freeing [[Working Context]] budget for new information. The key innovation is **substitutability**: gists can stand in for their original tokens inside the base LLM's context with minimal impact on predictions.

## Purpose

- **Replace raw tokens** with substitutable gists to free context budget.
- **Enable compression** of 32 tokens → 1 gist at each layer.
- **Hierarchical stacking** provides 1024× compression (32² with two layers).
- **Align embeddings** with the base LLM's embedding space for seamless integration.

## High-Level Architecture

GistNet uses a **two-stage refinement process** to compress token spans:

### 32→1→32→1 Refinement Pipeline

1. **Local self-attention (32 → 32)**
   - Standard self-attention + MLP blocks within 32-token windows
   - RoPE positional encodings for local ordering
   - Produces locally contextualized embeddings `E1`

2. **First compression (32 → 1)**
   - Learned slot query `Q₁` (shared across spans)
   - Cross-attention: slot reads from tokens
   - Produces first gist `G1`

3. **Expansion for refinement (1 → 32)**
   - Cross-attention: tokens read from gist
   - Diffuses gist information back across token space
   - Produces refined embeddings `E2`

4. **Final compression (32 → 1)**
   - Second learned slot query `Q₂`
   - Cross-attention: slot reads from refined tokens
   - Produces final gist `g_final`

### Hierarchical Stacking

- Two 32→1 layers stacked hierarchically
- Lower layer: operates on raw token embeddings
- Upper layer: operates on lower-layer gist outputs
- Result: 1024 tokens compressed to 1 top-level gist (32² compression)

## Key Properties

| Property | Description |
|----------|-------------|
| **Substitutability** | Gists can replace original tokens in LLM context with minimal prediction change |
| **Limited scope** | Operates within fixed 32-token windows; no long-range attention |
| **Parameter sharing** | Shared weights across all spans |
| **Aligned embeddings** | Output dimension matches base LLM embedding size |
| **Efficient** | O(32²·d) per span — negligible compared to base LLM |
| **Positioned** | Gist inserted at central token index for RoPE alignment |

## Training

GistNet is trained to ensure **substitutability**: replacing a span with its gist should minimally change the base LLM's predictions. The primary objective minimizes [[ΔNLL]]@H between original and gist-replaced contexts.

For complete training details, see:
- [[GistNet Training]] — Training objectives, loss functions, curriculum, and pipeline
- [[POC Implementation]] — Specific parameters and settings

## Architecture Specifications

For detailed architectural specifications, see:
- [[GistNet Architecture Details]] — Layer counts, dimensions, attention configurations, complexity analysis
- [[POC Implementation]] — Complete POC parameters and implementation details

## Role in System

GistNet forms the foundation of MegaContext's compression hierarchy:

1. **Compresses tokens** into substitutable gist embeddings
2. **Builds gist tree** in the [[MegaContext Tree]] for hierarchical storage
3. **Feeds gists** to [[LensNet]] for tail compression scoring
4. **Enables focus loop** by providing compressed representations to [[Focus Allocator]]
5. **Frees context budget** for new information in [[Working Context]]

```
Tokens (32) → GistNet L1 → Gists (1)
             ↓
Gists L1 (32) → GistNet L2 → Gists L2 (1)
                              ↓
                    MegaContext Tree Storage
                              ↓
                    LensNet + Focus Allocator
```

## Inputs & Outputs

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `E ∈ R[32, d]` | 32 raw token embeddings (no contextualization) |
| `Q₁, Q₂ ∈ R[1, d]` | Learned slot queries for two compression passes |
| `g_final ∈ R[d]` | Final gist vector aligned with base LLM embedding dim |

## Related Pages

- [[GistNet Training]] — Training objectives, loss functions, and pipeline
- [[GistNet Architecture Details]] — Complete architectural specifications
- [[MegaContext Tree]] — Storage structure for gist hierarchy
- [[LensNet]] — Scores tail gists for compression decisions
- [[Focus Allocator]] — Selects gists for working context restoration
- [[Working Context]] — Where gists are decompressed back to tokens
- [[ΔNLL]] — Primary training metric for substitutability
- [[substitutability]] — Core principle guiding GistNet design
- [[POC Implementation]] — Proof of concept parameters and settings
- [[Training & Operations]] — Overall training strategy
