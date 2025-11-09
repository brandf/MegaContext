---
tags:
  - components
summary: Transformer-based 32→1 compressor that substitutes token spans with gists compatible with the base LLM.
---

![[GistNetDiagram.png]]

GistNet compresses each 32-token block into a single gist embedding aligned with the base model’s hidden space, enabling substitutable multi-level representations inside MegaContext.

## What is GistNet?

GistNet is a **local encoder for token spans** that replaces fixed-length token sequences with compact **gist embeddings** (“gists”) [1]. Each gist preserves the meaning of its original tokens while freeing [[Working Context]] budget for new information. The key innovation is **substitutability** [1]: gists can stand in for their original tokens inside the base LLM’s context with minimal impact on predictions.

## Purpose

- **Replace raw tokens** with substitutable gists to free context budget.
- **Enable compression** of 32 tokens → 1 gist at each level of the [[MegaContext Tree]].
- **Hierarchical stacking** provides 1024× compression (32² with two levels).
- **Align embeddings** with the base model’s embedding space for seamless substitution.

## Architecture Snapshot

GistNet now uses a lightweight stack of nanochat transformer blocks on top of a 32-token window. Pooling heads collapse the final activations into a single gist without modifying the backbone.

```mermaid
flowchart LR
    Tokens[32 token embeddings]
    CLS[(optional CLS token)]
    Blocks[Mini transformer blocks<br/>(RoPE + MHA + MLP)]
    Head[Pooling head<br/>(mean · query · CLS)]
    Gist[gist ∈ ℝ^d]

    Tokens --> Blocks
    Tokens -. prepend .-> CLS --> Blocks
    Blocks --> Head --> Gist
```

1. **Embedding + RoPE.** The 32-token slice is embedded and rotated with local RoPE metadata.
2. **Mini transformer stack.** 1–N nanochat blocks (attention + MLP) process the window. Heads share weights across all spans; adding a `[CLS]` token is optional per head.
3. **Pooling head.** We collapse the per-token states via one of three readouts:
   - **Mean pooling** + linear or MLP projection (baseline).
   - **Query pooling**: a learned query attends to the window once (Structured Self-Attentive pooling [4]).
   - **CLS pooling**: prepend a learnable `[CLS]` token and read it directly (BERT-style [5]).

### Current Code Path

| Depth | CLI flag (`--gistnet`) | Pooling mode | Projection | Notes |
| --- | --- | --- | --- | --- |
| 2 layers *(default family)* | `transformer2_mean_mlp` *(default)* | Mean | 2-layer MLP | Strong accuracy/cost balance. |
| | `transformer2_query_mlp` | Query | 2-layer MLP | Adds explicit attention pooling. |
| | `transformer2_cls_mlp` | CLS token | 2-layer MLP | Leverages `[CLS]` token. |
| | `transformer2_mean_linear` | Mean | Linear | Cheapest transformer head. |
| | `transformer2_query_linear` | Query | Linear | Query pooling w/out MLP. |
| | `transformer2_cls_linear` | CLS token | Linear | CLS pooling w/out MLP. |
| 4 layers *(heavier family)* | `transformer4_mean_mlp` | Mean | 2-layer MLP | Higher-quality / higher-FLOPs option. |
| | `transformer4_query_mlp` | Query | 2-layer MLP | Deeper backbone + query pooling. |
| | `transformer4_cls_mlp` | CLS token | 2-layer MLP | Deeper `[CLS]` summary. |
| | `transformer4_mean_linear` | Mean | Linear | Heavier backbone, cheap head. |
| | `transformer4_query_linear` | Query | Linear | Query pooling on 4-layer stack. |
| | `transformer4_cls_linear` | CLS token | Linear | CLS pooling on 4-layer stack. |
| Baseline | `mean_linear` | Mean | Linear | “Dumb” control with no transformer. |

> The legacy self-attention/slot variants were removed—only the transformer stack + pooling heads remain in code.

## Workflow in MegaContext

1. **Encode span.** The per-device embedding layer produces `[B, 32, d]` slices for each token block.
2. **Run GistNet.** The mini transformer + pooling head outputs a gist `[B, 1, d]`.
3. **Insert gist.** The [[MegaContext Tree]] stores the gist at LOD0 and higher, reducing on-device context size.
4. **Hierarchical stacking.** Re-run GistNet on sets of 32 gists to obtain higher LOD nodes (32² compression).
5. **Focus loop.** [[LensNet]] scores the resulting gists, and the [[Focus Allocator]] decides when to expand or collapse them.

## Training Summary

Training still minimizes [[Glossary#ΔNLL (Delta Negative Log-Likelihood)|ΔNLL]]@H using a frozen teacher LLM:

1. Sample a random 32-token window.
2. Run GistNet → gist.
3. Replace the window with the gist in-context.
4. Teacher runs on original vs. gist context, providing logits for KL/ΔNLL.
5. Optional contrastive term keeps neighbouring gists distinct.

See [[GistNet Training]] for objectives, loss weights, and optimizer settings.

## Inputs & Outputs

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `E ∈ ℝ^{32×d}` | Raw token embeddings from the base model |
| `CLS ∈ ℝ^{d}` | Optional learnable summary token (CLS pooling only) |
| `g ∈ ℝ^{d}` | Final gist vector aligned with the base model |

## Related Pages

- [[GistNet Training]] — Training objectives, loss functions, and pipeline.
- [[GistNet Architecture Details]] — Deep dive into transformer blocks, pooling heads, and complexity.
- [[MegaContext Tree]] — Hierarchical storage for multi-level gists.
- [[LensNet]] / [[Focus Allocator]] — Consumers of gists during focus allocation.
- [[Training & Operations]] — End-to-end training orchestration.

## References

1. **Gist Tokens** (Mu et al., 2023) — [[reference/papers/Gist Tokens - 2304.08467v3.md|Analysis]] — Learned prompt compression via attention masking.  
2. **Compressive Transformer** (Rae et al., 2019) — [[reference/papers/Compressive Transformer.md|Analysis]] — Long-term compressed memory for transformers.  
3. **Knowledge Distillation** (Hinton et al., 2015) — [[reference/papers/Knowledge Distillation.md|Analysis]] — Teacher-student framework for GistNet training.  
4. **Structured Self-Attentive Sentence Embedding** (Lin et al., 2017) — [[reference/papers/Structured Self-Attentive Sentence Embedding.md|Analysis]] — Query-based pooling inspiration.  
5. **BERT** (Devlin et al., 2018) — [[reference/papers/BERT - 1810.04805.md|Analysis]] — CLS token pooling inspiration.

See [[Related Work]] for the full bibliography used across the MegaContext documentation.
