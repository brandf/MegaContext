---
tags: [papers, reference]
summary: Efficient fine-tuning approach that extends context windows in large language models using shifted sparse attention and LoRA adaptation, achieving 100k context at minimal training cost.
---

# LongLoRA (arXiv:2309.12307) — Report

**PDF**: [LongLoRA - 2309.12307.pdf](LongLoRA%20-%202309.12307.pdf)

## Overview
- Introduces **shifted sparse attention (S²-Attn)** during training to enable efficient context extension while maintaining dense attention at inference, avoiding architecture modifications that would prevent use of existing inference optimizations.
- Combines sparse attention training with **LoRA (Low-Rank Adaptation)** to extend context windows from typical 2k-4k to 100k tokens with minimal parameter updates and training cost.
- Demonstrates that embedding and normalization layers are critical for long-context adaptation—fine-tuning only attention via LoRA is insufficient, but adding these trainable parameters enables successful extension.
- Achieves strong performance on long-context benchmarks (PG19, Proof-pile) with LLaMA2 models (7B-70B), showing that parameter-efficient methods can match or exceed full fine-tuning quality.

## Core Concepts
- **Shifted Sparse Attention (S²-Attn)**: During training, attention is restricted to local groups (e.g., 2048 tokens) with half the groups shifted by half the group size, creating overlapping windows. This reduces training memory and compute while still allowing information flow across the full context. At inference, standard dense attention is used.
- **Trainable parameter selection**: Beyond LoRA adapters on attention projections, LongLoRA trains embedding layers and normalization layers (LayerNorm/RMSNorm). These components are crucial for adapting position encodings and feature distributions to longer contexts.
- **Two-stage training**: First pretrain with short contexts using LoRA, then extend to long contexts with S²-Attn. This progressive approach stabilizes training and reduces convergence issues.
- **Position interpolation**: Leverages position interpolation techniques to smoothly extend RoPE embeddings from the pretrained context length to the target long context, preventing catastrophic interference with learned position representations.

## Relevance to MegaContext
- **Parameter-efficient training strategy**: LongLoRA's demonstration that LoRA suffices for attention while only embeddings and norms need full training directly informs [[GistNet Training]] and [[LensNet Training]]. We can freeze most base model parameters while training our added components.
- **Sparse attention during training**: S²-Attn's approach of using efficient attention patterns during training while maintaining dense inference aligns with MegaContext's need to train on long horizons for [[Glossary#ΔNLL@H (Delta NLL at Horizon)|ΔNLL@H]] measurement without prohibitive memory costs.
- **Context extension via position adaptation**: LongLoRA's focus on embedding and normalization layers for context extension validates our approach of carefully managing [[Positional Encoding]] when retrofitting base models for MegaContext. Their position interpolation techniques can guide our [[Glossary#NTK Scaling|NTK Scaling]] implementation.
- **Efficient long-context training**: LongLoRA's ability to train on 100k contexts with reasonable GPU memory provides a template for how we can conduct [[Alternating Optimization]] with long ΔNLL horizons without requiring massive infrastructure.

## What We Can Use
- **Adopt S²-Attn for GistNet training**: Implement shifted sparse attention during [[GistNet Training]] when computing ΔNLL@H over long horizons. This would reduce memory costs while still providing meaningful substitutability signals across extended contexts.
- **Apply LoRA strategically**: Following LongLoRA's findings, use LoRA adapters for attention layers in the [[Base Model]] adaptation while fully training GistNet and LensNet components. This minimizes interference with the frozen base model.
- **Prioritize embedding/norm training**: When fine-tuning the base model for MegaContext compatibility (if needed), focus training on embedding and normalization layers as LongLoRA shows these are critical for long-context adaptation.
- **Implement progressive context extension**: Use a two-stage approach for [[Training & Operations]]—first train components on shorter contexts (e.g., 8k tokens), then extend to full target contexts (32k+). This can improve stability and reduce training time.
- **Leverage position interpolation**: Integrate LongLoRA's position interpolation techniques into our [[Positional Encoding]] retrofit pipeline to smoothly extend RoPE-based models beyond their pretrained context limits.

## Limitations & Risks
- **Training-inference mismatch**: S²-Attn creates a distribution shift between training (sparse) and inference (dense) attention patterns. While LongLoRA shows this works, we must validate that [[GistNet]] and [[LensNet]] don't overfit to sparse attention artifacts during training.
- **Limited to decoder-only models**: LongLoRA's techniques are demonstrated primarily on decoder-only LLaMA models. Adaptation to encoder-decoder or other architectures may require modifications, though this is less relevant for MegaContext's focus on decoder-only base models.
- **Position interpolation degradation**: While position interpolation enables context extension, it can degrade quality for tasks requiring precise positional reasoning. MegaContext's [[Glossary#Gaussian RoPE|Gaussian RoPE]] may face similar challenges when representing uncertain positions in gist spans.
- **Embedding layer overhead**: Training embedding layers adds memory overhead since these layers are large (vocab_size × d_model). For MegaContext with frozen base models, we avoid this issue, but it's relevant if we need base model adaptation.
- **Context length scaling limits**: LongLoRA achieves 100k contexts, but perplexity degrades at extreme lengths. MegaContext's hierarchical approach should scale beyond these limits, but we must verify that integration with long-context base models doesn't introduce unexpected failure modes.

## Potential Follow-Up Reading
- **LoRA paper** (Hu et al., 2021): Original low-rank adaptation work—foundational for understanding LongLoRA's parameter-efficient approach and relevant for MegaContext's [[GistNet]] and [[LensNet]] architecture decisions.
- **Position Interpolation papers** (Chen et al., 2023): Detailed analysis of extending RoPE and other position encodings—critical for [[Positional Encoding]] strategies in MegaContext.
- **Landmark Attention** (Mohtashami & Jaggi, 2023): Alternative approach to long-context training using random landmark tokens, offering different trade-offs than LongLoRA's shifted sparse attention.
- **LongLLaMA** (Tworkowski et al., 2023): Uses FoT (Focused Transformer) to extend contexts via cross-attention to memory—complementary approach to both LongLoRA and MegaContext.
- **Efficient attention surveys** covering sparse, local, and approximate attention patterns to understand the full landscape of training-time optimizations.

## Open Questions for MegaContext
- Should we implement **S²-Attn or similar sparse patterns** during GistNet training to reduce memory when computing ΔNLL@H over 64-128 token horizons, or does the relatively modest horizon length make this unnecessary?
- Can we **combine LongLoRA's position interpolation with Gaussian RoPE**—using interpolation for the mean position while learning variance from data—to get both smooth extension and uncertainty representation?
- How should we **handle embedding layer adaptation** if base models pretrained at short contexts (e.g., 4k) are used in MegaContext? Should we apply LoRA to embeddings or freeze them entirely?
- What is the **optimal LoRA rank** for base model adaptation in MegaContext? LongLoRA uses ranks of 8-64; we need to determine appropriate values for our [[Focus Allocator]] and working-context assembly components.
- Can **LongLoRA's techniques extend to LensNet training**, where we need to handle sequences of mixed-LOD entries that may not follow standard attention patterns?

## Related Pages
- [[GistNet Training]]
- [[LensNet Training]]
- [[Alternating Optimization]]
- [[Training & Operations]]
- [[Positional Encoding]]
- [[LoRA]]
- [[Base Runtime]]
- [[POC Architecture]]
- [[GistNet]]
- [[LensNet]]
- [[Working Context]]
