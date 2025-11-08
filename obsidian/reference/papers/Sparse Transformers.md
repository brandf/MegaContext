---
tags: [papers, reference]
summary: Introduces sparse attention patterns (strided and fixed) to scale transformers to sequence lengths of 16k+ tokens, reducing O(n²) attention cost to O(n√n).
---

# Sparse Transformers (arXiv:1904.10509)

**PDF**: [Sparse Transformers - 1904.10509.pdf](Sparse%20Transformers%20-%201904.10509.pdf)

## Overview
- Introduces **sparse factorizations of the attention matrix** to reduce computational and memory requirements from O(n²) to O(n√n), enabling transformers to scale to sequences of 16,384+ tokens.
- Proposes two primary sparse attention patterns:
  - **Strided attention**: Each token attends to every k-th token (e.g., k=128), providing global context with fixed stride
  - **Fixed attention**: Each token attends to a fixed block of nearby tokens plus tokens that attend to it, creating local-global patterns
- Demonstrates strong performance on long-range tasks including text generation, image generation (ImageNet 64×64), and raw audio modeling (5 seconds at 12kHz).
- Uses gradient checkpointing and mixed precision to further reduce memory footprint during training.

## Core Concepts
- **Factorized Attention Patterns**: Rather than computing full n×n attention, decompose into multiple sparse attention heads that attend to complementary subsets of positions. The union of patterns in different heads provides connectivity across the full sequence.
- **Strided Pattern**: Token i attends to positions {i-l, i-l+k, i-l+2k, ...}, where l is the local window and k is the stride. Provides global context with regular sampling.
- **Fixed Pattern**: Token i attends to local positions {i-l, ..., i} plus specific fixed positions (e.g., multiples of k). Creates pyramid-like connectivity where information flows through hub positions.
- **Recomputation during Backprop**: Uses gradient checkpointing to trade computation for memory—stores only subset of activations during forward pass and recomputes as needed during backward pass.
- **Position-dependent patterns**: Different positions can use different attention patterns (e.g., early layers use fixed, later layers use strided), enabling architectural flexibility.

## Relevance to MegaContext
- **Direct inspiration for [[LensNet]]'s non-causal attention**: Sparse Transformers show that selective connectivity patterns can maintain quality while reducing cost—exactly what LensNet needs when scoring working-context entries.
- **Strided attention aligns with gist hierarchy**: LOD1/LOD2 gists naturally form strided patterns when collapsed—attending to every 32nd or 1024th position. We can leverage this structural sparsity without additional masking overhead.
- **Fixed attention informs [[Focus Allocator]] strategies**: The idea that certain "hub" positions (in our case, recently accessed or high-utility gists) should be densely connected while distant regions use sparse sampling maps to our expand/collapse decisions.
- **Memory-compute tradeoff insights**: Their gradient checkpointing approach is directly applicable to [[GistNet Training]], where we can recompute gist embeddings during backward pass rather than caching all intermediate states.
- **Long-range dependencies**: Demonstrates that sparse patterns can capture dependencies across 16k+ tokens without full attention—validates our approach of using mixed LOD (LOD0/LOD1/LOD2) rather than requiring dense attention over all history.

## What We Can Use
- **Implement strided LensNet attention**: When LensNet scans the [[Working Context]], use strided patterns to attend to distant LOD2 gists while maintaining dense attention over recent LOD0 blocks. This reduces LensNet's O(W²) cost to O(W√W).
- **Position-dependent attention in [[GistNet]]**: Early compression layers use local-only attention (fixed pattern) to build LOD1 gists, while later meta-gist layers use strided attention to aggregate across LOD1 blocks into LOD2.
- **Gradient checkpointing for training**: Apply their recomputation strategy during [[GistNet Training]] to reduce memory footprint by ~30-50%, allowing larger batch sizes or deeper compression networks.
- **Multi-head pattern mixing**: Use different sparse patterns in different [[LensNet]] attention heads—one head uses strided for global context, another uses fixed for local detail—then merge focus scores.
- **Adaptive stride scheduling**: Start with small strides during early training (more connectivity) and gradually increase stride as [[GistNet]] learns better substitutable compressions, reducing compute as quality improves.

## Limitations & Risks
- **Not truly adaptive**: Sparse patterns are fixed at architecture design time, not learned or adjusted based on content. MegaContext's content-aware focus allocation (via [[LensNet]]) is more flexible but also more complex to train.
- **Requires specialized CUDA kernels**: Efficient implementation demands custom kernels for blocked sparse matmuls. Standard PyTorch/JAX operations can't exploit the sparsity without significant engineering effort.
- **Pattern mismatch with data**: If the fixed stride doesn't align with actual dependency structure (e.g., k=128 but dependencies occur every 100 tokens), performance degrades. MegaContext must ensure gist boundaries align with semantic boundaries.
- **Limited hierarchy**: Only 2-level factorization (local + global). MegaContext's 3-level hierarchy (LOD0/LOD1/LOD2) may need 3-way factorization to fully exploit sparsity.
- **Training instability**: Sparse patterns can create disconnected subgraphs early in training before the model learns to route information through hub positions. Requires careful initialization and learning rate warmup.

## Potential Follow-Up Reading
- **Longformer** (2020.04085) - Sliding window + global tokens for document tasks; more adaptive than fixed sparse patterns
- **BigBird** (2007.14062) - Random + window + global attention; adds stochastic component to sparse patterns
- **Routing Transformers** (2003.05997) - Learned attention routing via clustering; dynamic sparsity based on content similarity
- **Block-Recurrent Transformers** - Combines sparse attention with recurrent memory; relevant for MegaContext's append-only growth
- **Synthesizer** (2005.00743) - Replaces content-based attention with learned patterns; explores data-independent attention

## Open Questions for MegaContext
- Should [[LensNet]] use fixed sparse patterns (cheaper, more predictable) or content-dependent patterns (current plan, more flexible)?
- Can we combine strided attention in LensNet with our gist hierarchy—treating LOD2 gists as natural stride anchors that define the sparse pattern?
- How to handle variable-length gist chains (some branches have LOD2, others only LOD1)? Fixed stride assumes uniform structure.
- Should [[GistNet]]'s cross-attention during compression use sparse patterns, or is full attention necessary for high-quality gist generation?
- Can we meta-learn the optimal stride schedule during [[Training & Operations]]—adjusting k based on observed ΔNLL@H distributions?

## Related Pages
- [[LensNet]]
- [[LensNet Scoring]]
- [[GistNet]]
- [[GistNet Training]]
- [[Focus Allocator]]
- [[Working Context]]
- [[MegaContext Tree]]
- [[Architecture Details]]
- [[System Properties]]
