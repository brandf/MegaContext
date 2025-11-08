---
tags: [papers, reference]
summary: Fast and memory-efficient exact attention using IO-aware tiling and kernel fusion, achieving 2-4× speedup and enabling 64k+ context training.
---

# FlashAttention (arXiv:2205.14135)

**PDF**: [Flash Attention - 2205.14135.pdf](Flash%20Attention%20-%202205.14135.pdf)

## Overview
- Introduces **IO-aware algorithm** for exact attention that reduces memory reads/writes between GPU HBM (high-bandwidth memory) and SRAM (on-chip cache) from O(n²) to O(n²/M) passes, where M is SRAM size.
- Uses **tiling** to divide Q, K, V into blocks that fit in SRAM, computing softmax incrementally using online softmax algorithm with running statistics.
- Implements **fused CUDA kernel** that performs attention computation (QK^T, softmax, dropout, output projection) in a single kernel without materializing intermediate n×n matrices.
- Achieves **2-4× training speedup** on GPT-2/BERT-sized models and enables context lengths up to 64k tokens (limited only by memory, not speed).
- Exact attention—no approximation or quality degradation compared to standard attention.

## Core Concepts
- **Memory Hierarchy Awareness**: Standard attention is memory-bound, not compute-bound. The bottleneck is moving data between slow HBM (40-80 GB, ~1.5 TB/s) and fast SRAM (20 MB, ~19 TB/s). FlashAttention minimizes HBM accesses.
- **Tiling Algorithm**: Divide Q into blocks of size Bᵣ (e.g., 128) and K,V into blocks of size Bᶜ (e.g., 64). For each Q block:
  - Load Q block to SRAM
  - Loop over K,V blocks, computing attention scores and outputs incrementally
  - Use online softmax to maintain numerically stable running max and sum
- **Online Softmax**: Compute softmax in streaming fashion without seeing all values first:
  - Track running max m and sum ℓ for each row
  - When processing new K block, update: m_new = max(m_old, m_block), ℓ_new = exp(m_old - m_new)·ℓ_old + exp(m_block - m_new)·ℓ_block
  - Rescale accumulated output accordingly
- **Kernel Fusion**: Combine QK^T matmul, softmax, attention dropout, and OV matmul into single CUDA kernel. Eliminates intermediate writes/reads of attention matrix.
- **Recomputation in Backward**: Don't store O(n²) attention matrix for backward pass. Instead, recompute attention on-the-fly during backprop using same tiling strategy. Trade computation for memory.

## Relevance to MegaContext
- **Critical for [[POC Implementation]]**: FlashAttention is baseline requirement for efficient [[GistNet]] and [[LensNet]] training. Without it, even 8k [[Working Context]] would be prohibitively slow.
- **Enables long working contexts**: W_max=32k becomes feasible with FlashAttention's linear memory scaling. Standard attention would require 32²×d = 1GB+ just for attention matrix; FlashAttention reduces to ~10MB.
- **[[GistNet]] compression efficiency**: When [[GistNet]] computes cross-attention between 32 input tokens and gist slot queries, FlashAttention reduces 32² reads to ~4 passes. Critical for real-time compression in [[Runtime Loop]].
- **[[LensNet]] scoring speedup**: Non-causal attention over W-length working context benefits from FlashAttention's tiling, reducing [[LensNet Scoring]] latency by 2-3×.
- **Enables deeper hierarchies**: Memory savings allow larger batch sizes during [[GistNet Training]], or longer lookahead horizons H for [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] computation.

## What We Can Use
- **Integrate FlashAttention-2/3 in all attention modules**: Replace PyTorch's scaled_dot_product_attention with FlashAttention implementation. Applies to:
  - [[GistNet]]'s self-attention and cross-attention layers
  - [[LensNet]]'s dual cross-attention over working context
  - [[Base Runtime]]'s frozen LLM attention (if we can modify inference code)
- **Tune block sizes for MegaContext workloads**: Default Bᵣ=128, Bᶜ=64 optimized for standard transformers. Profile MegaContext's specific attention patterns (non-causal LensNet, short cross-attention in GistNet) and adjust tile sizes for optimal HBM↔SRAM traffic.
- **Recomputation strategy for training**: Apply FlashAttention's backward pass recomputation to [[GistNet Training]] and [[LensNet Training]]. Reduces activation memory by 5-10×, enabling larger batch sizes or longer horizons.
- **Variable-length attention**: Use FlashAttention's support for variable sequence lengths to handle [[Working Context Assembly]]'s mixed LOD sequences efficiently—don't need to pad LOD0/LOD1/LOD2 entries to uniform length.
- **Fused dropout and masking**: Implement attention masking (for causal/non-causal switching) and dropout inside the FlashAttention kernel rather than as separate operations, reducing memory traffic by another 20-30%.
- **Multi-query attention optimization**: FlashAttention-2 includes optimizations for MQA/GQA patterns (shared KV across heads). If we adopt grouped-query attention in [[GistNet]] or [[LensNet]], use these optimizations.
- **Benchmark attention patterns**: Profile MegaContext's specific attention workloads (32-token compression, W-length scoring, H-horizon lookahead) against FlashAttention's block sizes to identify bottlenecks and optimize accordingly.

## Limitations & Risks
- **Requires CUDA**: FlashAttention is GPU-only (CUDA/Triton). No efficient CPU fallback, which limits development/debugging on non-GPU machines. Critical for [[POC Implementation]] deployment.
- **Backward pass recomputation overhead**: Trading memory for compute works during training but adds 10-20% wall-clock time. For very long sequences (H=128), this overhead compounds.
- **Causal vs. non-causal switching**: FlashAttention optimizes for causal attention. [[LensNet]]'s non-causal attention over working context may not benefit as much from tiling optimizations—need separate tuning.
- **Dropout during inference**: FlashAttention's fused dropout is training-only. If we use dropout-based uncertainty estimation during inference, need separate implementation path.
- **Mixed precision numerical issues**: Aggressive FP16/BF16 usage in FlashAttention can cause numerical instability with extreme attention distributions. May require careful scaling for [[LensNet]]'s signed focus scores.
- **Version compatibility**: FlashAttention-2/3 APIs differ significantly. Pinning to specific version creates tech debt; upgrading requires retesting all attention modules.

## Potential Follow-Up Reading
- **FlashAttention-2** (2307.08691) - 2× faster than v1, better parallelism for long sequences
- **FlashAttention-3** (2024.07.07) - Hardware-aware optimizations for Hopper GPUs (H100), asynchronous loads
- **Paged Attention** (vLLM) - Memory management for dynamic batching; complements FlashAttention for serving
- **Triton tutorials** - Understanding kernel implementation details; useful for custom MegaContext attention patterns
- **Memory-Efficient Attention (xformers)** - Alternative implementation with similar goals; comparison baseline
- **Ring Attention** (2310.01889) - Distributed attention across multiple devices; relevant for scaling beyond single GPU

## Open Questions for MegaContext
- What's the actual speedup/memory benefit for MegaContext's specific attention patterns (non-causal LensNet, small cross-attention GistNet) vs. standard causal LM attention?
- Should we fork FlashAttention to add custom support for mixed LOD sequences (LOD0/LOD1/LOD2 with different embedding scales)?
- Can we combine FlashAttention's tiling with sparse attention patterns from [[Sparse Transformers]]—tile within sparse blocks for maximum efficiency?
- How to handle FlashAttention's recomputation during counterfactual ΔNLL@H computation in [[LensNet Training]]—double recomputation overhead?
- Should [[POC Implementation]] have fallback to standard attention for debugging, or always require FlashAttention (simpler but less portable)?
- What's the memory/speed tradeoff between FlashAttention's recomputation and [[Reformer]]'s reversible layers? Can we combine both?

## Related Pages
- [[GistNet]]
- [[GistNet Training]]
- [[GistNet Architecture Details]]
- [[LensNet]]
- [[LensNet Scoring]]
- [[LensNet Training]]
- [[Working Context]]
- [[Working Context Assembly]]
- [[POC Implementation]]
- [[Base Runtime]]
- [[Runtime Loop]]
- [[Training & Operations]]
- [[Performance Sketch]]
- [[Sparse Transformers]]
- [[Reformer]]
