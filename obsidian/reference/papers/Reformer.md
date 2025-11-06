---
tags: [papers, reference]
summary: Memory-efficient transformer using locality-sensitive hashing (LSH) for O(n log n) attention and reversible layers to eliminate activation storage.
---

# Reformer (arXiv:2001.04451)

**PDF**: [Reformer - 2001.04451.pdf](Reformer%20-%202001.04451.pdf)

## Overview
- Introduces **locality-sensitive hashing (LSH) attention** to reduce attention complexity from O(n²) to O(n log n) by hashing queries and keys into buckets and only computing attention within buckets.
- Uses **reversible layers** (RevNets) to eliminate the need to store activations during training—each layer can be reconstructed from the next layer's activations during backprop, reducing memory by 50%+.
- Applies **chunked feed-forward layers** that process inputs in small batches to trade computation for memory in the FFN components.
- Enables training on sequences of 64k tokens on a single device with 16GB memory, demonstrating 16× longer sequences than standard transformers.
- Validates on multiple domains: long-document modeling (enwik8), image generation, and synthetic copying tasks.

## Core Concepts
- **LSH Attention**: Hash queries q and keys k using random projection LSH (angular hashing). Tokens with similar hashes (falling into the same bucket) are likely to have high attention scores. Compute attention only within each bucket, sorting tokens by hash to group them.
- **Multi-round hashing**: Use multiple independent hash functions (e.g., n_rounds=4) to reduce chance of missing important pairs. Average attention scores across rounds.
- **Shared QK**: Set Q=K (same projection matrix) to ensure qi and ki hash to the same bucket when i=i (self-attention). This reduces hash collisions and memory.
- **Reversible Residual Layers**: Split activations into (x₁, x₂) and compute:
  - y₁ = x₁ + Attention(x₂)
  - y₂ = x₂ + FFN(y₁)
  - Backward: Reconstruct x₂ = y₂ - FFN(y₁), x₁ = y₁ - Attention(x₂)
- **Chunked FFN**: Process FFN(x) in chunks of size c (e.g., 128 tokens) to limit peak memory. Trade: more kernel launches vs. lower memory footprint.

## Relevance to MegaContext
- **LSH for gist retrieval**: Rather than [[LensNet]] scoring every entry in the [[Working Context]], use LSH to quickly identify which gists are likely relevant to the current query. This converts O(W²) LensNet attention to O(W log W).
- **Reversible [[GistNet]]**: Apply RevNet architecture to GistNet's compression layers to reduce memory during [[GistNet Training]]. Since we process 32→1→32→1, we can make both compression stages reversible, halving activation memory.
- **Chunked compression**: When building LOD2 gists from 32 LOD1 gists, process in chunks of 8 L1s at a time rather than materializing full 32-block cross-attention matrices. Reduces peak memory during hierarchical compression.
- **Hash-based tree navigation**: Store LSH signatures for each gist node in the [[MegaContext Tree]]. When assembling [[Working Context]], hash the current query and retrieve gists with matching hashes, then apply [[Focus Allocator]] only to retrieved candidates.
- **Memory-efficient staging context**: [[Staging Context]] with 100k entries would benefit enormously from reversible attention—current dense attention over staging would be prohibitively expensive.

## What We Can Use
- **Implement LSH attention for LensNet-G**: When [[LensNet]] operates over the large [[Staging Context]], use LSH to bucket entries by semantic similarity. Only compute full attention scores within top-k buckets, reducing compute by 10-50×.
- **Reversible GistNet layers**: Rewrite [[GistNet Architecture Details]] to use reversible residuals around the self-attention and cross-attention blocks. Reduces training memory for GistNet by ~40-60%.
- **Hash-augmented node metadata**: Add LSH hash codes (e.g., 64-bit signatures) to [[Node Metadata]] for each gist. During [[Working Context Assembly]], filter candidates using hash similarity before running expensive ΔNLL computations.
- **Chunked LOD2 generation**: Modify [[GistNet Training]] pipeline to generate LOD2 gists in chunks—compress 32 L1s in 4 passes of 8 L1s each, accumulating cross-attention incrementally. Enables deeper compression hierarchy (LOD3, L4) without memory explosion.
- **Shared QK in LensNet**: Simplify [[LensNet]] by using shared query/key projections for self-attention over working context. Reduces parameter count by 33% and aligns with Reformer's stability findings.

## Limitations & Risks
- **Approximation errors**: LSH attention is approximate—some high-attention pairs may be missed if they hash to different buckets. For [[LensNet]], this could cause focus misallocation, collapsing important regions or expanding irrelevant ones.
- **Hash collision sensitivity**: If hash functions poorly separate distinct content, LSH degenerates to random pairing. Requires careful tuning of hash dimension, number of rounds, and bucket sizes for MegaContext's embedding space.
- **Not compatible with RoPE**: Standard LSH works on raw embeddings, but MegaContext uses [[RoPE]] for positional encoding. Hashing RoPE-modified queries/keys would break position-invariance—need to hash base embeddings separately, then apply RoPE after bucketing.
- **Reversibility restricts architecture**: RevNets require specific residual structure (two streams) and rule out certain activation functions (e.g., in-place operations). May constrain [[GistNet]] design choices.
- **Chunking increases overhead**: Chunked FFN reduces memory but increases computation time due to repeated kernel launches and less efficient batching. For real-time [[Runtime Loop]], latency may increase.
- **Poor performance on short contexts**: LSH overhead (hashing, sorting, bucketing) dominates when n is small. Not useful for [[Working Context]] (W_max=8k), only for large [[Staging Context]] or disk-backed [[MegaContext Tree]] scans.

## Potential Follow-Up Reading
- **Performer** (2006.10902) - FAVOR+ kernel approximation for linearized attention; O(n) complexity with no hashing
- **Linformer** (2006.04768) - Low-rank projection of attention; complementary to LSH for memory reduction
- **Routing Transformers** (2003.05997) - k-means clustering for attention; learned alternative to fixed LSH
- **Combiner** (2103.14031) - Hierarchical memory with local + global attention; similar to MegaContext's LOD approach
- **Memory Efficient Attention (xformers)** - Practical implementation reference for chunked attention in PyTorch

## Open Questions for MegaContext
- Can we combine LSH with our gist hierarchy—using coarse LOD2 hashes to prune search space before computing fine-grained LOD0 attention?
- How to adapt LSH for temporal relevance? Standard LSH ignores position; we want recent tokens to be "closer" regardless of content similarity.
- Should [[LensNet Training]] include hash-based negative sampling—train on mismatched hash buckets as hard negatives to improve focus discrimination?
- What's the right granularity for reversible layers? Full reversible [[GistNet]] vs. selective reversibility only in deepest layers?
- Can we use LSH hash codes as additional features for [[Focus Allocator]]—e.g., prioritize expanding gists with high hash similarity to current query?
- Is LSH applicable to [[MegaContext Tree]] persistence? Hash-based indexing for faster retrieval of historical gists from disk?

## Related Pages
- [[LensNet]]
- [[LensNet Scoring]]
- [[LensNet Training]]
- [[GistNet]]
- [[GistNet Training]]
- [[GistNet Architecture Details]]
- [[Node Metadata]]
- [[Working Context Assembly]]
- [[Staging Context]]
- [[MegaContext Tree]]
- [[Storage Format]]
- [[Performance Sketch]]
- [[Focus Allocator]]
