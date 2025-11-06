---
tags: [papers, reference]
summary: Extends Transformer-XL with learned compression of past activations into compressed memory for extended context with bounded computation
---

# Compressive Transformer (arXiv:1911.05507)

**PDF**: [Compressive Transformer - 1911.05507.pdf](Compressive%20Transformer%20-%201911.05507.pdf)

**Paper**: "Compressive Transformers for Long-Range Sequence Modelling" (DeepMind, 2019)
**ArXiv**: 1911.05507

## Overview

The Compressive Transformer extends [[Transformer-XL]]'s segment-level recurrence with a lossy compression mechanism for past activations. Instead of discarding old memories when they exceed a fixed cache size, it **compresses them** into a smaller compressed memory using learned compression functions. This enables the model to retain information from much longer sequences (up to 3× context extension) with bounded memory and computation costs.

Key innovation: Add a second layer of memory management where oldest cached activations are compressed rather than discarded, creating a two-tier memory hierarchy (recent cache + compressed memory).

## Core Concepts

### Two-Tier Memory Hierarchy

Building on Transformer-XL's architecture, Compressive Transformer maintains:

1. **Regular Memory (M)**: Recent segment activations, identical to Transformer-XL
   - Size: n_m tokens worth of cached hidden states
   - Full-resolution: complete layer-wise activations
   - Attention: model attends to this with full precision

2. **Compressed Memory (CM)**: Older activations compressed at fixed ratio
   - Size: n_cm compressed representations
   - Compression ratio: typically c=3 (3 activations → 1 compressed token)
   - Created by compressing oldest activations from regular memory
   - Attention: model attends to compressed representations

```
[Current Segment] → [Regular Memory M] → [Compressed Memory CM]
                         (recent)              (older, compressed)
```

### Memory Management Flow

As new segments are processed:

1. **New segment** arrives (length n_s tokens)
2. **Shift regular memory**:
   - Oldest n_s activations in M are **compressed** into CM
   - Recent n_s activations from current segment enter M
3. **Shift compressed memory**:
   - Oldest n_s/c compressed memories are discarded
   - Newly compressed activations are appended
4. **Attention**:
   - Current segment attends to both M and CM
   - Total attention span: n_s + n_m + (n_cm × c)

This creates a sliding window with graceful forgetting rather than abrupt truncation.

### Compression Functions

The paper explores multiple compression mechanisms:

#### 1. Max/Mean Pooling (Simple)
- **Operation**: Fixed pooling over groups of c consecutive activations
- **Max pooling**: `CM[i] = max(M[c*i : c*i+c], dim=0)`
- **Mean pooling**: `CM[i] = mean(M[c*i : c*i+c], dim=0)`
- **Pros**: Simple, no parameters, deterministic
- **Cons**: Lossy, no learned adaptation

#### 2. 1D Convolution
- **Operation**: Learnable convolution with stride c
- **Formula**: `CM = Conv1D(M, kernel_size=c, stride=c)`
- **Pros**: Learned compression, relatively simple
- **Cons**: Limited receptive field per compressed token

#### 3. Dilated Convolution
- **Operation**: Convolution with dilation to increase receptive field
- **Purpose**: Each compressed token can aggregate information from larger spans
- **Pros**: Better long-range aggregation
- **Cons**: More parameters, computational cost

#### 4. Most Used Memories (Best Performing)
- **Operation**: Select activations based on attention patterns from previous segment
- **Formula**: Track attention weights, compress the most-attended-to positions
- **Intuition**: Keep what was actually useful
- **Pros**: Content-aware, task-adaptive
- **Cons**: Requires tracking attention stats, more complex

The paper finds **most-used** and **1D convolution** perform best in practice.

### Attention Mechanism

The attention remains similar to Transformer-XL but spans two memory regions:

```python
# Conceptual attention
def compressed_transformer_attention(query, M, CM):
    # Query: current segment hidden states
    # M: regular memory (uncompressed recent activations)
    # CM: compressed memory (older compressed activations)

    # Concatenate memories
    keys = concat([CM, M, query])
    values = concat([CM, M, query])

    # Standard attention with relative positional encoding
    attention_output = scaled_dot_product_attention(
        query=query,
        key=keys,
        value=values,
        relative_positions=compute_relative_positions(keys)
    )

    return attention_output
```

**Key property**: Attention is computed over *both* compressed and uncompressed memories, allowing the model to use long-range context while keeping recent context at full resolution.

### Positional Encoding

Uses Transformer-XL's **relative positional encoding**:
- Positions are relative to current token
- Works seamlessly with compressed memories
- Compressed tokens maintain their temporal position
- No special handling needed for compression boundaries

## Training Approach

### Architecture Modifications

Starting from Transformer-XL:

1. **Add compression module**:
   - Convolution layers or attention-tracking mechanism
   - Applied at compression boundaries
   - Trained end-to-end with main model

2. **Extend memory management**:
   - Implement two-tier cache system
   - Add compression logic when shifting memories
   - Maintain compressed memory alongside regular memory

3. **Modify attention masks**:
   - Ensure causal masking spans both memory types
   - Compressed memories are visible to all subsequent positions

### Training Objectives

- **Primary**: Standard causal language modeling (cross-entropy loss)
- **No auxiliary losses**: Compression is learned implicitly through LM objective
- **End-to-end**: Compression functions trained jointly with transformer layers

### Training Details

**Setup**:
- Segment length: 512 tokens typical
- Regular memory: 512-1024 cached activations
- Compressed memory: 512 compressed tokens (1536-3072 effective range with c=3)
- Compression ratio: c = 3 most common
- Batch processing: process multiple segments per sequence

**Gradient Flow**:
- Backprop through current segment
- Gradients flow through regular memory (like Transformer-XL)
- Gradients also flow through compression functions
- No gradient through compressed memories (treated as constants)

This is similar to Transformer-XL's approach but with additional compression learning.

### Loss Masking

Important detail: The paper explores whether to compute loss over:
1. **All tokens** (including those using compressed memory)
2. **Only recent tokens** (where regular memory provides full context)

They find computing loss over all tokens works better, as it provides learning signal for the compression mechanism.

## Results and Performance

### Key Findings (from paper)

1. **Context Extension**:
   - Effective context 3× longer than Transformer-XL with same memory budget
   - Example: 1536 token effective context vs. 512 with regular Transformer-XL
   - Memory bounded to same size as baseline

2. **Perplexity Improvements**:
   - Consistent improvements on PG-19, enwik8, text8 benchmarks
   - Larger gains on tasks requiring longer dependencies
   - Best with "most-used" compression method

3. **Compression Method Comparison**:
   - **Most-used**: Best performance, adaptive to content
   - **1D Convolution**: Strong performance, simple to implement
   - **Max/Mean pooling**: Worse than learned methods but still useful
   - **Dilated convolution**: Good but more compute

4. **Scaling Behavior**:
   - Performance improves with larger compressed memory
   - Diminishing returns beyond certain sizes
   - Compression ratio c=3 is a good balance

5. **Task Performance**:
   - **Long-document modeling**: Significant gains
   - **Short-range tasks**: Similar to Transformer-XL
   - **Factual recall**: Better retention of distant facts

### Performance Metrics

- **PG-19 (books)**: ~5-7% perplexity reduction over Transformer-XL
- **enwik8 (Wikipedia)**: ~2-3% improvement
- **Long-range dependency probing**: Substantially better than baselines

## Relevance to MegaContext

### Conceptual Alignment

Compressive Transformer addresses a core challenge MegaContext tackles: **how to maintain long context with bounded resources**. Both systems:

1. **Hierarchical memory**: Multiple levels of detail (recent vs. compressed)
2. **Learned compression**: Neural compression of past information
3. **Graceful degradation**: Older context becomes coarser, not discarded
4. **Attention over hierarchy**: Model attends to mixed-resolution representations

### Direct Parallels

| Compressive Transformer | MegaContext | Mapping |
|-------------------------|-------------|---------|
| Regular Memory (M) | LOD0 (raw tokens) in Working Context | Recent, full-resolution |
| Compressed Memory (CM) | LOD1/LOD2 gists in Working Context | Older, compressed |
| Compression functions | [[GistNet]] | Learned compression |
| Two-tier hierarchy | Multi-level tree (LOD0→LOD1→LOD2) | Hierarchical abstraction |
| Bounded memory | [[W_max]] budget | Resource constraint |
| Most-used compression | [[LensNet]] focus scoring | Content-aware selection |

### Key Differences

| Aspect | Compressive Transformer | MegaContext |
|--------|------------------------|-------------|
| **Structure** | Linear sequence (recent → compressed) | Tree with hierarchical branching |
| **Granularity** | Fixed compression ratio (c=3) | Multi-level (32×, 1024×) |
| **Selection** | Temporal recency + attention stats | Learned focus scoring + budget allocation |
| **Compression** | Fixed-ratio pooling/conv | Learned gists via GistNet |
| **Scope** | Single sequence with sliding window | Multi-turn conversations, global tree |
| **Decompression** | Not possible (lossy compression) | Possible (gists are pointers to source) |
| **Training** | Implicit via LM loss | Explicit ΔNLL@H optimization |

## What We Can Use

### 1. Two-Tier Memory Architecture

**Concept**: Maintain recent full-resolution + older compressed memory
- MegaContext already does this but more explicitly
- Validates our approach of mixing LOD0 and LOD1/LOD2 in working context
- Suggests 3× context extension is achievable minimum

**Implementation**:
- Working context naturally divides into LOD0 (recent) and LOD1/LOD2 (compressed)
- Consider whether recent context should *always* be LOD0
- Or allow [[LensNet]] to decide dynamically (more flexible)

### 2. Compression Trigger Logic

**Concept**: Compress oldest activations when memory limit reached
- Similar to MegaContext's budget-based collapse

**Application**:
- [[Focus Allocator]] already implements this
- When W_max is exceeded, collapse oldest/lowest-utility entries
- Compressive Transformer validates temporal-priority approach

### 3. Most-Used Memory Selection

**Concept**: Track attention patterns to decide what to compress
- Brilliant idea: let the model's own attention reveal importance

**MegaContext adaptation**:
- [[LensNet]] learns this more explicitly via focus scores
- Could augment with attention statistics from base model
- Track which gists are attended to, boost their retention scores
- Use as auxiliary signal for counterfactual labeling

**Implementation idea**:
```python
# After base model forward pass
attention_weights = extract_attention_weights(model_output)
gist_attention_scores = compute_attention_to_gists(attention_weights)

# Feed to LensNet as additional features
lensnet_input = concat([
    working_context_embeddings,
    gist_attention_scores,  # NEW: actual usage stats
    position_features
])
```

### 4. Compression Ratio Insights

**Finding**: c=3 works well (3:1 compression)
- MegaContext's 32:1 is much more aggressive
- But we have multiple levels: LOD1 is 32:1, but LOD2 is 1024:1

**Implications**:
- Consider intermediate compression ratios (e.g., LOD0.5 with 8:1 or 16:1)
- Or make compression ratio adaptive based on content importance
- High-utility spans could use less aggressive compression

### 5. Training Without Auxiliary Losses

**Finding**: Standard LM loss sufficient for learning compression
- No need for complex reconstruction losses
- Compression quality emerges from prediction task

**MegaContext validation**:
- Our ΔNLL@H objective is similar in spirit
- Validates that prediction quality is the right metric
- Suggests we don't need complex regularizers for [[GistNet]]

### 6. Gradient Flow Through Compression

**Approach**: Backprop through compression functions, not through compressed memories
- Learns how to compress, not what was compressed

**Application to GistNet**:
- Train GistNet with gradients from prediction loss
- Compressed gists themselves are not backprop targets
- Focus on learning good compression mappings

## Limitations & Risks

### From the Paper

1. **Fixed Compression Schedule**:
   - Oldest memories always compressed, regardless of importance
   - No content-aware retention (except "most-used" variant)
   - MegaContext's [[LensNet]] addresses this

2. **Sequential Structure**:
   - Strictly time-ordered memory
   - Can't selectively decompress or reorganize
   - MegaContext's tree structure is more flexible

3. **Limited Abstraction**:
   - Compression is mostly summarization, not abstraction
   - No multi-level hierarchy (only 2 tiers)
   - MegaContext's multi-level tree provides more structure

4. **Compute Overhead**:
   - Still processes full sequences incrementally
   - Compression adds computational cost
   - MegaContext's budget system provides more explicit control

### Risks for MegaContext

1. **Compression Artifacts**:
   - Lossy compression can lose important details
   - Paper shows graceful degradation but quality still drops
   - **Mitigation**: Use ΔNLL@H to ensure substitutability

2. **Optimal Compression Ratio**:
   - c=3 works for them, but MegaContext uses 32:1
   - Much more aggressive = higher risk of information loss
   - **Mitigation**: Hierarchical structure and learned focus allocation

3. **Training Complexity**:
   - Learning compression functions adds training complexity
   - Needs careful initialization and learning rates
   - **Mitigation**: Separate GistNet training pipeline

## Potential Follow-Up Reading

### Papers Building on Compressive Transformer

- **Compressive Transformer Extensions**: Variants with different compression schemes
- **Memory-Compressed Attention**: Alternative memory compression approaches

### Related Compression Approaches

- **∞-former**: Infinite context with bounded memory using similar ideas
- **RMT (Recurrent Memory Transformer)**: Memory tokens that pass between segments
- **Memorizing Transformers**: k-NN over compressed memories (see [[Memorizing Transformers]])

### Hierarchical Memory Systems

- **Neural Turing Machines**: More complex memory addressing (see [[Neural Turing Machines]])
- **Differentiable Neural Computer**: Structured memory operations (see [[DNC]])
- **Memory Networks**: Explicit memory for reasoning

### Context Extension Methods

- **Transformer-XL**: Foundation for Compressive Transformer (see [[Transformer-XL]])
- **Longformer, BigBird**: Sparse attention patterns for long context
- **Reformer**: LSH attention for efficiency (see [[Reformer]])

## Open Questions for MegaContext

### 1. Hybrid Compression Strategies

Should MegaContext blend time-based and importance-based compression?
- **Temporal**: Always keep very recent context at LOD0 (like Compressive Transformer)
- **Importance**: Use [[LensNet]] for older context (MegaContext approach)
- **Hybrid**: Recent = temporal, older = importance-based

**Exploration**:
- Experiment with "keep last N tokens at LOD0" constraint
- Compare vs. fully dynamic focus allocation
- Measure impact on prediction quality

### 2. Attention-Guided Focus Scoring

Should we incorporate base model attention statistics into [[LensNet]]?
- Extract attention weights from frozen base model
- Feed as features to LensNet
- Combine learned scoring with actual usage patterns

**Prototype**:
```python
class AttentionAugmentedLensNet(LensNet):
    def forward(self, working_context, attention_stats):
        # attention_stats: which gists were actually attended to
        base_scores = super().forward(working_context)
        attention_boost = self.attention_mlp(attention_stats)
        return base_scores + attention_boost
```

### 3. Adaptive Compression Ratios

Should different content compress at different ratios?
- High-entropy spans: lower compression (preserve detail)
- Repetitive spans: higher compression (aggressive)
- Currently: uniform 32:1 for all LOD1 gists

**Implementation path**:
- Train GistNet variants with different compression ratios (8:1, 16:1, 32:1, 64:1)
- [[LensNet]] selects which compressor to use per block
- More complex but potentially more efficient

### 4. Intermediate Compression Levels

Should we add LOD0.5 between LOD0 and LOD1?
- LOD0.5: 8:1 compression (4 LOD0 tokens → 1 LOD0.5 gist)
- LOD1: 32:1 compression (remains unchanged)
- Gentler transition, less aggressive initial compression

**Trade-offs**:
- More flexible granularity
- But more complexity, more budget management
- May not provide enough benefit vs. cost

### 5. Compression Training Curriculum

Should GistNet training follow a curriculum?
- **Stage 1**: Easy compression (c=3, like Compressive Transformer)
- **Stage 2**: Medium compression (c=8)
- **Stage 3**: Target compression (c=32)
- Gradual increase in difficulty

**Hypothesis**: Might learn better compression functions than direct 32:1 training

## Related Pages

- [[Transformer-XL]] - Foundation architecture for Compressive Transformer
- [[GistNet]] - MegaContext's learned compression module
- [[LensNet]] - Focus scoring (analogous to "most-used" compression)
- [[Focus Allocator]] - Budget-aware memory management
- [[Working Context]] - Mix of LOD0 and compressed gists
- [[RETRO]] - Alternative retrieval-based approach
- [[Memorizing Transformers]] - k-NN over external memory
- [[Related Work]] - Broader context of long-context methods
