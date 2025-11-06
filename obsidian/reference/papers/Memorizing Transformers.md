---
tags: [papers, reference]
summary: Augments transformers with k-NN lookup over external key-value memory to extend effective context beyond the attention window
---

# Memorizing Transformers (arXiv:2203.08913)

**PDF**: [Memorizing Transformers - 2203.08913.pdf](Memorizing%20Transformers%20-%202203.08913.pdf)

**Paper**: "Memorizing Transformers" (Wu et al., Google Research, 2022)
**ArXiv**: 2203.08913

## Overview

Memorizing Transformers augment standard transformer models with **k-nearest neighbor (k-NN) lookup over external key-value memory** to dramatically extend effective context length. The key innovation is integrating approximate nearest neighbor search directly into the attention mechanism, allowing the model to attend to relevant past context from arbitrarily large external memories (tested up to 262K tokens) without changing the model architecture or increasing sequence length.

Unlike [[RETRO]] which requires chunked cross-attention and retraining, Memorizing Transformers can be applied to **pretrained models** by simply adding k-NN augmented attention layers and fine-tuning only the new components.

**Key results**:
- Extends context from 8K to 262K tokens with minimal overhead
- Improves perplexity by 2-4× on long-document tasks
- Works with frozen base model (only fine-tune k-NN integration)

## Core Concepts

### k-NN Augmented Attention

The core idea is to **enhance standard attention** with retrieved memories from an external store:

**Standard Attention**:
```
Attention(Q, K, V) = softmax(Q K^T / √d) V
```

**k-NN Augmented Attention**:
```
1. Local attention: A_local = Attention(Q, K_local, V_local)
2. k-NN retrieval: retrieve k nearest keys from external memory
3. k-NN attention: A_knn = Attention(Q, K_retrieved, V_retrieved)
4. Merge: A_final = gate(A_local, A_knn)
```

This allows the model to attend to both:
- **Local context**: Recent tokens in the current sequence (standard attention)
- **Retrieved context**: Relevant past tokens from external memory (k-NN attention)

### External Key-Value Memory

The system maintains an **external memory store** containing:

**Structure**:
- **Keys**: Hidden states from intermediate layers (not input embeddings)
- **Values**: Corresponding hidden states (same as keys in their implementation)
- **Indexing**: FAISS approximate nearest neighbor index
- **Scope**: Can store millions of key-value pairs

**Population**:
- Memory is populated as the model processes text
- For long sequences: cache all past activations
- For multi-document: aggregate activations across documents
- Memory persists across forward passes (unlike standard attention)

**Key design choice**: Use **intermediate layer activations** (e.g., layer 6 of 12) as keys rather than input embeddings, as they're more semantically meaningful.

### k-NN Retrieval Process

For each query token:

1. **Query Formation**:
   - Use hidden state from current layer as query
   - Same representation space as stored keys

2. **Approximate NN Search**:
   - Query FAISS index with approximate search (IVF, HNSW)
   - Retrieve k nearest neighbors (typically k=32-256)
   - Use inner product or LOD2 distance

3. **Retrieved Context**:
   - Get corresponding value vectors for retrieved keys
   - Compute attention over retrieved values
   - Typically much smaller k than local context size

4. **Attention Computation**:
   ```python
   # Simplified conceptual code
   def knn_attention(query, memory_keys, memory_values, k=32):
       # 1. Retrieve k nearest neighbors
       distances, indices = faiss_index.search(query, k)

       # 2. Get corresponding keys and values
       retrieved_keys = memory_keys[indices]      # (batch, k, dim)
       retrieved_values = memory_values[indices]  # (batch, k, dim)

       # 3. Compute attention over retrieved context
       scores = (query @ retrieved_keys.T) / sqrt(d)  # (batch, k)
       attention_weights = softmax(scores)

       # 4. Weighted sum of retrieved values
       output = attention_weights @ retrieved_values  # (batch, dim)

       return output
   ```

### Integration with Transformer Layers

**Architecture Strategy**:

The paper explores where to add k-NN augmentation:

1. **Augment All Layers**: Add k-NN to every attention layer
   - Most powerful but most expensive

2. **Augment Subset of Layers**: Only add k-NN to specific layers (e.g., every 3rd layer)
   - Good balance of performance and cost
   - Similar to [[RETRO]]'s CCA placement

3. **Top Layers Only**: Add k-NN only to upper layers
   - Cheapest option
   - Upper layers are more semantic, benefit more from long-range context

**Integration Pattern**:

```python
class KnnAugmentedAttentionLayer(nn.Module):
    def __init__(self, base_attention, knn_memory):
        self.local_attn = base_attention  # Standard attention
        self.knn_attn = KnnAttention(knn_memory)
        self.gate = nn.Linear(hidden_dim, 1)  # Learned gating

    def forward(self, x):
        # Local attention (standard)
        local_output = self.local_attn(x)

        # k-NN attention (external memory)
        knn_output = self.knn_attn(x)

        # Learned gating to combine both
        gate_value = sigmoid(self.gate(x))
        output = gate_value * local_output + (1 - gate_value) * knn_output

        return output
```

### Memory Gating

A critical component is **learned gating** that decides how much to rely on:
- **Local attention**: Recent context, high resolution
- **k-NN attention**: Retrieved context, potentially more relevant but noisier

**Gating Mechanisms Explored**:

1. **Token-level gating**: Each token decides its own mixture
   ```python
   gate = sigmoid(W_gate @ query)
   output = gate * local + (1 - gate) * knn
   ```

2. **Layer-level gating**: Fixed mixture per layer
   ```python
   output = α * local + (1 - α) * knn  # α is learned parameter
   ```

3. **Attention-score gating**: Based on retrieval quality
   ```python
   # If k-NN retrieval is confident (high similarity), use it more
   gate = softmax([max(local_scores), max(knn_scores)])
   ```

The paper finds **token-level gating** works best, allowing dynamic adaptation based on query content.

## Training Approach

### Training Methodology

**Key Insight**: Can fine-tune pretrained models with k-NN augmentation **without retraining from scratch**.

**Training Pipeline**:

1. **Start with Pretrained Model**:
   - Use any pretrained transformer (GPT-2, GPT-3, etc.)
   - Freeze base model parameters initially

2. **Add k-NN Components**:
   - Insert k-NN attention modules at chosen layers
   - Initialize gating networks
   - Build initial memory index from training data

3. **Fine-tuning Strategy**:
   - **Option A (Cheap)**: Train only k-NN components + gates
     * Freeze all base model parameters
     * Only optimize new k-NN attention and gating layers
     * Fastest, but limited adaptation

   - **Option B (Full)**: Fine-tune entire model
     * Train base model + k-NN components end-to-end
     * Better performance but more expensive
     * Used for main paper results

4. **Training Objective**:
   - Standard causal language modeling loss
   - No auxiliary losses needed
   - Model learns to use k-NN retrieval through LM objective

### Memory Population During Training

**Challenge**: How to populate memory while training?

**Strategies**:

1. **Pre-populate from Past Data**:
   - Process training data in multiple passes
   - First pass: populate memory with all activations
   - Second pass: train with populated memory
   - Most accurate but requires multiple passes

2. **Online Population**:
   - Populate memory as training progresses
   - Earlier batches don't have full memory
   - Faster but less stable early in training

3. **Frozen Memory**:
   - Populate memory once from pretrained model
   - Keep memory fixed during fine-tuning
   - Only train gating and integration

The paper uses **strategy 1** for main results: pre-populate memory then train.

### Training Details

**Hyperparameters**:
- k (neighbors retrieved): 32-256, typically 128
- Memory size: 8K-262K tokens
- Learning rate: Lower than pretraining (1e-5 typical)
- Batch size: Similar to base model training

**Computational Considerations**:
- k-NN search adds latency (~10-20ms per query with FAISS)
- Can batch k-NN queries for efficiency
- GPU for model, CPU for k-NN search (or GPU with optimized kernels)

## Results and Performance

### Key Findings

1. **Context Extension**:
   - Extends effective context from 8K to 262K tokens
   - Linear scaling: longer memory = better performance
   - No plateau observed up to 262K tokens tested

2. **Perplexity Improvements**:
   - **PG-19 (books)**: 2-4× perplexity reduction on long sequences
   - **arXiv papers**: 3-5× improvement on technical documents
   - **C4 (web)**: 1.5-2× improvement on general text
   - Gains increase with document length

3. **Memory Size Scaling**:
   - 8K memory: modest gains over baseline
   - 32K memory: significant improvements
   - 65K memory: strong performance on most tasks
   - 262K memory: best performance, especially on book-length texts

4. **Retrieval Efficiency**:
   - k=128 neighbors sufficient for most tasks
   - k=32 works reasonably well (cheaper)
   - k=256 provides marginal additional gains

5. **Layer Placement**:
   - Augmenting all layers: best but expensive
   - Every 3rd layer: 80-90% of full performance, much cheaper
   - Top 1/3 of layers only: 60-70% of performance, cheapest

### Performance Metrics

**Perplexity (PG-19 books, 65K memory)**:
- Baseline (8K context): 18.2
- With k-NN (65K effective): 7.4
- **2.5× improvement**

**Long-range dependency tasks**:
- Document-level reasoning: +40% accuracy
- Coreference resolution across chapters: +35% accuracy
- Factual recall from early context: +50% accuracy

**Inference Speed**:
- Local attention only: 100ms per sequence
- With k-NN (k=128): 120ms per sequence
- **~20% overhead** with optimized FAISS

## Relevance to MegaContext

### Conceptual Similarities

Memorizing Transformers and MegaContext both:

1. **Extend effective context** beyond attention window
2. **Use external storage** for past information
3. **Selective retrieval** based on relevance
4. **Frozen base model** (can work with pretrained LMs)
5. **Hierarchical information access**: local (recent) vs. global (retrieved)

### Architectural Parallels

| Memorizing Transformers | MegaContext | Mapping |
|-------------------------|-------------|---------|
| External key-value memory | [[MegaContext Tree]] | Long-term storage |
| k-NN retrieval | [[LensNet]] scoring + tree traversal | Relevance-based selection |
| Local attention context | LOD0 tokens in [[Working Context]] | Recent high-res content |
| Retrieved k-NN context | LOD1/LOD2 gists in Working Context | Older compressed content |
| Gating mechanism | [[Focus Allocator]] budget management | Allocation of attention |
| FAISS indexing | Tree structure navigation | Memory organization |
| Intermediate layer keys | [[GistNet]] embeddings | Semantic representations |

### Key Differences

| Aspect | Memorizing Transformers | MegaContext |
|--------|------------------------|-------------|
| **Memory Structure** | Flat key-value store with indexing | Hierarchical tree with multiple LODs |
| **Retrieval** | k-NN similarity search (frozen) | Learned focus scoring (adaptive) |
| **Compression** | None (stores full activations) | Learned gist compression (32:1, 1024:1) |
| **Memory Size** | Limited by RAM/VRAM (100K-1M tokens) | Unbounded (disk-backed, billions possible) |
| **Granularity** | Token-level retrieval | Block-level with multi-scale gists |
| **Training** | Fine-tune pretrained models | Train GistNet/LensNet from scratch |
| **Integration** | Gating between local and k-NN | Budget-based focus allocation |
| **Computational Model** | k-NN search overhead | Tree traversal + gist generation |
| **Adaptivity** | Fixed similarity metric | Learned retrieval via LensNet |

## What We Can Use

### 1. k-NN as Complement to Learned Retrieval

**Idea**: Augment [[LensNet]] with k-NN similarity as an additional signal

**Implementation**:
```python
class HybridLensNet(LensNet):
    def __init__(self, base_lensnet, faiss_index):
        self.lensnet = base_lensnet
        self.faiss_index = faiss_index
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query, working_context):
        # Learned scoring (current approach)
        learned_scores = self.lensnet(query, working_context)

        # k-NN similarity (new signal)
        knn_scores = self.compute_knn_scores(query)

        # Fuse both signals
        combined = self.fusion_layer(concat([learned_scores, knn_scores]))
        return combined
```

**Benefits**:
- Learned LensNet provides task-adaptive scoring
- k-NN provides immediate similarity-based retrieval
- Hybrid approach: best of both worlds

### 2. FAISS for Efficient Tree Search

**Concept**: Use FAISS to accelerate tree node retrieval

**Current MegaContext approach**:
- Sequential tree traversal guided by LensNet
- Potentially slow for very large trees

**k-NN augmentation**:
- Build FAISS index over all tree nodes (LOD1/LOD2 gists)
- Use for fast candidate selection
- Refine candidates with LensNet for final selection

**Two-stage retrieval**:
```python
def retrieve_relevant_nodes(query, tree, lensnet, k_candidates=100):
    # Stage 1: Fast k-NN candidate selection
    candidate_nodes = faiss_index.search(query, k=k_candidates)

    # Stage 2: LensNet refinement for final selection
    scores = lensnet(query, candidate_nodes)
    top_nodes = select_top_k(candidate_nodes, scores, k=10)

    return top_nodes
```

### 3. Gating Mechanisms for Focus Allocation

**Concept**: Learned gating between local and retrieved context

**Application to MegaContext**:
- Current: [[Focus Allocator]] uses discrete expand/collapse decisions
- Alternative: Soft gating over mixed-LOD working context

**Soft focus allocation**:
```python
class SoftFocusAllocator:
    def compute_attention_weights(self, query, working_context):
        # Each entry gets a soft weight (not binary expand/collapse)
        weights = self.gate_network(query, working_context)

        # Normalize to W_max budget
        normalized_weights = weights / weights.sum() * W_max

        # Attend with weighted mixture
        output = weighted_attention(query, working_context, normalized_weights)
        return output
```

**Trade-offs**:
- More flexible than discrete allocation
- But harder to interpret and control
- Might combine both: discrete structure + soft weighting

### 4. Intermediate Layer Representations

**Finding**: Using intermediate layer activations as keys works better than input embeddings

**MegaContext application**:
- [[GistNet]] could operate on intermediate layers of base model
- Instead of compressing input embeddings, compress mid-layer states
- Richer semantic content for retrieval

**Modified GistNet architecture**:
```python
class MidLayerGistNet(GistNet):
    def forward(self, tokens, base_model):
        # Run base model to intermediate layer
        hidden_states = base_model.forward_to_layer(tokens, layer=6)

        # Compress intermediate representations (not input embeddings)
        gist = self.compress(hidden_states)

        return gist
```

**Benefits**:
- Gists capture higher-level semantics
- Better for retrieval and substitutability
- Aligns with how Memorizing Transformers use mid-layer keys

### 5. Memory Pre-population Strategy

**Concept**: Pre-populate memory before task execution

**Memorizing Transformers approach**:
- Process documents in advance to populate memory
- Then perform task with populated memory

**MegaContext adaptation**:
- Pre-build tree from known corpus (e.g., codebase, documentation)
- Load pre-built tree at runtime
- Supports "instant" long-context awareness

**Use cases**:
- Code completion: pre-index entire codebase
- Document QA: pre-index document collection
- Chat with persistent memory: load previous conversation tree

## Limitations & Risks

### From the Paper

1. **Memory Storage Costs**:
   - Must store all key-value pairs in memory
   - 262K tokens × hidden_dim × num_layers = substantial RAM/VRAM
   - Not scalable to billions of tokens without compression
   - **MegaContext advantage**: Gist compression reduces storage 1000×

2. **Retrieval Latency**:
   - k-NN search adds 10-20ms per query
   - Becomes bottleneck for large memories or real-time applications
   - **MegaContext consideration**: Tree traversal may be slower but more targeted

3. **Index Building Time**:
   - FAISS index construction is slow for large datasets
   - Must rebuild when memory changes
   - **MegaContext advantage**: Incremental tree updates

4. **No Hierarchical Structure**:
   - Flat memory = no multi-scale reasoning
   - All content at same level of detail
   - **MegaContext advantage**: Multi-level abstraction (LOD0/LOD1/LOD2)

5. **Fixed Similarity Metric**:
   - k-NN uses fixed similarity (inner product, LOD2)
   - Can't learn task-specific notion of relevance
   - **MegaContext advantage**: LensNet learns relevance

6. **Cold Start Problem**:
   - Needs pre-populated memory to be useful
   - Empty memory = no benefit
   - **MegaContext consideration**: Same issue applies

### Risks for MegaContext

1. **Computational Overhead**:
   - Adding FAISS indexing to MegaContext adds complexity
   - May not be worth it if tree traversal is sufficient
   - **Mitigation**: Use k-NN only for very large trees (>1M nodes)

2. **Index-Tree Synchronization**:
   - If using FAISS, must keep index in sync with tree structure
   - Tree updates require index updates
   - **Mitigation**: Rebuild index periodically, not on every update

3. **Gating Complexity**:
   - Soft gating is less interpretable than discrete focus allocation
   - Harder to debug and control
   - **Mitigation**: Start with discrete, add soft gating as optional extension

## Potential Follow-Up Reading

### Related k-NN and Memory Approaches

- **kNN-LM** (Khandelwal et al., 2020): Original k-NN language model, foundation for this work
- **SPALM** (Yogatama et al., 2021): Sparse memory for language models
- **Unlimiformer** (Bertsch et al., 2023): Retrieval augmentation for any transformer

### Memory-Augmented Architectures

- **Neural Turing Machines** (Graves et al., 2014): Differentiable memory access [[Neural Turing Machines]]
- **Differentiable Neural Computer** (Graves et al., 2016): Structured memory [[DNC]]
- **Memory Networks** (Weston et al., 2015): Explicit memory for QA

### Efficient Retrieval Systems

- **FAISS** (Johnson et al., 2019): Library used for k-NN search
- **ScaNN** (Google): Alternative approximate NN library
- **HNSW**: Hierarchical navigable small world graphs for NN search

### Alternative Long-Context Approaches

- **RETRO** (Borgeaud et al., 2021): Chunked cross-attention with retrieval [[RETRO]]
- **Compressive Transformer** (Rae et al., 2019): Compressed memory [[Compressive Transformer]]
- **Transformer-XL** (Dai et al., 2019): Segment-level recurrence [[Transformer-XL]]

## Open Questions for MegaContext

### 1. Hybrid Retrieval Architecture

Should MegaContext combine learned tree traversal with k-NN search?

**Approach A: Pure Tree Traversal** (current plan)
- LensNet guides navigation through tree
- No external indexing required
- Interpretable, controllable

**Approach B: k-NN Augmented Tree**
- FAISS index over all tree nodes
- Use k-NN for fast candidate selection
- LensNet for refinement
- Faster but more complex

**Exploration**:
- Implement both approaches in prototype
- Benchmark tree traversal speed for trees with 1K, 10K, 100K, 1M nodes
- If tree traversal is fast enough (< 50ms), stick with pure approach
- If too slow, add k-NN acceleration

### 2. Soft vs. Hard Focus Allocation

Should focus allocation be discrete (expand/collapse) or continuous (soft weights)?

**Current (Discrete)**:
- Binary decisions: expand or collapse
- Clear budget accounting
- Interpretable actions

**Alternative (Soft)**:
- Continuous attention weights over mixed LOD
- Smoother transitions
- Inspired by Memorizing Transformers' gating

**Hybrid Proposal**:
- Use discrete for tree structure management
- Add soft gating for attention computation within working context
- Best of both worlds

### 3. Layer-Specific k-NN Integration

Should k-NN retrieval be integrated at specific layers like Memorizing Transformers?

**Options**:
- **All layers**: Full integration but expensive
- **Middle layers**: Balance cost/performance (layers 4-8 of 12)
- **Top layers**: Cheapest, for semantic retrieval only

**MegaContext context**:
- Base model is frozen, so can't modify its layers
- But could add k-NN between GistNet and base model
- Or use k-NN within LensNet for focus scoring

### 4. Attention Statistics for Focus Scoring

Should MegaContext track which gists are actually attended to by the base model?

**Memorizing Transformers insight**: "Most-used" memories are most valuable

**Implementation**:
```python
class AttentionAwareLensNet:
    def __init__(self):
        self.attention_tracker = AttentionStatisticsTracker()

    def score_entries(self, working_context, base_model_output):
        # Extract which gists were attended to
        attention_stats = self.attention_tracker.extract(base_model_output)

        # Boost scores for gists that were actually used
        base_scores = self.compute_scores(working_context)
        attention_boosted = base_scores + α * attention_stats

        return attention_boosted
```

**Benefits**:
- Learns from actual model behavior
- Reinforces useful gists
- Self-improving focus allocation

### 5. Pre-built Tree Infrastructure

Should MegaContext support loading pre-built trees for common corpora?

**Use cases**:
- Documentation QA: pre-index entire docs
- Code completion: pre-index codebase
- Research assistant: pre-index paper collection

**Architecture**:
```python
class PrebuiltTreeLoader:
    def load_codebase_tree(self, repo_path):
        # Load pre-built tree for codebase
        tree = TreeStore.load(f"{repo_path}/.megacontext/tree.bin")
        return tree

    def build_and_cache_tree(self, documents):
        # Build tree from documents and cache
        tree = MegaContextTree()
        for doc in documents:
            tree.append(doc)
        tree.save_to_disk()
        return tree
```

**Trade-offs**:
- Fast cold start
- But stale if corpus changes
- Need versioning and update strategy

## Related Pages

- [[RETRO]] - Retrieval with chunked cross-attention
- [[Compressive Transformer]] - Learned compression of past activations
- [[GistNet]] - MegaContext's learned compression (vs. k-NN's no compression)
- [[LensNet]] - Learned focus scoring (vs. k-NN's similarity)
- [[Focus Allocator]] - Budget management (vs. k-NN's gating)
- [[Working Context]] - Mix of LOD0 and gists (vs. k-NN's local+retrieved)
- [[MegaContext Tree]] - Hierarchical structure (vs. k-NN's flat memory)
- [[Neural Turing Machines]] - Alternative memory-augmented architecture
- [[DNC]] - Structured memory operations
- [[Related Work]] - Broader context of long-context research
