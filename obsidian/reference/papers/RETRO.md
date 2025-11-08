---
tags:
  - papers
  - reference
  - retrieval
  - architecture
summary: Retrieval-Enhanced Transformer that uses nearest-neighbor lookup to augment context from large external databases via chunked cross-attention
---

# RETRO (Retrieval-Enhanced Transformer)

**Paper**: "Improving language models by retrieving from trillions of tokens" (DeepMind, 2021)
**ArXiv**: 2112.04426
**Illustrated Guide**: https://jalammar.github.io/illustrated-retrieval-transformer/

## Overview

RETRO is a retrieval-enhanced transformer architecture from DeepMind that combines traditional language modeling with a retrieval mechanism over a large external database. The key innovation is **chunked cross-attention (CCA)**, which allows the model to condition on retrieved text passages during generation, enabling it to leverage trillions of tokens at inference time without increasing model parameters.

## Core Architecture Design

### Hybrid Encoder-Decoder Structure

RETRO uses an encoder-decoder-like setup where:

1. **Retrieval Database**: A massive database of text chunks (2 trillion tokens)
2. **Chunk Encoder**: Processes retrieved neighbor documents
3. **Main Decoder**: The primary language model with interleaved self-attention and chunked cross-attention

### Key Components

```
Input Text → [Chunking] → [Retrieval] → Retrieved Neighbors
                ↓                              ↓
         Self-Attention              Chunk Encoder (CA-Enc)
                ↓                              ↓
         [Chunked Cross-Attention (CCA)] ← Encoded Neighbors
                ↓
            Output
```

### How Retrieval Integrates with Transformer Layers

**RETRO interleaves two types of attention:**

1. **Self-Attention** (standard transformer): Operates at the document level
2. **Chunked Cross-Attention**: Operates at a finer passage/chunk level with retrieved neighbors

This interleaving pattern is the architectural innovation:
- Not every layer has CCA - they're inserted at specific intervals
- CCA layers allow the model to attend to retrieved context
- Regular self-attention layers process local context

## Database Structure and Retrieval Mechanism

### Database Construction

1. **Chunking Strategy**:
   - Split documents into fixed-size chunks (e.g., 64 tokens)
   - Each chunk becomes a retrievable unit
   - Chunks are embedded using frozen BERT embeddings

2. **Indexing**:
   - Create dense vector representations for all chunks
   - Store in a nearest-neighbor index (e.g., ScaNN, FAISS)
   - Database size: 2 trillion tokens from web-scale corpora

3. **Storage**:
   - Both the embeddings AND the raw text chunks are stored
   - Embeddings used for retrieval
   - Raw text fed to the encoder

### Retrieval Process

**At Training/Inference Time:**

1. **Query Formation**:
   - Current chunk being processed becomes the query
   - Use its frozen embedding for retrieval

2. **Nearest Neighbor Search**:
   - Retrieve k nearest neighbors (typically k=2)
   - Uses approximate nearest neighbor search for efficiency
   - Retrieval is deterministic given the same query

3. **Neighbor Processing**:
   - Retrieved chunks are fed to the Chunk Encoder (CA-Enc)
   - CA-Enc is a smaller transformer that encodes neighbors
   - These encodings become keys/values for cross-attention

**Important**: Retrieval happens at the chunk level, not token level. Each chunk retrieves its own neighbors.

## Chunked Cross-Attention (CCA) Mechanism

### What Makes It "Chunked"

Traditional cross-attention: every token attends to every retrieved token.

**CCA**: Attention is localized by chunk boundaries:
- Input is split into chunks (e.g., 64 tokens each)
- Each chunk only attends to its own retrieved neighbors
- Neighboring chunks retrieve different documents
- This creates a "chunked" attention pattern

### CCA Architecture Details

```python
# Conceptual flow
for chunk in input_chunks:
    # 1. Retrieve neighbors for this chunk
    neighbors = retrieve(chunk)

    # 2. Encode neighbors
    encoded_neighbors = chunk_encoder(neighbors)

    # 3. Cross-attend
    # Query: current chunk hidden states
    # Key/Value: encoded neighbor representations
    output = cross_attention(
        query=chunk_hidden_states,
        key=encoded_neighbors,
        value=encoded_neighbors
    )
```

### CCA Layer Placement

- **Not every layer has CCA** - too expensive
- RETRO typically inserts CCA layers at intervals (e.g., every 3rd layer)
- First few layers: standard self-attention only
- Middle/later layers: interleaved self-attention and CCA
- This balances local context (self-attention) and retrieved context (CCA)

### Chunk Encoder (CA-Enc)

- Separate smaller transformer that encodes retrieved neighbors
- Takes retrieved text chunks as input
- Produces key/value representations for CCA
- Frozen or trained with specific objectives
- Much smaller than main model (parameter efficiency)

## Training Approach

### Pre-training

**Objective**: Standard causal language modeling with retrieval augmentation

1. **Data Preparation**:
   - Build retrieval database from training corpus
   - Index all chunks with embeddings

2. **Training Loop**:
   ```
   For each training example:
     - Split into chunks
     - For each chunk:
       * Retrieve k nearest neighbors from database
       * Encode neighbors with CA-Enc
       * Process through RETRO layers (self-attention + CCA)
       * Predict next tokens
     - Compute loss (cross-entropy)
     - Backpropagate through main model + CA-Enc
   ```

3. **Key Training Details**:
   - Retrieval database is frozen (embeddings don't change during training)
   - Main decoder and CA-Enc are trained end-to-end
   - Can retrieve from training data (with proper safeguards to avoid train/test leakage)

### Training Efficiency Considerations

- Retrieval is the bottleneck - done once per chunk
- Caching retrieved neighbors speeds up training
- Approximate nearest neighbor search trades accuracy for speed
- Can pre-compute and cache retrievals for static training data

### Fine-tuning

RETRO can be fine-tuned like standard LMs:
- Keep retrieval database fixed or update it
- Fine-tune on task-specific data
- CCA layers provide pathway for task-relevant retrieval

## Results and Performance

### Key Findings (from paper)

1. **Parameter Efficiency**:
   - RETRO models achieve comparable performance to much larger standard transformers
   - 7.5B RETRO ~ 25B GPT-3 on many benchmarks
   - Effective model compression through retrieval

2. **Scaling Behavior**:
   - Performance improves with:
     * Larger retrieval databases
     * More retrieved neighbors (up to a point)
     * Better quality retrieval (embedding models)

3. **Factual Knowledge**:
   - Significantly better at fact-based tasks
   - Can update knowledge by updating database (no retraining)
   - More interpretable - can inspect what was retrieved

4. **Long-Range Dependencies**:
   - CCA helps with longer-context understanding
   - Retrieved chunks provide relevant context beyond window size
   - Better continuations for factual text

### Performance Metrics

- **Perplexity**: Lower than comparable-sized baselines
- **Factual QA**: Strong improvements (retrieving relevant facts)
- **Generation Quality**: More accurate and factual continuations
- **Zero-shot Transfer**: Competitive with larger models

## Limitations and Computational Costs

### Computational Costs

1. **Training**:
   - Retrieval adds significant overhead
   - Need to maintain and query large index
   - Memory: store embeddings + raw text for 2T tokens
   - Retrieval latency multiplied by number of chunks

2. **Inference**:
   - Every chunk requires retrieval query
   - Approximate NN search still has cost
   - Chunk encoder adds parameters and compute
   - CCA layers add FLOPs vs. standard transformer

3. **Infrastructure**:
   - Requires serving infrastructure for retrieval
   - Database storage (TB-scale)
   - Need fast NN search system (ScaNN, FAISS)

### Limitations

1. **Retrieval Quality Dependency**:
   - Model performance tied to retrieval quality
   - Poor embeddings → poor retrievals → worse generation
   - Sensitive to database composition

2. **Chunk Boundary Effects**:
   - Fixed chunking may split important context
   - Retrieval granularity is coarse (chunks, not tokens)
   - Boundary artifacts possible

3. **Static Database**:
   - Database is frozen during training/inference
   - Can't adapt retrieval based on learned preferences
   - Knowledge cutoff tied to database construction

4. **Complexity**:
   - More complex than standard LMs
   - Harder to debug and interpret
   - Infrastructure dependencies

5. **Retrieval Overhead**:
   - Latency increase from retrieval
   - Not suitable for extremely low-latency applications
   - Batch processing may be inefficient

6. **Limited Interaction**:
   - CCA is one-way: decoder attends to retrieved
   - No feedback from decoder to improve retrieval
   - Retrieval is deterministic, not learned

## Related Work Referenced

RETRO builds on and references several research areas:

### Retrieval-Augmented Methods
- **REALM** (Retrieval-Augmented Language Model): Earlier retrieval + LM work
- **RAG** (Retrieval-Augmented Generation): Seq2seq with retrieval
- **DPR** (Dense Passage Retrieval): Dense retrieval methods
- **kNN-LM**: Nearest neighbor lookup for language modeling

### Efficient Transformers
- **Transformer-XL**: Recurrent memory for long context
- **Compressive Transformer**: Compressed memory for long sequences
- **Routing Transformer**: Learned sparse attention patterns

### Memory-Augmented Models
- **Neural Turing Machines**: External memory with read/write
- **Differentiable Neural Computer**: Structured memory access
- **Memory Networks**: Explicit memory for reasoning

### Context Compression
- Ideas related to compressing or summarizing context
- Soft prompt tuning and prefix methods

## Comparison with MegaContext

### Similarities

1. **External Memory**: Both use external storage beyond model parameters
2. **Selective Attention**: Both selectively attend to relevant information
3. **Hierarchical Processing**: Both have notion of chunks/segments and finer-grained tokens
4. **Parameter Efficiency**: Both achieve more with fewer parameters through external knowledge

### Key Differences

| Aspect | RETRO | MegaContext |
|--------|-------|-------------|
| **Structure** | Flat database of chunks | Hierarchical tree with multiple abstraction levels |
| **Retrieval** | kNN similarity search | Tree traversal with learned gist representations |
| **Granularity** | Fixed chunk size (e.g., 64 tokens) | Multi-scale (leaf→branch→root) |
| **Integration** | Chunked cross-attention at intervals | Focus allocator with dynamic budget distribution |
| **Database** | Static, externally indexed | Dynamic tree that grows with context |
| **Abstraction** | None - retrieves raw text | Learned gists at multiple levels |
| **Training** | Pre-training on massive corpus | Online learning from task data |
| **Retrieval Mechanism** | Frozen embeddings + kNN | Learned LensNet scoring + tree traversal |
| **Context Updates** | Requires rebuilding index | Incremental tree updates |
| **Attention Pattern** | Chunk-local cross-attention | Hierarchical with multi-level focus |

### MegaContext Advantages Over RETRO

1. **Hierarchical Abstraction**:
   - MegaContext learns multi-level gists, not just raw retrieval
   - Tree structure provides natural coarse-to-fine reasoning
   - Can efficiently represent information at multiple scales

2. **Dynamic Context**:
   - Tree grows organically with conversation/task
   - No need to rebuild massive external database
   - Adaptive to specific task/user

3. **Learned Retrieval**:
   - LensNet learns what to retrieve for task
   - Not dependent on frozen embeddings
   - Can adapt retrieval strategy through training

4. **Efficient Memory**:
   - Don't need to store/index trillions of tokens
   - Tree pruning and compression
   - Compact gist representations

5. **Fine-Grained Control**:
   - Focus allocator provides explicit budget management
   - Can trade off breadth vs. depth dynamically
   - More interpretable attention allocation

### RETRO Advantages Over MegaContext (Initial Design)

1. **Proven at Scale**:
   - Demonstrated on 2T token databases
   - Production-ready infrastructure (ScaNN, etc.)
   - Known training recipes

2. **Simpler Architecture**:
   - Fewer novel components
   - Standard transformers + cross-attention
   - Easier to implement and debug

3. **Zero-shot Retrieval**:
   - Can leverage any corpus without task-specific training
   - Frozen embeddings work out-of-the-box
   - No need to build task-specific trees

4. **Broad Knowledge**:
   - Access to web-scale knowledge
   - Good for factual tasks
   - Can retrieve diverse information

## Relevance to MegaContext

### What MegaContext Can Learn from RETRO

1. **Chunked Cross-Attention Pattern**:
   - Efficient way to integrate retrieved/external context
   - Balance between local (self-attention) and global (cross-attention)
   - Sparse attention to retrieved content

2. **Encoder-Decoder Split**:
   - Separate encoding of external context (CA-Enc) from main model
   - Allows different architectures/sizes for different roles
   - GistNet could play similar role to CA-Enc

3. **Training Methodology**:
   - End-to-end training of retrieval + generation
   - Language modeling objective works with retrieval
   - Caching strategies for efficiency

4. **Chunk-Level Processing**:
   - Granular retrieval (chunks not full documents)
   - Each chunk can have different neighbors
   - Informs MegaContext's node granularity decisions

### How MegaContext Improves on RETRO

1. **Hierarchical vs. Flat**:
   - Tree structure provides better organization than flat database
   - Multi-level gists vs. raw text chunks
   - Supports reasoning at different abstraction levels

2. **Learned Compression**:
   - GistNet learns to compress, not just retrieve
   - Gists are task-aware and adaptive
   - More efficient than storing raw text

3. **Dynamic Allocation**:
   - Focus allocator vs. fixed CCA pattern
   - Budget-aware attention distribution
   - Can adapt to query complexity

4. **Incremental Updates**:
   - Tree grows incrementally vs. batch indexing
   - Supports streaming and online scenarios
   - No need to rebuild massive indexes

## Technical Takeaways for Implementation

### For MegaContext Architecture

1. **Cross-Attention Integration**:
   - Consider CCA-style chunked attention for tree nodes
   - Separate encoder (GistNet) for compressing nodes
   - Interleave self-attention (working context) with cross-attention (tree)

2. **Chunking Strategy**:
   - Fixed-size chunks work well for RETRO
   - MegaContext's variable-sized nodes are more flexible
   - But fixed sizes simplify implementation

3. **Retrieval Caching**:
   - Cache retrieved nodes/gists like RETRO caches neighbors
   - Recompute only when tree changes
   - Tradeoff memory for speed

4. **Training Objectives**:
   - Language modeling is sufficient (no complex retrieval training)
   - End-to-end training works
   - Start simple, add complexity as needed

### For GistNet Design

1. **Role Clarity**:
   - GistNet is like RETRO's CA-Enc but more sophisticated
   - Encode tree nodes into compact gist representations
   - Provide keys/values for cross-attention in working context

2. **Size Considerations**:
   - CA-Enc in RETRO is much smaller than main model
   - GistNet can be parameter-efficient
   - Focus compute on main task model

3. **Pretraining**:
   - RETRO freezes embeddings initially
   - Consider whether to pretrain GistNet separately
   - Then fine-tune end-to-end

## References and Further Reading

- **Paper**: https://arxiv.org/abs/2112.04426
- **Illustrated Guide**: https://jalammar.github.io/illustrated-retrieval-transformer/
- **Related**:
  - REALM (Retrieval-Augmented Language Models)
  - RAG (Retrieval-Augmented Generation)
  - kNN-LM (Nearest Neighbor Language Models)
  - Memorizing Transformers (similar ideas with different implementation)

## Related MegaContext Pages

- [[MegaContext & RAG]] - How MegaContext differs from retrieval approaches
- [[Comparisons]] - Detailed architecture comparisons
- [[Working Context Assembly]] - How retrieved information is integrated
- [[GistNet]] - MegaContext's analog to RETRO's CA-Enc
- [[Related Work]] - Broader context of related research
