---
tags:
  - architecture
summary: Core system properties that distinguish MegaContext - constant compute, constant memory, dynamic focus, reversibility, and learned optimization.
---
# MegaContext System Properties

This document defines the fundamental properties that characterize MegaContext as a system and distinguish it from alternative approaches.

---

## 1. Constant Compute Property

**Definition:** Per-step computational complexity remains constant regardless of total context size.

### Mathematical Expression

For a [[MegaContext Tree]] containing N total tokens:

```
Compute_per_step = O(W_max²)  where W_max is fixed

Not dependent on N
```

**Standard LLM:**
```
Compute_per_step = O(N²)  (quadratic attention)
```

### Breakdown

For [[POC Implementation|POC]] with W_max = 8,192 tokens:

| Operation | Time | Frequency | Amortized Cost |
|-----------|------|-----------|----------------|
| **Base model forward** | ~15 ms | Every token | 15 ms/token |
| **[[GistNet]] compression** | ~0.3 ms | Every K=32 tokens | ~0.01 ms/token |
| **[[LensNet]] scoring** | ~2.5 ms | Every K=32 tokens | ~0.08 ms/token |
| **[[Focus Allocator]]** | ~0.1 ms | Every K=32 tokens | ~0.003 ms/token |
| **Total** | | | **~15.1 ms/token** |

**Overhead:** ~0.7% regardless of whether the [[MegaContext Tree]] contains 10k or 10M tokens.

### Why It Matters

**Predictable latency:**
```
1M token context:  ~15.1 ms/token
10M token context: ~15.1 ms/token  (same!)
1B token context:  ~15.1 ms/token  (same!)
```

**Contrast with alternatives:**
- Standard LLM with 100k context: ~100× slower than 10k
- Sparse attention: Grows linearly O(N)
- RAG: Constant model time, but adds 50-200ms retrieval

---

## 2. Constant Memory Property

**Definition:** GPU memory usage remains constant regardless of total context size.

### Mathematical Expression

```
GPU_memory = O(W_max)  (constant)
Disk_storage = O(N)  (linear, but off-GPU)
```

### Breakdown

For [[POC Implementation|POC]]:

**GPU Memory (constant):**
```
Working Context:     512 MB  (8k tokens × 2048 dim × fp16)
KV-cache:           512 MB  (depends on model, not context size)
Model weights:      6 GB    (frozen, loaded once)
GistNet params:     2 MB    (tiny auxiliary network)
LensNet params:     2 MB    (tiny auxiliary network)
Total:              ~7 GB   (constant regardless of N)
```

**Disk/RAM Storage (linear):**
```
MegaContext Tree:   136 MB per 1M tokens
                    1.36 GB per 10M tokens
                    13.6 GB per 100M tokens
```

### Why It Matters

**Scalability:**
- 1M token context: 7 GB GPU ✓
- 10M token context: 7 GB GPU ✓ (same!)
- 100M token context: 7 GB GPU ✓ (same!)

**Standard LLM:**
- 32k context: ~2 GB KV-cache
- 100k context: ~6 GB KV-cache
- 1M context: ~60 GB KV-cache (impossible on single GPU)

**Economics:**
- MegaContext: Constant GPU cost + cheap disk storage
- Standard LLM: Exponentially expensive GPU memory

---

## 3. Dynamic Focus Property

**Definition:** The system continuously and automatically adjusts level of detail based on learned relevance predictions.

### Components

**[[LensNet]]:** Predicts which regions need detail
**[[Focus Allocator]]:** Applies expand/collapse operations
**Continuous:** Refocuses every K tokens (not query-time only)

### How It Works

```
Every K=32 tokens:
  1. LensNet scores all Working Context entries
       entry[i] → score ∈ [-1, +1]
       positive = needs detail (expand)
       negative = can be compressed (collapse)

  2. Focus Allocator applies top-scoring operations
       Expand: LOD1 → LOD0 (add detail, costs +31 tokens)
       Collapse: LOD0 → LOD1 (remove detail, saves -31 tokens)

  3. Budget maintained
       sum(entry_costs) ≤ W_max at all times
```

### Contrast with Alternatives

| System | Focus Mechanism |
|--------|----------------|
| **RAG** | Query-time retrieval (stateless) |
| **Sparse Attention** | Fixed patterns (e.g., every 64th token) |
| **Compressive Transformers** | Static aging (oldest first) |
| **MegaContext** | Continuous learned prediction (content-aware) |

### Example

```
Turn 1: User asks about login code
  → LensNet scores login regions: +0.8
  → Focus Allocator expands login from LOD1 → LOD0
  → Model sees login in full detail

Turn 2: User asks about database schema
  → LensNet scores login regions: -0.6 (no longer relevant)
  → Focus Allocator collapses login from LOD0 → LOD1
  → Login still in tree, just compressed in Working Context

Turn 3: User returns to login question
  → LensNet scores login regions: +0.7 (relevant again!)
  → Focus Allocator expands login from LOD1 → LOD0
  → No information lost, just re-expanded
```

See [[Examples]] for detailed walkthrough.

---

## 4. Reversibility Property

**Definition:** Focus changes are reversible—compressed content can be re-expanded without information loss.

### How It Works

The [[MegaContext Tree]] stores all content at LOD0 (full detail) permanently:

```
MegaContext Tree (disk):
  LOD0: [all tokens ever seen]
  LOD1: [learned gists for LOD0 blocks]
  LOD2: [learned gists for LOD1 blocks]

Working Context (GPU):
  Mix of LOD0/LOD1/LOD2 drawn from tree

Collapse LOD0→LOD1:
  - LOD0 tokens remain in tree
  - Working Context shows LOD1 gist instead
  - Saves 31 tokens in budget

Expand LOD1→LOD0:
  - Fetch LOD0 tokens from tree
  - Replace LOD1 gist in Working Context
  - Costs 31 tokens in budget
  - Original information restored
```

### Contrast with Alternatives

**Compressive Transformers:**
- Compression is **one-way lossy**
- Compressed memories cannot be recovered
- Once compressed with mean-pooling, original tokens are lost

**RAG:**
- Not applicable—no compression, just retrieval
- Retrieved chunks are always full text

**MegaContext:**
- **Two-way lossless** (at LOD0 storage level)
- Compressed in [[Working Context]] but preserved in [[MegaContext Tree]]
- Can expand/collapse any region dynamically

### Why It Matters

**Adaptability:**
```
Conversation evolves → relevance changes → focus adapts
```

Without reversibility, you'd need to decide upfront what to compress permanently. With reversibility, the system can change its mind based on new information.

**Example:**
```
T=0:   Topic is authentication → login code at LOD0
T=100: Topic shifts to database → login collapsed to LOD1
T=200: Bug in login mentioned → login re-expanded to LOD0
```

The system doesn't need to predict the future—it adapts as the conversation unfolds.

---

## 5. Learned Optimization Property

**Definition:** Focus policies and compression strategies are learned from data, not hand-crafted heuristics.

### What Is Learned

**[[GistNet]]:** How to compress 32 tokens into 1 [[Glossary#Gist / Gist Embedding|gist]]
- Objective: Minimize [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] (substitutability)
- Training: Teacher-student with frozen base model
- Result: Learns task-relevant abstractions

**[[LensNet]]:** Which regions need detail
- Objective: Maximize prediction quality within budget
- Training: Counterfactual labeling (what if we expanded/collapsed here?)
- Result: Learns relevance prediction

### Contrast with Heuristics

**Hand-crafted approaches:**
- Compress oldest content (FIFO aging)
- Keep first/last N tokens
- Attend to sentence boundaries
- Fixed attention patterns (every 64th token)

**Limitations:**
- Task-agnostic (same policy for all tasks)
- Cannot adapt to content
- Requires domain expertise to tune

**Learned approaches:**
- Data-driven (adapts to corpus statistics)
- Task-specific (different policies for QA vs summarization)
- Continuous improvement (retrain as data evolves)

### Example

**Heuristic policy:**
```
Always compress content older than 1000 tokens
```
**Problem:** What if token 800 contains the answer to current question?

**Learned policy:**
```
LensNet predicts token 800 is highly relevant to current query
→ Keep expanded even though old
```

### Training Signals

**[[GistNet]] sees:**
- Next-token prediction loss
- Hidden state similarity
- Actual model performance with/without [[Glossary#Gist / Gist Embedding|gist]]

**[[LensNet]] sees:**
- Counterfactual ΔNLL (what would happen if we expanded/collapsed?)
- Budget utilization efficiency
- Historical access patterns

Result: Policies that optimize for actual task performance, not proxy metrics.

---

## 6. Substitutability Property

**Definition:** [[Glossary#Gist / Gist Embedding|Gists]] can replace their source tokens in the [[Working Context]] with minimal impact on model predictions.

### Mathematical Expression

For a [[Glossary#Gist / Gist Embedding|gist]] G representing tokens T = [t₀, t₁, ..., t₃₁]:

```
ΔNLL@H(G, T) = NLL(next_tokens | context_with_gist)
              - NLL(next_tokens | context_with_tokens)

Target: ΔNLL@H < 0.1  (negligible degradation)
```

### How It's Achieved

**[[GistNet]] training:**
1. Encode 32 tokens into hidden states using frozen base model
2. Compress into 1 [[Glossary#Gist / Gist Embedding|gist]] embedding via [[GistNet]]
3. Pass [[Glossary#Gist / Gist Embedding|gist]] through frozen base model layers
4. Minimize difference in:
   - Next-token predictions
   - Hidden layer activations
   - Attention patterns

**Result:** [[Glossary#Gist / Gist Embedding|Gist]] "looks like" a token to the base model

### Why It Matters

**Seamless integration:**
```python
# Base model doesn't know some entries are gists
context = [token, token, gist, token, gist, gist, token]
logits = base_model(context)  # Just works!
```

**No architectural changes:**
- No special [[Glossary#Gist / Gist Embedding|gist]] embedding layer
- No modified attention
- Frozen base model completely unaware

**Quality preservation:**
- ΔNLL@H < 0.1 means predictions barely change
- Model "sees" approximately the same information
- Just more compactly represented

---

## Property Interactions

These properties **reinforce each other:**

```
Constant Compute + Constant Memory
  → System can scale to any context size

Dynamic Focus + Reversibility
  → Can adapt to changing relevance without information loss

Learned Optimization + Substitutability
  → Focus policy optimizes actual task performance

All Together
  → Unbounded context at constant cost with automatic management
```

### System Coherence

**Example end-to-end:**
```
1. User provides 1M token codebase
   → Constant Memory: Only W_max on GPU, rest in tree

2. User asks question about login code
   → Dynamic Focus: LensNet scores login high
   → Reversibility: Expand login from LOD1 → LOD0

3. Model generates answer
   → Substitutability: Distant code shown as gists
   → Constant Compute: ~15ms per token regardless

4. Learned Optimization: LensNet learned what matters
   → Better than "always expand recent" heuristic
```

---

## Verification & Measurement

### How to Verify These Properties

**1. Constant Compute:**
```python
for N in [10k, 100k, 1M, 10M]:
    tree = ingest_tokens(N)
    latency = measure_decode_step(tree)
    assert latency < 20ms  # Should be ~15ms ± margin
```

**2. Constant Memory:**
```python
for N in [10k, 100k, 1M]:
    gpu_mem_before = torch.cuda.memory_allocated()
    tree = ingest_tokens(N)
    working_context = assemble_wc(tree)
    gpu_mem_after = torch.cuda.memory_allocated()
    assert (gpu_mem_after - gpu_mem_before) < 1GB  # Constant WC size
```

**3. Dynamic Focus:**
```python
# Measure refocus rate
focus_changes = count_expand_collapse_operations(episode)
assert focus_changes > 0  # Should adapt over time
```

**4. Reversibility:**
```python
# Expand then collapse
original_l0 = get_l0_block(tree, block_id=100)
collapse(working_context, block_id=100)  # LOD0 → LOD1
expand(working_context, block_id=100)    # LOD1 → LOD0
recovered_l0 = get_l0_block(tree, block_id=100)
assert np.allclose(original_l0, recovered_l0)  # Lossless
```

**5. Substitutability:**
```python
# Measure ΔNLL
nll_with_tokens = compute_nll(context_L0)
nll_with_gist = compute_nll(context_L1)
delta_nll = nll_with_gist - nll_with_tokens
assert delta_nll < 0.1  # Minimal degradation
```

---

## Summary Table

| Property | Definition | Benefit | Measured By |
|----------|------------|---------|-------------|
| **Constant Compute** | O(W_max²) per step | Predictable latency at any scale | ms/token |
| **Constant Memory** | O(W_max) on GPU | Unbounded context on fixed hardware | GB GPU RAM |
| **Dynamic Focus** | Continuous refocusing | Adapts to changing relevance | Swap rate |
| **Reversibility** | Lossless expand/collapse | No information loss from compression | Reconstruction error |
| **Learned Optimization** | Data-driven policies | Better than heuristics | ΔNLL, task accuracy |
| **Substitutability** | [[Glossary#Gist / Gist Embedding|Gists]] ≈ tokens | No base model changes needed | [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] |

---

## Related Pages

- [[Architecture Details]] - How properties emerge from system design
- [[Performance Sketch]] - Detailed performance analysis
- [[Comparisons]] - How properties compare to alternatives
- [[How MegaContext Works]] - Properties in action
- [[Grand Vision]] - Future property enhancements

---

*These six properties work together to enable MegaContext's core promise: **effectively infinite context at constant compute**.*
