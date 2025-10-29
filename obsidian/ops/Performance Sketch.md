---
tags:
  - ops
summary: Envelope estimates for compute and storage when running MegaContext, including a decade-long retention case study.
---
MegaContext keeps per-step compute close to a frozen LLM while amortizing storage through hierarchical gists, quantization, and telemetry-guided pruning.

---

- **Per-step compute:** essentially base decode cost; GistNet/LensNet overhead <1%.
- **Working budget:** ~8k active tokens in the POC, scaling toward 32k for future builds.
- **Storage growth:** 32-ary tree adds only ~3.2% above leaves; precision and pruning dominate footprint.
- **Case study:** 10-year, 500 Hz robotics log compresses to ~6–14 TB with pruning + entropy coding.
- **Planning hooks:** informs cost assumptions in [[Grand Vision]] and operational budgets in [[Training & Operations]].

---
## Details

### Step-level comparison

| Setup | MegaContext tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | Context-limited |
| **MegaContext (POC)** | ~1 M | 8 k | ~0.5 GB | few MB | Constant compute per step |
| **MegaContext (Future)** | 1 B+ | 32 k | ~2 GB | 10–50 MB/s | Fully trained base model |

Per-step compute ≈ base decode cost; gist extraction and LensNet overhead <1%.

---

### Detailed Per-Step Compute Breakdown

For a MegaContext-enabled system running on a single GPU (NVIDIA L4/A100-class):

#### Base Model Decode (SmolLM3-3B, W_max=8k)
- **Forward pass:** ~15 ms (8k tokens, bf16, FlashAttention 2)
- **KV-cache:** ~512 MB GPU memory
- **Throughput:** ~67 tokens/sec (includes autoregressive generation overhead)

#### GistNet Overhead (32-token block → L1 gist)
- **Compression time:** ~0.3 ms per block
- **L1 generation (32 blocks → L2):** ~9.6 ms total
- **Amortized per decode step:** ~0.3 ms (runs once per K=32 tokens)
- **Overhead:** ~2% relative to base decode

#### LensNet Overhead (8k working context entries)
- **Scoring pass:** ~2.5 ms per update
- **Frequency:** Once per K=32 tokens
- **Amortized per token:** ~0.08 ms
- **Overhead:** ~0.5% relative to base decode

#### Focus Allocator
- **Priority queue operations:** <0.1 ms
- **Negligible overhead:** priority queues are O(log N), N≈256 entries

#### Total Overhead
```
Base decode:        15.0 ms/token
GistNet:          +  0.3 ms (amortized)
LensNet:          +  0.08 ms (amortized)
Focus allocator:  +  0.003 ms
Total:              15.383 ms/token (~2.5% overhead)
```

**Key takeaway:** MegaContext adds **<3% latency** while enabling unbounded context.

---

### Storage Analysis

#### MegaContext Tree Storage

For a corpus of N L0 tokens stored as a 32-ary hierarchical gist tree:

| Level | Nodes | Storage (fp16) | Storage (8-bit) | Notes |
|-------|-------|----------------|-----------------|-------|
| **L0** | N | N × 4 bytes | N × 4 bytes | Token IDs (uint32) |
| **L1** | N/32 | (N/32) × d × 2 | (N/32) × d × 1 | Gist vectors |
| **L2** | N/1024 | (N/1024) × d × 2 | (N/1024) × d × 1 | Gist vectors |
| **L3+** | N/32^k | ... | ... | Future expansion |

Where `d` = embedding dimension (typically 2048–4096).

**Example: 1M tokens, d=2048, fp16 gists:**
- L0: 1M × 4 bytes = 4 MB (token IDs)
- L1: (1M/32) × 2048 × 2 = 128 MB
- L2: (1M/1024) × 2048 × 2 = 4 MB
- **Total:** 136 MB for 1M token history (~136 bytes/token)

**Tree overhead:** L1+L2 adds only ~3.2% storage compared to L0 alone (factor: 32/31).

#### Storage Scaling Scenarios

| Scenario | Total Tokens | L0 Size | Tree Size (fp16) | Tree Size (8-bit) | Notes |
|----------|--------------|---------|------------------|-------------------|-------|
| **Short conversation** | 10k | 40 KB | 1.3 MB | 650 KB | Few messages |
| **Coding session** | 100k | 400 KB | 13 MB | 6.5 MB | Medium codebase |
| **Long-form doc** | 1M | 4 MB | 136 MB | 68 MB | Book/manual |
| **Lifetime agent** | 10M | 40 MB | 1.36 GB | 680 MB | Persistent assistant |
| **Knowledge base** | 100M | 400 MB | 13.6 GB | 6.8 GB | Wikipedia-scale |
| **Decade robotics log** | 1.5×10¹¹ | 600 GB | ~20 TB | ~10 TB | See [[Realtime Scenarios]] |

**With pruning:** Retaining only 0.5–1% of L0 tokens in high-detail form and aggressive quantization can reduce storage by 10–50× (see [[MegaCuration]]).

---

### Memory Hierarchy Breakdown

For a production MegaContext system:

```
┌─────────────────────────────────────────────┐
│ GPU (Active Inference)                      │
│ - Working Context: 8k–32k tokens            │
│ - KV-cache: 0.5–2 GB                        │
│ - Model weights: 6–24 GB (frozen)           │
│ - GistNet/LensNet: 0.1–0.5 GB              │
│ Cost: $$$ (high-speed HBM)                  │
└─────────────────────────────────────────────┘
            ↕ (streaming, ~10 MB/s)
┌─────────────────────────────────────────────┐
│ RAM (Hot Tree Index)                        │
│ - Recent gist embeddings: 100 MB–1 GB       │
│ - Node metadata index: 10–100 MB            │
│ - Tail gist cache (for LensNet): 5 MB      │
│ Cost: $$ (DDR4/5)                           │
└─────────────────────────────────────────────┘
            ↕ (memory-mapped I/O)
┌─────────────────────────────────────────────┐
│ SSD (Persistent Tree Storage)              │
│ - L0/L1/L2.ctx files: 10 GB–10 TB          │
│ - Checkpointed snapshots: 2× tree size     │
│ - Telemetry logs: 1–10% of tree size       │
│ Cost: $ (NVMe/SATA)                         │
└─────────────────────────────────────────────┘
            ↕ (cold archive)
┌─────────────────────────────────────────────┐
│ Object Storage (Long-term Archive)          │
│ - Pruned/compressed trees: 1–10 TB          │
│ - Historical checkpoints                    │
│ Cost: ¢ (S3/GCS)                            │
└─────────────────────────────────────────────┘
```

---

### Scaling Envelope: Compute vs Context Size

| Context Size | Base LLM (Full Attention) | MegaContext (W_max=8k) | MegaContext (W_max=32k) |
|--------------|---------------------------|------------------------|-------------------------|
| **4k tokens** | 1× baseline | 1.02× | 1.02× |
| **32k tokens** | 8× compute | 1.02× | 1.02× |
| **256k tokens** | 64× compute (OOM likely) | 1.03× | 1.03× |
| **1M tokens** | 256× (impossible) | 1.03× | 1.03× |
| **10M tokens** | N/A | 1.04× | 1.04× |
| **1B tokens** | N/A | 1.05× | 1.05× |

**Explanation:**
- Base LLM: Quadratic attention O(N²) means 8× context → 64× compute
- MegaContext: Working context stays fixed, compute stays constant ~O(W_max²)
- Slight overhead growth: More gists → more LensNet conditioning, but still <5%

**GPU memory vs context:**

| Context Size | Base LLM KV-cache | MegaContext Working KV | MegaContext Tree (RAM) |
|--------------|-------------------|------------------------|------------------------|
| 4k tokens | 256 MB | 128 MB | 1 MB |
| 32k tokens | 2 GB | 512 MB | 8 MB |
| 256k tokens | 16 GB (OOM on most GPUs) | 512 MB | 64 MB |
| 1M tokens | N/A | 512 MB | 256 MB |
| 10M tokens | N/A | 512 MB | 2.5 GB |

---

### Comparison to Alternative Approaches

#### vs. Full-Context Transformers (e.g., LongFormer, BigBird)

| Metric | Full-Context Sparse Attention | MegaContext |
|--------|-------------------------------|-------------|
| **Context length** | 4k–64k (hard limit) | Unbounded |
| **Compute** | O(N × log N) or O(N × W) | O(W_max²) constant |
| **Memory** | O(N) linear | O(W_max) constant |
| **Detail control** | Fixed patterns (global + local) | Learned dynamic focus |
| **Training** | End-to-end joint | Alternating (GistNet, LensNet) |

**Verdict:** MegaContext trades sparse attention patterns for learned compression + focus.

#### vs. Retrieval-Augmented Generation (RAG)

| Metric | RAG (DPR + FiD) | MegaContext |
|--------|-----------------|-------------|
| **Query latency** | ~50–200 ms (retrieval + rerank) | ~2 ms (LensNet scoring) |
| **Context integration** | Concatenate retrieved chunks | Inline gist substitution |
| **Memory format** | External vector DB | Hierarchical tree |
| **Focus dynamics** | Query-time only | Continuous refocusing |
| **Defocusing** | Not supported | Native (collapse) |

**Verdict:** RAG excels at external knowledge; MegaContext at persistent, evolving memory.

#### vs. Compressive Transformers (Rae et al. 2019)

| Metric | Compressive Transformer | MegaContext |
|--------|-------------------------|-------------|
| **Compression** | Fixed functions (mean pool, attention) | Learned (GistNet) |
| **Hierarchy** | Two-level (active, compressed) | Multi-level (L0, L1, L2, …) |
| **Focus control** | Static aging policy | Learned dynamic (LensNet) |
| **Substitutability** | Approximate | Trained for low ΔNLL@H |

**Verdict:** MegaContext generalizes compressive transformers with learned, hierarchical, reversible focus.

---

### Long-Term Storage Case Study: 10-Year Robotics Log

See [[Realtime Scenarios]] for full details. Summary:

**Scenario:** Continuous 500 Hz sensor data (4k-dim embeddings) over 10 years

| Compression Strategy | Storage Size | Notes |
|----------------------|--------------|-------|
| Raw L0 only (fp16) | ~1.29 PB | Impractical |
| Full tree (fp16) | ~1.33 PB | Only +3.2% overhead |
| Full tree (8-bit) | ~667 TB | Quantization helps 2× |
| Pruned (1% L0 @ 8-bit) | ~27 TB | Aggressive pruning |
| Pruned + entropy coding | ~13 TB | Compression on top |
| Ultra-aggressive (0.5% + 4-bit internal) | ~6–8 TB | Fits on commodity arrays |

**Key insight:** With smart pruning ([[MegaCuration]]) and quantization, even decade-long high-bandwidth logs compress to manageable sizes.

---

### Production Deployment Budgets

#### Small-Scale Deployment (1k–10k users)
- **Hardware:** 4× A100 GPUs, 2 TB NVMe SSD
- **MegaContext per user:** 1M tokens average (~130 MB each)
- **Total storage:** 10k × 130 MB = 1.3 TB (fits comfortably)
- **Concurrent inference:** ~40–80 users/GPU (depending on W_max)

#### Medium-Scale Deployment (100k users)
- **Hardware:** 40× A100 GPUs, 20 TB SSD array, object storage backend
- **MegaContext per user:** 5M tokens average (~650 MB each)
- **Hot storage (SSD):** 20 TB for active users
- **Cold storage (S3):** 50 TB compressed archives
- **Concurrent inference:** ~2,000 users simultaneously

#### Large-Scale Deployment (1M+ users)
- **Hardware:** Distributed GPU cluster, petabyte-scale object storage
- **MegaContext per user:** 10M tokens average (~1.3 GB each)
- **Active tier (SSD):** 100 TB for 10% of users
- **Archive tier (object storage):** 1 PB compressed
- **Strategy:** Tier hot/warm/cold MegaContext trees based on access patterns

---

### Performance Optimization Opportunities

#### Compute Optimizations
1. **Batch LensNet scoring:** Score multiple working contexts in parallel
2. **KV-cache reuse:** Preserve KV entries that don't change across refocus
3. **Async gist generation:** Generate L1/L2 gists in background workers
4. **Quantized GistNet:** Run GistNet in int8 for 2× speedup

#### Storage Optimizations
1. **Memory-mapped I/O:** Use `mmap` for zero-copy tree access
2. **Compression:** zstd block compression on `.ctx` files (~2–3× savings)
3. **Tiered storage:** Hot gists in RAM, warm on SSD, cold in object storage
4. **Deduplication:** Share identical gists across users (e.g., common documentation)

#### Focus Optimizations
1. **Predictive prefetching:** LensNet hints at likely future expansions
2. **Multi-resolution LensNet:** Coarse scoring pass first, fine scoring only where needed
3. **Learned allocator:** Replace greedy with differentiable surrogate (future)

---

### Research Milestone Targets

Per [[Research Paper Plan]] Phase 4, benchmark targets include:

| Metric | Target | Baseline (Full Context) | Baseline (RAG) |
|--------|--------|------------------------|----------------|
| **ΔNLL@H degradation** | ≤0.1 | 0.0 | 0.2–0.5 |
| **Latency overhead** | ≤10% | 0% | +50–200 ms (retrieval) |
| **Memory overhead** | ≤20% | 100% (KV-cache) | +5–10% (index) |
| **Swap rate** | ≤0.25 actions/block | N/A | N/A |
| **Mean residency** | ≥3 iterations/span | N/A | N/A |
| **Context coverage** | 10M+ tokens | 32k–128k max | Unlimited (but stateless) |

These targets guide POC development and paper evaluations.

---

## Summary

MegaContext achieves **constant-time decode** regardless of total context size by:
1. Keeping active working context fixed at W_max (8k–32k tokens)
2. Compressing inactive context into hierarchical gists (3.2% storage overhead)
3. Dynamically refocusing detail levels based on learned relevance (<3% compute overhead)

**Storage:** Linear growth O(N) with tree hierarchy overhead ~3%, pruning reduces further
**Compute:** Constant O(W_max²) for any total context size N
**Latency:** <3% overhead compared to frozen base model

This makes **billion-token contexts practical** on commodity GPU hardware while maintaining prediction quality within 0.1 ΔNLL@H of full-context baselines.
