# Performance Sketch

A back-of-the-envelope look at how MegaContext keeps compute constant while virtualizing memory, plus a storage case study for decade-scale logs.

## Step-level comparison

| Setup | MegaContext tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | Context-limited |
| **MegaContext (POC)** | ~1 M | 8 k | ~0.5 GB | few MB | Constant compute per step |
| **MegaContext (Future)** | 1 B+ | 32 k | ~2 GB | 10–50 MB/s | Fully trained base model |

Per-step compute ≈ base decode cost; gist extraction and LensNet overhead < 1 %.

## Long-term storage example: 24/7 robot (10 years)

**Assumptions**

- Sampling: 500 feature vectors / sec
- Feature size: 4,096-dim; stored as fp16 (2 bytes) unless noted
- Duration: 10 years ≈ 3.1536×10⁸ seconds ⇒ `N = 500 × 3.1536×10⁸ ≈ 1.5768×10¹¹` leaf vectors
- Tree arity: 32→1 at each level (no overlap)
- Tree depth: `log₃₂(N) ≈ 8` levels (root near level 8)
- Node payload: one vector per node (same width as leaves, different precision per scenario)

### Storage breakdown

| Scenario | Estimated storage | How to read this |
|---|---:|---|
| Raw leaves only (fp16) | ~1.29 PB | `N × 4096 × 2` bytes = 1.5768e11 × 8192 B |
| Full 32-ary tree at fp16 (leaves + all internal levels) | ~1.33 PB | Geometric factor for all nodes: (1 + 1/32 + 1/32² + …) = 32/31 ≈ 1.032× overhead over leaves |
| Full tree, 8-bit everywhere | ~667 TB | Leaves 8-bit: ~646 TB; internal nodes count = `N/31 ≈ 5.09e9`; internal 8-bit adds ~20.8 TB; total ≈ 646 + 20.8 |
| Pruned: keep only 1% of leaves @ 8-bit; keep all internal nodes @ 8-bit | ~27 TB | Leaves: 0.01 × 646 TB ≈ 6.46 TB; internal 8-bit ≈ 20.8 TB |
| Pruned + compressed: 1% leaves @ 8-bit with entropy coding (~×0.5); internal @ 8-bit with entropy coding (~×0.5) | ~13–14 TB | Leaves ≈ 3.2 TB + internal ≈ 10.4 TB |
| Aggressive: 0.5% leaves @ 8-bit + entropy (~×0.5); internal @ 4-bit + entropy (~×0.5) | ~6–8 TB | Leaves: 0.005 × 646 TB × 0.5 ≈ 1.6 TB; internal: `(20.8 TB × 0.5 for 4-bit) × 0.5` entropy ≈ 5.2 TB; total ~6.8 TB |

**Key takeaways**

- A full 32-ary tree only adds ~3.2% storage over leaves when stored at the same precision (factor 32/31); precision and pruning dominate total footprint.
- With 8-bit quantization and reasonable pruning of raw leaves (e.g., keep only salient 0.5–1%), plus straightforward entropy coding, a decade of continuous 500 Hz, 4k-dim features compresses to single-digit TBs—practical for local SSD arrays.
- This makes a lifelong, high-bandwidth memory feasible: raw details can be recovered where preserved; elsewhere, multilevel gists maintain global context with the working context handling on-demand re-expansion.
