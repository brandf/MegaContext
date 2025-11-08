---
tags:
  - plans
  - prd
summary: Defines the wLOD-aware KV caching scheme that keeps MegaAttention inference consistent as the working context tree expands and collapses.
---
# Hierarchical KV Caching

> **Status:** Plan of record (POR). Use this spec when implementing KV invalidation for MegaAttention/MegaPrediction; older caching discussions are informational only.

## Overview

This document defines a caching strategy for **[[MegaAttention Training|MegaAttention]]**, a hierarchical, multi-scale attention mechanism operating over the **[[Working Context]] (WC) tree** composed of multiple levels of detail (wLOD0, wLOD1, wLOD2, ...). It describes how to maintain and update Key/Value (KV) caches efficiently when the model's input sequence dynamically expands and contracts through gist substitutions from [[LensNet]] + the [[Focus Allocator]]—while also handling the temporal invalidation of all subsequent nodes whose attention dependencies occur *after* the edited regions. The policy keeps both base inference and [[MegaPrediction]] speculative spans numerically consistent without blowing away the entire cache.

---
## Problem Statement

[[MegaContext]]/[[MegaAttention Training|MegaAttention]] inference involves dynamic editing of the WC tree at the **leaf level (wLOD0)** as [[GistNet]] swaps gists and raw tokens.

These edits have **two orthogonal invalidation effects**:

1. **Hierarchical Invalidation:** Cached K/V tensors for the edited wLOD0 nodes and their ancestors (wLOD1, wLOD2) become invalid and must be recomputed.

2. **Temporal Invalidation:** Any node (across all wLOD levels) representing *time after* the edited regions depends on the modified history and must also have its K/V values discarded or refreshed.

Without accounting for both axes — hierarchical and temporal — the cache would yield inconsistent attention results. Efficiently updating both dimensions without recomputing the entire sequence is the core challenge.

---
## Design Principles

1. **Locality of Change:** Each edit affects only its own local subtree and all subsequent tokens in time.

2. **Hierarchical Consistency:** Each wLOD maintains its own KV cache; invalidation propagates upward and forward in time.

3. **Incremental Regeneration:** Only dirty and temporally affected subtrees are recomputed; clean, earlier nodes remain cached.

4. **Near-Causality:** Low-level attention is strictly causal; higher levels may include limited non-causal summaries for efficiency.

5. **Temporal Integrity:** Later timestamps must never depend on outdated earlier summaries.

---
## Working Context Tree

```

          wLOD2

          /   \

     wLOD1   wLOD1

      /  \     /  \

   wLOD0 wLOD0 wLOD0 wLOD0

```

Each node in the tree represents a temporal span (a runtime slice of the larger [[MegaContext Tree]]):
- **wLOD0 nodes**: individual tokens or local gists.
- **wLOD1+ nodes**: summaries of their child spans.

---
## Hierarchical and Temporal KV Caching

Each wLOD level maintains its own KV cache, but the cache must also respect *temporal dependencies*. This means that when an edit occurs, invalidation must propagate:

- **Upward (hierarchical):** to all ancestor nodes summarizing the changed region.
- **Forward (temporal):** to all nodes — at any wLOD — representing tokens *after* the edited region in causal order.


| Level | Granularity | Cache Contents | Update Trigger |
|--------|--------------|----------------|----------------|
| **wLOD0** | Tokens/gists | Fine-grained K/V tensors | Expansion/collapse edits at leaves, or earlier edits in time |
| **wLOD1** | 32-token summaries | Aggregated K/V from child wLOD0s via [[GistNet]] pooling | Any edit in child span or earlier timestamp |
| **wLOD2** | Global summaries | Aggregated K/V from wLOD1 summaries | Any change in subordinate regions or earlier time ranges |


Each cache entry includes:

- Key tensor `K` and Value tensor `V`

- Positional metadata `(t_center, σ)` representing span center and width

- Dirty flag for invalidation propagation (spatial + temporal)

---
## Cache Invalidation Rules

1. **Expansion:** When the [[Focus Allocator]] expands a gist, a single wLOD0 node becomes 32 new wLOD0 tokens. Their parent wLOD1 and wLOD2 summaries are invalidated, along with *all later temporal positions* that attend to or derive from this span.

2. **Collapse:** When 32 wLOD0 tokens collapse into one gist, the replaced tokens and their ancestor summaries are invalidated, and any later tokens must also be invalidated.

3. **Upward and Forward Propagation:**
   - Upward: recompute summaries through wLOD1 → wLOD2.
   - Forward: discard K/V caches for all tokens after the earliest edited timestamp.

4. **Downward Independence:** Earlier, unaffected nodes remain valid.

5. **Fallback Policy:** If edits occur early in the context (invalidating most of the sequence), fall back to full recompute for efficiency and simplicity.

---
## Implementation Strategies

### **Baseline Solution: Full Recompute**

- Recompute all wLOD levels each step.
- Clear all KV caches.

**Pros:**
- Simple and exact.
- Establishes correctness baseline.

**Cons:**
- O(T²) cost.
- Ignores hierarchical and temporal structure.

**Performance:** 1× baseline.

---
### **Proposed Solution: Hierarchical Incremental Cache

- Maintain KV caches for all wLOD levels.
- On each step:
  1. Detect expansions/collapses at wLOD0.
  2. Determine the *earliest timestamp* affected.
  3. Invalidate all nodes (across all wLODs) from that timestamp forward.
  4. Mark dirty leaves and ancestor nodes for recompute.
  5. Recompute affected subtrees using [[Flash Attention]] kernels.
  6. Propagate updated summaries upward (wLOD1 → wLOD2).
  7. Merge recomputed nodes into existing caches.

**Pros:**
- Handles both spatial (hierarchical) and temporal invalidations.
- Unified, production-ready design.
- Leverages tree locality for minimal recompute.
- Retains correctness identical to full recompute.
- 5–20× faster depending on edit density and edit timing.

**Cons:**
- Requires hierarchical + temporal cache management.
- Moderate engineering complexity.

**Performance:** expected 5–20× faster than baseline (context length dependent).

---
## Engineering Notes

- Use **[[Flash Attention]]-3** for dirty block recompute.
- Adopt **FP8 + Multi-Query Attention** to reduce memory.
- Batch edit processing to coalesce overlapping spans.
- Leverage **CUDA Graphs** for predictable latency.
- Profile both spatial and temporal invalidation ratios.
