---
tags:
  - module
summary: The hierarchical gist tree that stores the complete interaction history on disk, built incrementally through GistNet compression with block-aligned ingest and deterministic storage layout.
---
![[MegaContext Tree Diagram.png]]

The MegaContext Tree is the complete, append-only history of all tokens and their hierarchical gist summaries. It separates long-term memory (potentially millions or billions of tokens) from the fixed-size [[Working Context]] that the frozen base model actually sees.

---

- **Purpose:** Store unbounded context history at multiple levels of detail (L0, L1, L2, …).
- **Structure:** 32-ary tree where each parent gist compresses 32 children (tokens or lower-level gists).
- **Storage:** Persisted as binary files (`{L0,L1,L2}.ctx`) with deterministic offsets; RAM-resident in POC, disk-backed in future.
- **Updates:** Incremental ingest as 32-token blocks arrive; [[GistNet]] generates new gist nodes.
- **Interfaces:** Feeds [[Working Context]] assembly, [[LensNet]] conditioning, and [[Focus Allocator]] decisions.

---
## Details

### What is the MegaContext Tree?

The MegaContext Tree is a **hierarchical data structure** that stores the complete interaction or document history as a tree of gists. Unlike traditional LLM context windows that are limited to a fixed size, the MegaContext Tree can grow unboundedly while providing different levels of detail on demand.

Think of it like a pyramid:
- **Base layer (L0):** Every raw token ever seen
- **Level 1 (L1):** Each group of 32 L0 tokens compressed into a single gist
- **Level 2 (L2):** Each group of 32 L1 gists compressed into a single gist (representing 1,024 L0 tokens)
- **Level 3+ (future):** Additional layers for even coarser summaries

At any moment, the [[Working Context]] draws from this tree, selecting which parts to show as raw tokens and which as compressed gists based on [[LensNet]]'s focus scores.

---

### Tree Structure & Hierarchy

#### Node Types

Each node in the MegaContext Tree represents a contiguous span of the original token sequence at a specific level of detail:

| Level | Content | Compression Ratio | Token Coverage |
|-------|---------|-------------------|----------------|
| **L0** | Raw token embeddings | 1:1 (no compression) | 1 token per entry |
| **L1** | [[GistNet]] 32→1 gist | 32:1 | 32 L0 tokens |
| **L2** | [[GistNet]] 32→1 gist of gists | 1024:1 | 1,024 L0 tokens |
| **L3+** | (Future) Higher-level gists | 32^n:1 | 32^n L0 tokens |

#### Tree Properties

- **Fixed branching factor:** Every internal node has exactly 32 children (block size K=32 in POC)
- **Perfect alignment:** Node boundaries align with 32-token blocks; no partial spans
- **Contiguous coverage:** Nodes at each level tile the timeline without gaps or overlaps
- **Append-only:** New tokens extend the rightmost branch; historical nodes are immutable (except for refreshing gists when GistNet is retrained)
- **Balanced growth:** Tree depth grows logarithmically (log₃₂(N) levels for N tokens)

#### Example Tree Fragment

```
L2 [0:1024]
├─ L1 [0:32]
│  ├─ L0 [0] "The"
│  ├─ L0 [1] "quick"
│  ├─ L0 [2] "brown"
│  └─ ... (29 more L0 tokens)
├─ L1 [32:64]
│  └─ ... (32 L0 tokens)
└─ ... (30 more L1 gists)
```

---

### Node Metadata

Each node stores or can compute the following metadata:

| Field | Type | Meaning |
|-------|------|---------|
| `span_id` | `uint64` | Unique identifier for this span |
| `level` | `uint8` | 0 (L0), 1 (L1), 2 (L2), etc. |
| `start_token` | `uint64` | Absolute position of first token in this span |
| `end_token` | `uint64` | Absolute position of last token (exclusive) |
| `parent_id` | `uint64` | Span ID of parent node (null for root) |
| `child_ids` | `uint64[32]` | Span IDs of children (null for L0 leaves) |
| `data_offset` | `uint64` | Byte offset in storage file for this node's payload |
| `timestamp` | `uint64` | Ingestion time (optional, for temporal decay in pruning) |
| `access_count` | `uint32` | Number of times this span was in working context (telemetry) |
| `gist_version` | `uint16` | Which GistNet checkpoint generated this gist (for versioning) |

**Storage strategy:**
- Lightweight metadata (IDs, offsets, pointers) lives in RAM as an in-memory index
- Heavy payloads (token IDs, gist embeddings) live in binary `.ctx` files
- Because block sizes are fixed, offsets can be computed deterministically without external indexes

---

### Ingest & Update APIs

#### Ingesting New Tokens

The tree grows incrementally as tokens arrive:

1. **Buffer tokens:** Accumulate incoming tokens until a complete 32-token L0 block is ready
2. **Store L0 block:** Append token IDs to `L0.ctx` and create metadata entries
3. **Generate L1 gist:** When 32 L0 blocks complete, run [[GistNet]] to produce an L1 gist; append to `L1.ctx`
4. **Propagate upward:** When 32 L1 gists complete, generate an L2 gist; continue recursively
5. **Update index:** Add new nodes to the in-memory tree index with parent/child pointers

**Pseudocode:**
```python
def ingest_tokens(tree, tokens):
    tree.l0_buffer.extend(tokens)

    while len(tree.l0_buffer) >= 32:
        block = tree.l0_buffer[:32]
        l0_node = tree.add_l0_block(block)
        tree.l0_buffer = tree.l0_buffer[32:]

        if l0_node.parent_full():  # 32 siblings ready
            l1_gist = gistnet.compress(l0_node.siblings())
            l1_node = tree.add_l1_gist(l1_gist)

            if l1_node.parent_full():
                l2_gist = gistnet.compress(l1_node.siblings())
                tree.add_l2_gist(l2_gist)
```

#### Refreshing Gists

When [[GistNet]] is retrained (during alternating optimization), existing gist nodes may be updated:

1. **Iterate over L1 nodes:** For each L1 node, retrieve its 32 L0 children
2. **Recompute gist:** Run the new GistNet checkpoint to generate a fresh gist
3. **Update in place:** Overwrite the old gist at the same offset in `L1.ctx`
4. **Propagate upward:** Recursively refresh L2 gists whose children changed
5. **Version tracking:** Increment `gist_version` metadata to track which checkpoint generated each gist

**Important:** Gist refreshing is optional during the POC. The system can operate with gists frozen to a single GistNet checkpoint. Refreshing becomes important during joint training phases (see [[Training & Operations]]).

---

### Storage Format

The MegaContext Tree persists to disk as three binary files with deterministic layout:

#### Binary Layout (`{L0,L1,L2}.ctx`)

Each file begins with a 64-byte header (see [[POC Architecture#Binary storage layout]]):

```
Offset  Field            Type      Meaning
------  -----            ----      -------
0       magic            uint32    0x4D434354 ("MCCT")
4       version          uint16    Format revision (start at 1)
6       level            uint16    0 (L0), 1 (L1), or 2 (L2)
8       block_size       uint16    K = 32
10      embedding_dim    uint16    Base model embedding width d
12      dtype_code       uint16    0=uint32, 1=fp16, 2=bf16
14      model_name       char[32]  "SmolLM3-3B" etc.
46      reserved         18 bytes  Zeroed for future use
```

**Payload layout:**
- **L0.ctx:** Contiguous `uint32` token IDs (vocabulary indices)
  - Offset for block `i`: `64 + i × 32 × 4` bytes
- **L1.ctx, L2.ctx:** Contiguous `fp16` or `bf16` gist vectors
  - Offset for gist `i`: `64 + i × embedding_dim × 2` bytes

**Benefits:**
- **Deterministic access:** Compute any node's offset via simple arithmetic
- **Memory-mapped I/O:** Future disk-backed versions can use `mmap` for efficient random access
- **Portability:** Self-describing headers allow cross-platform sharing
- **Compression-ready:** Fixed-width records can be block-compressed (zstd, etc.) in future versions

---

### Block Alignment & Contiguity Invariants

The MegaContext Tree enforces strict **block alignment** to maintain the [[Glossary#Contiguity Invariant]]:

1. **Block boundaries:** All nodes start and end at multiples of K=32 tokens
2. **No partial spans:** A node never covers 17 or 48 tokens—always exact multiples of 32
3. **Sibling groups:** Every 32 consecutive nodes at level L share the same parent at level L+1
4. **Timeline coverage:** At each level, nodes tile the full token timeline without gaps or overlaps

**Why this matters:**
- [[Focus Allocator]] can swap LOD levels (expand/collapse) without breaking continuity
- [[RoPE]] positional encodings remain consistent when gists replace tokens
- [[Working Context]] assembly is simplified—just concatenate contiguous node ranges

---

### Relationship to Working Context

The MegaContext Tree and [[Working Context]] serve complementary roles:

| Aspect | MegaContext Tree | [[Working Context]] |
|--------|------------------|---------------------|
| **Scope** | Complete history (unbounded) | Fixed window (W_max tokens) |
| **Location** | Disk (future) or RAM (POC) | GPU memory |
| **Content** | All levels (L0, L1, L2) stored | Mixed levels selected dynamically |
| **Mutability** | Append-only, immutable nodes | Entries swap LOD every block |
| **Purpose** | Long-term memory substrate | Immediate context for base LLM |

**Flow:**
1. All tokens are ingested into the MegaContext Tree
2. [[LensNet]] analyzes the current [[Working Context]] and recent tree state
3. [[Focus Allocator]] selects which tree nodes to include in the working context and at what LOD
4. The working context is a **view** into the tree, not a separate copy

---

### Role in the System

The MegaContext Tree is the **foundational data structure** that enables everything else:

- **For [[GistNet]]:** Training data source and storage target for hierarchical gists
- **For [[LensNet]]:** Provides tail gists (recent L1/L2) used as conditioning signals
- **For [[Focus Allocator]]:** Supplies candidate spans for expand/collapse operations
- **For [[Runtime Loop]]:** Orchestrates ingest → gist → focus → decode cycle
- **For telemetry:** Tracks access patterns, ΔNLL sensitivity, and pruning signals (see [[MegaCuration]])

---

### POC Implementation Notes

In the proof-of-concept:
- **RAM-resident:** Tree index and all payloads live in RAM (no disk I/O yet)
- **Two levels:** L0 tokens + L1 gists + single L2 root (sufficient for moderate contexts)
- **Synchronous updates:** Gist generation happens inline during ingest (no background workers)
- **Fixed GistNet:** Gists frozen to initial checkpoint (no retraining during demo runs)

See [[POC Scope]] for detailed constraints and [[POC Architecture]] for module interfaces.

---

### Future Enhancements

Post-POC improvements include:
- **Disk-backed storage:** Memory-mapped `.ctx` files with async streaming (see [[Research Paper Plan]] Phase 3)
- **Deeper hierarchies:** L3, L4, … for billion-token contexts
- **Incremental updates:** Rebuild only affected subtrees when files change (see [[Cognitive Core#Curating the core knowledge corpus]])
- **Provenance tracking:** Attach source IDs, timestamps, retrieval scores per node (see [[MegaCuration]])
- **Soft deletes:** Mark low-utility spans as inactive without removing them (pruning tier)
- **Version management:** Handle multiple GistNet checkpoints coexisting in the tree

---

## Summary

The MegaContext Tree is MegaContext's "hard drive"—a persistent, hierarchical memory that stores the complete interaction history at multiple levels of detail. By separating long-term storage (tree) from short-term attention ([[Working Context]]), the system achieves **effectively infinite context** while keeping per-step compute constant. All roads in MegaContext lead through the tree: [[GistNet]] builds it, [[LensNet]] reads it, [[Focus Allocator]] navigates it, and the frozen base model consumes refined views drawn from it.
