---
tags:
  - operations
  - incremental-updates
summary: Complete procedures for ingesting tokens, generating gists, updating the tree structure, and refreshing gists when GistNet is retrained—the operational heart of the MegaContext Tree.
---

Tree Operations defines the complete operational lifecycle of the [[MegaContext Tree]]: how raw tokens flow into [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]] blocks, how [[GistNet]] compresses them into hierarchical [[Glossary#Gist / Gist Embedding|gists]], how the tree structure grows incrementally, and how gists are refreshed when [[GistNet]] is retrained during [[Training & Operations#Alternating optimization|alternating optimization]].

---

- **Scope:** Token ingest buffering, L0/L1/L2 gist generation, tree index updates, gist refresh procedures
- **Key invariant:** [[Glossary#Contiguity Invariant|Block alignment]] maintained throughout all operations—always 32-token boundaries
- **Interfaces:** Consumes tokens from input stream, invokes [[GistNet]] for compression, updates tree index
- **Performance:** Incremental O(log₃₂ N) updates per block; gist refresh is O(N) but rare
- **POC constraints:** Synchronous ingest, fixed GistNet checkpoint, no background workers

---

## Details

### Overview: Operational Flow

Tree Operations encompasses four primary procedures:

1. **Token Ingest:** Buffering and writing raw tokens to [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]] storage
2. **Gist Generation:** Running [[GistNet]] when complete 32-block groups form at any level
3. **Tree Structure Updates:** Maintaining parent/child pointers and metadata index
4. **Gist Refresh:** Recomputing gists with new [[GistNet]] checkpoints during training

All operations maintain the [[MegaContext Tree#Block Alignment & Contiguity Invariants|block alignment invariants]]—node boundaries always fall on 32-token multiples, ensuring the [[Glossary#Contiguity Invariant]] holds when the [[Focus Allocator]] swaps [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|levels of detail]].

---

### Token Ingest Pipeline

#### Buffering Strategy

The [[MegaContext Tree]] grows in discrete 32-token blocks. To handle arbitrary-length input streams:

1. **Accumulation buffer:** Maintain an `l0_buffer` that collects incoming tokens
2. **Block threshold:** When buffer reaches ≥32 tokens, extract a complete 32-token block
3. **Partial residue:** Remaining tokens (< 32) stay buffered until next ingest call
4. **No partial writes:** Never write incomplete blocks—preserves deterministic offsets

**Example:**
- Ingest 50 tokens → Write 32 to [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]], buffer 18
- Ingest 20 more → Write 32 (18+14), buffer 6
- Ingest 100 more → Write 96 (3 complete blocks), buffer 10

#### L0 Block Storage

When a complete 32-token block is ready:

1. **Allocate span ID:** Generate unique `span_id` for this L0 node
2. **Compute offset:** Determine byte position in `L0.ctx` file (block_index × 32 × sizeof(token_id))
3. **Write payload:** Append 32 token IDs to `L0.ctx` (or overwrite at computed offset if preallocated)
4. **Create metadata:** Add node entry to in-memory tree index with:
   - `level = 0`
   - `start_token`, `end_token` (absolute positions in global sequence)
   - `parent_id` (computed from block index ÷ 32)
   - `data_offset` (byte position in file)
5. **Update sibling tracker:** Mark this block as ready in parent's child list

**Storage layout:** See [[Storage Format#L0 Storage|Storage Format]] for binary encoding details.

---

### Hierarchical Gist Generation

#### L1 Gist Creation

When 32 consecutive [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]] blocks complete:

1. **Check parent full:** Detect when an L1 parent has all 32 L0 children ready
2. **Load L0 embeddings:** Retrieve token embeddings for the 32×32 = 1,024 token span
3. **Run GistNet:** Invoke `gistnet.compress(l0_embeddings, level=1)` to produce single L1 [[Glossary#Gist / Gist Embedding|gist]] vector
4. **Write L1 gist:** Append gist embedding to `L1.ctx` file
5. **Create L1 metadata:** Add node to tree index with 32 `child_ids` pointing to L0 children
6. **Version tracking:** Record which [[GistNet]] checkpoint (`gist_version`) generated this gist

**Key insight:** [[GistNet]] operates on raw embeddings at each level—L0 embeddings for L1 gists, L1 gist embeddings for L2 gists.

#### L2 Gist Creation

When 32 consecutive L1 gists complete:

1. **Check L2 parent full:** Detect when 32 L1 siblings are ready
2. **Load L1 gist embeddings:** Retrieve the 32 L1 gist vectors (span of 1,024 L0 tokens)
3. **Run GistNet:** Invoke `gistnet.compress(l1_gists, level=2)` to produce single L2 gist
4. **Write L2 gist:** Append to `L2.ctx` file
5. **Create L2 metadata:** Add node with 32 `child_ids` pointing to L1 gists
6. **Recursive propagation:** If 32 L2 gists complete, continue to L3 (future)

**Compression hierarchy:** Each L2 gist represents 32 L1 gists × 32 L0 blocks = 1,024 raw tokens.

#### Propagation Rules

After writing any level-L node:

```python
if node.siblings_complete():  # All 32 children of parent exist
    parent_gist = gistnet.compress(node.siblings(), level=node.level+1)
    parent_node = tree.add_gist(parent_gist, level=node.level+1)
    # Recursively check parent's parent
    propagate_upward(parent_node)
```

**Lazy evaluation:** Gist generation happens only when all 32 children are ready—no partial gists.

---

### Ingest Pseudocode

Complete ingest procedure with recursive gist generation:

```python
def ingest_tokens(tree, tokens):
    """
    Incrementally ingest tokens into the MegaContext Tree.

    Args:
        tree: MegaContextTree instance
        tokens: List of new token IDs to append

    Returns:
        Number of complete L0 blocks written
    """
    tree.l0_buffer.extend(tokens)
    blocks_written = 0

    while len(tree.l0_buffer) >= 32:
        # Extract complete 32-token block
        block = tree.l0_buffer[:32]
        tree.l0_buffer = tree.l0_buffer[32:]

        # Write L0 block to tree
        l0_node = tree.add_l0_block(block)
        blocks_written += 1

        # Check if L1 parent is now complete
        if l0_node.parent_full():  # 32 L0 siblings ready
            l1_gist = gistnet.compress(l0_node.siblings())
            l1_node = tree.add_l1_gist(l1_gist)

            # Check if L2 grandparent is now complete
            if l1_node.parent_full():  # 32 L1 siblings ready
                l2_gist = gistnet.compress(l1_node.siblings())
                tree.add_l2_gist(l2_gist)

                # Continue recursively for L3+ (future)

    return blocks_written


def add_l0_block(tree, block):
    """
    Append a complete 32-token L0 block to the tree.

    Args:
        block: List of exactly 32 token IDs

    Returns:
        L0 node metadata object
    """
    assert len(block) == 32, "L0 blocks must be exactly 32 tokens"

    # Generate unique span ID
    span_id = tree.next_span_id()
    tree.next_span_id += 1

    # Compute absolute token range
    start_token = tree.total_l0_tokens
    end_token = start_token + 32
    tree.total_l0_tokens += 32

    # Compute parent span ID (deterministic from index)
    block_index = start_token // 32
    parent_id = compute_l1_parent_id(block_index)

    # Write to storage
    offset = tree.l0_storage.append(block)  # Returns byte offset

    # Create metadata node
    node = TreeNode(
        span_id=span_id,
        level=0,
        start_token=start_token,
        end_token=end_token,
        parent_id=parent_id,
        child_ids=None,  # L0 nodes are leaves
        data_offset=offset,
        timestamp=now(),
        gist_version=None  # L0 stores raw tokens, not gists
    )

    tree.nodes[span_id] = node
    tree.register_child(parent_id, span_id)

    return node


def add_l1_gist(tree, gist_embedding):
    """
    Append an L1 gist node to the tree.

    Args:
        gist_embedding: GistNet output vector (d_model dimensions)

    Returns:
        L1 node metadata object
    """
    span_id = tree.next_span_id()
    tree.next_span_id += 1

    # L1 gists represent 32 L0 blocks = 1,024 tokens
    l1_index = tree.total_l1_gists
    start_token = l1_index * 1024
    end_token = start_token + 1024
    tree.total_l1_gists += 1

    # Compute parent (L2) and children (32 L0 blocks)
    parent_id = compute_l2_parent_id(l1_index)
    child_ids = [tree.find_l0_child(start_token + i*32) for i in range(32)]

    # Write gist embedding to L1 storage
    offset = tree.l1_storage.append(gist_embedding)

    # Create metadata
    node = TreeNode(
        span_id=span_id,
        level=1,
        start_token=start_token,
        end_token=end_token,
        parent_id=parent_id,
        child_ids=child_ids,
        data_offset=offset,
        timestamp=now(),
        gist_version=tree.gistnet_checkpoint_id
    )

    tree.nodes[span_id] = node

    # Update parent pointers in children
    for child_id in child_ids:
        tree.nodes[child_id].parent_id = span_id

    tree.register_child(parent_id, span_id)

    return node
```

**Notes:**
- Similar logic applies to `add_l2_gist()` and higher levels
- `parent_full()` checks if all 32 siblings exist in tree index
- `siblings()` retrieves embeddings for the 32 sibling nodes
- Block alignment enforced by fixed 32-token groups

---

### Gist Refresh Operations

#### When to Refresh

Gist refresh occurs during [[Training & Operations#Alternating optimization|alternating optimization]] when [[GistNet]] is retrained:

1. **Training phase:** [[GistNet]] updates on batches sampled from tree (gist reconstruction + [[ΔNLL]] objectives)
2. **Save checkpoint:** New `gistnet_v{N}.pt` checkpoint written to disk
3. **Refresh decision:** System can optionally refresh existing gists with new checkpoint
4. **Propagation:** Refreshed L1 gists trigger L2 refresh; L2 gists trigger L3 refresh (future)

**POC constraint:** Gist refresh is **not implemented** in POC—all gists frozen to initial [[GistNet]] checkpoint. Refresh becomes critical during full training loops.

#### Refresh Procedure

When a new [[GistNet]] checkpoint is loaded:

```python
def refresh_gists(tree, new_gistnet_checkpoint):
    """
    Recompute all gist nodes using a newly trained GistNet.

    Args:
        new_gistnet_checkpoint: Path to new GistNet weights

    Returns:
        Statistics: nodes refreshed per level, time elapsed
    """
    gistnet.load_weights(new_gistnet_checkpoint)
    checkpoint_id = extract_version(new_gistnet_checkpoint)

    stats = {level: 0 for level in [1, 2, 3]}  # Track refreshed nodes

    # Refresh L1 gists first (bottom-up propagation)
    for l1_node in tree.get_level_nodes(level=1):
        # Retrieve 32 L0 children
        l0_children = [tree.nodes[cid] for cid in l1_node.child_ids]
        l0_embeddings = [tree.load_embedding(node) for node in l0_children]

        # Recompute gist with new GistNet
        new_gist = gistnet.compress(l0_embeddings, level=1)

        # Overwrite in place (same offset in L1.ctx)
        tree.l1_storage.write_at(l1_node.data_offset, new_gist)

        # Update metadata
        l1_node.gist_version = checkpoint_id
        l1_node.timestamp = now()  # Optional: track refresh time

        stats[1] += 1

    # Refresh L2 gists (now that L1 gists are updated)
    for l2_node in tree.get_level_nodes(level=2):
        l1_children = [tree.nodes[cid] for cid in l2_node.child_ids]
        l1_gists = [tree.load_embedding(node) for node in l1_children]

        new_gist = gistnet.compress(l1_gists, level=2)
        tree.l2_storage.write_at(l2_node.data_offset, new_gist)

        l2_node.gist_version = checkpoint_id
        stats[2] += 1

    # Continue for L3+ if present

    return stats
```

#### Incremental Refresh (Future)

For efficiency, refresh only affected subtrees:

1. **Track dirty nodes:** Mark L0 spans that changed since last refresh
2. **Propagate dirty flags:** Mark L1 parents of dirty L0 blocks, L2 grandparents, etc.
3. **Selective recomputation:** Refresh only dirty gists, skip clean subtrees
4. **Versioning:** Tree can contain mixed `gist_version` values during partial refresh

**Use case:** When [[MegaCuration]] updates a subset of the tree (e.g., file edits in [[Cognitive Core]]), avoid refreshing entire tree.

#### Version Metadata

Each gist node tracks which [[GistNet]] checkpoint generated it:

| Field | Type | Purpose |
|-------|------|---------|
| `gist_version` | `uint16` | Checkpoint ID (e.g., 0 = initial, 1 = after first retrain) |
| `timestamp` | `uint64` | Unix timestamp of last gist computation |

**Backward compatibility:** Older gists remain valid—[[Working Context]] can mix gist versions. [[LensNet]] may learn to discount stale gists (low `gist_version`) during focus scoring.

---

### Tree Index Updates

#### Metadata Synchronization

After each write operation (L0 block or L1/L2 gist):

1. **Add node to index:** Insert metadata entry in `tree.nodes[span_id]` dictionary
2. **Update parent's children list:** Append `span_id` to parent's `child_ids` array
3. **Update child's parent pointer:** Set `parent_id` field in newly written node
4. **Increment counters:** Update `total_l0_tokens`, `total_l1_gists`, etc.

**Index structure (POC):**
```python
class MegaContextTree:
    nodes: Dict[span_id, TreeNode]  # Metadata index
    l0_storage: BinaryFile  # L0.ctx file handle
    l1_storage: BinaryFile  # L1.ctx file handle
    l2_storage: BinaryFile  # L2.ctx file handle

    l0_buffer: List[token_id]  # Partial block accumulator
    total_l0_tokens: int  # Global token count
    total_l1_gists: int  # Number of L1 nodes
    total_l2_gists: int  # Number of L2 nodes

    next_span_id: int  # Monotonic span ID allocator
    gistnet_checkpoint_id: int  # Current GistNet version
```

#### Deterministic Span IDs

Span IDs can be computed deterministically from level + index:

```python
def compute_span_id(level, index_at_level):
    """
    Generate deterministic span ID for hierarchical tree.

    Args:
        level: 0 (L0), 1 (L1), 2 (L2), etc.
        index_at_level: Sequential index within that level

    Returns:
        Unique span ID (uint64)
    """
    # Encode level in high bits, index in low bits
    return (level << 56) | index_at_level
```

**Advantage:** Given a token position, can compute which L0/L1/L2 node contains it without index lookups.

#### Offset Calculation

Because all blocks are 32-aligned, byte offsets are deterministic:

```python
def l0_offset(block_index, token_size=4):
    """Byte offset in L0.ctx for block_index."""
    return block_index * 32 * token_size  # 4 bytes/token = 128 bytes/block

def l1_offset(gist_index, embedding_size=4096*4):
    """Byte offset in L1.ctx for gist_index."""
    return gist_index * embedding_size  # 4 bytes/float × 4096 dims
```

**Implication:** Can memory-map `.ctx` files and compute addresses directly—no external B-tree or index file needed. See [[Storage Format#Deterministic offsets|Storage Format]].

---

### API Interface

The [[MegaContext Tree]] exposes these operations to other modules:

#### Public Methods

```python
class MegaContextTree:
    # Ingest API
    def ingest_tokens(tokens: List[int]) -> int:
        """Append tokens to tree; returns number of complete blocks written."""

    def flush_buffer() -> None:
        """Force write partial buffer (< 32 tokens) with padding—used at EOD."""

    # Query API
    def get_node(span_id: int) -> TreeNode:
        """Retrieve metadata for a node by span ID."""

    def get_embedding(span_id: int) -> np.ndarray:
        """Load embedding (token or gist) from storage."""

    def get_span_at_position(token_pos: int, level: int) -> TreeNode:
        """Find which L0/L1/L2 node contains this absolute token position."""

    def get_level_nodes(level: int, start_pos: int, end_pos: int) -> List[TreeNode]:
        """Retrieve all nodes at a level covering a token range."""

    # Update API
    def refresh_gists(gistnet_checkpoint: str) -> Dict[int, int]:
        """Recompute all gists with new GistNet; returns stats per level."""

    def refresh_span(span_id: int) -> None:
        """Recompute a single gist node (incremental refresh)."""

    # Telemetry API
    def mark_accessed(span_id: int) -> None:
        """Increment access_count for this span (for MegaCuration)."""

    def get_statistics() -> Dict[str, Any]:
        """Return tree size, depth, gist version distribution, etc."""
```

#### Integration Points

- **[[Runtime Loop]]:** Calls `ingest_tokens()` after each model forward pass
- **[[Working Context]] assembly:** Calls `get_embedding()` to load L0/L1/L2 content
- **[[LensNet]]:** Calls `get_level_nodes(level=1)` to fetch tail L1 gists for conditioning
- **[[Focus Allocator]]:** Calls `get_span_at_position()` to locate nodes for expand/collapse
- **[[Training & Operations]]:** Calls `refresh_gists()` after [[GistNet]] retraining

---

### Performance Characteristics

#### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `ingest_tokens(B)` | O(B/32 × log₃₂ N) | Amortized O(1) per token; propagates log₃₂ N levels |
| `get_node(span_id)` | O(1) | Hash table lookup in memory index |
| `get_embedding(span_id)` | O(1) | Direct offset calculation + file read |
| `get_span_at_position(pos, level)` | O(1) | Deterministic: `index = pos // 32^(level+1)` |
| `refresh_gists()` | O(N) | Must recompute every gist node; parallelizable |

**Key insight:** Incremental ingest is O(log N) per block, but amortizes to O(1) per token because most ingests don't trigger gist generation.

#### Space Complexity

| Component | Size | Scaling |
|-----------|------|---------|
| L0 storage (`L0.ctx`) | N × 4 bytes | Linear in total tokens |
| L1 storage (`L1.ctx`) | N/32 × 16 KB | ~0.5 KB per 32 tokens (4096-dim × fp32) |
| L2 storage (`L2.ctx`) | N/1024 × 16 KB | ~16 bytes per token at L2 |
| Metadata index | N/32 × ~128 bytes | ~4 bytes per token (POC in-memory) |

**Total:** ~0.54 KB per token for full tree (L0+L1+L2+metadata). For 1M tokens: ~540 MB.

#### Disk I/O (Future)

With memory-mapped files:
- **Ingest writes:** Sequential append, ~128 bytes per L0 block + ~16 KB per L1 gist
- **Random reads:** O(1) seeks via deterministic offsets
- **Refresh writes:** Random overwrites at existing offsets (in-place update)

See [[Storage Format#Memory-mapped I/O|Storage Format]] for mmap strategies.

---

### POC Simplifications

The proof-of-concept omits several production features:

| Feature | POC Status | Production Plan |
|---------|------------|-----------------|
| **L3+ levels** | Not implemented | Add when N > 32K tokens (~3 levels sufficient for 1M tokens) |
| **Gist refresh** | Disabled | Implement during alternating optimization phase |
| **Async ingest** | Synchronous | Background worker threads for gist generation |
| **Disk storage** | RAM-only | Memory-mapped `.ctx` files in Phase 2 |
| **Incremental refresh** | N/A | Dirty-tracking for subtree refresh |
| **Partial block padding** | Not handled | Pad final buffer with special token at EOD |

See [[POC Scope]] for complete constraints and [[POC Architecture]] for module boundaries.

---

### Error Handling & Edge Cases

#### Incomplete Blocks

**Problem:** Input ends mid-block (e.g., 17 tokens in buffer at end-of-document).

**Solutions:**
1. **Pad with special token:** Append 15 copies of `<PAD>` token to complete block
2. **Leave buffered:** Keep partial block in memory until next ingest (session resumes)
3. **Flush API:** Expose `flush_buffer(pad_token)` for explicit handling

**POC approach:** Leave buffered—assume continuous interaction. No explicit EOD handling.

#### GistNet Errors

**Problem:** [[GistNet]] forward pass fails (e.g., OOM, NaN outputs).

**Mitigation:**
1. **Checkpoint validation:** Test [[GistNet]] on sample inputs before refresh
2. **Rollback:** Keep old gist version if new computation fails
3. **Graceful degradation:** Skip failed gist, continue with stale version

#### Concurrent Modifications

**Problem:** Multiple threads ingesting tokens or refreshing gists simultaneously.

**Mitigation:**
1. **Ingest lock:** Single writer for L0 buffer and tree index
2. **Read-only refresh:** Gist refresh doesn't add nodes, only updates embeddings
3. **Versioned reads:** [[Working Context]] snapshots span IDs before reading embeddings

**POC simplification:** Single-threaded—no concurrency in demo runs.

---

## Summary

Tree Operations defines the **operational heart** of the [[MegaContext Tree]]: the incremental ingest pipeline that buffers tokens into 32-aligned blocks, the hierarchical [[GistNet]] compression that produces L1/L2 gists, the metadata updates that maintain tree structure, and the gist refresh procedures that keep the tree synchronized with retrained [[GistNet]] checkpoints. These operations maintain the [[Glossary#Contiguity Invariant|block alignment invariants]] that enable seamless [[Working Context]] [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|LOD]] transitions. All tree growth is **incremental** (O(log N) per block) and **deterministic** (offsets computed without indexes), making the tree both efficient and debuggable. During the POC, operations are synchronous and gist refresh is disabled—full [[Training & Operations#Alternating optimization|alternating optimization]] with live refresh is a post-POC enhancement.

---

## Related Pages

### Parent/Overview Pages
- [[MegaContext Tree]] – Parent page: hierarchical tree structure, compression hierarchy, and core concepts
- [[POC Architecture]] – System-wide architecture showing how Tree Operations fits into the overall design
- [[Architecture Details]] – Complete system architecture and component interactions

### Sibling Detail Pages
- [[Node Metadata]] – Metadata schema for tree nodes (span IDs, parent/child relationships, offsets)
- [[Storage Format]] – Binary file layouts, deterministic offsets, and memory-mapped I/O strategies
- [[GistNet Architecture Details]] – Detailed architecture of the compression model used in operations

### Related System Components
- [[GistNet]] – The compression model that generates L1/L2 gists during hierarchical operations
- [[GistNet Training]] – How GistNet is trained and produces gists for tree operations
- [[Working Context]] – Primary consumer of tree content; uses operations to load spans
- [[Focus Allocator]] – Triggers span expansion/collapse operations using tree queries
- [[LensNet]] – Uses tree operations to fetch tail L1 gists for conditioning

### Implementation Guides
- [[POC Implementation]] – Practical implementation details and constraints for the proof-of-concept
- [[Training & Operations]] – Alternating optimization framework including gist refresh procedures
- [[Alternating Optimization]] – How tree operations integrate with the training loop

### Related Concepts
- [[Glossary#Contiguity Invariant]] – Block alignment maintained by all tree operations
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)]] – Level of detail hierarchy managed by operations
- [[Glossary#Gist / Gist Embedding]] – The compressed representations generated during operations
- [[substitutability]] – Core principle ensuring gists can replace tokens
- [[ΔNLL]] – Metric used during gist refresh to validate quality
