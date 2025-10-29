---
tags:
  - components
summary: Complete metadata schema for MegaContext Tree nodes, including field definitions, storage strategies, usage patterns, and versioning information.
---
Node metadata is the lightweight structural information that defines each span in the [[MegaContext Tree]]. While the heavy payloads (token embeddings, [[GistNet|gist]] embeddings) live in binary `.ctx` files, metadata lives in a RAM-resident index that enables O(1) navigation, parent-child traversal, and deterministic offset computation.

This document details the complete metadata schema, storage strategies, and how metadata is used throughout the system.

---

## Metadata Schema

Each node in the [[MegaContext Tree]] stores or can compute the following metadata:

| Field | Type | Meaning |
|-------|------|---------|
| `span_id` | `uint64` | Unique identifier for this span |
| `level` | `uint8` | 0 ([[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]]), 1 ([[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]]), 2 ([[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L2]]), etc. |
| `start_token` | `uint64` | Absolute position of first token in this span |
| `end_token` | `uint64` | Absolute position of last token (exclusive) |
| `parent_id` | `uint64` | Span ID of parent node (null for root) |
| `child_ids` | `uint64[32]` | Span IDs of children (null for L0 leaves) |
| `data_offset` | `uint64` | Byte offset in storage file for this node's payload |
| `timestamp` | `uint64` | Ingestion time (optional, for temporal decay in pruning) |
| `access_count` | `uint32` | Number of times this span was in working context (telemetry) |
| `gist_version` | `uint16` | Which GistNet checkpoint generated this gist (for versioning) |

---

## Field Definitions

### Core Identification

#### `span_id` (uint64)
**Purpose:** Globally unique identifier for this span across all levels and time.

**Properties:**
- Immutable once assigned
- Used as the primary key in the in-memory index
- Referenced by parent/child relationships
- Used in [[Working Context]] to track which spans are currently loaded

**Generation Strategy:**
- Simple monotonic counter in POC
- Could be position-encoded (level + offset) for deterministic reconstruction
- Must remain stable even when [[GistNet]] is retrained

**Example:**
```python
# L0 span covering tokens [0:32)
span_id = 1000

# L1 span covering tokens [0:32) compressed
span_id = 1001  # Different from its L0 children
```

#### `level` (uint8)
**Purpose:** Indicates the [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|level of detail (LOD)]] this node represents.

**Valid Values:**
- `0` = L0 (raw tokens, no compression)
- `1` = L1 (32:1 compression, covers 32 L0 tokens)
- `2` = L2 (1024:1 compression, covers 1,024 L0 tokens)
- `3+` = Future higher-level gists (32^n:1 compression)

**Usage:**
- Determines which storage file to read from (`L0.ctx`, `L1.ctx`, `L2.ctx`)
- Used by [[Focus Allocator]] to decide expand/collapse operations
- Critical for maintaining the [[Glossary#Contiguity Invariant|contiguity invariant]]

**Invariants:**
- Parent level = child level + 1
- All siblings share the same level
- Level determines node size: `span_size = 32^level` tokens

---

### Span Positioning

#### `start_token` (uint64)
**Purpose:** The absolute position of the first token covered by this span in the global timeline.

**Properties:**
- Zero-based indexing (first token ever ingested is position 0)
- Aligned to block boundaries: always a multiple of 32
- Immutable once node is created

**Example:**
```python
# First L1 gist
start_token = 0      # Covers tokens [0:32)

# Second L1 gist
start_token = 32     # Covers tokens [32:64)

# First L2 gist
start_token = 0      # Covers tokens [0:1024)
```

#### `end_token` (uint64)
**Purpose:** The absolute position of the last token (exclusive) covered by this span.

**Properties:**
- Exclusive upper bound (Python slice semantics)
- Always equals `start_token + 32^level`
- Block-aligned: always a multiple of 32

**Derivation:**
```python
end_token = start_token + (32 ** level)

# L0 node: covers 1 token
# L1 node: covers 32 tokens
# L2 node: covers 1024 tokens
```

**Usage:**
- Span size checking: `size = end_token - start_token`
- Contiguity validation: ensure adjacent spans have `end₁ = start₂`
- Range queries: "Which spans overlap with tokens [1000:2000)?"

---

### Tree Structure

#### `parent_id` (uint64)
**Purpose:** References the span ID of this node's parent in the tree hierarchy.

**Properties:**
- `null` (or sentinel value like `0`) for root nodes at the highest level
- Points to a node at `level + 1`
- Immutable once assigned

**Usage:**
- Bottom-up tree traversal (e.g., "What L2 gist does this L0 token belong to?")
- Propagating gist updates during [[MegaContext Tree#Refreshing Gists|gist refresh]]
- Validating tree structure integrity

**Example:**
```python
# 32 L0 nodes [0] through [31] all share the same parent
parent_id = 1001  # L1 node covering [0:32)

# 32 L1 nodes all point to their L2 parent
parent_id = 2001  # L2 node covering [0:1024)
```

#### `child_ids` (uint64[32])
**Purpose:** Array of span IDs for this node's 32 children.

**Properties:**
- Fixed size: always exactly 32 entries
- `null` entries for L0 leaf nodes (no children)
- Partially filled during tree growth (rightmost parent may have < 32 children)
- Immutable once all 32 children are assigned

**Usage:**
- Top-down tree traversal
- Fetching children for [[GistNet]] compression
- Expand operations: [[Focus Allocator]] replaces a gist with its 32 children

**Example:**
```python
# L1 node covering tokens [0:32)
child_ids = [
    1000,  # L0[0] "The"
    1001,  # L0[1] "quick"
    1002,  # L0[2] "brown"
    ...    # 29 more L0 span IDs
]

# L0 leaf node
child_ids = [null, null, ..., null]  # No children
```

---

### Storage Mapping

#### `data_offset` (uint64)
**Purpose:** Byte offset in the corresponding binary storage file where this node's payload begins.

**Properties:**
- Deterministic: can be computed from `span_id` and level without external indexes
- Immutable (except during gist refresh, where data is overwritten at the same offset)
- Enables O(1) random access via memory-mapped I/O

**Computation:**
```python
def compute_offset(level, start_token):
    """Deterministic offset calculation."""
    block_size = 32 ** level
    block_index = start_token // block_size

    if level == 0:
        # L0: each entry is 2 bytes (token ID)
        return block_index * 64  # 32 tokens × 2 bytes
    else:
        # L1+: each entry is 2048 bytes (gist embedding)
        return block_index * 2048
```

**Usage:**
- Memory-mapped file access: `mmap[data_offset : data_offset + entry_size]`
- Bulk reads: fetch multiple contiguous nodes without index lookups
- Gist updates: overwrite data at known offset during [[MegaContext Tree#Refreshing Gists|refresh]]

**See also:** [[Storage Format]] for complete binary layout details.

---

### Telemetry & Optimization

#### `timestamp` (uint64)
**Purpose:** Records when this node was ingested into the tree.

**Properties:**
- Unix timestamp (seconds since epoch) or monotonic tick counter
- Immutable once set
- Optional in POC, required for temporal decay features

**Usage:**
- **Temporal decay:** [[Focus Allocator]] can prioritize recent content over stale history
- **Recency bias:** [[LensNet]] conditioning can weight recent gists more heavily
- **Pruning decisions:** [[MegaCuration]] can identify old, unused spans for archival
- **Debugging:** Track when specific interactions occurred

**Example:**
```python
# L0 block ingested at 2024-01-15 10:30:00
timestamp = 1705318200

# Calculate age
age_seconds = current_time - timestamp
if age_seconds > 7 * 24 * 3600:  # Older than 1 week
    apply_temporal_decay()
```

#### `access_count` (uint32)
**Purpose:** Tracks how many times this span has been included in the [[Working Context]].

**Properties:**
- Starts at 0 when node is created
- Incremented each time the span is loaded into working context
- Persisted across sessions (stored in metadata index)
- Reset to 0 during gist refresh (optional design choice)

**Usage:**
- **Hotness tracking:** Identify frequently accessed spans for caching/optimization
- **Pruning signals:** Low access count + old timestamp = candidate for archival
- **ΔNLL weighting:** [[MegaCuration]] can prioritize high-access spans during training data curation
- **Analytics:** Track which parts of conversations are revisited

**Example:**
```python
# Span rarely accessed → candidate for compression/pruning
if access_count < 3 and age_days > 30:
    mark_for_pruning()

# Frequently accessed span → keep in hot cache
if access_count > 100:
    pin_to_fast_storage()
```

**Increment Logic:**
```python
def load_into_working_context(span_id):
    """Load a span and update its access count."""
    node = tree.get_node(span_id)
    node.access_count += 1
    working_context.add(node)
```

#### `gist_version` (uint16)
**Purpose:** Tracks which [[GistNet]] checkpoint generated this gist.

**Properties:**
- Starts at 0 or 1 (initial GistNet checkpoint)
- Incremented each time [[GistNet]] is retrained and this gist is refreshed
- Only relevant for L1+ nodes (L0 tokens don't have gists)
- Enables version coexistence during gradual refresh

**Usage:**
- **Version auditing:** Know which GistNet model produced each gist
- **Gradual rollout:** Refresh gists incrementally across tree after retraining
- **A/B testing:** Compare quality of gists from different checkpoints
- **Debugging:** Trace reconstruction errors back to specific model versions

**Example:**
```python
# Initial gist generation
gist_version = 1  # Baseline GistNet-v1

# After first retraining cycle
gist_version = 2  # GistNet-v2 (alternating optimization iteration 1)

# Mixed versions during gradual refresh
tree.get_node(100).gist_version = 2  # Refreshed
tree.get_node(200).gist_version = 1  # Not yet refreshed
```

**Versioning Strategy:**
- **Eager refresh:** Update all gists immediately when new checkpoint is deployed
- **Lazy refresh:** Update gists on-demand when accessed
- **Hybrid:** Refresh hot/recent spans first, leave cold spans on old version

---

## Storage Strategy

Node metadata follows a **two-tier storage model** that separates lightweight structural data from heavy payload data:

### RAM-Resident Metadata Index

**What Lives in RAM:**
- All fields listed in the schema above
- Parent/child pointers for O(1) tree navigation
- Lightweight telemetry (access counts, timestamps)
- Offset pointers into storage files

**Structure:**
```python
class NodeMetadata:
    """In-memory metadata for a single tree node."""
    span_id: uint64
    level: uint8
    start_token: uint64
    end_token: uint64
    parent_id: uint64
    child_ids: list[uint64]  # Length 32
    data_offset: uint64
    timestamp: uint64
    access_count: uint32
    gist_version: uint16

# Global index
tree_index: dict[uint64, NodeMetadata] = {}
```

**Memory Footprint:**
- Per-node overhead: ~320 bytes
- For 1M nodes: ~305 MB RAM
- Scales linearly with node count (not token count)

**Advantages:**
- O(1) lookups by span ID
- Fast tree traversal without disk I/O
- Supports range queries and spatial indexing
- Can be serialized to disk for crash recovery

### Disk-Backed Payloads

**What Lives on Disk:**
- L0: Token IDs (2 bytes each)
- L1+: Gist embeddings (2048 bytes each in POC)

**Access Pattern:**
1. Look up metadata in RAM index
2. Use `data_offset` to seek to correct position in `.ctx` file
3. Memory-map or read payload into GPU memory
4. Optional: LRU cache for hot payloads

**Benefits:**
- Metadata lookups don't require disk I/O
- Deterministic offsets enable O(1) random access
- Large histories (millions of tokens) stay on disk until needed
- [[Working Context]] loads only what's actively used

**See:** [[Storage Format]] for complete binary file layout specifications.

---

## Metadata Usage Patterns

### Tree Navigation

**Bottom-Up Traversal:**
```python
def get_ancestors(span_id: uint64) -> list[NodeMetadata]:
    """Get all ancestors from node to root."""
    ancestors = []
    current = tree_index[span_id]

    while current.parent_id is not None:
        parent = tree_index[current.parent_id]
        ancestors.append(parent)
        current = parent

    return ancestors
```

**Top-Down Traversal:**
```python
def get_all_leaves(span_id: uint64) -> list[NodeMetadata]:
    """Recursively get all L0 leaves under this node."""
    node = tree_index[span_id]

    if node.level == 0:
        return [node]

    leaves = []
    for child_id in node.child_ids:
        if child_id is not None:
            leaves.extend(get_all_leaves(child_id))

    return leaves
```

### Range Queries

**Find Spans Covering Token Range:**
```python
def find_spans_in_range(start: uint64, end: uint64, level: uint8) -> list[NodeMetadata]:
    """Find all spans at given level that overlap [start, end)."""
    spans = []

    for span_id, meta in tree_index.items():
        if meta.level != level:
            continue

        # Check overlap
        if meta.start_token < end and meta.end_token > start:
            spans.append(meta)

    return sorted(spans, key=lambda x: x.start_token)
```

### Gist Update Propagation

**Refresh After GistNet Retraining:**
```python
def refresh_gists(new_gistnet_version: uint16):
    """Update all L1+ gists with new GistNet checkpoint."""

    # Start with L1 nodes
    for span_id, meta in tree_index.items():
        if meta.level == 1:
            # Fetch L0 children
            children_data = [load_payload(cid) for cid in meta.child_ids]

            # Recompute gist
            new_gist = gistnet.compress(children_data)

            # Overwrite at same offset
            write_payload(meta.data_offset, new_gist)

            # Update version
            meta.gist_version = new_gistnet_version

    # Propagate to L2+
    propagate_upward(level=2, new_gistnet_version)
```

### Access Tracking

**Increment on Load:**
```python
def assemble_working_context(focus_map: dict[uint64, float]) -> WorkingContext:
    """Build working context and track access."""
    wc = WorkingContext()

    for span_id in focus_map.keys():
        meta = tree_index[span_id]

        # Increment telemetry
        meta.access_count += 1
        meta.last_access_time = current_time()

        # Load payload
        payload = load_payload(meta.data_offset)
        wc.add(span_id, payload)

    return wc
```

### Pruning Candidate Selection

**Identify Low-Value Spans:**
```python
def find_pruning_candidates() -> list[uint64]:
    """Find spans that are old and rarely accessed."""
    candidates = []

    for span_id, meta in tree_index.items():
        age_days = (current_time() - meta.timestamp) / 86400

        # Old + cold = prune candidate
        if age_days > 30 and meta.access_count < 5:
            candidates.append(span_id)

    return candidates
```

---

## Versioning & Evolution

### GistNet Checkpoint Tracking

**Problem:** During alternating optimization, [[GistNet]] is periodically retrained. Existing gists in the tree may become stale.

**Solution:** Use `gist_version` to track which checkpoint generated each gist.

**Strategies:**

#### Eager Refresh
- Update all gists immediately when new checkpoint is deployed
- Ensures consistency but requires full tree scan
- Best for small trees or when quality is critical

#### Lazy Refresh
- Update gists on-demand when accessed
- Amortizes cost across usage patterns
- Risk: inconsistent gist versions within tree

#### Versioned Coexistence
- Store multiple gist versions per span
- Let [[Focus Allocator]] choose which version to use
- Memory overhead: 2×-3× payload storage

**POC Approach:**
- Freeze gists to initial checkpoint (no retraining during demo)
- Post-POC: implement lazy refresh with `gist_version` tracking

### Schema Evolution

**Adding New Fields:**
```python
# Before
class NodeMetadata:
    span_id: uint64
    level: uint8
    # ... existing fields

# After
class NodeMetadata:
    span_id: uint64
    level: uint8
    # ... existing fields

    # New fields with defaults
    provenance_score: float = 0.0  # For MegaCuration
    soft_delete: bool = False       # For pruning
```

**Migration Strategy:**
- Add new fields with sensible defaults
- Backfill asynchronously for existing nodes
- Version the index schema (serialize with version tag)

---

## Examples

### Example 1: L0 Token Node

```python
NodeMetadata(
    span_id=1000,
    level=0,
    start_token=64,
    end_token=65,           # Covers 1 token
    parent_id=1032,         # Belongs to L1 [64:96)
    child_ids=[null] * 32,  # Leaf node
    data_offset=128,        # 64th token × 2 bytes
    timestamp=1705318200,
    access_count=5,
    gist_version=0          # N/A for L0
)
```

### Example 2: L1 Gist Node

```python
NodeMetadata(
    span_id=1032,
    level=1,
    start_token=64,
    end_token=96,           # Covers 32 tokens
    parent_id=2001,         # Belongs to L2 [0:1024)
    child_ids=[
        1000, 1001, 1002, ..., 1031  # 32 L0 children
    ],
    data_offset=4096,       # 2nd L1 gist × 2048 bytes
    timestamp=1705318232,   # 32 seconds after L0
    access_count=12,
    gist_version=1          # GistNet-v1
)
```

### Example 3: L2 Root Node

```python
NodeMetadata(
    span_id=2001,
    level=2,
    start_token=0,
    end_token=1024,         # Covers 1024 tokens
    parent_id=null,         # Root node (no parent)
    child_ids=[
        1000, 1032, 1064, ..., 1992  # 32 L1 children
    ],
    data_offset=0,          # First L2 gist
    timestamp=1705319000,
    access_count=50,        # Frequently accessed
    gist_version=2          # Updated to GistNet-v2
)
```

---

## POC Implementation Notes

**Current Constraints:**
- All metadata lives in RAM (no persistence between runs)
- Simple monotonic `span_id` generation
- `access_count` and `timestamp` collected but not used for decisions
- `gist_version` fixed at 1 (no retraining)
- `child_ids` stored as Python lists (not fixed-size arrays)

**Post-POC:**
- Serialize index to disk for crash recovery
- Implement deterministic `span_id` from (level, start_token)
- Add LRU cache for hot metadata lookups
- Enable metadata-driven pruning (access count + timestamp)
- Support gist versioning and gradual refresh

---

## Related Pages

### Parent/Overview Pages
- [[MegaContext Tree]] – The hierarchical tree structure that these metadata nodes describe and navigate
- [[Architecture Details]] – Complete system architecture showing metadata's role in the system
- [[POC Architecture]] – System-wide design and how metadata fits into component interactions

### Sibling Detail Pages
- [[Tree Operations]] – How metadata is created, updated, and maintained during tree operations
- [[Storage Format]] – Binary payload layout and deterministic offset calculation strategies
- [[GistNet Architecture Details]] – Architecture of the model that generates gist payloads

### Related System Components
- [[Working Context]] – Primary consumer of metadata; uses it to select which spans to load
- [[Working Context Assembly]] – How metadata drives working context construction
- [[Focus Allocator]] – Queries metadata for expand/collapse decisions based on span positions
- [[Focus Allocator Strategies]] – Specific algorithms that use metadata for focus decisions
- [[GistNet]] – Generates the gist payloads that metadata points to via data_offset
- [[LensNet]] – Uses metadata to locate and load L1 gists for conditioning

### Implementation Guides
- [[POC Implementation]] – Practical implementation of metadata structures in the proof-of-concept
- [[Training & Operations]] – How metadata versioning works during alternating optimization
- [[Alternating Optimization]] – Metadata updates during GistNet retraining cycles

### Operations & Optimization
- [[MegaCuration]] – Uses telemetry fields (access_count, timestamp) for pruning decisions
- [[Telemetry]] – How metadata tracks usage patterns for optimization
- [[GistNet Training]] – How gist_version metadata tracks training checkpoints

### Related Concepts
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)]] – Level hierarchy tracked in metadata
- [[Glossary#Contiguity Invariant]] – Block alignment enforced through metadata positioning
- [[substitutability]] – Design principle that metadata versioning helps maintain

---

## Summary

Node metadata is the **nervous system** of the [[MegaContext Tree]]—lightweight, RAM-resident information that enables fast navigation, deterministic storage mapping, and telemetry-driven optimization. By separating metadata (structural) from payloads (content), the system achieves O(1) lookups and scalable storage for potentially billions of tokens while keeping the index small enough to fit in RAM.
