---
tags:
  - components
summary: How the Working Context is assembled from the MegaContext Tree - selecting spans, choosing LODs, fetching data, materializing embeddings, and concatenating into the contiguous tensor the base model sees.
---
# Working Context Assembly

This document details how the [[Working Context]] is dynamically assembled from the [[MegaContext Tree]] as a contiguous tensor that the frozen base model consumes.

---

## Overview

The [[Working Context]] is not a separate data structure—it's a **dynamic view** into the [[MegaContext Tree]]:

```
MegaContext Tree (persistent storage)
       ↓
  Focus decisions (which spans at what LODs)
       ↓
Working Context (materialized tensor for GPU)
```

Assembly happens:
- **Initially:** When starting a conversation/session
- **After refocus:** When [[Focus Allocator]] changes LOD levels
- **After ingest:** When new tokens are added to the tree

---

## Assembly Steps

### 1. Select Span

Choose which portion of the [[MegaContext Tree]] timeline to cover.

**Strategy (POC):**
- **Recency-based:** Include the most recent T tokens
- **Query-aware:** If there's a specific query, may include non-contiguous relevant spans (future)

**Example:**
```python
# Simple recency strategy
total_tokens_in_tree = 1_000_000
W_max = 8_192

# Cover most recent tokens
start_token = max(0, total_tokens_in_tree - estimate_covered_range(W_max))
end_token = total_tokens_in_tree
```

**Considerations:**
- Can't literally include all 1M tokens (budget: 8k)
- Most recent tokens typically at LOD0 (full detail)
- Older tokens typically at LOD1/LOD2 (compressed)

---

### 2. Choose LODs

For each position in the selected span, decide detail level.

**Inputs:**
- Previous [[Working Context]] state (if exists)
- [[LensNet]] focus scores (if available)
- Initialization heuristics (if first assembly)

**Outputs:**
- List of `(span, level)` tuples:
  ```python
  [
    (Span(0, 32), 0),      # LOD0 block
    (Span(32, 64), 1),     # LOD1 gist
    (Span(64, 96), 1),     # LOD1 gist
    (Span(96, 1120), 2),   # LOD2 gist (covers 1024 tokens)
    ...
  ]
  ```

**Initial heuristics (before [[LensNet]] is trained):**
```python
def initial_lod_selection(start_token, end_token, W_max):
    """
    Simple recency-based LOD selection for cold start.
    """
    decisions = []

    # Most recent 256 tokens at LOD0 (full detail)
    recent_start = max(start_token, end_token - 256)
    for block_start in range(recent_start, end_token, 32):
        decisions.append((Span(block_start, block_start + 32), 0))

    # Next 2048 tokens at LOD1 (medium compression)
    mid_start = max(start_token, recent_start - 2048)
    for block_start in range(mid_start, recent_start, 32):
        decisions.append((Span(block_start, block_start + 32), 1))

    # Remaining at LOD2 (heavy compression)
    for block_start in range(start_token, mid_start, 1024):
        decisions.append((Span(block_start, block_start + 1024), 2))

    return decisions
```

---

### 3. Fetch Data

Retrieve the selected nodes from [[MegaContext Tree]] storage.

**From files:**
```python
class TreeStorage:
    def get_l0_block(self, block_id: int) -> np.ndarray:
        """Fetch 32 token IDs from LOD0.ctx"""
        offset = 64 + (block_id * 32 * 4)  # Header + tokens
        self.l0_file.seek(offset)
        return np.fromfile(self.l0_file, dtype=np.uint32, count=32)

    def get_l1_gist(self, gist_id: int) -> torch.Tensor:
        """Fetch LOD1 gist vector from LOD1.ctx"""
        offset = 64 + (gist_id * self.embedding_dim * 2)  # fp16
        self.l1_file.seek(offset)
        data = np.fromfile(self.l1_file, dtype=np.float16, count=self.embedding_dim)
        return torch.from_numpy(data)

    def get_l2_gist(self, gist_id: int) -> torch.Tensor:
        """Fetch LOD2 gist vector from LOD2.ctx"""
        offset = 64 + (gist_id * self.embedding_dim * 2)  # fp16
        self.l2_file.seek(offset)
        data = np.fromfile(self.l2_file, dtype=np.float16, count=self.embedding_dim)
        return torch.from_numpy(data)
```

See [[Storage Format]] for binary layout details.

---

### 4. Materialize Embeddings

Convert fetched data into embeddings in the model's embedding space.

#### For LOD0 Tokens

```python
def materialize_l0_embeddings(token_ids: np.ndarray,
                              embedding_layer: nn.Embedding) -> torch.Tensor:
    """
    Convert token IDs to embeddings using base model's embedding layer.

    Args:
        token_ids: [32] uint32 array
        embedding_layer: Base model's token embedding layer

    Returns:
        embeddings: [32, d] tensor
    """
    token_tensor = torch.from_numpy(token_ids).long()
    return embedding_layer(token_tensor)  # [32, d]
```

#### For LOD1/LOD2 Gists

```python
def materialize_gist_embeddings(gist_vector: torch.Tensor) -> torch.Tensor:
    """
    Gists are already in embedding space - just reshape.

    Args:
        gist_vector: [d] tensor from storage

    Returns:
        embeddings: [1, d] tensor (batch dimension for concatenation)
    """
    return gist_vector.unsqueeze(0)  # [1, d]
```

**Key insight:** [[GistNet]] outputs live in the same space as token embeddings, so no transformation needed.

---

### 5. Concatenate

Stack all entries into a single contiguous tensor.

```python
def assemble_working_context(tree: TreeStorage,
                             focus_decisions: List[Tuple[Span, int]],
                             embedding_layer: nn.Embedding) -> torch.Tensor:
    """
    Assemble Working Context from tree and focus decisions.

    Args:
        tree: MegaContext Tree storage interface
        focus_decisions: List of (span, level) tuples
        embedding_layer: Base model's embedding layer

    Returns:
        context: [N_total, d] tensor for base model
    """
    entries = []

    for span, level in focus_decisions:
        if level == 0:  # LOD0: raw tokens
            block_id = span.start // 32
            token_ids = tree.get_l0_block(block_id)
            embs = materialize_l0_embeddings(token_ids, embedding_layer)  # [32, d]
            entries.append(embs)

        elif level == 1:  # LOD1: gist
            gist_id = span.start // 32
            gist = tree.get_l1_gist(gist_id)  # [d]
            embs = materialize_gist_embeddings(gist)  # [1, d]
            entries.append(embs)

        elif level == 2:  # LOD2: gist
            gist_id = span.start // 1024
            gist = tree.get_l2_gist(gist_id)  # [d]
            embs = materialize_gist_embeddings(gist)  # [1, d]
            entries.append(embs)

    return torch.cat(entries, dim=0)  # [N_total, d]
```

---

## Position IDs for RoPE

The base model needs position IDs for [[Glossary#RoPE (Rotary Position Embedding)|RoPE]].

### Rules

- **LOD0 blocks:** Use actual sequential token positions
- **LOD1/LOD2 gists:** Use central position of their span

```python
def compute_position_ids(focus_decisions: List[Tuple[Span, int]]) -> torch.Tensor:
    """
    Compute position IDs for RoPE encoding.

    Returns:
        position_ids: [N_total] tensor of absolute positions
    """
    position_ids = []

    for span, level in focus_decisions:
        if level == 0:  # LOD0: sequential positions
            positions = list(range(span.start, span.end))
            position_ids.extend(positions)  # [32] positions

        else:  # LOD1/LOD2: central position
            central_pos = span.start + (span.end - span.start) // 2
            position_ids.append(central_pos)  # [1] position

    return torch.tensor(position_ids, dtype=torch.long)
```

**Why central position for gists?**
- Minimizes maximum distance from [[gist]] to any token it represents
- For span [64, 96), central position is 80, at most 16 tokens from any edge
- Edge positions (64 or 95) would be up to 31 tokens away

See [[Invariants#RoPE Invariant]] for details.

---

## Attention Mask

Standard causal attention mask—all entries attend to previous entries.

```python
def compute_attention_mask(N: int) -> torch.Tensor:
    """
    Standard causal mask: entry i can attend to entries 0..i

    Returns:
        mask: [N, N] bool tensor, True = attend, False = mask
    """
    return torch.tril(torch.ones(N, N, dtype=torch.bool))
```

**No special handling for gists:** They participate in causal attention like tokens.

---

## Complete Assembly Example

```python
class WorkingContextManager:
    def __init__(self, tree: TreeStorage, base_model):
        self.tree = tree
        self.embedding_layer = base_model.get_input_embeddings()
        self.W_max = 8192
        self.current_focus = []

    def assemble(self, focus_decisions: List[Tuple[Span, int]]):
        """
        Assemble Working Context from focus decisions.
        """
        # Validate budget
        total_cost = sum(32 if level == 0 else 1
                        for _, level in focus_decisions)
        assert total_cost <= self.W_max, f"Budget violation: {total_cost} > {self.W_max}"

        # Materialize embeddings
        embeddings = assemble_working_context(
            self.tree,
            focus_decisions,
            self.embedding_layer
        )

        # Compute position IDs
        position_ids = compute_position_ids(focus_decisions)

        # Compute attention mask
        N = embeddings.shape[0]
        attention_mask = compute_attention_mask(N)

        # Store state
        self.current_focus = focus_decisions
        self.embeddings = embeddings
        self.position_ids = position_ids
        self.attention_mask = attention_mask

        return embeddings, position_ids, attention_mask

    def forward_pass(self, base_model):
        """
        Pass assembled Working Context through base model.
        """
        outputs = base_model(
            inputs_embeds=self.embeddings,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids
        )
        return outputs.logits
```

---

## Optimization Strategies

### Incremental Assembly

When only a few entries change during refocus, avoid full reassembly:

```python
def incremental_update(old_context, changes: List[Tuple[int, Span, int]]):
    """
    Update only changed entries.

    Args:
        old_context: Previous Working Context state
        changes: List of (index, span, new_level) for changed entries
    """
    new_embeddings = old_context.embeddings.clone()

    for idx, span, new_level in changes:
        if new_level == 0:  # Expanded to LOD0
            new_embs = fetch_and_materialize_l0(span)  # [32, d]
        else:  # Collapsed to LOD1/LOD2
            new_embs = fetch_and_materialize_gist(span, new_level)  # [1, d]

        # Splice into context
        new_embeddings = splice_embeddings(new_embeddings, idx, new_embs)

    return new_embeddings
```

### Caching

Cache frequently accessed [[gist|gists]]:

```python
class GistCache:
    def __init__(self, capacity=1000):
        self.cache = {}  # {(level, gist_id): tensor}
        self.lru = []

    def get_gist(self, level: int, gist_id: int, tree: TreeStorage):
        key = (level, gist_id)

        if key in self.cache:
            # Cache hit
            self.lru.remove(key)
            self.lru.append(key)
            return self.cache[key]

        # Cache miss - fetch from tree
        if level == 1:
            gist = tree.get_l1_gist(gist_id)
        else:
            gist = tree.get_l2_gist(gist_id)

        # Update cache
        self.cache[key] = gist
        self.lru.append(key)

        # Evict if over capacity
        if len(self.cache) > self.capacity:
            evict_key = self.lru.pop(0)
            del self.cache[evict_key]

        return gist
```

---

## [[POC Implementation|POC]] Notes

**Simplifications:**
- **Synchronous assembly:** No background workers or prefetching
- **RAM-resident:** Tree in memory, no disk I/O latency
- **No caching:** Fetch every time (sufficient for moderate contexts)
- **Recency-only:** Initial LOD selection uses simple recency heuristic

**Future enhancements:**
- **Async assembly:** Prefetch likely-to-be-expanded [[gist|gists]]
- **Memory-mapped I/O:** Zero-copy access to disk-backed tree
- **Smart caching:** LRU or learned caching policy
- **Query-aware span selection:** Non-contiguous relevant regions

---

## Related Pages

### Core Process
- [[Working Context]] — Overview, role in system, and relationship to MegaContext Tree
- [[Working Context Refocusing]] — How the assembled context evolves through continuous LOD adjustments
- [[MegaContext Tree]] — Source data structure providing nodes at all LOD levels

### Assembly Dependencies
- [[LensNet]] — Provides focus scores that determine which LODs to select during assembly
- [[Focus Allocator]] — Executes expand/collapse decisions that trigger reassembly
- [[GistNet]] — Produces the gist embeddings materialized during assembly

### Technical Details
- [[Storage Format]] — Binary layout for fetching LOD0 tokens and LOD1/LOD2 gists from disk/RAM
- [[Node Metadata]] — Metadata fields used to locate and validate tree nodes during fetch
- [[Tree Operations]] — APIs for reading blocks and gists from the MegaContext Tree

### System Constraints
- [[Invariants]] — Budget, contiguity, block alignment, and RoPE constraints that assembly enforces
- [[POC Implementation]] — POC-specific assembly parameters (W_max, caching, synchronous fetch)
- [[Runtime Loop]] — When assembly happens in the decode-ingest-score-refocus cycle

### Performance
- [[Performance Sketch]] — Assembly latency, caching strategies, and optimization opportunities
- [[Telemetry]] — Metrics tracking assembly frequency, cache hit rates, and LOD distributions

---

*Assembly is the bridge between persistent [[MegaContext Tree]] storage and active [[Working Context]] attention—materializing the exact view the base model needs to see.*
