---
tags:
  - architecture
  - technical
summary: System-wide invariants that MegaContext maintains across all operations, including budget constraints, contiguity requirements, block alignment, and positional encoding consistency.
---

# MegaContext System Invariants

This document defines the fundamental invariants that must hold throughout MegaContext's operation. These constraints ensure correctness, predictability, and maintain the [[substitutability]] property that allows [[gist|gists]] to replace tokens seamlessly.

---

## Core Invariants

### 1. Budget Invariant

**Definition:**
```
sum(entry_costs in Working Context) ≤ W_max
```

**What it means:**
- The total token-equivalent cost of all entries in the [[Working Context]] must never exceed the budget [[Glossary#W_max (Token Budget)|W_max]]
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0 blocks]] cost 32 tokens each
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] and [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L2]] [[gist|gists]] cost 1 token each

**Why it matters:**
- Ensures constant GPU memory usage regardless of total history size
- Prevents out-of-memory errors during inference
- Makes performance predictable and scalable

**Enforcement:**
- [[Focus Allocator]] checks budget before every [[expand]] operation
- Every [[expand]] must be balanced by corresponding [[collapse|collapses]]
- System rejects operations that would violate the budget

**Example:**
```python
# Valid working context
entries = [
    L0_block,  # costs 32 tokens
    L0_block,  # costs 32 tokens
    L1_gist,   # costs 1 token
    L2_gist,   # costs 1 token
]
total_cost = 32 + 32 + 1 + 1 = 66 tokens
W_max = 8192
assert total_cost <= W_max  # ✓ Valid
```

---

### 2. Contiguity Invariant

**Definition:**
```
entry[i].end_token == entry[i+1].start_token  (for all consecutive entries)
```

**What it means:**
- Entries in the [[Working Context]] must tile the timeline without gaps or overlaps
- Every token position in the covered range appears in exactly one entry
- Entries appear in chronological order (older → newer)

**Why it matters:**
- Ensures the base model sees a coherent narrative flow
- Maintains [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] positional encoding consistency
- Prevents discontinuous jumps that would confuse the model
- Allows the model to understand temporal relationships

**Enforcement:**
- [[Focus Allocator]] only performs block-aligned swaps
- [[expand|Expand]]/[[collapse]] operations preserve temporal adjacency
- Tree assembly checks for contiguity before materializing [[Working Context]]

**Visual Example:**
```
✓ Valid (contiguous):
[L0: 0-32] [L1: 32-64] [L0: 64-96] [L2: 96-1120]
     └───────┘└────────┘└────────┘└──────────┘
     No gaps, perfect adjacency

✗ Invalid (gap):
[L0: 0-32] [L1: 32-64] [L0: 96-128]
     └───────┘└────────┘      └────────┘
                         ↑ GAP (tokens 64-96 missing)

✗ Invalid (overlap):
[L0: 0-32] [L1: 28-60] [L0: 60-92]
     └───────┘└────────┘└────────┘
          ↑ OVERLAP (tokens 28-32 appear twice)
```

---

### 3. Block Alignment Invariant

**Definition:**
```
entry.start_token % K == 0
entry.end_token % K == 0
```

Where [[Glossary#K (Block Size)|K]] = 32 in the [[POC Scope|POC]].

**What it means:**
- All entry boundaries must align with [[Glossary#K (Block Size)|K]]-token block boundaries
- No entry can start or end mid-block (e.g., at token 17 or token 50)
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0 blocks]] are exactly 32 tokens
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] [[gist|gists]] represent exactly 32 tokens (one L0 block)
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L2]] [[gist|gists]] represent exactly 1,024 tokens (32 L0 blocks)

**Why it matters:**
- Matches [[GistNet]] compression granularity (32→1)
- Enables clean [[expand]]/[[collapse]] operations without partial blocks
- Simplifies [[MegaContext Tree]] storage (deterministic offsets)
- Ensures [[gist|gists]] always represent complete, meaningful units

**Enforcement:**
- [[MegaContext Tree]] ingests tokens in 32-token batches
- [[Focus Allocator]] only swaps complete blocks
- [[GistNet]] only generates [[gist|gists]] for full 32-token spans

**Example:**
```python
# Valid entries
L0_block_1 = Entry(start=0, end=32)      # ✓ Aligned (0 % 32 == 0, 32 % 32 == 0)
L0_block_2 = Entry(start=32, end=64)     # ✓ Aligned
L1_gist = Entry(start=64, end=96)        # ✓ Aligned (represents 32 tokens)
L2_gist = Entry(start=96, end=1120)      # ✓ Aligned (represents 1024 tokens)

# Invalid entries
bad_entry = Entry(start=17, end=49)      # ✗ Misaligned (17 % 32 != 0)
```

---

### 4. Level Consistency Invariant

**Definition:**
```
entry covers span [s, e) → level ∈ {0, 1, 2} is legal for that span size
```

**What it means:**
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]] entries must cover exactly 32 tokens
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] entries must cover exactly 32 tokens (compressing one L0 block)
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L2]] entries must cover exactly 1,024 tokens (compressing 32 L0 blocks)
- Cannot have an L1 [[gist]] representing 64 tokens or an L2 [[gist]] representing 512 tokens

**Why it matters:**
- Ensures [[gist|gists]] accurately represent their source spans
- Maintains the hierarchical 32-ary tree structure
- Allows [[Focus Allocator]] to predict the effect of [[expand]]/[[collapse]] operations
- Simplifies [[MegaContext Tree]] navigation

**Enforcement:**
- [[GistNet]] only produces [[gist|gists]] from 32-child spans
- [[MegaContext Tree]] stores [[gist|gists]] at fixed levels corresponding to span size
- [[Focus Allocator]] checks level legality before operations

**Example:**
```python
# Valid level assignments
entry_1 = Entry(level=0, start=0, end=32)        # ✓ L0 covers 32 tokens
entry_2 = Entry(level=1, start=32, end=64)       # ✓ L1 covers 32 tokens
entry_3 = Entry(level=2, start=64, end=1088)     # ✓ L2 covers 1024 tokens

# Invalid level assignments
bad_1 = Entry(level=1, start=0, end=64)          # ✗ L1 can't cover 64 tokens
bad_2 = Entry(level=2, start=0, end=512)         # ✗ L2 can't cover 512 tokens
```

---

### 5. [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] Invariant

**Definition:**
```
For gists: position_id = start_token + (K / 2)
For L0 blocks: position_ids = [start_token, start_token+1, ..., end_token-1]
```

**What it means:**
- [[gist|Gists]] are positioned at the **central token index** of their span for [[Glossary#RoPE (Rotary Position Embedding)|RoPE]]
- [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0 tokens]] use their actual sequential positions
- This ensures [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] phase information remains consistent when swapping [[LOD|LODs]]

**Why it matters:**
- [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] encodes positional information as sinusoidal phase rotations
- Misaligned positions would break relative position relationships
- Using central position for [[gist|gists]] minimizes phase error
- Preserves the base model's ability to attend correctly across different [[LOD|LODs]]

**Enforcement:**
- [[Working Context]] assembly applies [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] position IDs during materialization
- [[gist|Gists]] inherit the central position of their span from the [[MegaContext Tree]]
- [[Focus Allocator]] preserves absolute token positions during swaps

**Example:**
```python
# L1 gist representing tokens [64, 96)
gist_position = 64 + (32 / 2) = 80  # Central position

# L0 block representing tokens [64, 96)
token_positions = [64, 65, 66, ..., 95]  # Sequential positions

# When swapping L1→L0 or L0→L1 for span [64, 96):
# - RoPE still sees positions centered around 80
# - Relative distances to other spans remain consistent
```

**Why Central Position?**
- Minimizes maximum distance from [[gist]] to any token it represents
- For a 32-token span [0, 32), central position 16 is at most 16 tokens from any original token
- Edge positions (0 or 31) would be up to 31 tokens away, increasing [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] phase error

---

## Derivative Invariants

These invariants follow from the core invariants above:

### 6. Tree-Context Consistency

**Definition:**
Every entry in the [[Working Context]] corresponds to a node in the [[MegaContext Tree]] at the appropriate level.

**Implications:**
- Cannot have an [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0]] entry for a span that doesn't exist in `L0.ctx`
- Cannot have an [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] [[gist]] that wasn't generated by [[GistNet]]
- [[expand|Expand]] operations require children to exist in the tree
- [[collapse|Collapse]] operations require parent [[gist]] to exist in the tree

---

### 7. Monotonic Timeline

**Definition:**
```
entry[i].start_token < entry[i+1].start_token  (for all i)
```

**Implications:**
- [[Working Context]] entries always proceed forward in time
- No time-travel or out-of-order entries
- Follows from Contiguity Invariant + Block Alignment Invariant

---

### 8. No Partial Swaps

**Definition:**
When [[expand|expanding]] or [[collapse|collapsing]], all 32 children/parent must be swapped atomically.

**Implications:**
- Cannot [[expand]] half of an [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] [[gist]] (e.g., only 16 of its 32 tokens)
- Cannot [[collapse]] just 10 of 32 [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0 blocks]] into an incomplete [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L1]] [[gist]]
- Follows from Block Alignment Invariant + Level Consistency Invariant

---

## Invariant Violations & Recovery

### Detection

**Budget Violation:**
```python
if sum(entry.cost for entry in working_context) > W_max:
    raise BudgetViolationError("Working context exceeds W_max")
```

**Contiguity Violation:**
```python
for i in range(len(working_context) - 1):
    if working_context[i].end != working_context[i+1].start:
        raise ContiguityViolationError(f"Gap between entry {i} and {i+1}")
```

**Block Alignment Violation:**
```python
for entry in working_context:
    if entry.start % K != 0 or entry.end % K != 0:
        raise AlignmentViolationError(f"Entry {entry} not K-aligned")
```

### Recovery Strategies

If an invariant is violated:

1. **Rollback:** Revert to previous valid [[Working Context]] state
2. **Recompute:** Rebuild [[Working Context]] from [[MegaContext Tree]]
3. **Logging:** Record violation for debugging and telemetry
4. **Graceful degradation:** Fall back to simpler focus policy (e.g., recency-only)

---

## Testing Invariants

See `tests/test_invariants.py` for comprehensive invariant checks run during [[POC Plan|POC]] development:

- `test_budget_invariant()` - Verifies budget never exceeds [[Glossary#W_max (Token Budget)|W_max]]
- `test_contiguity_invariant()` - Checks for gaps and overlaps
- `test_block_alignment()` - Validates [[Glossary#K (Block Size)|K]]-alignment
- `test_level_consistency()` - Ensures [[LOD|LODs]] match span sizes
- `test_rope_positions()` - Validates [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] position assignments

---

## Summary

These invariants are the **foundation of MegaContext's correctness**:

1. **Budget Invariant** → Constant memory usage
2. **Contiguity Invariant** → Coherent narrative flow
3. **Block Alignment Invariant** → Clean [[LOD]] swaps
4. **Level Consistency Invariant** → Hierarchical structure integrity
5. **[[Glossary#RoPE (Rotary Position Embedding)|RoPE]] Invariant** → Position encoding consistency

By maintaining these invariants, MegaContext ensures that [[gist|gists]] can seamlessly [[substitutability|substitute]] for tokens without breaking the [[frozen base model]]'s expectations.

See [[Architecture Details]] for how these invariants relate to the overall system design, and [[Focus Allocator]] for how they guide operational decisions.
