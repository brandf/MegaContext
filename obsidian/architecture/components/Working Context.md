---
tags:
  - module
summary: The fixed-size GPU window that the frozen base LLM actually sees, assembled dynamically from the MegaContext Tree with mixed-LOD entries that are refocused continuously to maintain relevance within a strict token budget.
---
![[WorkingContext.png]]

The Working Context is the **active memory** that the base LLM consumes at inference time. Unlike the unbounded [[MegaContext Tree]], the working context maintains a **fixed token budget** (W_max) by dynamically mixing raw tokens and compressed gists based on relevance.

---

- **Purpose:** Provide the base LLM with a contiguous, budget-constrained context window.
- **Budget:** Fixed at W_max tokens (8k–32k in POC); sum of entry costs must stay ≤ W_max.
- **Content:** Mixed levels of detail—raw L0 tokens where focus is needed, L1/L2 gists elsewhere.
- **Assembly:** Drawn from contiguous spans in the [[MegaContext Tree]], respecting temporal ordering.
- **Dynamics:** [[LensNet]] scores entries; [[Focus Allocator]] expands/collapses to adjust detail level.
- **Invariants:** No gaps, no overlaps, respects block alignment, maintains RoPE consistency.

---
## Details

### What is the Working Context?

The Working Context is the **fixed-size attention window** that sits in GPU memory and gets fed directly into the frozen base model for inference. While the [[MegaContext Tree]] can grow unboundedly on disk (or RAM in the POC), the working context must fit within a strict token budget.

Think of it as the **"spotlight"** that illuminates different parts of your memory at different resolutions:
- **High resolution (L0):** Raw tokens for the most relevant parts
- **Medium resolution (L1):** 32:1 gists for moderately relevant regions
- **Low resolution (L2):** 1024:1 gists for distant or less important context

This spotlight is **continuously refocused** by [[LensNet]] and [[Focus Allocator]] as new information arrives and priorities shift.

---

### Budget Constraints

#### W_max: The Token Budget

The working context is governed by a **strict token budget** called W_max:

| Parameter | POC Value | Future Targets | Meaning |
|-----------|-----------|----------------|---------|
| W_max | 8,192 | 16k–32k | Maximum token-equivalent cost allowed in working context |

**Token-equivalent costs** per entry:
- **L0 block (32 tokens):** costs 32 tokens
- **L1 gist:** costs 1 token (replaces 32 L0 tokens, saves 31)
- **L2 gist:** costs 1 token (replaces 32 L1 gists / 1024 L0 tokens, saves 1023)

#### Budget Enforcement

The [[Focus Allocator]] maintains the invariant:

```
sum(entry_costs) ≤ W_max
```

Every time an entry is **expanded** (L1→L0 or L2→L1):
- Token cost increases
- Requires corresponding **collapses** elsewhere to stay within budget

Every time an entry is **collapsed** (L0→L1 or L1→L2):
- Token cost decreases
- Frees budget for other entries to expand

#### Example Budget Calculation

Consider a working context with 8k token budget:

```
Entries:
- 100 × L0 blocks (32 tokens each) = 3,200 tokens
- 120 × L1 gists (1 token each)   =   120 tokens
- 50 × L2 gists (1 token each)    =    50 tokens
                          Total = 3,370 tokens ≤ 8,192 ✓
```

If [[LensNet]] wants to expand one L1 gist to L0 (cost: +31 tokens), the allocator must first collapse at least one L0 block to L1 (savings: -31 tokens) to maintain the budget.

---

### Entry Types & Tiling

#### Working Context Entries

Each entry in the working context represents a **contiguous block** from the [[MegaContext Tree]]:

| Entry Type | Content | Token Cost | Coverage |
|------------|---------|------------|----------|
| **L0 block** | 32 raw token embeddings | 32 | 32 tokens |
| **L1 gist** | Single gist vector from [[GistNet]] | 1 | 32 L0 tokens |
| **L2 gist** | Single gist vector from [[GistNet]] | 1 | 1,024 L0 tokens |

#### Contiguous Tiling

The working context maintains **perfect temporal contiguity**—entries tile the MegaContext history without gaps or overlaps:

```
Timeline: [0 ─────────────────────────────────────────────────── T]

Working Context (example):
[L0: 0-32] [L1: 32-64] [L1: 64-96] [L0: 96-128] [L2: 128-1152] ...
```

**Key properties:**
- Every position in the covered timeline appears in exactly one entry
- Entries appear in chronological order (older → newer)
- No gaps: next entry starts where previous one ends
- No overlaps: entries don't share any tokens

This **contiguity invariant** (see [[Glossary#Contiguity Invariant]]) is fundamental—it ensures:
- The base model sees a coherent narrative flow
- RoPE positional encodings remain consistent
- Focus changes don't introduce discontinuities

---

### Assembly from MegaContext Tree

The working context is not a separate data structure—it's a **dynamic view** into the [[MegaContext Tree]]:

#### Assembly Process

1. **Select span:** Choose which portion of the MegaContext timeline to cover (typically the most recent T tokens)
2. **Choose LODs:** For each position in that span, decide whether to use L0 tokens, L1 gist, or L2 gist
3. **Fetch data:** Retrieve the selected nodes from the tree's storage (`{L0,L1,L2}.ctx` files)
4. **Materialize embeddings:** Convert token IDs to embeddings (for L0) or use gist vectors directly (for L1/L2)
5. **Concatenate:** Stack all entries into a single contiguous tensor `[N_entries, d]`

#### Example Assembly

```python
def assemble_working_context(tree, focus_decisions):
    """
    Args:
        tree: MegaContext Tree with L0/L1/L2 nodes
        focus_decisions: List of (span, level) tuples

    Returns:
        embeddings: Tensor[N, d] for base model
    """
    entries = []

    for span, level in focus_decisions:
        if level == 0:  # L0: raw tokens
            token_ids = tree.get_l0_block(span)
            embs = embed_tokens(token_ids)  # [32, d]
            entries.append(embs)
        elif level == 1:  # L1: gist
            gist = tree.get_l1_gist(span)  # [1, d]
            entries.append(gist)
        elif level == 2:  # L2: gist
            gist = tree.get_l2_gist(span)  # [1, d]
            entries.append(gist)

    return torch.cat(entries, dim=0)  # [N_total, d]
```

---

### Interaction with Base LLM

#### What the Base Model Sees

From the base model's perspective, the working context is **just another context window**—it doesn't know that some embeddings are gists rather than raw tokens. This is possible because:

1. **Dimensionality match:** Gists live in the same embedding space as tokens (dimension `d`)
2. **RoPE compatibility:** Gists are positioned at the central token index of their span for consistent positional encoding
3. **Substitutability:** [[GistNet]] is trained so gists produce similar hidden states to their original tokens (low ΔNLL@H)

#### Forward Pass

```python
# Standard base model forward pass
outputs = base_model(
    inputs_embeds=working_context,  # [N, d] - mixed tokens & gists
    attention_mask=attention_mask,   # [N] - all entries attend causally
    position_ids=position_ids        # [N] - absolute token indices for RoPE
)

logits = outputs.logits  # [N, vocab_size]
```

**Key detail:** Even though entry 50 might be an L1 gist representing tokens 1600-1632, its `position_ids` entry is 1616 (the central token index), ensuring RoPE sees consistent phase information.

---

### The Refocus Process

The working context is not static—it **evolves continuously** as the conversation progresses and [[LensNet]] identifies new focus priorities.

#### Refocus Cadence

Refocusing happens **once per K tokens** (K=32 in POC):

1. **Decode K tokens:** Generate 32 new tokens using the current working context
2. **Ingest:** Add new tokens to [[MegaContext Tree]], potentially creating new gist nodes
3. **Score:** [[LensNet]] evaluates all working context entries and emits signed focus scores
4. **Adjust:** [[Focus Allocator]] applies up to N_diff expand/collapse operations (default 4)
5. **Repeat:** Use the updated working context for the next K tokens

#### Example Refocus Sequence

```
Step T=0:
Working Context: [L0: 0-32] [L1: 32-64] [L1: 64-96] [L0: 96-128]
                  │          │           │           │
                  └──────────┴───────────┴───────────┘
                         Token cost: 32+1+1+32 = 66

Step T=32 (after LensNet scoring):
LensNet scores:  [+0.1]    [-0.5]      [+0.7]      [+0.1]
                     ↓         ↓           ↓           ↓
                  neutral  collapse!    expand!     neutral

Step T=32 (after Focus Allocator):
Working Context: [L0: 0-32] [L2: 32-1056] [L0: 64-96] [L0: 96-128]
                  │          │             │           │
                  └──────────┴─────────────┴───────────┘
                         Token cost: 32+1+32+32 = 97

Changes:
- Entry at 32-64 collapsed to L2 (freed 31 tokens)
- Entry at 64-96 expanded to L0 (used 31 tokens)
- Net budget change: 0 tokens ✓
```

#### Why Refocus?

**Dynamic relevance:** What was important 1000 tokens ago may no longer matter. Refocusing lets the system:
- **Zoom in** on newly relevant regions (expand gists to tokens)
- **Zoom out** on now-irrelevant regions (collapse tokens to gists)
- **Maintain budget** by balancing detail across the window
- **Avoid distractors** by compressing irrelevant details

This is fundamentally different from fixed-window LLMs (which forget old context entirely) or RAG (which appends new context but never removes old distractors).

---

### Invariants

The working context must maintain several critical invariants at all times:

#### 1. Budget Invariant
```
sum(entry_costs) ≤ W_max
```
Total token-equivalent cost never exceeds the budget.

#### 2. Contiguity Invariant
```
entry[i].end_token == entry[i+1].start_token
```
No gaps or overlaps in the temporal coverage.

#### 3. Block Alignment Invariant
```
entry.start_token % K == 0
entry.end_token % K == 0
```
All boundaries align with K-token blocks (K=32 in POC).

#### 4. Level Consistency Invariant
```
entry covers span [s, e) → level ∈ {0, 1, 2} is legal for that span size
```
- L0 blocks cover exactly 32 tokens
- L1 gists cover exactly 32 tokens (one L0 block)
- L2 gists cover exactly 1024 tokens (32 L0 blocks)

#### 5. RoPE Invariant
```
position_ids[entry] = entry.start_token + K/2  (for gists)
position_ids[entry] = [entry.start_token, ..., entry.end_token-1]  (for L0)
```
Gists use the central position index; L0 blocks use actual token positions.

Violating any of these invariants could cause:
- Budget explosion (GPU OOM)
- Discontinuous narrative flow
- RoPE misalignment
- Invalid expand/collapse operations

---

### Relationship to MegaContext Tree

The working context and [[MegaContext Tree]] are complementary:

| Aspect | [[MegaContext Tree]] | Working Context |
|--------|---------------------|-----------------|
| **Storage** | Persistent (disk/RAM) | Ephemeral (GPU) |
| **Scope** | Complete history | Recent window |
| **Size** | Unbounded | Fixed (W_max) |
| **Content** | All LODs stored | Selective LODs |
| **Mutability** | Append-only | Dynamic refocus |
| **Role** | Long-term memory | Active attention |

**Analogy:** The MegaContext Tree is your brain's long-term memory (everything you've ever learned). The working context is your conscious attention right now (the small subset you're actively thinking about).

---

### Role in the System

The Working Context is the **central coordination point** for all components:

- **For the base model:** The only context it ever sees—a seemingly normal attention window
- **For [[LensNet]]:** Input to analyze and score for focus adjustments
- **For [[Focus Allocator]]:** The budget-constrained space where it applies expand/collapse decisions
- **For [[GistNet]]:** Provides examples of which gists are actually used (for on-policy training)
- **For [[Runtime Loop]]:** The working state that persists across decode steps

---

### POC Implementation Notes

In the proof-of-concept:
- **Size:** W_max = 8,192 tokens (configurable via YAML)
- **Update frequency:** Refocus every K=32 tokens
- **Initial assembly:** Start with most recent tokens/gists from the tree
- **No streaming:** Entire working context resides in GPU memory (no paging yet)
- **Simple heuristics:** Initial focus policy may use recency bias before LensNet is trained

See [[POC Scope]] for constraints and [[POC Architecture]] for implementation details.

---

### Future Enhancements

Post-POC improvements:
- **Larger budgets:** Scale to W_max = 32k–64k tokens with efficient attention (FlashAttention, ring attention)
- **Streaming assembly:** Memory-map working context entries for lower GPU memory usage
- **Multi-head contexts:** Multiple working contexts with different focus policies for diverse tasks
- **Attention biasing:** Learn task-specific attention masks that guide the base model's focus within the working context
- **KV-cache reuse:** Intelligently preserve KV-cache across refocus steps when most entries don't change

---

## Summary

The Working Context is the **active window** that bridges MegaContext's infinite memory with the finite reality of GPU resources. By dynamically blending raw tokens (high detail) and hierarchical gists (compressed summaries) within a strict budget, it gives the frozen base model effectively infinite context while keeping per-step compute constant. Every K tokens, [[LensNet]] re-evaluates relevance and the [[Focus Allocator]] adjusts the spotlight—ensuring the most important information always appears at the right level of detail. The working context is not merely a view into the [[MegaContext Tree]]—it's the **lens** through which the model perceives its unbounded memory.
