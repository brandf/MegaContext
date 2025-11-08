---
tags:
  - components
summary: The fixed-size GPU window that the frozen base LLM actually sees, assembled dynamically from the MegaContext Tree with mixed-LOD entries that are refocused continuously to maintain relevance within a strict token budget.
---
![[WorkingContext.png]]

The Working Context is the **active memory** that the base LLM consumes at inference time. Unlike the unbounded [[MegaContext Tree]], the working context maintains a **fixed token budget** ([[Glossary#W_max (Token Budget)|W_max]]) by dynamically mixing raw tokens and compressed gists based on relevance.

---

## What is the Working Context?

The Working Context is the **fixed-size attention window** that sits in GPU memory and gets fed directly into the frozen base model for inference. While the [[MegaContext Tree]] can grow unboundedly on disk (or RAM in the [[POC Architecture|POC]]), the working context must fit within a strict token budget.

Think of it as the **"spotlight"** that illuminates different parts of your memory at different resolutions:
- **High resolution (LOD0):** Raw tokens for the most relevant parts
- **Medium resolution (LOD1):** 32:1 gists for moderately relevant regions
- **Low resolution (LOD2):** 1024:1 gists for distant or less important context

This spotlight is **continuously refocused** by [[LensNet]] and [[Focus Allocator]] as new information arrives and priorities shift.

---

## Core Properties

### Fixed Budget
The working context operates under a **strict token budget** called [[Glossary#W_max (Token Budget)|W_max]]. In the [[POC Architecture|POC]], this is 8,192 tokens (configurable to 16k–32k). The system maintains the invariant that the sum of all entry costs never exceeds this budget.

For detailed budget mechanics, see [[Invariants#Budget Invariant]].

### Mixed Levels of Detail
The working context contains entries at different LODs based on relevance:

| Entry Type | Token Cost | Coverage |
|------------|------------|----------|
| **LOD0 block** | 32 tokens | 32 raw tokens |
| **LOD1 gist** | 1 token | 32 tokens (32:1 compression) |
| **LOD2 gist** | 1 token | 1,024 tokens (1024:1 compression) |

### Contiguous Tiling
Entries tile the [[MegaContext Tree]] timeline **without gaps or overlaps**, maintaining perfect temporal continuity:

```
Timeline: [0 ─────────────────────────────────────────────────── T]
Working Context: [LOD0: 0-32] [LOD1: 32-64] [LOD1: 64-96] [LOD0: 96-128] ...
```

This [[Glossary#Contiguity Invariant|contiguity]] ensures:
- Coherent narrative flow for the base model
- Consistent RoPE [1] positional encodings
- No discontinuities during focus changes

---

## Relationship to MegaContext Tree

The working context is not a separate data structure—it's a **dynamic view** into the [[MegaContext Tree]]:

| Aspect | [[MegaContext Tree]] | Working Context |
|--------|---------------------|-----------------|
| **Storage** | Persistent (disk/RAM) | Ephemeral (GPU) |
| **Scope** | Complete history | Recent window |
| **Size** | Unbounded | Fixed ([[Glossary#W_max (Token Budget)|W_max]]) |
| **Content** | All LODs stored | Selective LODs |
| **Mutability** | Append-only | Dynamic refocus |
| **Role** | Long-term memory | Active attention |

**Analogy:** The [[MegaContext Tree]] is your brain's long-term memory (everything you've ever learned). The working context is your conscious attention right now (the small subset you're actively thinking about).

---

## Assembly Process

The working context is assembled from the [[MegaContext Tree]] by:
1. Selecting a temporal span to cover
2. Choosing appropriate LOD for each region based on relevance (using sparse attention patterns [2])
3. Fetching data from the tree's storage
4. Materializing embeddings (tokens → embeddings, or gist vectors)
5. Concatenating into a single contiguous tensor

For full assembly details, see **[[Working Context Assembly]]**.

---

## Refocusing

The working context **evolves continuously** as the conversation progresses:
- **Cadence:** Refocus every K tokens (K=32 in [[POC Architecture|POC]])
- **Process:** [[LensNet]] scores entries, [[Focus Allocator]] applies expand/collapse operations
- **Budget:** Expansions and collapses are balanced to maintain the [[Glossary#W_max (Token Budget)|W_max]] constraint

**Why refocus?** Dynamic relevance means what was important 1000 tokens ago may no longer matter. Refocusing allows the system to zoom in on newly relevant regions and zoom out on now-irrelevant ones.

For full refocusing mechanics, see **[[Working Context Refocusing]]**.

---

## Base LLM Interaction

From the base model's perspective, the working context is **just another context window**. It doesn't know some embeddings are gists rather than raw tokens:

1. **Dimensionality match:** Gists live in the same embedding space as tokens
2. **RoPE compatibility:** Gists positioned at central token index for consistent encoding [1]
3. **Substitutability:** [[GistNet]] trained so gists produce similar hidden states to original tokens

```python
# Standard forward pass
outputs = base_model(
    inputs_embeds=working_context,  # [N, d] - mixed tokens & gists
    attention_mask=attention_mask,
    position_ids=position_ids        # Absolute indices for RoPE [1]
)
```

---

## System Invariants

The working context maintains several critical invariants:
- **Budget Invariant:** Total token cost ≤ [[Glossary#W_max (Token Budget)|W_max]]
- **Contiguity Invariant:** No gaps or overlaps in temporal coverage
- **Block Alignment Invariant:** All boundaries align with K-token blocks
- **Level Consistency Invariant:** Entry LOD matches span size
- **RoPE Invariant:** Consistent positional encoding

For detailed invariant definitions, see **[[Invariants]]**.

---

## Role in the System

The Working Context is the **central coordination point** for all components:

- **For the base model:** The only context it ever sees—a seemingly normal attention window
- **For [[LensNet]]:** Input to analyze and score for focus adjustments
- **For [[Focus Allocator]]:** The budget-constrained space where it applies expand/collapse decisions
- **For [[GistNet]]:** Provides examples of which gists are actually used (for on-policy training)
- **For [[Runtime Loop]]:** The working state that persists across decode steps

---

## POC Implementation

In the [[POC Architecture|proof-of-concept]]:
- **Size:** [[Glossary#W_max (Token Budget)|W_max]] = 8,192 tokens
- **Update frequency:** Refocus every K=32 tokens
- **Initial assembly:** Start with most recent tokens/gists from the tree
- **No streaming:** Entire working context in GPU memory
- **Simple heuristics:** Initial focus policy with recency bias

See **[[POC Implementation]]** for full details and constraints.

---

## Related Pages

- **[[Working Context Assembly]]** - Detailed assembly process and algorithms
- **[[Working Context Refocusing]]** - Refocusing mechanics and [[LensNet]]/[[Focus Allocator]] interaction
- **[[Invariants]]** - System-wide constraints including budget, contiguity, and alignment
- **[[MegaContext Tree]]** - The persistent memory structure backing the working context
- **[[LensNet]]** - Neural network that scores entries for relevance
- **[[Focus Allocator]]** - Component that applies expand/collapse operations
- **[[GistNet]]** - Neural network that produces compressed gist embeddings
- **[[POC Implementation]]** - Proof-of-concept constraints and implementation details

---

## References

1. **RoPE** (Su et al., 2021) — [[reference/papers/RoPE.md|Analysis]] — Rotary position embeddings used throughout MegaContext
2. **Sparse Transformers** (Child et al., 2019) — [[reference/papers/Sparse Transformers.md|Analysis]] — Factorized sparse attention patterns

See [[Related Work]] for the complete bibliography of all research papers referenced throughout the documentation.
