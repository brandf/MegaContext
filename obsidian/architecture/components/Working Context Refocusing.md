---
tags:
  - architecture
  - technical
summary: How the Working Context dynamically refocuses over time - the continuous process of expanding and collapsing LODs based on LensNet predictions to adapt to changing relevance.
---

# Working Context Refocusing

This document explains how the [[Working Context]] **continuously adapts** its level of detail as conversations progress and relevance shifts.

---

## Overview

The [[Working Context]] is not static—it **evolves continuously** through a process called **refocusing**:

```
Query: "Show me login code"
   ↓
LensNet: login regions → high relevance scores
   ↓
Focus Allocator: Expand login from L1 → L0
   ↓
Model sees login at full detail

Time passes, new query...

Query: "What about database schema?"
   ↓
LensNet: login → low relevance, database → high relevance
   ↓
Focus Allocator: Collapse login L0 → L1, Expand database L1 → L0
   ↓
Model now sees database at full detail, login compressed
```

This happens **automatically** every K tokens without manual intervention.

---

## Refocus Cadence

Refocusing happens at regular intervals:

**POC Configuration:**
- **Frequency:** Every K=32 tokens
- **Trigger:** After decoding a 32-token block
- **Operations:** Up to N_diff=4 expand/collapse actions per cycle

```
Decode block [0, 32)
   ↓
Refocus (adjust LODs)
   ↓
Decode block [32, 64)
   ↓
Refocus (adjust LODs)
   ↓
...
```

See [[POC Implementation]] for parameter values.

---

## Refocus Process

### Step-by-Step

```
1. Decode Phase
   - Generate next K=32 tokens using current Working Context
   - Model predicts based on whatever LODs are currently active

2. Ingest Phase
   - Add new tokens to MegaContext Tree
   - Run GistNet to create/update gist nodes if needed

3. Score Phase
   - LensNet evaluates all Working Context entries
   - Outputs signed focus scores: u_i ∈ [-1, +1]
     * Positive: needs more detail (expand)
     * Negative: can be compressed (collapse)

4. Allocate Phase
   - Focus Allocator selects top-scoring operations
   - Applies up to N_diff=4 expand/collapse actions
   - Maintains budget invariant

5. Repeat
   - Updated Working Context used for next decode phase
```

---

## Example Refocus Sequence

### Initial State (T=0)

```
User: "Show me the authentication code"

Working Context:
  [L0: auth_module tokens 0-256]     cost: 256 tokens
  [L1: utils gist]                   cost: 1 token
  [L1: database gist]                cost: 1 token
  [L2: distant_code gist]            cost: 1 token
                            Total: 259 tokens ≤ 8192 ✓
```

### After 32 Tokens (T=32)

Model generates response about authentication.

**LensNet scores:**
```
Entry                   Score    Interpretation
----------------------------------------------------
auth_module L0          +0.1     Keep at L0 (still relevant)
utils L1                +0.7     EXPAND! (became relevant)
database L1             -0.3     Collapse (less relevant now)
distant_code L2         -0.1     Keep at L2 (stay compressed)
```

**Focus Allocator actions:**
```
1. Expand utils L1 → L0         (+31 tokens)
2. Collapse database L1 → L2    (-31 tokens saved by higher compression)
   Net budget change: 0 tokens
```

**Updated Working Context:**
```
  [L0: auth_module 0-256]        cost: 256 tokens
  [L0: utils 256-288]            cost: 32 tokens (expanded!)
  [L2: database gist]            cost: 1 token (collapsed!)
  [L2: distant_code gist]        cost: 1 token
                        Total: 290 tokens ≤ 8192 ✓
```

### After Another 32 Tokens (T=64)

User asks: "How does the database schema handle this?"

**LensNet scores:**
```
Entry                   Score    Interpretation
----------------------------------------------------
auth_module L0          -0.6     COLLAPSE! (no longer relevant)
utils L0                -0.4     COLLAPSE! (less relevant)
database L2             +0.9     EXPAND! (very relevant now)
distant_code L2         +0.2     Keep at L2 (slight relevance)
```

**Focus Allocator actions:**
```
1. Expand database L2 → L1 → L0  (+31 tokens for L2→L1, then +31 for L1→L0)
2. Collapse auth_module L0 → L1  (-31 tokens)
3. Collapse utils L0 → L1        (-31 tokens)
   Net budget change: +62 - 62 = 0 tokens
```

**Updated Working Context:**
```
  [L1: auth_module gist]         cost: 1 token (collapsed!)
  [L1: utils gist]               cost: 1 token (collapsed!)
  [L0: database 0-32]            cost: 32 tokens (expanded!)
  [L2: distant_code gist]        cost: 1 token
                        Total: 35 tokens ≤ 8192 ✓
```

---

## Why Continuous Refocusing?

### Problem Without Refocusing

**Static focus policies:**
- **Recency-only:** Always keep recent tokens detailed
  - Problem: Recent tokens may not be relevant to current query
- **Query-time retrieval (RAG):** Fetch relevant docs when queried
  - Problem: No way to remove irrelevant docs mid-generation
  - Problem: Stateless—doesn't remember what was important before

### Benefits of Continuous Refocusing

1. **Adapts to changing relevance**
   ```
   T=0:   Topic A relevant → show at L0
   T=100: Topic B relevant → expand B to L0, collapse A to L1
   T=200: Back to topic A  → re-expand A to L0, collapse B
   ```

2. **Budget-aware optimization**
   - Limited to W_max tokens total
   - Must trade off detail across different regions
   - LensNet learns optimal allocation

3. **Reversible without information loss**
   - Collapsing L0→L1 doesn't delete L0 tokens
   - They remain in MegaContext Tree
   - Can re-expand if they become relevant again

4. **Learned, not heuristic**
   - LensNet trained on actual task performance
   - Discovers what matters for predictions
   - Better than hand-crafted rules

---

## LensNet's Role

[[LensNet]] is the **predictor** that drives refocusing.

### Inputs

| Input | Description |
|-------|-------------|
| `context` | Current Working Context entries (≈8k embeddings) |
| `tail_gists` | Recent history (L2 root + 5 latest L1 gists) |
| `levels` | Current LOD of each entry (0/1/2) |
| `spans` | Token coverage of each entry |
| `positions` | Distance from decode cursor |

### Outputs

| Output | Description |
|--------|-------------|
| `scores` | Signed focus score per entry, range [-1, +1] |

**Interpretation:**
- **score > +0.2:** Expand this entry (needs more detail)
- **score < -0.2:** Collapse this entry (can be compressed)
- **-0.2 ≤ score ≤ +0.2:** Leave as is (current LOD is fine)

See [[LensNet Scoring]] for computation details and [[LensNet Training]] for how scores are learned.

---

## Focus Allocator's Role

[[Focus Allocator]] is the **executor** that applies refocusing decisions.

### Algorithm

```python
def refocus_working_context(working_context, lens_scores, W_max, N_diff=4):
    """
    Apply up to N_diff expand/collapse operations.
    """
    # 1. Collect candidates
    expand_candidates = [
        (score, entry)
        for entry, score in zip(working_context, lens_scores)
        if score > 0.2 and can_expand(entry)
    ]
    collapse_candidates = [
        (score, entry)
        for entry, score in zip(working_context, lens_scores)
        if score < -0.2 and can_collapse(entry)
    ]

    # 2. Sort by magnitude (highest priority first)
    expand_candidates.sort(key=lambda x: x[0], reverse=True)
    collapse_candidates.sort(key=lambda x: x[0])  # Most negative first

    # 3. Apply operations while maintaining budget
    actions = []
    for _ in range(N_diff):
        # Try expand if budget allows
        if expand_candidates and budget_allows_expansion(working_context, W_max):
            score, entry = expand_candidates.pop(0)
            expand(entry)  # L1→L0 or L2→L1
            actions.append(("expand", entry))

        # Balance with collapse
        if collapse_candidates:
            score, entry = collapse_candidates.pop(0)
            collapse(entry)  # L0→L1 or L1→L2
            actions.append(("collapse", entry))

    return working_context, actions
```

### Constraints

1. **Budget invariant:** `sum(entry_costs) ≤ W_max` at all times
2. **Legality:** Can't expand L0 further, can't collapse L2 higher (in POC)
3. **Cooldown:** Entry must wait 2 iterations before flipping expand→collapse
4. **Block alignment:** All operations respect K=32 block boundaries

See [[Focus Allocator]] for algorithm details and [[Invariants]] for constraints.

---

## Refocusing Metrics

### Swap Rate

**Definition:** Average number of expand/collapse operations per refocus cycle

```python
swap_rate = total_actions / total_refocus_cycles
```

**Typical values:**
- **Low (0.05–0.1):** Mostly static, occasional adjustments
- **Medium (0.1–0.3):** Moderate adaptation to changing topics
- **High (0.3–0.5):** Aggressive refocusing, frequent topic shifts

**POC target:** 0.1–0.3 actions/block (2–4 changes per 32 tokens with N_diff=4)

### Residency Time

**Definition:** How long an entry stays at a particular LOD before changing

```python
mean_residency = sum(time_at_lod) / num_lod_changes
```

**Interpretation:**
- **Short residency (<3 blocks):** Noisy, unstable focus
- **Medium residency (3–10 blocks):** Healthy adaptation
- **Long residency (>20 blocks):** May be too sticky, not adapting enough

### Focus Coherence

**Definition:** What fraction of high-scoring entries are actually expanded?

```python
coherence = expanded_high_scores / total_high_scores
```

**Goal:** >0.8 (most high-scoring opportunities are taken)

---

## Refocusing vs. Retrieval

### Retrieval (RAG)

```
User query
   ↓
Retrieve relevant docs (one-time, query-based)
   ↓
Concatenate to context
   ↓
Generate
   ↓
(Retrieved docs stay in context until evicted)
```

**Characteristics:**
- **Query-time only:** No continuous adaptation
- **Stateless:** Each query retrieves independently
- **Append-only:** Can't remove irrelevant docs mid-generation
- **Binary:** Document either included or not (no LOD control)

### Refocusing (MegaContext)

```
Working Context exists
   ↓
Every K tokens:
   LensNet scores all entries
   Focus Allocator adjusts LODs
   ↓
Working Context continuously adapts
```

**Characteristics:**
- **Continuous:** Happens every K tokens, not just at query time
- **Stateful:** Remembers what was important before
- **Bidirectional:** Can add detail (expand) or remove (collapse)
- **Granular:** LOD control (L0/L1/L2), not binary include/exclude

See [[Comparisons#vs. RAG]] for detailed comparison.

---

## Refocusing Patterns

### Pattern 1: Topic Shift

```
T=0:    Topic A at L0, Topic B at L2
T=100:  User shifts to topic B
        → Collapse A (L0→L1), Expand B (L2→L0)
T=200:  Still on topic B
        → A stays at L1, B stays at L0
```

**Trigger:** User explicitly changes topic

### Pattern 2: Zooming In

```
T=0:    Overview at L1
T=50:   User asks for details
        → Expand specific regions (L1→L0)
T=100:  User asks for more details
        → Expand additional regions (L1→L0)
```

**Trigger:** Progressively deeper questions

### Pattern 3: Zooming Out

```
T=0:    Detailed view at L0
T=50:   User satisfied, moves on
        → Collapse details (L0→L1)
T=100:  Focus shifts elsewhere
        → Further collapse (L1→L2)
```

**Trigger:** Loss of relevance over time

### Pattern 4: Oscillation

```
T=0:    Region A at L0
T=50:   Collapse A (L0→L1)
T=100:  Re-expand A (L1→L0)
```

**Problem:** Unstable focus (noisy LensNet scores or low cooldown)
**Solution:** Increase cooldown_steps, add hysteresis in thresholds

---

## POC Simplifications

See [[POC Implementation]] for full parameters.

**What's simplified:**
- **Synchronous refocusing:** Happens inline, no background workers
- **Fixed thresholds:** τ_expand and τ_collapse hardcoded at ±0.2
- **Greedy allocator:** Not learned/differentiable (future work)
- **Simple cooldown:** Fixed 2-iteration wait (future: learned delays)

**Future enhancements:**
- **Async refocusing:** Background thread scores while model decodes
- **Adaptive thresholds:** Learn τ_expand and τ_collapse per task
- **Learned allocator:** Differentiable policy network (see [[Focus Allocator Strategies]])
- **Anticipatory expansion:** Prefetch likely-to-be-needed gists
- **Context-aware cooldowns:** Longer cooldowns for stable regions

---

## Related Pages

### Core Process
- [[Working Context]] — Overview, fixed-size properties, and role as the refocused window
- [[Working Context Assembly]] — How the working context is initially built and reassembled after refocusing
- [[Runtime Loop]] — Complete decode-ingest-score-refocus cycle orchestration

### Refocusing Components
- [[LensNet]] — Neural predictor that scores entries for relevance every K tokens
- [[LensNet Scoring]] — Detailed scoring computation, interpretation, and calibration
- [[LensNet Training]] — How LensNet learns to predict optimal refocusing decisions
- [[Focus Allocator]] — Greedy executor that applies expand/collapse actions
- [[Focus Allocator Strategies]] — Strategy comparison (greedy vs. learned vs. optimization)

### Data Sources
- [[MegaContext Tree]] — Persistent source of all LOD levels accessed during refocusing
- [[GistNet]] — Produces the gists that refocusing expands and collapses
- [[Tree Operations]] — APIs for fetching nodes when refocusing changes LOD selections

### System Constraints
- [[Invariants]] — Budget, contiguity, cooldown, and legality constraints enforced during refocusing
- [[POC Implementation]] — Refocus cadence (K=32), action limits (N_diff=4), threshold values
- [[System Properties]] — Dynamic focus property and its impact on system behavior

### Training & Operations
- [[Alternating Optimization]] — How refocusing behavior improves through joint LensNet/GistNet training
- [[Telemetry]] — Metrics for swap rate, residency time, focus coherence, and stability
- [[Performance Sketch]] — Latency breakdown and optimization opportunities for refocusing

### User Experience
- [[Examples]] — Walkthrough of refocusing patterns in real conversations
- [[Comparisons]] — Refocusing vs. RAG retrieval strategies

---

*Refocusing is MegaContext's **adaptive attention mechanism**—continuously adjusting detail levels to match evolving relevance, maintaining constant budget while optimizing for prediction quality.*
