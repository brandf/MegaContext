---
tags:
  - components
summary: Deep dive into LensNet's focus scoring mechanism—how scores are computed, interpreted, and used to guide working context refocusing.
---
# LensNet Scoring

[[LensNet]] produces **signed focus scores** that tell the [[Focus Allocator]] where to expand (zoom in) or collapse (zoom out) within the [[Working Context]]. This document explains how these scores are computed, what they mean, and how they guide focus decisions.

## Overview

Every entry in the [[Working Context]] receives a scalar **focus score** `u_i`:
- **Positive scores (`u_i > 0`)**: expand / increase detail (go down one level in the [[MegaContext Tree]])
- **Negative scores (`u_i < 0`)**: collapse / reduce detail (go up one level in the tree)
- **Zero or near-zero**: maintain current representation

These scores are **signed utilities** that balance exploration (expanding gists into raw tokens) against compression (collapsing tokens into gists), keeping the [[Working Context]] within its fixed token budget.

## Score Computation: Forward Pass Through LensNet

### Input Preparation

LensNet operates on embeddings from two sources:

1. **Context embeddings** (`N × d`): All ~8,000 entries currently in the [[Working Context]]
   - Mix of LOD0 (raw tokens), LOD1 gists, and LOD2 gists
   - Down-projected to LensNet width `d_lens ≈ 512`

2. **Tail gists** (`K × d`): Small conditioning set (default `K = 6`)
   - LOD2 root gist (full document summary)
   - Last 5 LOD1 gists (recent high-level summaries)
   - Encodes the upcoming query/task context

3. **Auxiliary features** (per entry):
   - `levels`: 0/1/2 markers indicating tree level
   - `span_width`: number of LOD0 tokens this entry represents
   - `distance_to_cursor`: blocks from decode cursor

### Three-Stage Architecture

**Stage 1 — Tail gists attend to context**

The tail gists (representing the query/task) read the entire working context:

```python
q_g = tail_gists @ W_qg          # [K, d_lens]
k_x = context @ W_kx             # [N, d_lens]
v_x = context @ W_vx             # [N, d_lens]
attn_g = softmax(q_g @ k_x.T / sqrt(d_lens))
gist_context = attn_g @ v_x      # [K, d_lens]
```

This produces query-aware gist representations that know what's relevant in the context.

**Stage 2 — Context queries enriched gists**

Each context entry queries the enriched gist set:

```python
q_x = context @ W_qx             # [N, d_lens]
k_g = gist_context @ W_kg        # [K, d_lens]
v_g = gist_context @ W_vg        # [K, d_lens]
attn_x = softmax(q_x @ k_g.T / sqrt(d_lens))
context_update = attn_x @ v_g    # [N, d_lens]
```

Each entry now encodes how relevant it is to the query/task.

**Stage 3 — Scoring head**

Auxiliary features are normalized to `[0, 1]` and combined with the context update:

```python
features = stack([levels.float(), span_width.float(), distance_to_cursor.float()])
inputs = cat([context_update, features @ W_feat], dim=-1)
scores = head(inputs).squeeze(-1)  # [N] signed scalars
```

The output is one **signed focus score** per entry.

### Computational Complexity

- **Complexity**: `O(N × K × d_lens)` per pass
- With `N ≈ 8k`, `K = 6`, `d_lens = 512`: ~25M multiply-adds
- **Runtime**: < 3 ms per update @ 8k tokens
- **Negligible** compared to base LLM decode

## Score Interpretation

### Sign and Magnitude

| Score Range | Meaning | Action |
|-------------|---------|--------|
| `u_i >> 0` | Strong desire to expand | High-priority expansion (gist → tokens) |
| `u_i > 0` | Positive focus signal | Candidate for expansion |
| `u_i ≈ 0` | Neutral | Maintain current representation |
| `u_i < 0` | Negative focus signal | Candidate for collapse |
| `u_i << 0` | Strong desire to collapse | High-priority collapse (tokens → gist) |

### What Positive Scores Mean

A **positive score** indicates that expanding this entry will likely improve the base LLM's predictions:

- For **LOD1 gists**: expand into constituent LOD0 tokens (≈256 tokens)
- For **LOD2 gists**: expand into constituent LOD1 gists (covers larger span)
- **Intuition**: "We need more detail here to answer the upcoming query"

### What Negative Scores Mean

A **negative score** indicates that collapsing this span into a gist will reduce token usage without harming predictions:

- For **LOD0 token sequences**: collapse into their parent LOD1 gist
- For **LOD1 gist sequences**: collapse into their parent LOD2 gist
- **Intuition**: "This detail isn't needed; we can summarize it"

### Legality Constraints

Not all actions are legal:

- **LOD0 tokens cannot expand** (already at finest detail)
- **LOD2 root cannot collapse** (already at coarsest detail)

Illegal directions are **hard-masked to zero** at inference time.

## How the Focus Allocator Uses Scores

The [[Focus Allocator]] receives signed scores from LensNet and applies them greedily:

### 1. Legality Masking

```python
# Prevent illegal actions
if level[i] == 0:
    scores[i] = min(scores[i], 0)  # LOD0 can't expand
if level[i] == 2:
    scores[i] = max(scores[i], 0)  # LOD2 can't collapse
```

### 2. Sorting by Magnitude

Actions are ranked by absolute score magnitude:
- Most positive scores → expansion candidates
- Most negative scores → collapse candidates

### 3. Greedy Execution

The allocator alternates between expansions and collapses:

1. **Expand** the highest-scoring legal gist (if budget allows)
2. **Collapse** the lowest-scoring legal span to free tokens
3. Repeat until budget is balanced

### 4. Budget Balancing

Total working context size must stay within bounds:

```
tokens_gained = sum(expansion_costs)
tokens_freed = sum(collapse_refunds)
net_change = tokens_gained - tokens_freed ≈ 0
```

The allocator stops when:
- Net token change is within budget
- No legal actions remain
- Hysteresis thresholds prevent thrashing

### 5. Hysteresis Rules

To prevent repeated expand/collapse on the same span:
- Track recently modified spans
- Apply cooldown periods
- Require score threshold before reversing recent action

## Score Distribution and Calibration

### Training Targets

During training, scores are supervised with **signed target utilities** `y_i`:

- **Expandable items** (LOD1/LOD2 with children): positive `y_i > 0`
- **Collapsible spans**: negative `y_i < 0`
- Others: 0 or masked

Target utilities are derived from **counterfactual ΔNLL**:
- "What NLL improvement would we get if we expanded this gist?"
- "What NLL cost would we pay if we collapsed this span?"

### Zero-Sum Budget Regularizer

To maintain constant working context size, LensNet is trained with:

```
P = sum_i cost_expand[i] * ReLU(u_i)      # expansion mass
N = sum_i refund_collapse[i] * ReLU(-u_i) # collapse mass
L_budget = ((P - N) / (eps + P + N))^2
```

This encourages **net-zero token changes** per block update.

### Typical Score Distributions

After training, score distributions typically look like:

- **Mean**: close to 0 (budget balancing)
- **Std dev**: 0.5–2.0 (depending on context complexity)
- **Mode**: near 0 (most entries remain unchanged)
- **Tails**: ±3–10 (high-priority expand/collapse)

### Optional Rebalancing at Inference

Before sending to the allocator, scores can be rescaled:

```python
pos_mass = sum(cost[i] * relu(scores[i]) for i in expandable)
neg_mass = sum(refund[i] * relu(-scores[i]) for i in collapsible)

# Rescale to balance masses
if pos_mass > neg_mass:
    scores[expandable] *= (neg_mass / pos_mass)
elif neg_mass > pos_mass:
    scores[collapsible] *= (pos_mass / neg_mass)
```

## Examples of Scoring Different Entries

### Example 1: Recent Query-Relevant Section

**Entry**: LOD1 gist covering "database schema definitions" (256 LOD0 tokens)

**Context**: Upcoming query is "What fields does the User table have?"

**LensNet reasoning**:
- Tail gists contain recent query context
- Cross-attention highlights schema-related gists
- This gist is highly relevant

**Score**: `u_i = +4.2` (strong expand signal)

**Action**: [[Focus Allocator]] expands this LOD1 gist into 256 LOD0 tokens, providing detailed schema information to the base LLM.

---

### Example 2: Irrelevant Historical Span

**Entry**: LOD0 tokens from early document (boilerplate copyright notice, 128 tokens)

**Context**: Document is long; working context is at capacity; query is about technical content far ahead

**LensNet reasoning**:
- Low attention from tail gists
- Far from decode cursor
- Legal to collapse (LOD0 tokens)

**Score**: `u_i = -2.8` (collapse signal)

**Action**: [[Focus Allocator]] collapses these 128 tokens into their parent LOD1 gist, freeing tokens for more relevant content.

---

### Example 3: LOD2 Root Gist

**Entry**: LOD2 root gist (summarizes entire document)

**Context**: Always present in working context

**LensNet reasoning**:
- Provides global context
- Cannot collapse (already at highest level)
- May expand if we need more LOD1 detail

**Score**: `u_i = +0.3` (weak expand signal, but likely not high-priority)

**Legality**: Can expand into constituent LOD1 gists, but cannot collapse further.

---

### Example 4: Mid-Document LOD1 Gist

**Entry**: LOD1 gist covering "methodology section" (256 LOD0 tokens compressed)

**Context**: Query is about results section; methodology not currently relevant

**LensNet reasoning**:
- Moderate attention from tail gists (some relevance)
- Could expand or maintain
- Not critical for immediate query

**Score**: `u_i = -0.1` (near-neutral, slight collapse bias)

**Action**: Likely maintained as LOD1 gist (score magnitude too low to prioritize action).

---

### Example 5: Recently Decoded Context

**Entry**: LOD0 tokens just decoded (last 128 tokens of generation)

**Context**: Generation in progress; these tokens are part of model's immediate context

**LensNet reasoning**:
- Very close to decode cursor (`distance_to_cursor ≈ 0`)
- Essential for causal generation
- Should never collapse

**Score**: `u_i = +0.05` (neutral-positive, protected from collapse)

**Action**: Maintained as LOD0 tokens regardless of score.

---

### Example 6: Boundary Case—Just Above Threshold

**Entry**: LOD1 gist with marginal relevance

**Context**: Working context slightly over budget

**Score**: `u_i = -0.9` (just above collapse threshold)

**Hysteresis check**:
- Was this recently expanded? → Yes (2 blocks ago)
- Cooldown period: 5 blocks
- **Action blocked**: Hysteresis prevents thrashing

**Outcome**: Entry maintained despite negative score.

## Training Loss Components

LensNet's scoring is shaped by four loss terms:

### 1️⃣ Regression Loss

```
L_reg = (1 / |M|) * sum_{i in M} (u_i - y_i)^2
```

Teaches scores to match counterfactual target utilities.

### 2️⃣ Ranking Loss

```
L_rank = softplus(-(u_i - u_j))  # for ordered pairs (i, j)
```

Ensures relative ordering: items with higher target utilities get higher scores.

### 3️⃣ Budget Regularizer

```
L_budget = ((P - N) / (eps + P + N))^2
```

Encourages balanced expand/collapse masses.

### 4️⃣ Legality Penalties

```
L_illegal = alpha * sum_{LOD0} ReLU(u_i) + beta * sum_{LOD2} ReLU(-u_i)
```

Discourages impossible actions during training (alpha, beta ≈ 0.3).

### Total Loss

```
L_total = L_reg + 0.5 * L_rank + 0.1 * L_budget + L_illegal
```

## Update Cadence

LensNet runs **once every K tokens** (POC: K = 32):

1. Gather latest tail gists
2. Run LensNet forward pass → produce scores
3. [[Focus Allocator]] executes actions
4. Updated [[Working Context]] is frozen for next 32 tokens

This matches the [[Working Context Refocusing|block-wise refocus]] strategy: no per-token recomputation.

## Key Insights

1. **Signed scores enable bidirectional control**: Unlike binary classifiers, signed utilities allow simultaneous expand/collapse planning.

2. **Non-causal attention is essential**: LensNet must see future context (via tail gists) to know what past detail matters—compensating for the base LLM's causal blindness.

3. **Budget balancing is built into training**: The zero-sum regularizer ensures LensNet doesn't systematically over-expand or over-collapse.

4. **Scores are utilities, not probabilities**: Magnitudes represent expected NLL improvement/cost, not confidence levels.

5. **Greedy allocation is sufficient**: The [[Focus Allocator]] doesn't need complex optimization—sorted score traversal works well in practice.

## Related Pages

### Core Components
- [[LensNet]] — Overall architecture, dual cross-attention design, and system role
- [[LensNet Training]] — How scores are trained via counterfactual supervision and multi-objective loss
- [[Focus Allocator]] — Execution engine that converts scores into expand/collapse operations
- [[Focus Allocator Strategies]] — Greedy vs. learned allocation policies for interpreting scores

### Integration Points
- [[Working Context]] — The fixed-size GPU window that LensNet optimizes
- [[Working Context Refocusing]] — Block-wise update process and refocus cadence (every K=32 tokens)
- [[Working Context Assembly]] — How scored entries are materialized from the MegaContext Tree
- [[MegaContext Tree]] — The hierarchical structure providing candidates for scoring

### Training & Data
- [[GistNet]] — Produces gist embeddings that LensNet learns to expand/collapse appropriately
- [[GistNet Training]] — Complementary compression training that enables multi-LOD scoring
- [[Alternating Optimization]] — Joint training regime coordinating GistNet and LensNet updates

### System Context
- [[POC Implementation]] — Concrete score thresholds, update frequency, and runtime parameters
- [[Invariants]] — Budget, legality, and contiguity constraints enforced by scoring
- [[Runtime Loop]] — Full decode-ingest-score-allocate cycle where scoring operates

---

> **Core principle**: LensNet scoring treats focus allocation as a **budget-constrained utility maximization problem**. By learning to predict counterfactual ΔNLL values, LensNet guides the [[Focus Allocator]] to maintain maximal relevance within a constant token budget.
