[](Focus%20Allocator.md)---
tags:
  - module
summary: Dual cross-attention controller that scores working-context entries for expansion or collapse.
---
![[LensNet Diagram.png]]
LensNet reads the working context plus tail gists to emit signed utilities that tell the [[Focus Allocator]] where to zoom in or back off, keeping the window relevant at constant compute.

- **Operates on:** working-context embeddings (≈8k entries) and a tail of gists.
- **Outputs:** signed focus scores per entry; positive to expand, negative to collapse.
- **Architecture:** dual cross-attention blocks (`context ↔ tail gists`) followed by scalar heads.
- **Cadence:** runs every `K` tokens (32 in POC) before allocator actions.
- **Training:** counterfactual ΔNLL utilities, budget regularizers, legality penalties.
- **Interfaces:** works alongside [[GistNet]] outputs and the greedy [[Focus Allocator]].

## Details

LensNet acts like an optical lens that dynamically **focuses** and **defocuses** regions within the MegaContext while keeping total compute constant. It predicts where to spend detail (expand gists into raw tokens) and where to blur (collapse raw tokens into gists), ensuring the **fixed-size working context** maintains maximal relevance.

## Operating assumptions

- LensNet reads the **working context**, not the MegaContext tree. It analyzes the embeddings currently fed into the base LLM — the only state resident on GPU.
- It outputs one **focus score** per entry (token embedding or gist).
- The [[Architecture Details#Key terms & invariants|contiguity invariant]] ensures each score maps to a single, non-overlapping lifetime span, so expand/collapse actions remain block-aligned.

### Why non-causal is essential

LensNet must understand *future queries* to know which past facts matter. Because the base LLM is causal, hidden states for earlier tokens cannot “see” upcoming questions; LensNet compensates by operating on the full working context.

## Conceptual overview

- LensNet runs independently of the frozen base LLM.
- It operates directly on the **working context embeddings** (≈ 8k entries), not on live LLM hidden states.
- It conditions on a small **gist set** (`L2 + last 5 L1` gists, total ≈ 6) taken from the end of the context, which implicitly encodes the upcoming query/task.
- The model outputs one **signed focus score** `u_i` per entry:
  - `u_i > 0`: expand / focus (increase detail, go one level down)
  - `u_i < 0`: collapse / defocus (reduce detail, go one level up)

> **Diagram needed — `assets/lensnet_focus.png`:** Show LensNet reading a tail slice of gists plus the working context, then emitting signed scores that the allocator converts into expand/collapse actions.

At runtime, the **focus allocator** interprets these scores to expand and collapse spans while keeping the working context within its token budget.

## Architecture (POC: dual cross-attention LensNet)

1. **Inputs**
   - `context`: `torch.FloatTensor[N, d]` — embeddings of all entries in the working context (≈8 000 tokens/gists).
   - `tail_gists`: `torch.FloatTensor[K, d]` — L2 root plus the latest `K-1` L1 gists (default `K=6`).
   - `levels`: `torch.LongTensor[N]` — 0/1/2 markers for legality masking.
   - `span_width`: `torch.LongTensor[N]` — number of L0 tokens represented by each entry.
   - `distance_to_cursor`: `torch.LongTensor[N]` — block distance from the decode cursor (optional feature; treat as integer tensor).
   - All embeddings are down-projected to a LensNet width `d_lens ≈ 512`.

2. **Stage 1 — Tail gists read the context**

```python
q_g = tail_gists @ W_qg          # [K, d_lens]
k_x = context @ W_kx             # [N, d_lens]
v_x = context @ W_vx             # [N, d_lens]
attn_g = torch.softmax(q_g @ k_x.T / math.sqrt(d_lens), dim=-1)
gist_context = attn_g @ v_x      # [K, d_lens]
```

3. **Stage 2 — Context queries updated gists**

```python
q_x = context @ W_qx             # [N, d_lens]
k_g = gist_context @ W_kg        # [K, d_lens]
v_g = gist_context @ W_vg        # [K, d_lens]
attn_x = torch.softmax(q_x @ k_g.T / math.sqrt(d_lens), dim=-1)
context_update = attn_x @ v_g    # [N, d_lens]
```

4. **Stage 3 — Scoring head**

Concatenate simple scalar features (levels, span width, distance) after normalizing them to `[0, 1]` and emit signed utilities:

```python
features = torch.stack(
    [levels.float(), span_width.float(), distance_to_cursor.float()], dim=-1
)
inputs = torch.cat([context_update, features @ W_feat], dim=-1)
scores = head(inputs).squeeze(-1)  # torch.FloatTensor[N]
```

5. **Stacks / refinement**

Stacking 1–3 such dual-attention blocks improves stability; parameters `(W_qg, W_kx, …)` are shared or re-initialized per block depending on capacity.

**Complexity:** `O(N × K × d_lens)` per pass. With `N ≈ 8k`, `K = 6`, `d_lens = 512`, the update costs ~25 M multiply-adds—negligible relative to the base model decode.

## Update cadence (block-wise refocus)

LensNet runs **once every K tokens** (POC: K = 32). During each block update:

1. Gather the latest gists `G`.
2. Run LensNet to produce signed scores `u_i`.
3. The focus allocator executes expansions/collapses subject to the working-context budget.
4. The updated context is frozen for the next K tokens.

This matches the intended inference cadence (no per-token recompute).

## Training objectives

### 1️⃣ Signed focus supervision
Each entry receives a **signed target utility** `y_i` derived from counterfactual ΔNLL deltas:

- Expandable items (L1/L2 children) ⇒ positive `y_i > 0`
- Collapsible spans ⇒ negative `y_i < 0`
- Others ⇒ 0 / masked.

LensNet learns to regress and rank these utilities.

```
L_reg  = (1 / |M|) * sum_{i in M} (u_i - y_i)^2
L_rank = softplus(-(u_i - u_j))  # for ordered pairs
```

### 2️⃣ Zero-sum budget regularizer
To maintain constant working-context size:

```
P = sum_i c_i_plus * ReLU(u_i)
N = sum_i c_i_minus * ReLU(-u_i)
L_budget = ((P - N) / (eps + P + N))^2
```

(`c_i^+` / `c_i^-` = token cost / refund.) This encourages net-zero expand/defocus mass per block.

### 3️⃣ Legality penalties

Prevent impossible actions:

```
L_illegal = alpha * sum_{L0} ReLU(u_i) + beta * sum_{L2} ReLU(-u_i)
```

(`alpha`, `beta` ≈ 0.3). At inference, invalid directions are hard-masked to 0.

### 4️⃣ Total loss

```
L_total = L_reg + 0.5 * L_rank + 0.1 * L_budget + L_illegal
```

## Inference procedure

1. **Mask** illegal sides (L0 can’t expand; L2 can’t collapse).
2. **Optional rebalance:** rescale positive/negative masses to match before sending to the focus allocator.
3. The [[Focus Allocator]] greedily applies expand/collapse actions within the token budget, honoring hysteresis rules.

## Summary of POC parameters

| Item | Value / Notes |
|------|----------------|
| Input embeddings | ≈8 k entries (mixed L0/L1/L2) |
| Conditioning gists | 6 (L2 + 5 L1) |
| Down-projection width | 512 |
| Attention heads | 8 |
| Stacks | 1–3 |
| Update cadence | every 32 tokens |
| Output | Signed focus score `u_i` per entry |
| Runtime | < 3 ms per update @ 8 k tokens |
| Params | ≈ 100 k – 200 k total |

**In short:** LensNet is a compact, non-causal controller built as a dual cross-attention network (`8k → 6 → 8k`). It runs once per block, predicts balanced signed focus scores for every entry, and guides the [[Focus Allocator]] to keep the working context sharp, legal, and budget-neutral.
