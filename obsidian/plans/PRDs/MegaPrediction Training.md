---
tags:
  - plans
  - prd
summary: Aligns MegaPrediction training with MegaAttention's wLOD working-context tree so one shared LM head learns multi-scale forecasts without bespoke auxiliary heads.
---
# [[MegaPrediction]]: MegaAttention-Aligned Training Plan

> **Status:** Plan of record (POR). This PRD governs speculative planning + shared readouts; older POC notes are historical context only.

**Scope:** Train MegaPrediction end-to-end with the same forward pass that powers [[MegaAttention Training|MegaAttention]] so no standalone LOD heads are required.
**Goal:** Reuse the mixed-LOD [[Working Context]] tree to supervise next-token and next-gist predictions directly from the latest hidden state at each wLODn.
**Why now:** The WC already carries wLOD0→wLOD2 leaves assembled by [[LensNet]] + the [[Focus Allocator]], meaning the model naturally exposes the features we previously tried to learn by bolting on auxiliary projection heads.

---

## 1) Design summary

1. Run the normal MegaAttention forward pass over `[WC || teacher-forced horizon]` (identical to [[MegaContext End-to-End Training]] Step 4).
2. Record the **latest node at each wLOD** along the teacher-forced continuation. Each carries its own positional center `(t)` and scale `(σ)` thanks to [[Positional Encoding#Gaussian RoPE stack|Gaussian RoPE]].
3. Apply the shared LM head (tying weights with the input embedding) to those hidden states. Interpret outputs according to level:
   - wLOD0 → token logits
   - wLOD1 → 32-token gist vector (regression)
   - wLOD2 → 1024-token gist vector (regression)
4. Supervise with ground-truth targets generated on-the-fly by [[GistNet]] over the same window.
5. Route gradients through the base transformer and GistNet—no extra adapters, no duplicated heads.

This mirrors the intent of [[MegaPrediction]]'s speculative planner while satisfying the latest POR: prediction time scales now come "for free" from the WC tree that MegaAttention already traverses.

---

## 2) Working-context readouts

Let `wc_tree` store the packed sequence returned by [[Working Context Assembly]] after MegaAttention ingest.

| Level | Node selector | Readout rule | Notes |
|-------|---------------|--------------|-------|
| wLOD0 | Last WC index (`wc_tree.latest_token()`) | Shared LM head → logits | Matches base next-token loss. |
| wLOD1 | Last gist child covering the current causal window | Shared LM head → logits projected into embedding space → interpret as vector | Equivalent to old "LOD1 head" but now fully shared. |
| wLOD2 | Root-most node whose span intersects the cursor | Same shared head | Provides ultra-long forecast for speculative planning. |

A node's positional metadata ensures causal correctness; [[Hierarchical KV Caching Strategy]] guarantees their cached K/Vs were recomputed when earlier spans changed.

---

## 3) Loss surfaces & weighting

| Level | Target builder | Loss | Suggested weight |
|-------|----------------|------|------------------|
| wLOD0 tokens | Ground-truth next tokens | Cross-entropy (NLL) | `1.0` |
| wLOD1 gist | `GistNet(gt_tokens[b:b+32])` | Cosine distance or MSE | `α1 = 0.02 → 0.05` |
| wLOD2 gist | `GistNet(gt_tokens[b:b+1024])` | Cosine distance or MSE | `α2 = 0.01 → 0.03` |

Guidelines:
- Start with cosine loss; switch to MSE only if gradients become unstable.
- Keep `α1 > α2` so block-level predictions matter more than ultra-long summaries.
- Optional **consistency loss** can penalize disagreement between incremental soft-gist reconstructions (from token logits) and the direct LM-head projection, but keep it low (`≤0.01`) to avoid over-constraining the trunk.

ΔNLL@H remains the acceptance metric for gist substitutability—log it per level so we can see whether errors stem from token accuracy or coarse summaries.

---

## 4) Training loop integration

1. Assemble `[WC || H horizon tokens]` (teacher forcing) exactly as in [[MegaContext End-to-End Training]].
2. Run a single MegaAttention forward pass with shared [[Flash Attention]] kernels for wLOD0–wLOD2 slices.
3. Extract the hidden states at:
   - the last `H` tokens (wLOD0)
   - the final gist node per 32-token block (wLOD1)
   - the final gist node per 1024-token window (wLOD2)
4. Apply the shared LM head to each hidden state.
5. Build targets via [[GistNet]] inside `torch.no_grad()` to start; unfreeze later for full end-to-end coupling.
6. Compute `L_tok + α1·L_lod1 + α2·L_lod2 (+ L_cons)` and backprop once.
7. Log ΔNLL, gist alignment metrics, and speculation-specific KPIs so [[LensNet]] can reuse the signals during focus scoring.

No structural edits are required elsewhere in the alternating optimization schedule; this drop-in replaces the dual-head experiment described in the prior revision.

---

## 5) Reference implementation sketch

```python
# wc_ids: [W] working-context token ids (mixed wLOD0)
# gt_ids: [H] teacher-forced future tokens (H % 32 == 0)
seq_ids = torch.cat([wc_ids, gt_ids], dim=0)
logits, hidden, tree = megaattention_forward(seq_ids, return_tree=True)

# Shared LM head
tied_head = model.lm_head  # tied to embeddings

# wLOD0 token loss
logits_H = logits[-H:]
L_tok = F.cross_entropy(logits_H.view(-1, logits_H.size(-1)), gt_ids.view(-1))

# Gather latest nodes per block/level
lod1_states = tree.latest(level=1, horizon=H)  # [K, d]
lod2_states = tree.latest(level=2, horizon=H)

g1_pred = F.normalize(tied_head(lod1_states), dim=-1)
g2_pred = F.normalize(tied_head(lod2_states), dim=-1)

with torch.no_grad():
    gt_emb = model.embed_tokens(gt_ids)
    g1_true = gistnet(gt_emb.view(-1, 32, gt_emb.size(-1)))  # [K, d]
    g2_true = gistnet(gt_emb.view(-1, 1024, gt_emb.size(-1)))  # slow path, sample sparsely
    g1_true = F.normalize(g1_true, dim=-1)
    g2_true = F.normalize(g2_true, dim=-1)

L_lod1 = (1 - (g1_pred * g1_true).sum(dim=-1)).mean()
L_lod2 = (1 - (g2_pred * g2_true).sum(dim=-1)).mean()

total = L_tok + alpha1 * L_lod1 + alpha2 * L_lod2
(total / grad_accum).backward()
```

Implementation tips:
- Cache `g2_true` sparsely (every N batches) if 1024-token gist windows are too expensive.
- Pin the shared LM head to BF16 to avoid precision drift between token/gist predictions.
- Track per-level gradient norms to ensure the shared trunk does not overfit to any single scale.

---

## 6) Evaluation & telemetry

| Metric | Why it matters | Notes |
|--------|----------------|-------|
| ΔNLL@H per level | Proves gists remain substitutable while sharing the same head | Compare against the frozen baseline before enabling MegaPrediction. |
| LOD agreement | `(g1_pred · g1_true)` trending upward indicates better block planning | Feed into [[LensNet]] as an auxiliary reward for focus decisions. |
| Speculative acceptance rate | Fraction of speculative spans committed without re-generation | Depends on [[Hierarchical KV Caching Strategy]] keeping caches current. |

Log these metrics via the existing [[Telemetry]] conventions so downstream agents can reason about planner quality.

---

## 7) Inference / speculative usage

- The shared head outputs feed directly into the workflows described in [[MegaPrediction]] (gist-first decoding, constrained search, beam pruning).
- [[LensNet]] can score which speculative spans to refine using LOD agreement deltas rather than bespoke confidence heuristics.
- Because caches stay valid, speculative spans can be rolled back safely; if too many edits invalidate K/V buffers, trigger the full recompute fallback defined in [[Hierarchical KV Caching Strategy]].

---

## 8) Roadmap

1. **Baseline (this doc):** shared-head regression losses + ΔNLL monitoring.
2. **Quantized gists:** once continuous training is stable, revisit the Latentese/VQ plan for wLOD1 to unlock categorical decoding (ties into [[Compressive Transformer]]-style compressors).
3. **RL fine-tuning:** blend task rewards with compute/latency costs so MegaPrediction learns when to stop refining future spans.
4. **Serving optimizations:** integrate [[Flash Attention]]-3 and CUDA Graphs for low-latency speculative decoding loops.
