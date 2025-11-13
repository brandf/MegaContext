---
tags:
  - components
summary: Transformer-based controller that scores working-context entries for expansion or collapse, trained via pairwise preference comparisons.
---

LensNet reads the [[Working Context|working-context]] window and emits signed utilities that tell the [[Focus Allocator]] where to zoom in or back off, keeping the window relevant at constant compute.

- **Operates on:** current working-context embeddings plus per-entry metadata (LOD level, global position).
- **Outputs:** signed policy scores per entry (tanh-clamped to ±1, later temperature-scaled); positive ⇒ expand, negative ⇒ collapse.
- **Architecture:** mini transformer stack (2/4/8 layers) with Gaussian RoPE and a linear/MLP scoring head.
- **Cadence:** runs every `K` tokens before allocator actions. See [[POC Implementation]] for concrete values.
- **Training:** random-variant preference comparisons, Bradley–Terry (logistic) loss + optional budget/rank regularizers. See [[LensNet Training]].
- **Interfaces:** consumes [[GistNet]] outputs and feeds the greedy [[Focus Allocator]].

## Role in the System

LensNet is the “attention controller” of MegaContext: it predicts which entries in the working context deserve more or less detail so the allocator can keep the budgeted window maximally relevant. It operates solely on the embeddings that are already on device; no extra passes over the [[MegaContext Tree]] are required.

```
[[GistNet]] → compresses spans into gists
LensNet (transformer) → scores each WC entry
[[Focus Allocator]] → applies expand/collapse actions
[[Working Context]] → remains size-constrained but task-relevant
```

## Architectural Overview

```mermaid
flowchart LR
    subgraph Inputs
        WC[Working Context embeddings<br/>(B × W × d)]
        Meta[LOD levels + global positions]
    end
    Pos[Gaussian RoPE (cached)]
    Blocks[Transformer blocks (2/4/8 layers)]
    Head[Scoring head<br/>(linear or MLP)]
    Scores[tanh scores]

    WC --> Blocks
    Meta --> Pos --> Blocks
    Blocks --> Head --> Scores
```

1. **Metadata-aware inputs.** The [[Working Context]] supplies embeddings, LOD tensor (0 = raw tokens, 1 = LOD1 gists, …), and global positions. It also caches rotary embeddings so LensNet only recomputes when the window actually changes (append/replace/trim).
2. **Gaussian RoPE.** We reuse the Gaussian RoPE module from [[GistNet]] so frequency decay follows global positions + LOD levels.
3. **Backbone depth.** Controlled via `--lensnet_layers` (2 / 4 / 8). Depth primarily trades cost vs accuracy.
4. **Heads.** The scorer head is either a single linear projection or a two-layer MLP with hidden size `max(embed_dim // 2, 128)` (we shrink instead of expand to keep per-token compute low, since LensNet touches thousands of entries every refocus).
5. **Signed outputs.** The final scores pass through `tanh` to bound them in ±1, making downstream thresholds stable.

## API & CLI

- `--lensnet_type transformer` *(only option today)*
- `--lensnet_layers 2|4|8` (default 2)
- `--lensnet_head linear|mlp` (default mlp)

All variants share the same interface: `LensNet.forward(working_context: WorkingContext) -> torch.Tensor` returning `[B, W]` signed scores.

## Working Context Integration

The [[Working Context]] now exposes a `get_rotary_embeddings(cache_key, builder)` API. LensNet passes a cache key derived from its depth/head configuration plus a closure that calls `GaussianRoPE`. Whenever the working window mutates (`append`, `replace`, `_trim`), the cache is invalidated. This keeps positional recompute costs negligible even when LensNet runs every K tokens.

## Training & Scoring

Training remains counterfactual but now flows through **pairwise preference data**:

1. Build one LOD0 baseline WC plus `N` random compressions per sequence.
2. Run the base model on every variant to obtain next-token losses.
3. Convert the losses into advantages (`adv_delta`) relative to the baseline.
4. Form preference pairs (`preference_pairs`) and apply a Bradley–Terry loss with temperature `mc_lens_temperature`, plus optional rank/budget penalties.

The resulting policy scores feed the greedy [[Focus Allocator]], and telemetry (`mc/adv_delta_*`, `mc/preference_corr_*`) tracks how well scores align with observed ΔNLL improvements.

## References

1. **Neural Turing Machines** (Graves et al., 2014) — [[reference/papers/Neural Turing Machines.md|Analysis]] — Content-based memory controllers.  
2. **Differentiable Neural Computer** (Graves et al., 2016) — [[reference/papers/DNC.md|Analysis]] — Learned read/write policies for external memory.  
3. **Structured Self-Attentive Sentence Embedding** (Lin et al., 2017) — [[reference/papers/Structured Self-Attentive Sentence Embedding.md|Analysis]] — Inspiration for query-style pooling.  
4. **BERT** (Devlin et al., 2018) — [[reference/papers/BERT - 1810.04805.md|Analysis]] — CLS token summary mechanism reused in our pooling heads.

See [[Related Work]] for the complete bibliography of all research papers referenced throughout the documentation.
