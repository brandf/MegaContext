---
tags:
  - ops
summary: Alternating optimization, instrumentation, and validation practices for GistNet, LensNet, and runtime components.
---
Rotate through [[GistNet]], [[LensNet]], and LoRA [1] updates while collecting telemetry that keeps ΔNLL, swap rates, and latency within targets.

---

- **Alternating phases:** JT1 ([[GistNet]]), JT2 ([[LensNet]]), JT3 (LoRA).
- **On-policy labeling:** regenerate ΔNLL utilities after each [[GistNet]] update.
- **Telemetry:** log loss@H, swap rate, residency, latency for every block.
- **Validation:** acceptance criteria tie to ΔNLL degradation ≤0.1 at `W_max=8k`.
- **Tooling:** rely on `uv run pytest`, W&B logging, and artifacts directories for repeatability.

---
### Colab Runtime Setup

- Switch the notebook runtime to a GPU before launching the bootstrap cell (`Runtime → Change runtime type → GPU`).
- Execute the **Quick Start** bootstrap cell in [`notebooks/megacontext.ipynb`](../../notebooks/megacontext.ipynb) to clone the repo, install dependencies, and enable widget support inside Colab.
- The package targets Python 3.10+, matching Colab's default interpreter; restart and rerun the bootstrap cell whenever the VM resets.
- Keep WANDB and Hugging Face tokens handy—enter them via the widgets block to persist credentials for the current session.
- Use the **0. Setup Console** cell to select configs, tokens, storage roots, logging mode, resume checkpoints, and seeds before kicking off dataset prep or training.
- Set `MEGACONTEXT_FORCE_DATA_REBUILD=1` if you need to regenerate dataset shards even when Arrow outputs are already present; otherwise the notebook reuses cached data.

### Resuming Interrupted Runs

- Mount durable storage (e.g. Novita.ai network volumes) and point `MEGACONTEXT_ARTIFACT_ROOT` at it before starting; checkpoints and summaries land under `gistnet/`.
- After reconnecting, rerun the bootstrap + storage cells, then pick the most recent `.ckpt` in **Checkpoints & Resume** to continue training.
- `MEGACONTEXT_DATA_ROOT` keeps dataset shards on the same persistent volume; rerunning dataset prep simply validates existing files.
- Use `MEGACONTEXT_SEED` to fix randomness per experiment so resumed runs stay deterministic.

---
## Overview

MegaContext uses **alternating optimization** [similar to GAN training] to co-train [[GistNet]], [[LensNet]], and lightweight base-model adapters without full end-to-end backprop through the discrete [[Focus Allocator]].

### Why Alternating Optimization?

**Goal:** Let all three modules co-adapt while maintaining stability and computational efficiency.

**Challenge:** Full end-to-end training would require backprop through:
- Discrete [[Focus Allocator]] decisions (non-differentiable)
- Long sequence unrolls (prohibitively expensive)
- Coupled multi-module dependencies

**Solution:** Short alternating phases where some modules are frozen while others learn from on-policy signals produced by the frozen parts. This:
- Stabilizes joint training without end-to-end gradients
- Produces supervision/targets from current policy state
- Allows modules to co-evolve through multiple cycles
- Maintains computational tractability

### Training Modules

- **[[GistNet]]** (`Gist`) — 32→1 compression, two levels; substitutability objective [2].
- **[[LensNet]]** (`LensNet`) — dual cross-attn (8k→6→8k); signed focus scores.
- **Base-LoRA** (`LoRA`) [1] — tiny adapters on the base LLM to improve gist compatibility.
- **[[Focus Allocator]]** — remains discrete and greedy (no relaxation).

---
## High-Level Workflow

```
Cycle N:
  ┌─────────────────────────────────────────────────────┐
  │ JT1: Update GistNet (fix LensNet + LoRA)            │
  │   → On-policy substitutability on replaced spans    │
  │   → Make gists better drop-in replacements          │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │ JT2: Update LensNet (fix GistNet + LoRA)            │
  │   → Generate counterfactual ΔNLL labels             │
  │   → Learn better focusing policy                    │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │ JT3: Update LoRA (fix GistNet + LensNet)            │
  │   → Task NLL with discrete Working Context          │
  │   → Adapt base to "like" gist tokens                │
  └─────────────────────────────────────────────────────┘
                          ↓
  Repeat for 3-5 cycles → Validation → Stop when stable
```

---
## Training Phases (Brief)

### Phase JT1 — Update GistNet
**Fix:** `LensNet`, `LoRA` | **Update:** `Gist`

With the current focusing policy fixed, make gists better drop-in replacements for exactly the places the policy cares about.
- Build/refresh [[MegaContext Tree]] with current `Gist`
- Run `LensNet` + [[Focus Allocator]] to form [[Working Context]]
- Optimize on **substitutability loss** (KL or ΔNLL@`H`) for replaced spans

### Phase JT2 — Update LensNet
**Fix:** `Gist`, `LoRA` | **Update:** `LensNet`

Given the current gists, learn a better focusing policy.
- Generate counterfactual labels (ΔNLL/ΔKL for expand/collapse candidates)
- Train with signed regression + ranking + budget regularizer
- Update-every-`K` cadence (Lens runs once per block)

### Phase JT3 — Update Base-LoRA
**Fix:** `Gist`, `LensNet` | **Update:** `LoRA` (small ranks)

Slightly adapt the base to work well with gist tokens and current working-context geometry.
- Task NLL@`H` with discrete [[Working Context]]
- Weak substitutability keep-alive loss
- Target input embeddings + first/last 2 attention blocks

---
## Schedule & Hyperparameters

- **Cycle length:** JT1 → JT2 → JT3 = one cycle. Repeat 3–5 cycles.
- **Step counts per phase:**
  - JT1 ([[GistNet]]): 2–4k steps
  - JT2 ([[LensNet]]): 2–4k steps
  - JT3 (LoRA): 1–2k steps
- **Batching:** mixed long-context tasks; block size `K=32`; horizon `H=64`
- **Checkpoints:** save after each phase; early-stop on validation Loss@`H` vs token budget

---
## When to Stop

- Validation Loss@`H` vs budget improves then plateaus across cycles
- Swap rate stabilizes; no ping-pong
- Ablations: freezing any one of {[[GistNet]], [[LensNet]], LoRA} causes measurable drop

**Outcome:** All three modules co-learn without the cost/fragility of full end-to-end training.

---
## Related Pages

### Detailed Documentation
- **[[Alternating Optimization]]** — Full details on EM-style training, phase procedures, stability tips, and failure modes
- **[[Telemetry]]** — Instrumentation, metrics, artifact handling, and validation checklist
- **[[POC Implementation]]** — POC acceptance criteria and example walkthroughs

### Component Training
- **[[GistNet Training]]** — Substitutability objectives, architecture details, and loss functions
- **[[LensNet Training]]** — Signed regression, ranking, budget regularization, and counterfactual labeling

### System Components
- **[[GistNet]]** — 32→1 compression encoder
- **[[LensNet]]** — Non-causal focus scorer
- **[[Focus Allocator]]** — Discrete greedy allocation policy
- **[[MegaContext Tree]]** — Hierarchical storage structure
- **[[Working Context]]** — Active token buffer assembly

## References

1. **LoRA** (Hu et al., 2021) — [[papers/LoRA|Analysis]] — Low-rank adaptation used in GistNet/LensNet training
2. **Knowledge Distillation** (Hinton et al., 2015) — [[papers/Knowledge Distillation|Analysis]] — Teacher-student framework for GistNet training

See [[Related Work]] for the complete bibliography of all research papers referenced throughout the documentation.

**Note on alternating optimization:** This training approach is similar to techniques used in GAN training (Goodfellow et al., 2014), where generators and discriminators are trained alternately. In MegaContext, we alternate between updating compression (GistNet), focusing (LensNet), and adaptation (LoRA) modules, allowing them to co-evolve without expensive end-to-end gradients through discrete allocation decisions.
