---
tags:
  - ops
summary: Alternating optimization, instrumentation, and validation practices for GistNet, LensNet, and runtime components.
---
Rotate through [[GistNet]], [[LensNet]], and LoRA updates while collecting telemetry that keeps ΔNLL, swap rates, and latency within targets.

---

- **Alternating phases:** JT1 ([[GistNet]]), JT2 ([[LensNet]]), JT3 (LoRA).
- **On-policy labeling:** regenerate ΔNLL utilities after each [[GistNet]] update.
- **Telemetry:** log loss@H, swap rate, residency, latency for every block.
- **Validation:** acceptance criteria tie to ΔNLL degradation ≤0.1 at `W_max=8k`.
- **Tooling:** rely on `uv run pytest`, W&B logging, and artifacts directories for repeatability.

---
## Overview

MegaContext uses **alternating optimization** ("EM-style") to co-train [[GistNet]], [[LensNet]], and lightweight base-model adapters without full end-to-end backprop through the discrete [[Focus Allocator]].

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

- **[[GistNet]]** (`Gist`) — 32→1 compression, two levels; substitutability objective.
- **[[LensNet]]** (`LensNet`) — dual cross-attn (8k→6→8k); signed focus scores.
- **Base-LoRA** (`LoRA`) — tiny adapters on the base LLM to improve gist compatibility.
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
