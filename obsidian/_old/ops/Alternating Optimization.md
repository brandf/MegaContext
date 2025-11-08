---
tags:
  - ops
summary: Alternating optimization strategy for co-training GistNet, LensNet, and LoRA adapters through EM-style phase cycling.
---
Alternating optimization enables co-training of [[GistNet]], [[LensNet]], and base-model adapters without full end-to-end backpropagation through the discrete [[Focus Allocator]]. By cycling through phases where different components are frozen and updated, the system learns stable policies while maintaining computational efficiency.

---

## Why Alternating Optimization is Needed

**Core Challenge:** The [[MegaContext Tree]] architecture involves three interdependent modules—[[GistNet]] (compression), [[LensNet]] (focusing), and base-model adapters (LoRA)—plus a discrete [[Focus Allocator]] that makes non-differentiable expand/collapse decisions.

**Problem with End-to-End Training:**
- Full backpropagation through long unrolls of the [[Focus Allocator]] is computationally prohibitive
- The discrete allocator has no gradients to flow through
- Joint optimization causes instability: when all modules change simultaneously, they chase moving targets

**Solution: EM-Style Alternating Phases**
The alternating optimization schedule mimics Expectation-Maximization:
- **E-like step:** Hold policy components fixed to generate supervision signals (e.g., counterfactual ΔNLL utilities)
- **M-like step:** Update another module to better fit those targets

This approach:
- **Stabilizes training** by providing fixed targets for each update phase
- **Enables on-policy learning** without requiring differentiable allocators
- **Allows co-adaptation** of all components across multiple cycles
- **Reduces oscillation** compared to simultaneous updates

```
┌─────────────────────────────────────────────────────┐
│  Challenge: Three modules + discrete allocator      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ GistNet  │  │ LensNet  │  │   LoRA   │          │
│  │(compress)│  │ (focus)  │  │ (adapt)  │          │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        └──────────────┴──────────────┘              │
│                       │                             │
│              ┌────────▼────────┐                    │
│              │ Focus Allocator │                    │
│              │   (discrete!)   │                    │
│              └─────────────────┘                    │
│                                                      │
│  Solution: Alternate which module learns            │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1: GistNet Training (JT1)

**Objective:** Make gists better drop-in replacements for the spans currently being compressed by the policy.

**Fixed Components:** [[LensNet]], LoRA adapters
**Updated Component:** [[GistNet]]

### Procedure

1. **Build/refresh [[MegaContext Tree]]** using the current `Gist` model
2. **For each training block** (size `K=32`):
   - Run [[LensNet]] + [[Focus Allocator]] to determine expand/collapse actions
   - Form the [[Working Context]] used by the base LLM
3. **Optimize [[GistNet]]** on spans that were actually compressed:

**Loss Functions:**

```
L_GistNet = L_substitutability + λ₁·L_stability + λ₂·L_boundary

where:
  L_substitutability = KL(p_full || p_replaced) or ΔNLL@H
  L_stability       = ||gist_current - gist_checkpoint||₂
  L_boundary        = weighted ΔNLL on edge tokens
```

#### Substitutability Loss
- **Primary objective:** Minimize ΔNLL at horizon `H=32–128` for gists that were inserted into the [[Working Context]]
- **Measures:** How well the compressed gist preserves next-token prediction compared to full tokens
- **Computed:** Only on spans that the current policy actually used (on-policy)

#### Stability Loss (Optional)
- **Purpose:** Prevent catastrophic drift between training steps
- **Method:** LOD2 distance between current gist embeddings and previous checkpoint
- **Helps:** Maintain consistency as [[LensNet]] learns its policy

#### Boundary Auxiliary Loss (Optional)
- **Purpose:** Preserve semantics at compression boundaries
- **Method:** Upweight ΔNLL terms on edge tokens of compressed spans
- **Addresses:** Boundary artifacts where critical tokens straddle compression blocks

### Intuition

With the focusing policy fixed, [[GistNet]] learns to encode exactly what the allocator needs. Since [[LensNet]] chooses which spans to expand/collapse, [[GistNet]] gets signal on real-world usage patterns rather than arbitrary compressions.

### Hyperparameters

- **Steps per cycle:** 2–4k gradient steps
- **Horizon:** `H = 64` tokens
- **Block size:** `K = 32` tokens
- **Optimizer:** AdamW (bf16) with cosine LR schedule + warmup

---

## Phase 2: LensNet Training (JT2)

**Objective:** Learn a better focusing policy given the current quality of gist compressions.

**Fixed Components:** [[GistNet]], LoRA adapters
**Updated Component:** [[LensNet]]

### Procedure

1. **Generate counterfactual labels** on on-policy snapshots:
   - For each candidate expand/collapse action in the current [[Working Context]]
   - Compute ΔNLL or ΔKL utilities (batched for efficiency)
   - Convert to signed utility scores per token:
     - **Positive** for expand actions (decompression improves prediction)
     - **Negative** for collapse actions (compression doesn't hurt much)

2. **Train [[LensNet]]** with multiple objectives:

**Loss Functions:**

```
L_LensNet = L_utility + λ₁·L_budget + λ₂·L_legality

where:
  L_utility  = signed regression + ranking loss
  L_budget   = zero-sum regularizer (token-cost weighted)
  L_legality = penalties for invalid actions
```

#### Utility Loss
- **Signed regression:** Predict positive utility for helpful expands, negative for safe collapses
- **Ranking loss:** Within each snapshot, correctly order actions by utility
- **Teaches:** Which focus shifts provide the most value

#### Budget Regularizer
- **Zero-sum constraint:** Total token cost must stay within `W_max`
- **Token-cost weighted:** Each action weighted by its impact on budget
- **Enforces:** Resource constraints during training

#### Legality Penalties
- **Runtime masking:** Prevent invalid actions (e.g., expanding already-expanded nodes)
- **Structural constraints:** Respect tree topology and residency rules

### Update Cadence

[[LensNet]] runs **once per block** (every `K=32` tokens), so training updates follow the same `K`-step cadence to match operational deployment.

### Intuition

Given the current gist quality from [[GistNet]], [[LensNet]] learns which parts of the [[MegaContext Tree]] are most valuable to decompress. The signed utility labels provide direct supervision: "expand this because it helps prediction" or "collapse that because it's not being used."

### Hyperparameters

- **Steps per cycle:** 2–4k gradient steps
- **Counterfactual sampling:** Batch evaluation of expand/collapse candidates
- **Optimizer:** AdamW (bf16) with cosine LR schedule + warmup
- **Hysteresis:** Minimum residency prevents expand/collapse thrashing

---

## Phase 3: Base-LoRA Training (JT3)

**Objective:** Adapt the base LLM to work better with gist tokens and the current [[Working Context]] geometry.

**Fixed Components:** [[GistNet]], [[LensNet]]
**Updated Component:** LoRA adapters (low-rank)

### LoRA Placement

Keep adapters **small and targeted**:
- **Input embedding projection**
- **QKV/O matrices** of either:
  - First 2 attention blocks, OR
  - Last 2 attention blocks
  - (Pick one set, not both)

**Goal:** Interface alignment, not knowledge injection

### Loss Functions

```
L_LoRA = L_task + λ₁·L_keepalive + λ₂·L_teacher

where:
  L_task      = NLL@H with discrete Working Context
  L_keepalive = weak substitutability constraint
  L_teacher   = KL to teacher-with-MegaContext (optional)
```

#### Task Loss
- **Primary objective:** Standard next-token prediction at horizon `H`
- **Context:** Uses the discrete [[Working Context]] produced by [[LensNet]] + [[Focus Allocator]]
- **Teaches:** How to predict well given mixed-LOD inputs

#### Substitutability Keep-Alive (Weak)
- **Purpose:** Prevent gist semantics from drifting away from base model understanding
- **Weight:** Low `λ₁` to avoid interfering with task objective
- **Ensures:** Gists remain interpretable to the adapted model

#### Teacher Distillation (Optional)
- **If available:** Distill from a larger teacher model that also uses [[MegaContext Tree]]
- **Provides:** Additional supervision signal for quality

### Intuition

Slightly nudge the base LLM to "like" gist tokens and handle the positional geometry of mixed full/compressed contexts. The adapters help with:
- **Positional anchoring:** Understanding RoPE phases with inserted gists
- **Embedding variance:** Handling the distribution shift from gist vectors
- **Attention patterns:** Learning to attend across LOD boundaries

### Hyperparameters

- **Steps per cycle:** 1–2k gradient steps (shorter than JT1/JT2)
- **LoRA rank:** `r = 4–16` (keep tiny)
- **Learning rate:** Lower than other phases
- **Optimizer:** AdamW (bf16) with cosine LR schedule + warmup

---

## Iteration and Convergence

### Cycle Structure

One complete cycle consists of: **JT1 → JT2 → JT3**

```
┌────────────────────────────────────────────────────┐
│                  Training Cycle                     │
├────────────────────────────────────────────────────┤
│                                                     │
│  ╔═══════════════╗                                 │
│  ║ Phase JT1     ║  2-4k steps                     │
│  ║ Update GistNet║  Fix: LensNet, LoRA             │
│  ╚═══════╤═══════╝                                 │
│          │                                          │
│          │ (regenerate ΔNLL labels)                │
│          ▼                                          │
│  ╔═══════════════╗                                 │
│  ║ Phase JT2     ║  2-4k steps                     │
│  ║ Update LensNet║  Fix: GistNet, LoRA             │
│  ╚═══════╤═══════╝                                 │
│          │                                          │
│          │ (use updated policy)                    │
│          ▼                                          │
│  ╔═══════════════╗                                 │
│  ║ Phase JT3     ║  1-2k steps                     │
│  ║ Update LoRA   ║  Fix: GistNet, LensNet          │
│  ╚═══════╤═══════╝                                 │
│          │                                          │
│          │ (checkpoint all modules)                │
│          └─────────► Repeat cycle                  │
│                                                     │
└────────────────────────────────────────────────────┘
```

### Number of Cycles

**Recommended:** 3–5 complete cycles

**Rationale:**
- First cycle: Modules discover basic co-adaptation strategies
- Middle cycles: Refinement and stabilization
- Final cycles: Fine-tuning until convergence plateau

### Convergence Criteria

Stop training when:

1. **Validation Loss@H vs budget** improves then plateaus across cycles
2. **Swap rate stabilizes** — no expand/collapse ping-pong
3. **Ablation test passes** — freezing any single module causes measurable performance drop

**Metrics to Track:**
- `loss_at_h`: Task loss at horizon H
- `swap_rate`: Actions per block (target ≤ 0.25)
- `mean_residency`: Block lifetime in [[Working Context]] (target ≥ 3 iterations)
- `delta_nll_degradation`: Compression quality (target ≤ 0.1 at `W_max=8k`)

### Warm Start Strategy

**Before the first JT1:**
- Do a short sequential pretrain of [[GistNet]], then [[LensNet]]
- Reduces early oscillations by establishing reasonable initial policies
- Prevents modules from starting completely misaligned

---

## Coordination Between Phases

### Data Flow Across Phases

```
┌─────────────────────────────────────────────────────┐
│              Data Flow Per Cycle                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  JT1: Update GistNet                                │
│  ┌─────────────────────────────────┐                │
│  │ 1. Freeze LensNet & LoRA        │                │
│  │ 2. Decode blocks with current   │                │
│  │    Working Context              │                │
│  │ 3. Update Gist using on-policy  │                │
│  │    substitutability losses      │                │
│  └──────────────┬──────────────────┘                │
│                 │                                    │
│                 │ (same blocks)                     │
│                 ▼                                    │
│  JT2: Update LensNet                                │
│  ┌─────────────────────────────────┐                │
│  │ 1. Freeze GistNet & LoRA        │                │
│  │ 2. Compute counterfactual       │                │
│  │    utilities on expand/collapse │                │
│  │ 3. Update LensNet with signed   │                │
│  │    utilities + budget losses    │                │
│  └──────────────┬──────────────────┘                │
│                 │                                    │
│                 │ (updated policy)                  │
│                 ▼                                    │
│  JT3: Update LoRA                                   │
│  ┌─────────────────────────────────┐                │
│  │ 1. Freeze GistNet & LensNet     │                │
│  │ 2. Run normal blocks with       │                │
│  │    LensNet + Allocator active   │                │
│  │ 3. Update LoRA on Task NLL@H    │                │
│  │    + weak keep-alive loss       │                │
│  └─────────────────────────────────┘                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Critical Synchronization Points

#### After JT1: On-Policy Label Regeneration

**CRITICAL:** Always regenerate ΔNLL utility labels after completing JT1

**Why:** [[GistNet]] has updated, so the gist quality has changed. [[LensNet]] must train on counterfactuals computed with the *current* gists, not stale ones from the previous cycle.

**Process:**
1. Complete JT1 phase (update [[GistNet]])
2. Re-encode all training examples with updated gists
3. Recompute counterfactual ΔNLL for expand/collapse candidates
4. Begin JT2 with fresh labels

#### After JT2: Policy Update

**Effect:** [[LensNet]] now makes different focus decisions

**Implication:** The [[Working Context]] geometry changes, affecting which spans [[GistNet]] needs to compress well in the next cycle.

This feedback loop drives co-adaptation:
- [[LensNet]] focuses on high-value regions
- [[GistNet]] learns to compress those regions well
- LoRA adapts the base model to the current policy
- Repeat

### Checkpointing Strategy

**After each phase:**
- Save module weights under `artifacts/checkpoints/`
- Store counterfactual utility tables under `artifacts/deltas/` (Parquet/Arrow format)
- Mirror the unified experiment config under `configs/`

**Enables:**
- Rollback if a phase destabilizes training
- Ablation studies comparing different phase combinations
- Reproducibility of experiments

---

## Stability & Efficiency Tips

### Preventing Oscillation

**Hysteresis in [[Focus Allocator]]:**
- **Minimum residency steps:** Blocks must stay in [[Working Context]] for ≥3 iterations
- **Cooldown periods:** After collapse, wait before allowing re-expansion
- **Prevents:** Expand/collapse thrashing during JT2/JT3

**Stability Regularization:**
- Small LOD2 penalty on gist drift in JT1
- Low learning rates for LoRA in JT3
- Cosine LR schedules with warmup per phase

### Computational Efficiency

**Batching Strategy:**
- Mixed long-context tasks per batch
- Block size `K=32` for manageable memory
- Horizon `H=64` for next-token prediction
- Target ~8k effective tokens per microbatch
- Use gradient accumulation (e.g., 2 microbatches × 4 sequences) to fit 24GB GPUs

**Counterfactual Sampling:**
- Batch evaluate expand/collapse candidates in JT2
- Reuse base model forward passes across counterfactuals
- Cache intermediate activations when possible

**LoRA Efficiency:**
- Keep ranks small (`r=4–16`)
- Update only critical layers (embedding + 2 attention blocks)
- Goal is interface alignment, not knowledge injection

### Curriculum Strategy

**Start simple, add complexity:**

1. **Early cycles:** Narrative and document tasks (smoother gradients)
2. **Middle cycles:** Add structured content (lists, tables)
3. **Late cycles:** Add code and highly structured formats

**Rationale:** Let modules stabilize on easier tasks before introducing hard cases with strict boundary requirements.

---

## Telemetry & Monitoring

### Required Metrics (Per Iteration)

Stream to Weights & Biases or similar:

| Metric | Target | Purpose |
|--------|--------|---------|
| `loss_at_h` | Decreasing | Task performance at horizon H |
| `swap_rate` | ≤ 0.25 | Focus stability (actions/block) |
| `mean_residency` | ≥ 3 | Block lifetime in WC |
| `delta_nll_degradation` | ≤ 0.1 | Compression quality vs full context |
| `latency_ms` | Baseline +10% | Per-block processing time |
| `token_budget_utilization` | Near `W_max` | Efficiency of focus allocation |

### Diagnostic Traces

**Allocator Action Traces:**
- Log every expand/collapse decision with:
  - Block ID, action type, utility score
  - Token budget before/after
  - Residency time of affected blocks

**Purpose:** Diagnose regressions and oscillation patterns

### Validation Tests

**Run after each phase:**

1. **Substitutability test:** ΔNLL@H on held-out compression examples
2. **Focus quality test:** Compare utility predictions vs actual ΔNLL
3. **Boundary test:** Synthetic examples with critical tokens at block edges
4. **Non-causal test:** Verify [[LensNet]]'s C1/C2 attention patterns

---

## Outcome

After 3–5 cycles of alternating optimization:

**[[GistNet]]** learns to encode **what the policy needs** — compressing spans that [[LensNet]] will actually collapse, with quality matching the regions where expansion is rare.

**[[LensNet]]** learns to choose **expansions that actually help** — focusing on high-utility regions given current gist quality, with stable budget-respecting policies.

**LoRA** nudges the base LLM to be **friendlier to mixed-LOD inputs** — handling positional geometry and embedding variance from gist tokens without knowledge drift.

**System-level benefits:**
- Co-adapted modules work better together than independently trained components
- Stable policies without full end-to-end backpropagation cost
- Interpretable training dynamics (each phase has clear objectives)
- Efficient compute (no long unrolls or differentiable relaxations)

**Verification:** All three modules co-learn effectively — freezing any one causes measurable performance degradation in ablation tests.

---

## Related Pages

### Training & Operations
- [[Training & Operations]] — Full training, instrumentation, and validation guide
- [[Telemetry]] — Comprehensive metrics, logging, and monitoring infrastructure
- [[Base Runtime]] — Runtime loop and inference engine architecture
- [[Performance Sketch]] — Latency, memory, and throughput analysis

### Component Training
- [[GistNet Training]] — Phase 1 (JT1): Compression model training procedures
- [[LensNet Training]] — Phase 2 (JT2): Focus policy learning methodology
- [[GistNet Architecture Details]] — Encoder/decoder architecture and loss functions
- [[LensNet Scoring]] — Utility prediction and counterfactual evaluation

### Core Components
- [[GistNet]] — Compression module architecture
- [[LensNet]] — Focus scoring and attention mechanism
- [[Focus Allocator]] — Discrete expand/collapse decision logic
- [[Focus Allocator Strategies]] — Greedy selection, hysteresis, and cooldown strategies
- [[Working Context]] — Assembly of the active token window
- [[Working Context Assembly]] — Construction of mixed LOD token sequences
- [[Working Context Refocusing]] — Dynamic focus shift operations
- [[MegaContext Tree]] — Hierarchical compression structure

### Implementation & Planning
- [[POC Plan]] — Phase-by-phase implementation roadmap
- [[POC Implementation]] — Technical implementation details and milestones
- [[POC Architecture]] — Simplified architecture for proof-of-concept
- [[POC Scope]] — Feature scope and acceptance criteria

### System Architecture
- [[Architecture]] — Complete system overview
- [[Architecture Details]] — In-depth architectural specifications
- [[Runtime Loop]] — Per-block execution cycle
- [[System Properties]] — Constant-time guarantees and resource bounds
- [[Invariants]] — Critical system invariants and constraints

### Related Concepts
- [[Ops]] — Operations and training overview
- [[Components]] — Component architecture index
- [[How MegaContext Works]] — High-level system explanation
- [[Context Focus]] — Introduction and quick start guide

---

## References

**Phase definitions:** [[Training & Operations#Joint training (alternating / "EM-style")]]
**Schedule details:** [[Training & Operations#Schedule & hyperparameters]]
**Convergence criteria:** [[Training & Operations#When to stop]]
**Stability tips:** [[Training & Operations#Stability & efficiency tips]]
