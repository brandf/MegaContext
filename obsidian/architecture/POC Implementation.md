---
tags:
  - architecture
summary: Comprehensive POC implementation guide consolidating all parameters, configurations, constraints, and module-specific settings from across the project.
---
# POC Implementation Guide

> **Status:** Historical reference for the notebook-era prototype. Use [[Training & Operations]] and `TODO.md` for the nanochat workflow; this page remains for context when porting legacy assumptions.

This document consolidates ALL POC-specific implementation details, parameters, configurations, and constraints. When implementing POC components, this is the **single source of truth** for:
- Parameter values
- Simplifications vs full vision
- Module configurations
- Technology stack
- Testing requirements

See [[MegaContext PRD Index]] for the active roadmap, [[POC Scope]] for historical guardrails, and [[Migration Plan - Nanochat Integration]] for the nanochat-specific scaffolding.

---

## PRD alignment & nanochat integration

- **Role of this note:** captures legacy POC shortcuts (e.g., Lightning loops, manual storage formats) so we can reference them while porting features into the nanochat fork described in [[Migration Plan - Nanochat Integration]].
- **Mapping to PRDs:**
  - [[MegaContext End-to-End Training]] replaces the alternating loop; use these parameters when wiring the nanochat trainer.
  - [[MegaAttention Training]] consumes the working-context hierarchy and LOD budgets defined below.
  - [[MegaPrediction Training]] pulls its wLOD readouts, ΔNLL horizons, and gist losses directly from the sections that follow.
- **Nanochat hooks:** wherever this note mentions `notebooks/` or Lightning scripts, assume we now implement the same logic inside `nanochat/train.py`, `nanochat/model.py`, or CLI helpers; keep the configuration tables for reference when setting nanochat defaults.

---

## Global POC Parameters

### Core Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **K** (Block size) | 32 tokens | Fixed for POC |
| **W_max** (Working Context budget) | 8,192 tokens | Configurable via YAML; future: 16k–32k |
| **H** (ΔNLL Horizon) | 64 tokens | For [[GistNet Training\|GistNet]] evaluation |
| **N_diff** (Max focus changes per step) | 4 actions | Expand/collapse limit per refocus |
| **Cooldown steps** | 2 iterations | Min time before block can flip actions |

### Hierarchy Configuration

| Level | Compression | Coverage | Token Cost in [[Working Context\|WC]] |
|-------|-------------|----------|--------------------------------------|
| **LOD0** | 1:1 (no compression) | 1 token | 32 tokens per block |
| **LOD1** | 32:1 | 32 LOD0 tokens | 1 token per [[Glossary#Gist / Gist Embedding|gist]] |
| **LOD2** | 1024:1 | 1,024 LOD0 tokens | 1 token per [[Glossary#Gist / Gist Embedding|gist]] |

**POC Limitation:** Two levels only (LOD0, LOD1, LOD2 root) - sufficient for moderate contexts.

**Total compression:** 32² = 1024× (with two layers)

### Focus Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| **τ_expand** | 0.20 | Minimum signed score for expansion |
| **τ_collapse** | 0.20 | Symmetric collapse threshold |

### Update Cadence

- **Refocus frequency:** Every K=32 tokens
- **[[LensNet]] scoring:** Once per refocus cycle
- **[[GistNet]] compression:** Inline during ingest (synchronous)

---

## POC Simplifications & Constraints

### What's Frozen in POC

1. **[[Glossary#Frozen Base Model|Base LLM]]:** No fine-tuning during initial loop; LoRA is follow-up work
2. **[[GistNet]] checkpoint:** Gists frozen to initial checkpoint during demo runs (no retraining)
3. **Hierarchy depth:** Fixed at 2 levels (LOD0, LOD1, LOD2 root)
4. **Block size:** K=32 hardcoded (no variable-length blocks)
5. **Storage:** RAM-resident (no disk I/O or memory-mapping in POC)

### What's Simplified in POC

1. **Synchronous updates:** Ingest → refocus → decode happens inline (no background workers)
2. **No streaming:** Entire [[Working Context]] resides in GPU memory (no paging)
3. **Simple initial focus:** May use recency bias before [[LensNet]] is trained
4. **Fixed thresholds:** τ_expand and τ_collapse hardcoded at 0.2 (no adaptive)
5. **Single base model:** Not multi-model or MoE
6. **Toy corpus:** Project docs instead of large-scale datasets

### Deferred Features

Post-POC enhancements (see [[Future Plan]]):
1. Disk-backed storage with memory-mapped files
2. LOD3+ hierarchy levels for billion-token contexts
3. Incremental tree updates (rebuild only affected subtrees)
4. Provenance tracking per node
5. Soft deletes / pruning tiers (see [[MegaCuration]])
6. Version management for multiple [[GistNet]] checkpoints
7. Differentiable focus router (learned [[Focus Allocator]])
8. KV-cache reuse across refocus steps
9. Multi-head contexts with different focus policies
10. Attention biasing for task-specific guidance

---

## Technology Stack

### Core Dependencies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Python** | Python | 3.11 | Core language |
| **PyTorch** | PyTorch | ≥2.2 | Tensor operations |
| **Transformers** | HuggingFace | Latest | Base model interface |
| **FlashAttention** | FlashAttention 2 | Latest | Efficient attention |
| **Environment** | `uv` | Latest | Dependency management |
| **Logging** | Weights & Biases | Latest | Metrics tracking |

### Key Commands

```bash
# Setup environment
uv venv
uv sync

# Run tests
uv run pytest --maxfail=1 --disable-warnings

# Legacy notebook workflow (deprecated)
uv run jupyter lab  # run notebooks/megacontext.ipynb and customise phases
uv run python -m tools.decode_demo --config configs/SampleText_TinyGPT2.yaml
```

### Base Model Configuration

**Primary choice:** `HuggingFaceTB/SmolLM3-3B`
- Precision: **bf16** (bfloat16)
- GPU requirement: 24–48 GB (for model + working context + training)
- Alternative: `Qwen/Qwen3-1.7B`

**Loading:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
```

---

## Module-Specific Configurations

### [[GistNet]] Parameters

See [[GistNet]] for architecture overview, [[GistNet Architecture Details]] for layer specs, [[GistNet Training]] for training details.

#### Architecture

| Parameter | POC Value | Notes |
|-----------|-----------|-------|
| Window size | 32 tokens | Fixed |
| Slot queries | 2 shared learned queries (Q₁, Q₂) | |
| Layers per 32→1 block | 2 self + 2 cross-attention | |
| Refinement stack | 32→1→32→1 | Two-stage compression |
| Embedding dim | Same as base LLM (e.g., 4096) | Must match |
| Internal hidden width | 512 | Bottleneck |
| Attention heads | 8 | |
| RoPE | Applied to tokens only, slots omit it | |
| Activation | GELU | |
| Norm | Pre-LayerNorm | |
| Parameters | ~0.5M per layer | ~1M total |
| Runtime | <1 ms per 32-token span | On NVIDIA L4, bf16 |
| Output | Single `g_final` vector per span | Dimension = embedding_dim |

#### Training Loss

```python
# Primary: Substitutability
Loss_subst = KL(P_base || P_replaced)  # Or ΔNLL@H over H=64 tokens

# Optional: Contrastive (prevent collapse)
Loss_contrast = max(0, margin - cosine_sim(g_i, g_j))  # margin ≈ 0.2

# Total
Loss = Loss_subst + 0.05 * Loss_contrast
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Long-form text (4k–16k tokens), chunked into 32-token spans |
| Teacher | Frozen base LLM for [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] computation |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | Cosine decay |
| Precision | bf16 |
| Curriculum | Start with contiguous text, then structured data (lists, code, tables) |

#### Hierarchy

- **Two 32→1 layers** stacked hierarchically
- Lower layer runs on token [[Glossary#Embedding|embeddings]]
- Upper layer operates on lower-layer [[Glossary#Gist / Gist Embedding|gist]] outputs
- Result: 32² = 1024 tokens per LOD2 [[Glossary#Gist / Gist Embedding|gist]]

---

### [[LensNet]] Parameters

See [[LensNet]] for overview, [[LensNet Training]] for training, [[LensNet Scoring]] for score computation.

#### Architecture

| Parameter | POC Value | Notes |
|-----------|-----------|-------|
| Input embeddings | ≈8k entries | Mixed LOD0/LOD1/LOD2 from [[Working Context]] |
| Conditioning [[Glossary#Gist / Gist Embedding|gists]] | 6 total | LOD2 root + 5 latest LOD1 [[Glossary#Gist / Gist Embedding|gists]] |
| Down-projection width | 512 (d_lens) | Bottleneck dimension |
| Attention heads | 8 | |
| Stacks | 1–3 dual cross-attention blocks | |
| Update cadence | Every K=32 tokens | |
| Output | Signed focus score u_i per entry | Range: [-1, +1] |
| Runtime | <3 ms per update @ 8k tokens | On NVIDIA L4 |
| Parameters | ≈100k–200k total | Tiny auxiliary network |

#### Complexity

- **O(N × K × d_lens)** per forward pass
- With N ≈ 8k, K = 6, d_lens = 512 → ~25M multiply-adds
- **Negligible** compared to base model decode

#### Training Loss

```python
# 1. Regression on signed utility targets
L_reg = MSE(predictions, targets)

# 2. Ranking loss for ordered pairs
L_rank = softplus_ranking_loss(score_pairs)

# 3. Budget regularizer (zero-sum preference)
L_budget = ((P - N) / (P + N))²  # P=positive scores, N=negative

# 4. Illegality penalties
L_illegal = α * illegal_expand_penalty + β * illegal_collapse_penalty  # α, β ≈ 0.3

# Total
L_total = L_reg + 0.5 * L_rank + 0.1 * L_budget + L_illegal
```

#### Conditioning Inputs

| Input | Shape | Purpose |
|-------|-------|---------|
| `context` | [N, d] | [[Working Context]] entry [[Glossary#Embedding|embeddings]] (≈8k) |
| `tail_gists` | [6, d] | LOD2 root + 5 latest LOD1 [[Glossary#Gist / Gist Embedding|gists]] |
| `levels` | [N] | 0/1/2 markers for legality masking |
| `span_width` | [N] | LOD0 tokens represented per entry |
| `distance_to_cursor` | [N] | Block distance from decode cursor |

---

### [[Focus Allocator]] Parameters

See [[Focus Allocator]] for algorithm, [[Focus Allocator Strategies]] for variations.

#### Runtime Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| **τ_expand** | 0.20 | Min score magnitude for expansion |
| **τ_collapse** | 0.20 | Symmetric collapse threshold |
| **N_diff** | 4 | Max expand/collapse actions per iteration |
| **cooldown_steps** | 2 | Min iterations before block can flip actions |
| **lens_update_interval** | 32 tokens (K) | [[LensNet]] runs once per block |
| **tail_gist_window** | 5 LOD1 + current LOD2 | Conditioning set for [[LensNet]] |

#### Constraints

1. **Block alignment:** Every [[Working Context|WC]] entry covers exactly one full 32-token block at a single [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)|LOD]]
2. **Action budget:** Apply at most N_diff=4 operations per iteration
3. **Positional alignment:** [[Glossary#Gist / Gist Embedding|Gists]] reuse absolute token indices for [[RoPE]]; occupy central token index of span
4. **Legality:** LOD0 blocks can't expand further; LOD2 [[Glossary#Gist / Gist Embedding|gists]] can't collapse higher (in POC)

#### Greedy Algorithm

```python
def focus_allocator_step(working_context, policy_scores, W_max):
    # 1. Collect candidates
    expand_queue = [(score, entry) for entry, score in zip(working_context, policy_scores)
                    if score > τ_expand and can_expand(entry)]
    collapse_queue = [(score, entry) for entry, score in zip(working_context, policy_scores)
                      if score < -τ_collapse and can_collapse(entry)]

    # 2. Sort queues
    expand_queue.sort(reverse=True)    # Descending (highest first)
    collapse_queue.sort()              # Ascending (most negative first)

    # 3. Apply N_diff operations
    actions_taken = 0
    while actions_taken < N_diff and (expand_queue or collapse_queue):
        # Prioritize expansions if budget available
        if expand_queue and current_budget_allows_expansion():
            score, entry = expand_queue.pop(0)
            expand(entry)  # LOD1→LOD0 or LOD2→LOD1
            actions_taken += 1

        # Balance with collapses
        if collapse_queue:
            score, entry = collapse_queue.pop(0)
            collapse(entry)  # LOD0→LOD1 or LOD1→LOD2
            actions_taken += 1

    return working_context
```

---

### [[Working Context]] Parameters

See [[Working Context]] for overview, [[Working Context Assembly]] for materialization, [[Working Context Refocusing]] for focus changes.

#### Budget

| Parameter | POC Value | Future |
|-----------|-----------|--------|
| **W_max** | 8,192 tokens | 16k–32k |

#### Entry Costs

- **LOD0 block (32 tokens):** 32 tokens
- **LOD1 [[Glossary#Gist / Gist Embedding|gist]]:** 1 token (saves 31)
- **LOD2 [[Glossary#Gist / Gist Embedding|gist]]:** 1 token (saves 1023)

#### Budget Invariant

```python
assert sum(entry.cost for entry in working_context) <= W_max
```

See [[Invariants]] for all system [[invariants]].

#### Refocus Cycle

```
Every K=32 tokens:
  1. Decode K tokens using current WC
  2. Ingest new tokens to MegaContext Tree
  3. LensNet scores all WC entries
  4. Focus Allocator applies up to N_diff=4 operations
  5. Repeat with updated WC
```

---

### [[MegaContext Tree]] Parameters

See [[MegaContext Tree]] for overview, [[Storage Format]] for binary layout, [[Tree Operations]] for APIs.

#### Tree Structure

| Level | Compression | Coverage | Entry Type |
|-------|-------------|----------|------------|
| **LOD0** | 1:1 | 1 token | Token ID (uint32) |
| **LOD1** | 32:1 | 32 tokens | [[Glossary#Gist / Gist Embedding|Gist]] vector (fp16) |
| **LOD2** | 1024:1 | 1,024 tokens | [[Glossary#Gist / Gist Embedding|Gist]] vector (fp16) |

#### Tree Properties

- **Fixed branching factor:** 32 children per node
- **Perfect alignment:** Node boundaries align with 32-token blocks
- **Append-only:** Historical nodes immutable (except [[Glossary#Gist / Gist Embedding|gist]] refresh)
- **Balanced growth:** Depth grows as log₃₂(N)

#### Storage Layout

See [[Storage Format]] for complete details.

**Files:**
- `LOD0.ctx` - Raw token IDs (uint32)
- `LOD1.ctx` - LOD1 [[Glossary#Gist / Gist Embedding|gist]] vectors (fp16, dimension = embedding_dim)
- `LOD2.ctx` - LOD2 [[Glossary#Gist / Gist Embedding|gist]] vectors (fp16, dimension = embedding_dim)
- `metadata.json` - Tree metadata and configuration

**Header (64 bytes):**
| Offset | Field | Type | Value |
|--------|-------|------|-------|
| 0 | magic | uint32 | 0x4D434354 ("MCCT") |
| 4 | version | uint16 | 1 (POC) |
| 6 | level | uint16 | 0, 1, or 2 |
| 8 | block_size | uint16 | 32 |
| 10 | embedding_dim | uint16 | Base model dimension |
| 12 | dtype_code | uint16 | 0=uint32, 1=fp16, 2=bf16 |
| 14 | model_name | char[32] | UTF-8 null-terminated |
| 46 | reserved | 18 bytes | Zeroed |

**POC Simplification:** RAM-resident (no disk I/O or memory-mapping yet)

---

## Sample Configuration File

**File:** `configs/Gutenberg_SmolLM3.yaml`

```yaml
name: Gutenberg_SmolLM3
description: Gutenberg subset with SmolLM3-3B base model and two-stage GistNet training.

dataset:
  dataset_name: gutenberg_sample
  tokenizer: HuggingFaceTB/SmolLM2-360M-Instruct
  block_size: 32
  context_tokens: 512
  context_stride: 512
  horizon: 32
  teacher_model: HuggingFaceTB/SmolLM2-360M-Instruct
  splits:
    train:
      source: ../data/raw/gutenberg/**/*.txt
      output_path: ../data/gutenberg_sample/train.arrow

base_model:
  name: HuggingFaceTB/SmolLM3-3B
  torch_dtype: bfloat16
  run_name: poc_smollm3_l4

gistnet:
  model:
    hidden_size: auto
    block_size: 32
    num_heads: 16
    mlp_ratio: 4.0
  training:
    batch_size: 8
    precision: bf16-mixed
    phases:
      - name: pooling-pretrain
        objective: pooling_mse
        max_steps: 2000
        window_tokens: 512
        lr: 0.001
      - name: delta-finetune
        objective: delta_nll
        max_steps: 1000
        window_tokens: 512
        lr: 0.0005
```

---

## Testing Requirements

### Determinism

All POC tests must be deterministic:
- **Seeded RNG:** Fixed random seeds for reproducibility
- **Deterministic blocks:** 32-token blocks from dataset prep
- **Round-trip tests:** Tree persistence and recovery
- **Synthetic streams:** Deterministic test inputs

### Smoke Tests (CI-friendly)

```python
# Dataset tooling
test_dataset_prep_deterministic()

# Base model loading
test_base_model_loads()

# Tensor shapes
test_gistnet_output_shapes()
test_lensnet_output_shapes()

# Budget calculations
test_working_context_budget_calculation()

# Legality masks
test_focus_allocator_legality_masks()
```

### Unit Tests

```python
# Tree operations
test_megacontext_tree_ingest()
test_megacontext_tree_persistence()

# Focus allocator
test_focus_allocator_greedy_algorithm()
test_focus_allocator_cooldown()
test_focus_allocator_edge_cases()

# LensNet
test_lensnet_conditioning_inputs()
test_lensnet_score_computation()

# GistNet
test_gistnet_determinism()
test_gistnet_loss_computation()
test_gistnet_substitutability()  # ≤5% ΔNLL threshold
```

### Integration Tests

```python
# End-to-end
test_poc_loop_with_synthetic_stream()
test_budget_invariants_maintained()
test_focus_reallocation_logging()

# Dataset prep
test_dataset_prep_on_sample_corpus()

# Base model integration
test_base_model_forward_passes()
```

### Evaluation Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **[[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]]** | ≤0.1 | Compare base vs [[Glossary#Gist / Gist Embedding|gist]]-replaced predictions |
| **Overhead** | ≤5% | Latency with vs without MegaContext |
| **Swap rate** | 0.1–0.3 actions/block | Log focus changes per iteration |
| **Budget compliance** | 100% | Assert [[invariants]] never violated |

---

## Design Principles

### Tensor-First Philosophy

- Keep [[Glossary#Gist / Gist Embedding|gist]]-side components tensor-first
- Prefer thin Python wrappers around PyTorch modules
- Persist [[MegaContext Tree|MegaContext]] structures as **contiguous LOD0/LOD1/LOD2 tensors**
- Mirror on-disk layouts instead of dense Python object graphs

### Curriculum Training

- Masked-attention curriculum (per Gist Token paper)
- Progressively shrink working window during training
- Balance context richness against storage/compute budgets

### Working Context Management

**MegaContext wrapper:**
- Owns contiguous LOD0/LOD1/LOD2 buffers
- Encapsulates offsets/parent pointers
- Provides iterators for enumerating legal window-sized views

**WorkingContext wrapper:**
- Provides views for token [[Glossary#Embedding|embeddings]] vs [[Glossary#Gist / Gist Embedding|gist]] [[Glossary#Embedding|embeddings]]
- Utilities to materialize KV-cache keys for chosen slices
- Combinator utilities for span replacement with specific [[Glossary#Gist / Gist Embedding|gist]] levels

---

## Key Invariants

See [[Invariants]] for comprehensive list. POC must maintain:

1. **Budget Invariant:** `sum(entry_costs) ≤ W_max`
2. **Contiguity Invariant:** `entry[i].end_token == entry[i+1].start_token`
3. **Block Alignment Invariant:** All boundaries align with K=32 blocks
4. **Level Consistency Invariant:** LOD0=32 tokens, LOD1=32 tokens (compressed), LOD2=1024 tokens
5. **[[RoPE]] Invariant:** [[Glossary#Gist / Gist Embedding|Gists]] use central position index; LOD0 uses actual positions

---

## What Success Looks Like

A successful POC demonstrates:

1. **[[GistNet]]** compresses 32 tokens → 1 [[Glossary#Gist / Gist Embedding|gist]] with [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] ≤ 0.1
2. **[[LensNet]]** predicts relevance and guides focus changes
3. **[[Focus Allocator]]** maintains [[invariants]] while adapting [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)|LOD]]
4. **[[Working Context]]** stays within budget while handling dynamic context
5. **[[MegaContext Tree]]** grows unboundedly while access stays constant-time
6. **End-to-end system** achieves <5% overhead vs frozen base model
7. **Reproducible demos** show focus adapting to changing queries

---

## Related Pages

- [[MegaContext PRD Index]] - Current milestone roadmap and phases
- [[POC Scope]] - Capability guardrails and what's excluded
- [[POC Architecture]] - Module responsibilities and interfaces
- [[Training & Operations]] - Training cadence and telemetry
- [[Architecture Details]] - Full system design
- [[System Properties]] - Core properties (constant compute, etc.)

---

*This is the **single source of truth** for POC implementation details. When component files say "see POC Implementation for details," this is where to look.*
