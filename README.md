# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory. This document is both a conceptual overview and a technical design spec for an initial proof-of-concept (POC).*

---
## TL;DR — MegaContext
MegaContext is a proposed system architecture for virtualized LLM context - think “MegaTexture for text.”, if you're familiar with this graphics concept.

It separates a model’s context into a lifetime context (a hierarchical gist tree stored on disk) and a working context (a fixed-size mix of tokens and gists on GPU).  A standard (even pre-trained) LLM then operates on the working context.

A lightweight learned LensNet (and streaming focus allocator) continuously/incrementally refocus the full lifetime context onto the working context, giving the model effectively infinite memory at constant compute.

The next section walks through how the runtime loop stays within a fixed working context while tracking the entire lifetime history. Curious about the long-term implications? Jump to [Grand vision](#grand-vision-why-this-matters) near the end; the intervening sections drill into the proof-of-concept (POC) mechanics.

---

## How MegaContext Works

Large language models are constrained by a fixed context window.  
MegaContext removes this limit by separating:

- **Lifetime context** — the complete interaction or document history (potentially millions or billions of tokens) stored as a *hierarchical gist tree* on disk (RAM for the POC).  
- **Working context** — a fixed 8k–32k token budget on GPU, mixing raw tokens with gists drawn from the lifetime tree. The frozen base LLM sees only this window, which stays contiguous in “time” even as individual spans switch between token-level and gist-level representations.

### Core components

- **Lifetime gist tree** — built incrementally as text streams in (every 32 tokens → L1 gist; every 32 L1 gists → L2 gist; etc.).  
- **Working context** — contiguous window over the tree; total token cost is capped by `W_max`.  
- **GistNet** — a lightweight network that compresses local spans (e.g., 32→1) into **gists** that act as substitutable stand-ins for their source tokens. Stacking gists-of-gists yields a hierarchical, lossy representation of the full lifetime history.  
- **LensNet + focus allocator** — LensNet scores each working-context entry (token embedding or gist) for expansion or collapse; a block-aligned focus allocator applies those scores, streaming finer- or coarser-grained entries in and out while respecting contiguity and the budget.

### Analogy: MegaTexture → MegaContext
This is not required to understand MegaContext, but for those that are interested in learning about the inspiration [this video](https://www.youtube.com/watch?v=BiQCz2NjPR8) provides a good overview of the problems Mega Texture solves.
- In graphics, **MegaTexture** streams the visible portions of a vast texture mipmap into GPU memory at the appropriate resolution.  
- **MegaContext** mirrors that idea for language: instead of mipmap tiles, it maintains embeddings at multiple levels of detail (token L0, gist L1, gist L2, …), yielding effectively unbounded context for a frozen LLM.

### Intuitions / Motivation
The core intuition that's motivating this work is that long context is only useful if the model can focus on the relevant parts and ignore distractors (efficiently).  
- "Relevant parts" is inherently non-causal (something that wasn't previously relevant can become relevant), so this implies dynamic focusing/defocusing.  One-way compression/summarization schemes are fundamentally flawed.
- Exciting new future LLM scenarios will be unlocked at 100M+ context lengths, and at this scale both memory and compute requirements must be sub-linear to be practical for widespread consumer applications.

---

### Runtime lifecycle at a glance

```
Streaming text  ──► Lifetime Gist Tree  ──►  Focus Allocator  ──►  Working Context  ──►  Frozen Base LLM ──► Next Token Prediction
                               ▲                    ▲          
                               └───────── LensNet ──┘
```

1. **Ingest & summarize.** Buffer incoming tokens in 32-token blocks, roll them into new or updated gist nodes, and persist the lifetime tree (disk later, RAM for the POC).
2. **Assemble the working context.** Lay out a contiguous-in-time sequence of tokens and gists whose combined token-equivalent cost stays within `W_max`. Every position represents exactly one interval of the lifetime history at some level of detail.
3. **Refocus.** LensNet reads the current working context (plus tail gists), emits signed focus scores, and the (currently greedy) focus allocator applies block-aligned expansions/collapses without breaking contiguity or budget.
4. **Decode.** The frozen base LLM consumes the refreshed working context to predict the next token(s), feeding newly generated tokens back into step 1.

**Update cadence & buffering.**
- **Lifetime tree maintenance:** Both user tokens and model-generated tokens are buffered until a full 32-token block (L0) or 32 L1 children are available before rebuilding the corresponding gist nodes. This keeps gist updates block-aligned and prevents churn in the hierarchy.
- **LensNet conditioning gists:** LensNet only refreshes its conditioning set on its own schedule (e.g., every 256 working-context entries). Those gists can be read from the lifetime tree or recomputed lazily immediately before each LensNet call; either path observes the same block-aligned buffers.

> **Diagram needed — `assets/runtime_flow.png`:** Visualize the streaming loop from incoming tokens → lifetime gist tree → focus allocator → working context → frozen LLM, with LensNet providing feedback into the allocator.

The next sections unpack each stage: lifetime storage, compression (GistNet), focus control (LensNet + focus allocator), and the training schedule that keeps them aligned.

---

### Key terms & invariants

| Term | Meaning |
|------|---------|
| `Lifetime context` | Full, append-only history stored as a hierarchical gist tree (disk later, RAM for the POC). |
| `Working context` (`WC`) | Fixed-size GPU window (8k–32k token budget) that the base LLM sees; built from contiguous-in-time entries. |
| Working-context entry | Either a block of raw tokens (`L0`) or a gist summarizing that block or its ancestors (`L1`, `L2`, …). Exactly one entry covers each moment in the lifetime history. |
| `L0 / L1 / L2` | Level of detail (LOD): `L0`=tokens, `L1`=32→1 gist, `L2`=gist of gists. Higher `L` means coarser detail and lower token cost. |
| `W_max` | Token-equivalent budget for the working context (sum of entry costs ≤ `W_max`). |
| Block size `K` | Number of new tokens processed per update (POC: `K = 32`). |
| Horizon `H` | Lookahead range used when computing ΔNLL or task losses (defaults: 64 for narrative traces, 96 for mixed agent turns, 128 for code). |
| ΔNLL@`H` | Change in negative log-likelihood over horizon `H` when replacing a region with its gist; used for supervision. |

**Invariants**
- Working context entries tile the lifetime history without gaps or overlaps; switching LOD swaps entries but preserves temporal continuity.
- GistNet outputs **gists** that reuse the base embedding dimension and can replace their source token blocks directly in the working context.
- LensNet and the focus allocator update entries between decode steps while keeping the budget and contiguity invariants intact.

These definitions appear throughout the rest of the document; refer back here when new notation shows up later.

---

### Document roadmap

1. **POC architecture & interfaces** — maps the runtime modules and data structures you will build.  
2. **POC scope & performance sketch** — nails down the proof-of-concept boundaries before diving into components.  
3. **Module deep dives (GistNet, LensNet, allocator)** — explains how each subsystem works and how they cooperate.  
4. **Training & operations** — covers alternating optimization, labeling, and instrumentation.  
5. **Roadmap & vision** — situates the prototype within longer-term ambitions in [Grand vision](#grand-vision-why-this-matters) and future directions.

---

## POC architecture & interfaces

The runtime is divided into focused modules so each invariant from [Key terms & invariants](#key-terms--invariants) has an explicit owner. The table below outlines those responsibilities before the later sections unpack implementation details.

| Module | Suggested path | Responsibilities | Key inputs/outputs |
|--------|----------------|------------------|--------------------|
| GistNet | `src/gistnet/` | Train & serve 32→1 gists, populate lifetime tree nodes | Input: token embeddings; Output: gist vectors + metrics |
| Lifetime tree | `src/memory/tree.py` | Maintain contiguous-in-time hierarchy (L0/L1/L2) in RAM (future stream to disk) | Input: gists/tokens; Output: node handles, metadata |
| Focus allocator | `src/focus/allocator.py` | Apply LensNet scores to expand/collapse blocks | Input: working-context entries, scores; Output: refreshed WC |
| LensNet | `src/focus/lensnet.py` | Score each WC entry for detail adjustments | Input: WC entries + tail gists; Output: focus scores |
| Runtime loop | `src/runtime/engine.py` | Orchestrate ingest → refocus → decode | Input: streaming tokens; Output: next-token logits, telemetry |
| CLI tools | `tools/` | Command-line helpers for dataset prep, logging, evaluation | Input: CLI args/config; Output: reports, artifacts |
| Evaluation/tests | `tests/` mirrored per module | Validate substitutability, focus policy, end-to-end behavior | Input: synthetic + real traces |

> **Diagram needed — `assets/module_stack.png`:** Layer the modules (lifetime tree, working context, LensNet, allocator, base LLM) and annotate data moving between them each decode cycle.

### Suggested data structures (Python-style)

```python
@dataclass
class GistNode:
    span_id: int
    level: Literal["L0", "L1", "L2"]
    start_token: int
    end_token: int
    embedding: torch.Tensor  # [d]
    parent_id: Optional[int]
    child_ids: list[int]

@dataclass
class WorkingEntry:
    node_id: int
    level: Literal["L0", "L1", "L2"]
    cost: int  # token-equivalent cost
    span_width: int  # number of L0 tokens covered
    distance_to_cursor: int  # blocks from decode cursor

@dataclass
class FocusScore:
    node_id: int
    level: Literal["L0", "L1", "L2"]
    score: float  # signed utility
    legality: Literal["expandable", "collapsible", "static"]
```

### Runtime loop pseudocode

```python
def decode_stream(stream):
    lifetime = LifetimeTree()
    working = WorkingContext()
    lens = LensNet.load(...)
    allocator = FocusAllocator(...)
    lm = BaseModel.from_pretrained(...)

    # Optional helper to maintain tail gists for LensNet
    tail_gists = TailGistCache(window=5)

    for block in stream.iter_blocks(K=32):
        nodes = lifetime.ingest_block(block)           # updates L0/L1/L2 nodes
        working.patch(nodes)                           # ensure contiguous coverage

        g_t = tail_gists.fetch(lifetime, working)      # L2 + recent L1
        packed = working.pack()
        focus_scores = lens.score(
            context=packed["embeddings"],
            levels=packed["levels"],
            span_width=packed["span_width"],
            distance_to_cursor=packed["distance_to_cursor"],
            tail_gists=g_t,
        )
        working = allocator.apply(working, focus_scores, lifetime)
        logits = lm.forward(**working.to_tensors())
        yield logits.argmax()
        lifetime.append_generated_token(logits.argmax())
```

### Working context API (runtime)

The loop above relies on a small, explicit surface so implementation teams know which tensors to provide to downstream modules.

| Method | Returns | Shape / Notes |
|--------|---------|---------------|
| `working.entries()` | `list[WorkingEntry]` | Block-aligned entries sorted by time (`start_token`). |
| `working.tail_view(k=256)` | `list[WorkingEntry]` | Last `k` entries for instrumentation; defaults to the LensNet update window. |
| `working.pack()` | `dict[str, torch.Tensor]` | Keys: `embeddings` (`[W, d]`), `levels` (`[W]`), `span_width` (`[W]`), `distance_to_cursor` (`[W]`). |
| `working.to_tensors()` | `dict[str, torch.Tensor]` | Keys: `inputs_embeds` (`[W, d]` mixed token/gist embeddings), `attention_mask` (`[W]`), optional `position_ids` (`[W]`). The base LLM wrapper forwards them as `lm.forward(inputs_embeds=..., attention_mask=..., position_ids=...)`. |
| `TailGistCache.fetch(lifetime, working)` | `torch.FloatTensor[K, d]` | Returns the L2 root plus the latest `window` L1 gists (default 5) aligned to LensNet conditioning. |

The LensNet scorer mirrors this with the signature `lens.score(context, levels, tail_gists, span_width, distance_to_cursor)` and returns a tensor of signed utilities (`torch.FloatTensor[W]`). Adjust as needed if additional scalar features prove useful later.

### Framework & environment assumptions

- **Base model:** start with `HuggingFaceTB/SmolLM3-3B` (bf16) or, if compute is tighter, `
Qwen/Qwen3-1.7B`. Both run comfortably on a single 24–48 GB GPU.
- **Runtime stack:** PyTorch ≥ 2.2 with FlashAttention 2, Hugging Face `transformers`, `accelerate`, and `datasets`.
- **Environment bootstrap:** prefer [`uv`](https://github.com/astral-sh/uv) for reproducible installs: `uv venv`, `uv pip install -r requirements.txt`, then `uv run python -m pip install -e .` for editable modules if needed.
- **Logging:** use [Weights & Biases](https://wandb.ai) for metrics and counterfactual ΔNLL traces; keep raw gists in memory for the POC.
- **Precision:** bf16 for model forward/backward; fp16 for gist snapshots if you need serialization.
- **Configuration:** place experiment configs under `configs/` (YAML) documenting block size `K`, horizon `H`, ΔNLL sampling strategy, and thresholds (`τ_expand`, `τ_collapse`).
- **Dataset staging:** tokenize corpora into contiguous 32-token blocks and store them as `.arrow` shards under `data/<dataset>/<split>.arrow`; provide `uv run python -m tools.prepare_dataset --config configs/data/<name>.yaml` to regenerate them.
- **Storage layout:** persist lifetime memory as `{L0,L1,L2}.ctx` binary files with a fixed header plus packed data (see below). Fixed block sizes make byte offsets deterministic, so no external index is required.

### Binary storage layout (`{L0,L1,L2}.ctx`)

Each file begins with a 64-byte header followed by tightly packed payloads. The header uses little-endian encoding and the following fields:

| Offset | Field | Type | Meaning |
|--------|-------|------|---------|
| 0 | `magic` | `uint32` | Constant `0x4D434354` (`MCCT`) to detect corruption. |
| 4 | `version` | `uint16` | Format revision (start at `1`). |
| 6 | `level` | `uint16` | 0, 1, or 2 indicating `L0`, `L1`, or `L2`. |
| 8 | `block_size` | `uint16` | Number of L0 tokens per gist (default 32). |
| 10 | `embedding_dim` | `uint16` | Width `d` of gist vectors (for `L1`/`L2`). |
| 12 | `dtype_code` | `uint16` | 0=`uint32`, 1=`fp16`, 2=`bf16`. |
| 14 | `model_name` | `char[32]` | UTF-8 null-terminated identifier of the base model (e.g., `SmolLM3-3B`). |
| 46 | `reserved` | 18 bytes | Zeroed; available for future metadata (checksum, flags). |

Payload layout per level:
- **L0 (`dtype_code=0`):** contiguous `uint32` token ids matching the base tokenizer vocabulary. Each block stores exactly `block_size` entries.
- **L1/L2 (`dtype_code=1`):** contiguous `fp16` vectors of shape `[num_nodes, embedding_dim]`. Gists inherit the same orientation as the base embedding matrix, so random access is `offset = header_size + index * embedding_dim * 2`.

Per-node metadata (`span_id`, `start_token`, `level`, parent/child pointers) stays in the lifetime tree’s in-memory index; because the binary payloads are fixed-width, offsets can always be recomputed on the fly.

### Sample run config (`configs/runs/poc_smollm3.yaml`)

```yaml
run_name: poc_smollm3_l4
base_model: HuggingFaceTB/SmolLM3-3B
tokenizer: HuggingFaceTB/SmolLM3-3B
precision: bf16
block_size: 32                # K
working_budget: 8192          # W_max
horizon: 64                   # H for ΔNLL labeling
focus_thresholds:
  expand: 0.2
  collapse: 0.2
  cooldown_steps: 2
datasets:
  gistnet_pretrain:
    - pg19
    - booksum
  lensnet_traces:
    - synthetic_coding_sessions
    - longbench_narratives
optimizer:
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
logging:
  wandb_project: megacontext-poc
  log_interval: 50
artifacts_dir: artifacts/
cli_tools_dir: tools/
storage:
  lifetime_dir: artifacts/lifetime/
  files:
    L0: L0.ctx
    L1: L1.ctx
    L2: L2.ctx
```

The remaining sections reference these interfaces when describing training and evaluation pipelines.

---

## POC scope & constraints

With the module map in place, the POC narrows to the following guardrails to ensure we can verify behavior end to end without boiling the ocean:

- **Frozen base LLM** no fine-tuning initially, with LoRA finetuning as a follow up  
- **Two-level Lifetime gist tree:** The POC will be limited to moderate sized contexts so only 2 layers should be sufficient   
- **Synchronous updates.** Lifetime tree lives in RAM/GPU for POC (rather than disk); updates happen between autoregressive steps.
  
---

## Performance sketch

| Setup | Lifetime tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | context-limited |
| **MegaContext (POC)** | ~1 M | 8 k | ~0.5 GB | few MB | constant compute per step |
| **MegaContext (Future)** | 1 B+ | 32 k | ~2 GB | 10–50 MB/s | fully trained base model |

Per-step compute ≈ base decode cost; gist extraction and LensNet overhead < 1 %.

### Long-term storage example: lifetime memory for a 24/7 robot (10 years)

**Assumptions**

- **Sampling:** 500 feature vectors / sec  
- **Feature size:** 4,096-dim; stored as fp16 (2 bytes) unless noted  
- **Duration:** 10 years ≈ 3.1536×10⁸ seconds ⇒ **N = 500 × 3.1536×10⁸ ≈ 1.5768×10¹¹** leaf vectors  
- **Tree arity:** 32→1 at each level (no overlap)  
- **Tree depth:** log₃₂(N) ≈ **8 levels** (root near level 8)  
- **Node payload:** one vector per node (same width as leaves, different precision per scenario)

#### Storage breakdown

| Scenario | Estimated storage | How to read this |
|---|---:|---|
| **Raw leaves only (fp16)** | **~1.29 PB** | N × 4096 × 2 bytes = 1.5768e11 × 8192 B |
| **Full 32-ary tree at fp16 (leaves + *all* internal levels)** | **~1.33 PB** | Geometric factor for all nodes: (1 + 1/32 + 1/32² + …) = 32/31 ≈ **1.032×** overhead over leaves |
| **Full tree, 8-bit everywhere** | **~667 TB** | Leaves 8-bit: ~646 TB; internal nodes count = N/31 ≈ 5.09e9; internal 8-bit adds ~20.8 TB; total ≈ 646 + 20.8 |
| **Pruned: keep only 1% of leaves @ 8-bit; keep *all* internal nodes @ 8-bit** | **~27 TB** | Leaves: 0.01 × 646 TB ≈ 6.46 TB; internal 8-bit ≈ 20.8 TB; total ≈ 27.3 TB |
| **Pruned + compressed: 1% leaves @ 8-bit with entropy coding (~×0.5); internal @ 8-bit with entropy coding (~×0.5)** | **~13–14 TB** | Leaves ≈ 3.2 TB + internal ≈ 10.4 TB |
| **More aggressive: 0.5% leaves @ 8-bit + entropy (~×0.5); internal @ 4-bit + entropy (~×0.5)** | **~6–8 TB** | Leaves: 0.005 × 646 TB × 0.5 ≈ **1.6 TB**; internal: (20.8 TB × 0.5 for 4-bit) × 0.5 entropy ≈ **5.2 TB**; total ~6.8 TB |

**Key takeaways**

- A **full 32-ary tree** only adds ~**3.2%** storage over leaves when stored at the **same precision** (factor 32/31), so multilevel LOD itself is cheap; **precision and pruning dominate** total footprint.  
- With **8-bit quantization** and **reasonable pruning** of raw leaves (e.g., keep only salient 0.5–1%), plus straightforward **entropy coding**, **a decade of continuous 500 Hz, 4k-dim features** compresses to **single-digit TBs**—practical for local SSD arrays.  
- This makes a **lifelong, high-bandwidth memory** feasible: raw details can be recovered where preserved; elsewhere, multilevel gists maintain global context with the **working context** handling on-demand re-expansion.

---

## GistNet — local gist extraction (32→1, two-layer tree)

### Purpose
GistNet replaces short, fixed-length token sequences with compact **gist embeddings** ("gists") that can stand in for their original tokens inside the base LLM’s context.  
Each gist preserves the meaning of its 32-token span while freeing token budget for new information.  
Stacking two 32→1 layers provides **1024× compression** in the proof of concept (POC).

---

### Inputs & outputs
| Symbol | Shape | Meaning |
|---------|--------|---------|
| `E ∈ R[32, d]` | 32 raw token embeddings (no contextualization) |
| `Q₁, Q₂ ∈ R[1, d]` | learned slot queries for the two compression passes |
| `g_final ∈ R[d]` | final gist vector aligned with the base LLM embedding dim |

---

### POC architecture (32→32→1→32→32→1)

GistNet alternates **self-attention** and **cross-attention** to gradually compress and refine each 32-token span.

#### Stage 1 — Local token self-attention (32 → 32)
- Apply 1–2 standard self-attention + MLP blocks within the 32-token window.  
- Add RoPE or sinusoidal positional encodings for local ordering.  
- Output is `E1`, a locally contextualized version of the raw embeddings.

#### Stage 2 — Compression (32 → 1)
- Introduce the first learned slot query `Q₁` (shared across spans).  
- Perform cross-attention where the slot reads from the tokens:  

```
G1 = CrossAttn(query=Q1, key=E1, value=E1)
G1 = G1 + MLP(LN(G1)) # residual + feedforward
```

- `G1` is the first gist embedding for this 32-token span.

#### Stage 3 — Expansion (1 → 32)
- Expand information back into the 32-token space for refinement:  

```
E2 = CrossAttn(query=E1, key=G1, value=G1)
E2 = E1 + MLP(LN(E2))
```

- Optionally run one self-attention block over `E2` to diffuse the gist info across tokens.

#### Stage 4 — Final compression (32 → 1)
- Run a second cross-attention with the independent learned slot query `Q₂`:  

```
g_final = CrossAttn(query=Q2, key=E2, value=E2)
g_final = LN(MLP(g_final))
```
- The result `g_final` is the final gist vector for the span and becomes a node in the lifetime gist tree.

#### Stage 5 — Hierarchical stacking
- Two 32→1 layers are stacked hierarchically (32² = 1024 tokens per top-level gist).  
- The lower layer runs directly on token embeddings; the upper operates on lower-layer outputs.
- This per-block stacking preserves the [contiguity invariant](#key-terms--invariants) noted earlier—each gist still maps to an exact, non-overlapping span in the lifetime history.

> **Diagram needed — `assets/gist_hierarchy.png`:** Depict an L0 token block rolling up into an L1 gist and then into an L2 gist, with pointers back to the lifetime timeline.


### Architectural properties
| Property | Description |
|-----------|--------------|
| **Limited scope** | Operates strictly within 32-token windows; no long-range attention. |
| **Parameter sharing** | Shared weights across all spans; upper and lower layers may share or specialize. |
| **Complexity** | O(32²·d) per span — negligible compared to the base LLM. |
| **Dimensionality** | Outputs match the base model’s embedding size `d`. |
| **Positioning** | Gist inserted at the central token index for RoPE alignment. |
| **Precision** | bf16 or fp16; supports later quantization for storage. |


### Training objectives

#### 1. Substitutability (primary)
Train the model so that replacing a span with its gist minimally changes the base LLM’s predictions.

For each training example:

```
Loss_subst = KL( P_base(x_{t+1:T} | E)
|| P_base(x_{t+1:T} | E_replaced) )
```
or equivalently minimize the ΔNLL between the full and gist-replaced context over a short horizon (H = 32–128 tokens).

#### 2. Contrastive span separation (optional)
Discourage neighboring spans from collapsing to identical gists:

```
Loss_contrast = max(0, margin - cosine_similarity(g_i_final, g_j_final))
```
for adjacent spans (margin ≈ 0.2).

Total loss:
```
Loss = Loss_subst + 0.05 * Loss_contrast
```

### Implementation details (POC)
| Item | Setting |
|------|----------|
| Window size | 32 tokens |
| Slots | 2 shared learned queries (`Q₁`, `Q₂`) |
| Layers per 32→1 block | 2 self + 2 cross |
| Refinement stack | 32→1→32→1 |
| Embedding dim | same as base LLM (e.g., 4096) |
| Internal hidden width | 512 |
| Attention heads | 8 |
| RoPE | applied to token positions only (slots omit it) |
| Activation | GELU |
| Norm | Pre-LayerNorm |
| Parameters | ~0.5M per layer |
| Output | single `g_final` vector per span |
| Runtime | <1 ms per 32-token span on GPU |

Runtime figures assume a single NVIDIA L4 running bf16 inference with `HuggingFaceTB/SmolLM3-3B`; expect faster throughput on A100-class hardware.

### Training pipeline (POC)
1. **Dataset:** long-form text (4k–16k tokens), chunked into 32-token spans.  
2. **Teacher:** frozen base LLM used for ΔNLL@H computation.  
3. **Objective:** minimize ΔNLL@H between original and gist-replaced contexts.  
4. **Curriculum:** start with contiguous text, then include structured data (lists, code, tables).  
5. **Optimizer:** AdamW, lr = 1e-4, cosine decay, bf16 precision.  
6. **Output:** store 32→1 and 1024→1 gists in the lifetime gist tree for later use by LensNet and the focus allocator.

### Recap
GistNet is a **local encoder for token spans** whose only goal is to emit substitutable gist vectors aligned with the base model’s embedding space.  
It uses **self- and cross-attention refinement (32→1→32→1)** to squeeze each 32-token block into a single vector without ever decoding back to tokens.  
Stacked hierarchically, GistNet forms the **Lifetime Gist Tree** that supports scalable, virtualized context in MegaContext and supplies the tail gists that condition LensNet at its scheduled refreshes.

---

## LensNet — how focus is decided

### Why “Lens”?
LensNet acts like an optical lens that dynamically **focuses** and **defocuses** regions within the lifetime context while keeping total compute constant.  
It predicts where to spend detail (expand gists into raw tokens) and where to blur (collapse raw tokens into gists), ensuring that the **fixed-size working context** maintains maximal relevance.

### What it operates on
- LensNet reads the **working context** (not the lifetime tree).  
  It analyzes the embeddings currently fed into the base LLM — the only state that resides on GPU.
- It outputs one **focus score** per entry (token embedding or gist).
- The [contiguity invariant](#key-terms--invariants) from the glossary ensures each score maps to a single, non-overlapping lifetime span, so expand/collapse actions remain block-aligned.

### Why non-causal is essential
LensNet must understand *future queries* to know which past facts matter.

**Example**
```
C1: "My shirt is red. My pants are green."
C2: "My shirt is red. My pants are green. What color hat would match my shirt?"
```


Because the base LLM is causal, the hidden states for “shirt” and “pants” are identical in C1 and C2; they never see the question.  
A non-causal LensNet can look at the full working context (including the query) and boost focus on the “shirt” fact.

### Conceptual overview
- LensNet runs independently of the frozen base LLM.  
- It operates directly on the **working context embeddings** (≈ 8k entries), not on live LLM hidden states.  
- It conditions on a small **gist set** (`L2 + last 5 L1` gists, total ≈ 6) taken from the end of the context, which implicitly encodes the upcoming query/task.  
- The model outputs one **signed focus score** `u_i` per entry:
- `u_i > 0`: expand / focus (increase detail, go one level down)
- `u_i < 0`: collapse / defocus (reduce detail, go one level up)

> **Diagram needed — `assets/lensnet_focus.png`:** Show LensNet reading a tail slice of gists plus the working context, then emitting signed scores that the allocator converts into expand/collapse actions.

At runtime, the **focus allocator** interprets these scores to expand and collapse spans while keeping the working context within its token budget.

### Why dynamic LOD matters
Traditional context compression methods summarize once and lose detail forever.  
MegaContext continually re-evaluates importance: if a previously collapsed region becomes relevant again, it can be expanded back into its children gists or raw tokens.  
Note that this expansion is NOT a lossy decoding of the gist latent - the lifetime context preserves the full token-level details on disk (or in RAM for the POC), so the LLM has full access to the whole lifetime context, just not all at once.
This enables the model’s effective memory to **evolve over time** as new information arrives.  Similar to how you're now thinking about your first kiss 😘

### Architecture (POC: dual cross-attention LensNet)

1. **Inputs**
   - `context`: `torch.FloatTensor[N, d]` — embeddings of all entries in the working context (≈8 000 tokens/gists).  
   - `tail_gists`: `torch.FloatTensor[K, d]` — L2 root plus the latest `K-1` L1 gists (default `K=6`).  
   - `levels`: `torch.LongTensor[N]` — 0/1/2 markers for legality masking.  
   - `span_width`: `torch.LongTensor[N]` — number of L0 tokens represented by each entry.  
   - `distance_to_cursor`: `torch.LongTensor[N]` — block distance from the decode cursor (optional feature; treat as integer tensor).  
   - All embeddings are down-projected to a LensNet width `d_lens ≈ 512`.

2. **Stage 1 — Tail gists read the context**  
   Using standard attention primitives:

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

### Update cadence (block-wise refocus)

LensNet runs **once every K tokens** (POC: K = 32).  
During each block update:

1. Gather the latest gists `G`.  
2. Run LensNet to produce signed scores `u_i`.  
3. The focus allocator executes expansions/collapses subject to the working-context budget.  
4. The updated context is frozen for the next K tokens.

This matches the intended inference cadence (no per-token recompute).

### Training objectives

#### 1️⃣ Signed focus supervision
Each entry receives a **signed target utility** `y_i` derived from counterfactual NLL deltas:

- Expandable items (L1/L2 children) ⇒ positive `y_i > 0`  
- Collapsible spans ⇒ negative `y_i < 0`  
- Others ⇒ 0 / masked.

LensNet learns to regress and rank these utilities.

```
L_reg  = (1 / |M|) * sum_{i in M} (u_i - y_i)^2
L_rank = softplus(-(u_i - u_j))  # for ordered pairs
```

#### 2️⃣ Zero-sum budget regularizer
To maintain constant working-context size:

```
P = sum_i c_i_plus * ReLU(u_i)
N = sum_i c_i_minus * ReLU(-u_i)
L_budget = ((P - N) / (eps + P + N))^2
```
(`c_i^+` / `c_i^-` = token cost / refund.)  
This encourages net-zero expand/defocus mass per block.

#### 3️⃣ Legality penalties
Prevent impossible actions:

```
L_illegal = alpha * sum_{L0} ReLU(u_i) + beta * sum_{L2} ReLU(-u_i)
```
(alpha, beta ≈ 0.3).  
At inference, invalid directions are hard-masked to 0.

#### 4️⃣ Total loss

```
L_total = L_reg + 0.5 * L_rank + 0.1 * L_budget + L_illegal
```

### Inference procedure

1. **Mask** illegal sides (L0 can’t expand; L2 can’t collapse).  
2. **Optional rebalance**: rescale positive/negative masses to match before sending to the focus allocator.  
3. The focus allocator greedily applies expand/collapse actions within the token budget, honoring hysteresis rules.

### Summary of POC parameters

| Item | Value / Notes |
|------|----------------|
| Input embeddings | ≈8 k entries (mixed L0/L1/L2) |
| Conditioning gists | 6 (L2 + 5 L1) |
| Down-projection width | 512 |
| Attention heads | 8 |
| Stacks | 1–3 |
| Update cadence | every 32 tokens |
| Output | signed focus score `u_i` per entry |
| Runtime | < 3 ms per update @ 8 k tokens |
| Params | ≈ 100 k – 200 k total |

Timings were measured on an NVIDIA L4 with `SmolLM3-3B`. Scaling to larger GPUs (A100/H100) reduces latency proportionally.


**In short:**  
LensNet is a compact, non-causal controller built as a dual cross-attention network (`8k → 6 → 8k`).  
It runs once per block, predicts balanced signed focus scores for every entry, and guides the focus allocator to keep the working context sharp, legal, and budget-neutral.

---

## Focus allocator — block-aligned actions

LensNet alone only supplies signed focus scores. The allocator turns those scores into concrete expand/collapse actions while preserving contiguity, budget, and level-of-detail (LOD) constraints. It is the practical enforcer of the [contiguity invariant](#key-terms--invariants) inside the working context.

### POC constraints & terminology

- **Block alignment:** GistNet currently compresses 32-token blocks. In the POC, every working-context entry must cover exactly one full block at a single LOD (either 32 raw tokens or their 32→1 gist). Higher-level gists (e.g., L2) cover 32 contiguous L1 blocks.
- **Score granularity:** LensNet may emit per-entry scores, but the allocator aggregates them per block so that siblings share a single action score. A future LensNet variant can predict directly per block to avoid this aggregation.
- **Action budget:** Apply at most `N_diff` expand/collapse operations per iteration (default 4). This keeps the system near equilibrium and prevents thrashing.
- **Positional alignment:** When swapping L0/L1 entries, reuse the original absolute token indices for RoPE; gists occupy the central token index of their covered span so the base LLM receives consistent phase information.

### Greedy loop (POC)

1. **Collect candidates.** Partition focus scores by block and compute one score per expandable or collapsible unit:
   - Positive scores (`> τ_expand`, default 0.2) become expand candidates (e.g., replace an L1 gist with its 32 L0 tokens or expand an L2 gist into 32 L1 children).
   - Negative scores (`< -τ_collapse`, default 0.2) become collapse candidates (e.g., replace 32 L0 tokens with their L1 gist).
   - Ignore candidates that would violate block alignment (mixed LODs) or budget limits.
2. **Rank.** Maintain two priority queues: descending for expands, ascending for collapses. Tie-break by recency or distance to the cursor.
3. **Apply diff-limited updates.** Pop from the queues alternately (largest expand, largest collapse) until:
   - You have applied `N_diff` actions,
   - One queue empties, or
   - Applying the next action would break the `W_max` budget.
   Collapses refund token budget; expands consume it. If the net cost drifts away from `W_max`, bias the next iteration toward the side that restores balance.
4. **Re-run LensNet if needed.** Because changing LODs alters the scores, optionally iterate LensNet → allocator until either (a) no legal actions remain above thresholds or (b) you reach a maximum number of refinement steps (default 2–3).

### Hysteresis & guardrails

- **Action cooldown:** Track the last action applied per block and dampen (or mask out) the opposite action for `cooldown_steps = 2` iterations. This prevents jitter where the allocator repeatedly expands and collapses the same span.
- **Legality masks:** Blocks at minimum LOD (L0) cannot expand; blocks at maximum LOD (current root level) cannot collapse. These masks should be enforced both in LensNet’s output (runtime masking) and inside the allocator.
- **Consistency checks:** After every iteration, verify that working-context entries still tile the timeline without overlap and that every node’s children share the same LOD.

### Recommended runtime defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `τ_expand` | 0.20 | Minimum signed score magnitude before expanding an entry. |
| `τ_collapse` | 0.20 | Symmetric collapse threshold; keep equal to `τ_expand` until adaptive tuning is available. |
| `N_diff` | 4 | Maximum expand/collapse actions per iteration to cap churn. |
| `cooldown_steps` | 2 | Minimum iterations before a block can flip actions. |
| `lens_update_interval` | 32 tokens (`K`) | LensNet runs once per block and consumes cached tail gists. |
| `tail_gist_window` | 5 L1 nodes + current L2 | Conditioning set passed to LensNet. |

These defaults keep the working context near equilibrium while allowing meaningful detail movement; they are the baseline values for automated tests and ablations.

### Future directions

- Smarter action selection (e.g., matching total expand/collapse mass, soft assignments, or small linear programs) to balance budget and latency.
- Learning a differentiable surrogate (“focus router”) that could eventually replace the greedy loop.
- Adaptive thresholds (`τ_expand`, `τ_collapse`) based on recent utilization to keep the loop stable.

For now, the greedy, block-aligned allocator keeps the POC simple while leaving room for more sophisticated controllers later.

---

## Training & Operations

### Joint training (alternating / “EM-style”)

**Goal:** Let all three modules co-adapt without full end-to-end backprop through the discrete focus allocator or long unrolls.  
**Method:** Short alternating phases where some modules are frozen while others learn from on-policy signals produced by the frozen parts. Repeat for a few cycles.

#### What “EM-style” means here
We alternate optimization across modules:
- **E-like step:** hold policy parts fixed to produce supervision/targets (e.g., counterfactual utilities).
- **M-like step:** update another module to better fit those targets.
It’s not exact EM; it’s an **alternating optimization schedule** that stabilizes joint training.

#### Modules
- **GistNet** `Gist` (32→1, two levels; substitutability objective)
- **LensNet** `LensNet` (dual cross-attn 8k→6→8k; signed focus scores)
- **Base-LoRA** `LoRA` (tiny adapters on the base LLM to improve gist compatibility)
- **Focus allocator** is always **discrete greedy** (no relaxation needed)

#### Phase B1 — Update GistNet (fix LensNet + LoRA)
**Fix:** `LensNet`, `LoRA`
**Update:** `Gist`

**Procedure (on-policy):**
1. Build/refresh lifetime trees with current `Gist`.
2. For each training block (size K=32): run `LensNet` + focus allocator to pick expands/collapses; form the working context used by the base LLM.
3. Optimize **GistNet** on spans touched in this block using:
   - **Substitutability loss**: KL(full || replaced) or ΔNLL@H (H=32–128) for the gist that *was actually* inserted.
   - **Stability loss** (optional): L2 between current gist and previous checkpoint to avoid drift.
   - **Boundary aux** (optional): upweight ΔNLL terms on edge tokens so the encoder preserves boundary semantics.

**Intuition:** With the current focusing policy fixed, make gists better drop-in replacements for *exactly the places the policy cares about*.

#### Phase B2 — Update LensNet (fix GistNet + LoRA)
**Fix:** `Gist`, `LoRA`  
**Update:** `LensNet`

**Procedure:**
1. Using the fixed `Gist`, generate **counterfactual labels** on on-policy snapshots:
   - For candidate expands/collapses in the current working context, compute ΔNLL/ΔKL (batched).
   - Convert to **signed utility per token** (expand positive; collapse negative).
2. Train `LensNet` with:
   - **Signed regression + ranking** (within snapshot)
   - **Zero-sum budget** regularizer (token-cost weighted)
   - **Legality** penalties; keep runtime masking
   - **Update-every-K** cadence (Lens runs once per block)

**Intuition:** Given the current gists, learn a better focusing policy.


#### Phase B3 — Update Base-LoRA (fix GistNet + LensNet)
**Fix:** `Gist`, `LensNet`  
**Update:** `LoRA` (small ranks; keep it tiny)

**Where to place LoRA (recommended):**
- Input embedding projection
- QKV/O of the **first 2 attention blocks** *or* the **last 2** (pick one set; not both)

**Losses:**
- **Task NLL@H** with the *discrete* working context produced by `LensNet` + focus allocator
- **Substitutability keep-alive** (weak): prevents gist semantics drifting away from what the base understands
- (Optional) **KL to teacher** if you have a larger teacher-with-MegaContext

**Intuition:** Slightly adapt the base to “like” gist tokens and the current WC geometry (positional anchoring, variance, etc.).


#### Schedule & hyperparameters

- **Cycle length:** B1 → B2 → B3 = **one cycle**. Repeat **3–5 cycles**.
- **Step counts per phase (per cycle):**  
  - B1 (GistNet): 2–4k steps  
  - B2 (LensNet): 2–4k steps  
  - B3 (LoRA): 1–2k steps  
- **Batching:** mixed long-context tasks; block size K=32; horizon H=64.
- **Optimizers:** AdamW (bf16), cosine LR with warmup per phase.
- **Tokens / GPU:** target ~8k effective tokens per microbatch; use gradient accumulation (e.g., 2 microbatches × 4 sequences) to fit within 24 GB GPUs.
- **Tokenizer:** reuse the base model’s tokenizer and embedding matrix to avoid drift between gist vectors and token embeddings.
- **Checkpoints:** save after each phase; early-stop on validation **Loss@H vs. token-budget**.


#### Data flow per cycle (pseudo)

1. **B1:**  
   - Freeze `LensNet`, `LoRA`.  
   - Decode blocks with current WC (from LensNet + focus allocator).  
   - Update `Gist` using on-policy substitutability losses on the replaced spans.

2. **B2:**  
   - Freeze `Gist`, `LoRA`.  
   - From the same blocks, compute counterfactual utilities (expand/collapse candidates).  
   - Update `LensNet` with signed utilities + budget/legality losses.

3. **B3:**  
   - Freeze `Gist`, `LensNet`.  
   - Run normal blocks (LensNet + focus allocator active) and update `LoRA` on Task NLL@H (+ weak substitutability keep-alive).


#### Stability & efficiency tips

- **Warm starts:** Do a short **sequential pretrain** (GistNet then LensNet) before the first B1; it reduces early oscillations.
- **Small LoRA ranks:** r=4–16, low LR; the goal is interface alignment, not knowledge injection.
- **Hysteresis in focus allocator:** min residency steps to prevent expand/collapse thrash during B2/B3.
- **On-policy labeling:** Always regenerate ΔNLL labels *after* the last B1 so LensNet trains on current gists.
- **Curriculum:** start with narrative/doc tasks; add lists/tables/code once stable.
- **Telemetry:** track (a) Loss@H vs budget, (b) swap rate, (c) residency time, (d) non-causal C1/C2 tests.


#### When to stop
- **Validation Loss@H vs budget** improves then plateaus across cycles.
- **Swap rate** stabilizes; no ping-pong.
- **Ablations:** freezing any one of {GistNet, LensNet, LoRA} now causes a measurable drop.

**Outcome:** All three modules co-learn: **GistNet** encodes what the policy needs, **LensNet** chooses expansions that actually help, and **LoRA** nudges the base LLM to be friendlier to mixed-LOD inputs—without the cost/fragility of full end-to-end training.


---

## Comparison: MegaContext vs. RAG

| Aspect | RAG | MegaContext |
|---------|-----|-------------|
| **Storage** | External documents, often text chunks in a vector DB | Hierarchical learned gists (vectors) directly aligned to the model’s lifetime |
| **Retrieval trigger** | Query-time semantic search | Continuous, learned focus from LensNet |
| **Integration** | Concatenate retrieved text to prompt | Replace/expand in working context with proper positional encoding |
| **Training** | Separate retriever / generator | Single substitutability & focus training |
| **Memory type** | Stateless look-up | Persistent evolving memory with reversible summarization |

MegaContext is *structurally* similar to RAG in that both pull relevant data into a fixed context, but differs fundamentally: it treats compression and focus as an **integrated learned process** rather than retrieval over external text.

---

### Conversation: “Isn’t this just RAG with tools?”

**Alex (Devil’s Advocate):** We already have agents that retrieve docs, call APIs, and stitch context together. Why train GistNet/LensNet when heuristics plus vector search gives us long-term memory today?

**Sam (MegaContext advocate):** RAG retrieves; MegaContext *represents*. The lifetime tree keeps a contiguous, multi-resolution memory so the base model always sees a coherent mix of tokens and gists. Learned focus (LensNet) uses ΔNLL signals to decide what detail actually matters.

**Alex:** Agents already summarize conversation history and pull relevant snippets on demand. With nightly re-indexing and heuristics for cache hits, it works fine.

**Sam:** Until you scale to 100 M+ tokens. MegaContext ensures substitutability at every level, keeps latency bounded (working context stays 8–32 k), and provides traceability—you can pinpoint which gist drove a response. Retrieval snippets can’t guarantee positional alignment or stability.

**Alex:** Starting from scratch sounds heavy. With RAG, we ship features immediately.

**Sam:** MegaContext is about a platform shift. The lifetime tree becomes the knowledge substrate, enabling “cognitive core” models—small weights focused on reasoning while knowledge updates externally. It plays nicely with RAG: ingest retrieved results into the tree, and LensNet manages detail thereafter. Think of it as the memory layer that unifies tools, not a replacement.

**Alex:** So it complements existing agents?

**Sam:** Exactly. Tools and RAG stay in the loop, but MegaContext virtualizes memory with learned compression/focus. You trade ad-hoc prompt assembly for a consistent, updatable architecture built to scale.

---

### Training data & streaming behavior

- **GistNet training:** use long-context corpora such as `pg19`, `BookSum`, or code-heavy datasets (e.g., `the-stack-smol`) to expose diverse structures. Each 32-token window supplies (full vs gist) pairs.
- **LensNet training:** log working-context snapshots from the runtime loop (synthetic stories, coding sessions, agent traces). Replay them offline to compute counterfactual utilities.
- **Streaming loop:** as new tokens arrive,
  1. Buffer 32 tokens → create/update the corresponding L1 gist.
  2. Promote every set of 32 L1 nodes into an L2 gist.
  3. Invoke LensNet + focus allocator before the next decode step to rebalance detail.
- **Serialization (optional):** persist gists as fp16 or 8-bit vectors with JSON metadata (`span_id`, `level`, `timestamp`) if you need to resume runs; otherwise keep them in-memory for the POC.

### Counterfactual ΔNLL labeling

1. **Snapshot selection:** capture the working context every `K` tokens along with candidate expand/collapse actions that respect legality masks.
2. **Batch recompute:** for each candidate, rebuild the context with the alternative LOD and run the frozen base model over a short horizon (`H = 64` for narrative traces, up to 96–128 for code-heavy or cross-turn reasoning workloads). Reuse tokenizer outputs; caching KV states across candidates is optional but speeds things up.
3. **Utility extraction:** compute `ΔNLL = NLL_alt - NLL_base`. Positive values favor expansion (detail helps); negative values favor collapse. Normalize by span width so utilities are comparable across levels.
4. **Score targets:** store utilities alongside the scalar features LensNet expects (`level`, `span_width`, `distance_to_cursor`). These tensors feed the regression/ranking losses and keep legality masks aligned with training.

This batched approach keeps labeling tractable on a single A100 or L4 instance even for 3 B models.

### Joint training (alternating / “EM-style”)

Refer to the detailed phase descriptions above and track the following during each cycle:
- ΔNLL@`H` gap between full vs gist contexts (should shrink over cycles).
- Net expand/collapse mass per block (should stay near zero).
- LoRA loss on held-out prompts (ensures base compatibility).

### Instrumentation & artifact handling

- **Logging:** stream metrics (losses, swap rates, residency histograms) to Weights & Biases for later comparison; tag runs by dataset + thresholds.
- **Checkpoints:** save GistNet, LensNet, and LoRA weights under `artifacts/checkpoints/`. Store counterfactual utility tables under `artifacts/deltas/` (Parquet or Arrow for efficient slicing).
- **Configs:** mirror each run’s YAML under `configs/runs/` so experiments are reproducible.
- **Testing harness:** add PyTest suites under `tests/` (e.g., `tests/test_gistnet.py`, `tests/test_focus_allocator.py`) and document `uv run pytest --maxfail=1 --disable-warnings --cov=src` as the canonical invocation.
- **Local tooling:** provide Python entry points under `tools/` (e.g., `python -m tools.format`, `python -m tools.lint`) that wrap `ruff` and `black` so contributors can run `uv run python -m tools.format` / `uv run python -m tools.lint`.
- **CLI scripts:** expose dataset/labeling helpers as modules (`python -m tools.ingest_data`, `python -m tools.label_dnll`) and register hydra/typer CLIs if needed; keep lightweight wrappers under `scripts/` for automation.
- **Telemetry (required):** emit per-iteration metrics (`loss_at_h`, `swap_rate`, `mean_residency`, `latency_ms`, `token_budget_utilization`) and persist them alongside allocator action traces so regressions are diagnosable.

### Limitations & failure modes (watchlist)

- **Gist drift:** substitutability degrades if GistNet overfits; monitor ΔNLL@`H` gaps and refresh ΔNLL labels after each B1 phase.
- **Allocator oscillation:** repeated expand/collapse of the same block indicates thresholds/cooldown need adjustment; histogram residency times to catch this.
- **Boundary artifacts:** compressed spans that straddle critical tokens (e.g., function definitions) may cause performance cliffs; add targeted tests for boundary cases.
- **Latency spikes:** excessive counterfactual sampling or large `N_diff` values can break constant-time promises; record per-iteration latency in telemetry.
- **Positional aliasing:** swapping gists without reusing original indices can shift RoPE phases; ensure instrumentation validates positional consistency.

### Evaluation & validation checklist

**Accuracy & compression**
- **ΔNLL vs budget:** sweep `W_max` (4k → 16k) using held-out long-form tasks; target ΔNLL degradation ≤ 0.1 compared to full-context baselines at equivalent token budgets.
- **Compression stress:** verify substitutability at 32× and 1024× compression with narrative and code samples, ensuring ΔNLL@`H` stays within 0.2 of the uncompressed control.
- **Focus ablations:** compare causal vs non-causal LensNet and allocator variants (with/without cooldown) to confirm the non-causal controller yields ≥3% lower ΔNLL@`H`.

**Runtime & stability**
- **Resource trace:** log GPU memory, wall-clock latency per block, and total expand/collapse mass; keep latency within +10% of the frozen baseline at 8k active tokens.
- **Swap rate & residency:** track mean residency ≥3 iterations per block and swap rate ≤0.25 actions per block to avoid thrashing.
- **Boundary diagnostics:** run synthetic tests where important tokens align with block edges to ensure no catastrophic degradation (>0.2 ΔNLL jump).

**Benchmarks**
- Evaluate narrative QA (LongBench `NarrativeQA`), academic QA (`Qasper`), and coding/story tasks (InfiniteBench). Report ΔNLL@`H`, latency, and swap metrics alongside baseline LLM runs.
- Optional stretch: include HELM-LC suites once the pipeline stabilizes to benchmark against summarization/RAG strategies.

**POC acceptance criteria**
- Demonstrate ΔNLL degradation ≤0.1 at `W_max = 8k` with constant-time compute (latency overhead ≤10%) and stable swap metrics on at least one narrative and one coding benchmark relative to the frozen base model.

### Example walkthrough (toy coding session)

1. **Setup:** Load a small TypeScript project summary into lifetime memory (≈4k tokens) and seed the working context with the latest user/system gists.
2. **User turn:** “Add logging to the `fetchUser` helper.” Ingest tokens into the lifetime tree (32-token blocks) and update L1 gists.
3. **LensNet pass:** Scores the new query tokens highly (`u_i ≈ +0.4`) and suggests expanding the gist that summarizes `fetchUser`.
4. **Focus allocator:** Applies one expand action (L1→32×L0) and one collapse on distant chatter (`u_i ≈ -0.3`), staying within `W_max`.
5. **Decode:** The base LLM, now seeing raw tokens for `fetchUser`, produces the patch. Newly generated code is appended to the lifetime tree.
6. **Trace capture:** Log ΔNLL utilities, focus actions, and residency times to W&B for later analysis.

Document a similar narrative under `docs/walkthroughs/` once the POC code path is live so future contributors can replay it end to end.

---

## Related work

| Concept | Reference | Relevance |
|----------|------------|------------|
| MegaTexture (id Software, 2007) | Virtualized textures | Direct analogy |
| Perceiver / Perceiver IO (DeepMind 2021-22) | Latent cross-attention | Architectural similarity |
| Slot Attention (Locatello 2020) | Object-like latent slots | GistNet inspiration |
| Compressive Transformer (Rae 2019) | Long-term compressed memory | Temporal analog |
| Gist tokens / LLMLingua 2 (2023-24) | Prompt compression | Substitutability idea |
| RAG / Retrieval-Augmented Generation | Search-based retrieval | Conceptual cousin |
| MegaContext (this work) | — | Unified learned compression + focus over frozen LLMs |

---

## Implementation roadmap

1. **32→1 GistNet** — implement & train substitutability.  
2. **Lifetime Tree Builder** — streaming, 2-level hierarchy in RAM.  
3. **LensNet v1 (non-causal)** — implement query-conditioned scorer, train on offline labels.  
4. **Focus allocator** — greedy expand/collapse, hysteresis.  
5. **E2E POC** — run step-loop (score → allocate → update → decode).  
6. **Evaluate** — loss vs budget, C1/C2 relevance, stress tests.

---

## Grand vision: why this matters

The POC will prove the mechanism; this section zooms out to why it is worth the effort once the core loop is stable.

### ♾️ Virtually infinite memory
Lifetime context can grow unbounded while per-step compute and GPU RAM remain constant. A conversation could persist for years without retraining or forgetting.

### 🧩 Smaller, smarter models
An LLM trained end-to-end with MegaContext could shift parameter budget away from memorized facts toward reasoning, abstraction, and planning. Knowledge lives in the *lifetime memory* instead of the weights.

### 💻 Agentic coding & persistent tasks
Today, agents rely on brittle, lossy context management (manual summarization, sub-agents, RAG hacks). MegaContext treats context management as a **first-class architectural component**, allowing seamless long-term reasoning and creative iteration.

### 🌐 Core knowledge as dynamic system prompt
Shipping LLMs with a **core lifetime context** transforms in-context learning: the model boots with a massive “system prompt” of structured world knowledge that updates externally and without retraining weights.
- A cloud-hosted MegaContext model could refresh its understanding of the world continually, combining retrieval and reasoning in a unified pipeline.
- An agentic coding system could provide an entire codebase as a system prompt (lifetime context), eliminating the expensive / error prone processes of reading parts of the project's code.

---

## The end game: a cognitive core

> Inspired by Andrej Karpathy’s “cognitive core” idea — a compact reasoning engine whose weights specialize in abstraction while factual knowledge lives externally.

MegaContext offers a pragmatic path to this separation by treating the lifetime memory as an extensible knowledge substrate and keeping the working context small.

### What lives in the core?
- **Base model (<1 B params):** a compact transformer trained to reason over mixed token/gist embeddings delivered by the working context.
- **GistNet + LensNet stack:** keeps knowledge substitutable and focuses detail on demand.
- **Lifetime tree:** a curated, pre-gisted corpus of “core knowledge” (10 M–1 B tokens) spanning textbooks, documentation, ontologies, code—kept current without weight changes.

### Training the cognitive core
1. **Curate & gist the knowledge base:** preprocess the corpus into block-aligned spans, compute multi-level gists with a dedicated GistNet, and store them via the `{L0,L1,L2}.ctx` format.
2. **Warm start the controllers:** pretrain LensNet using counterfactual traces from a larger teacher model so the small core inherits a strong focusing policy; refresh GistNet on spans the policy touches most.
3. **Alternating training loop:** during each batch, the base model observes an 8–32 k working context assembled by LensNet/allocator from the knowledge tree. Optimize the model on task NLL, then refresh LensNet and GistNet in alternating phases (as described in [Training & Operations](#training--operations)).
4. **Encourage dependence on memory:** include tasks that require multi-hop reasoning across the knowledge tree; penalize ignoring relevant spans by comparing ΔNLL with/without expansions.
5. **Distill from a teacher:** use a larger LLM with direct access to the knowledge base to produce targets, distilling reasoning strategies into the smaller model.

### Why it matters
- **Smaller weights, richer knowledge:** the base model can focus on pattern recognition, logical composition, and planning while the lifetime tree handles fact storage and updates.
- **Continuous learning:** updating facts means ReGisting new documents, not modifying weights—ideal for domains with rapid change.
- **Composable systems:** multiple cognitive cores can share or federate lifetime memories, enabling collaborative agents without redundant retraining.
- **Traceability:** hallucinations or conflicting answers can be traced back to the specific gists and source documents surfaced in the working context, making attribution and debugging far more transparent than opaque weight memorization.

### Open research directions
- **Joint training:** exploring end-to-end differentiable surrogates that allow gradients to flow through expand/collapse actions.
- **Knowledge curation:** tools for versioning, deduplicating, and auditing the lifetime tree as it scales to billions of tokens.
- **Focus policies:** RL or bandit strategies that optimize accuracy × latency beyond the current greedy allocator.
- **Safety & alignment:** policies for moderating which knowledge segments are surfaced to the working context in sensitive domains.

### Curating the core knowledge corpus

- **Segmented ordering:** Group documents by domain or task (e.g., coding, scientific literature, product docs). Within each segment, order files so high-level gists correspond to coherent themes; for code, a chain like `README → design notes → module docs → source files` gives LensNet clear zoom targets.
- **Granularity & bridges:** Keep base blocks contiguous, but insert “bridge” gists when cross-document reasoning is common (API description ↔ implementation). These bridges live at higher levels (L3/L4) and help LensNet jump across related materials.
- **Metadata enrichment:** Tag each span with domain, file path, language, timestamp, recency, and trust scores. Feed these as features into LensNet so focus policies can prefer fresher or context-matching knowledge.
- **Quality control:** Deduplicate near-identical spans before gist extraction; monitor gist variance to detect noisy inputs. Track provenance IDs for every gist so hallucinations can be traced back to the original source and corrected.
- **Incremental updates:** Append new partitions instead of reprocessing the entire tree. Because offsets are deterministic, you can rebuild affected gists in place and avoid full re-ingest. Version each partition so rollbacks or audits remain manageable.
- **Curriculum for training:** As the corpus grows, schedule tasks that encourage the base model to rely on relevant segments (e.g., code tasks sample from the “code” partition). Penalize ignoring retrieved spans by comparing ΔNLL with and without expansions during training.

This “cognitive core” roadmap builds directly on the [core knowledge as dynamic system prompt](#-core-knowledge-as-dynamic-system-prompt) concept and frames MegaContext as the enabling substrate for long-lived, updatable reasoning systems.

---

## Pruning lifetime context

Even with disk-backed storage, a mature lifetime memory will accumulate redundant, outdated, or low-value spans. Pruning keeps the gist tree healthy without sacrificing recall.

### Signals to collect
- **Access telemetry:** track each span’s dwell time, expansion count, and last access step. Branches that never surface in the working context become pruning candidates.
- **ΔNLL sensitivity:** periodically replace spans (or whole subtrees) with coarser ancestors and measure ΔNLL@`H`. Low sensitivity indicates the detail can be safely collapsed or removed.
- **Query alignment:** maintain lightweight embeddings or tags for recent tasks; spans that never semantically align with active queries can be deprioritized.

### Pruning strategy
- **Level-aware shrinkage:** drop long-idle L0 tokens first (keeping their L1 gist). If L1 gists are never expanded, collapse them into L2, and so on.
- **Version compaction:** keep current file versions in high detail; archive historical revisions as coarse gists or diffs to preserve traceability without hoarding tokens.
- **Temporal decay:** assign domain-specific half-lives (e.g., fast decay for sensor logs, slow for specs) so the tree naturally thins over time.

### Guardrails & recovery
- **Soft delete tier:** move candidates to a cold or “inactive” partition before permanent removal. If future telemetry indicates renewed interest, restore the branch.
- **Event tagging:** allow the runtime or humans to tag spans (“bugfix”, “incident”, “reward spike”). Tagged spans bypass automated pruning.
- **Audit metadata:** retain compact descriptors (timestamp, checksum, parent ID) so pruned content remains discoverable in logs, even if embeddings are gone.

### Automation considerations
- Run pruning jobs asynchronously with the main decode loop, using accumulated telemetry to schedule compaction during low-load windows.
- Extend LensNet labeling to estimate utility loss if an ancestor disappears, providing data-driven pruning hints.
- Explore RL/bandit policies that treat storage as a constrained resource and learn which spans to retain for maximal downstream reward.

### Domain notes
- **Robotics:** tie retention to reward signals or anomaly detectors; keep high-resolution data around events (collisions, novel observations) and aggressively compress idle periods.
- **Codebases:** maintain a graph from code spans to tests/issues/PRs so spans with active dependencies stay detailed; collapse stale modules to L2/L3 summaries.
- **Documentation:** preserve canonical specs verbatim, but decay meeting notes or superseded plans once new revisions land.

Pruning is easiest if provenance, access counts, and tagging hooks exist from day one; the POC should wire these metrics even if pruning is a future feature.

---

## Future directions

- Async disk streaming of the lifetime tree.  
- RL-trained focus allocator optimizing accuracy × latency.  
- Multi-token gists for structured data.  
- Joint training of LLM + MegaContext from scratch.  
- Shared or federated lifetime memories between agents.
- Adaptive pruning of lifetime memory to keep knowledge fresh and storage bounded (see [Pruning lifetime context](#pruning-lifetime-context)).

---

## License & contributions

MIT License (suggested).  
PRs welcome — please include reproducible tests for GistNet, LensNet, the focus allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures — focusing detailed computation only where needed.  
It opens a path to persistent, updatable, and truly lifelong language models.*
