---
title: "POC Architecture & Interfaces"
type: "concept"
status: "active"
tags: ["architecture"]
summary: "Defines module boundaries, environment assumptions, and storage formats for the proof-of-concept."
links:
  - "[[MOC - Core Components]]"
  - "[[POC Scope]]"
  - "[[Runtime Loop]]"
  - "[[Training & Operations]]"
---

- Outlines how the proof-of-concept wires modules, datasets, and storage formats so the runtime loop can execute with minimal assumptions.

## TL;DR
- **Module table:** clarifies responsibilities across GistNet, LensNet, allocator, and runtime engine.
- **Environment:** PyTorch 2.2+, FlashAttention 2, `uv` for dependency management.
- **Storage:** `{L0,L1,L2}.ctx` binary layout with deterministic offsets.
- **Configs:** sample YAML showing run parameters and dataset wiring.
- **Linked plans:** aligns with [[plans/POC Plan]] phases and [[POC Scope]] constraints.

## Details

This note captures the module map, environment assumptions, and storage layout that previously lived in `README.md`. It complements the milestone targets in [[plans/POC Plan]].

## Module responsibilities

| Module | Suggested path | Responsibilities | Key inputs/outputs |
|--------|----------------|------------------|--------------------|
| GistNet | `src/gistnet/` | Train & serve 32→1 gists, populate MegaContext tree nodes | Input: token embeddings; Output: gist vectors + metrics |
| MegaContext tree | `src/megacontext/memory/tree.py` | Maintain contiguous-in-time hierarchy (L0/L1/L2) in RAM (future stream to disk) | Input: gists/tokens; Output: node handles, metadata |
| Focus allocator | `src/runtime/focus_allocator.py` | Apply LensNet scores to expand/collapse blocks | Input: working-context entries, scores; Output: refreshed WC |
| LensNet | `src/lensnet/` | Score each WC entry for detail adjustments | Input: WC entries + tail gists; Output: focus scores |
| Runtime loop | `src/runtime/engine.py` | Orchestrate ingest → refocus → decode | Input: streaming tokens; Output: next-token logits, telemetry |
| CLI tools | `tools/` | Command-line helpers for dataset prep, logging, evaluation | Input: CLI args/config; Output: reports, artifacts |
| Evaluation/tests | `tests/` mirrored per module | Validate substitutability, focus policy, end-to-end behavior | Input: synthetic + real traces |

> **Diagram needed — `assets/module_stack.png`:** Layer the modules (MegaContext tree, working context, LensNet, allocator, base LLM) and annotate data moving between them each decode cycle.

## Framework & environment assumptions

- **Base model:** start with `HuggingFaceTB/SmolLM3-3B` (bf16) or, if compute is tighter, `Qwen/Qwen3-1.7B`. Both run comfortably on a single 24–48 GB GPU.
- **Runtime stack:** PyTorch ≥ 2.2 with FlashAttention 2, Hugging Face `transformers`, `accelerate`, and `datasets`.
- **Environment bootstrap:** prefer [`uv`](https://github.com/astral-sh/uv) for reproducible installs: `uv venv`, `uv pip install -r requirements.txt`, then `uv run python -m pip install -e .` for editable modules if needed.
- **Logging:** use [Weights & Biases](https://wandb.ai) for metrics and counterfactual ΔNLL traces; keep raw gists in memory for the POC.
- **Precision:** bf16 for model forward/backward; fp16 for gist snapshots if you need serialization.
- **Configuration:** place experiment configs under `configs/` (YAML) documenting block size `K`, horizon `H`, ΔNLL sampling strategy, and thresholds (`τ_expand`, `τ_collapse`).
- **Dataset staging:** tokenize corpora into contiguous 32-token blocks and store them as `.arrow` shards under `data/<dataset>/<split>.arrow`; provide `uv run python -m tools.prepare_dataset --config configs/data/<name>.yaml` to regenerate them. Set `MEGACONTEXT_DATA_ROOT=/path/to/storage` (e.g., `/content/drive/MyDrive/MegaContext` in Colab) to redirect outputs to persistent storage.
- **GistNet training:** invoke `uv run python -m tools.train_gistnet --dataset … --config …` (or `python -m tools.train_gistnet` inside notebooks) so the package resolves without ad-hoc path edits.
- **Storage layout:** persist MegaContext memory as `{L0,L1,L2}.ctx` binary files with a fixed header plus packed data (see below). Fixed block sizes make byte offsets deterministic, so no external index is required.

## Binary storage layout (`{L0,L1,L2}.ctx`)

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

Per-node metadata (`span_id`, `start_token`, `level`, parent/child pointers) stays in the MegaContext tree’s in-memory index; because the binary payloads are fixed-width, offsets can always be recomputed on the fly.

## Sample run config (`configs/runs/poc_smollm3.yaml`)

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

Refer back to this configuration when wiring up the runtime loop in [[Runtime Loop]] or when validating scope boundaries in [[POC Scope]].

## Layer 3 · Change Log
- 2025-10-22: Added metadata and layered summaries after README refactor.
