---
tags:
  - architecture
summary: Defines module boundaries, environment assumptions, and storage formats for the proof-of-concept.
---
Outlines how the proof-of-concept wires modules, datasets, and storage formats so the [[Runtime Loop]] can execute with minimal assumptions.

---

- **Module table:** clarifies responsibilities across [[GistNet]], [[LensNet]], [[Focus Allocator]], and runtime engine.
- **Environment:** PyTorch 2.2+, FlashAttention 2, `uv` for dependency management.
- **Storage:** `{LOD0,LOD1,LOD2}.ctx` binary layout with deterministic offsets.
- **Configs:** sample YAML showing run parameters and dataset wiring.
- **Linked plans:** originally aligned with the legacy POC milestone; for current requirements see [[MegaContext PRD Index]] alongside [[POC Scope]] constraints.

---
## Details

This note captures the module map, environment assumptions, and storage layout that previously lived in `README.md`. Treat it as historical context alongside the active requirements in [[MegaContext PRD Index]].

## Module responsibilities

| Module | Suggested path | Responsibilities | Key inputs/outputs |
|--------|----------------|------------------|--------------------|
| [[GistNet]] | `src/megacontext/gistnet/` | Train & serve 32→1 gists, populate [[MegaContext Tree]] nodes (legacy notebook implementation) | Input: token embeddings; Output: gist vectors + metrics |
| [[MegaContext Tree]] | (design only; placeholder stubs under `src/megacontext/data/`) | Maintain contiguous-in-time hierarchy (LOD0/LOD1/LOD2) in RAM (future stream to disk) | Input: gists/tokens; Output: node handles, metadata |
| [[Focus Allocator]] | (not yet implemented in code) | Apply [[LensNet]] scores to expand/collapse blocks | Input: [[Working Context]] entries, scores; Output: refreshed WC |
| [[LensNet]] | (design only; nanochat implementation tracked in PRDs) | Score each WC entry for detail adjustments | Input: WC entries + tail gists; Output: focus scores |
| [[Runtime Loop]] | `src/megacontext/runtime/` (WorkingContext + BaseModel wrappers) | Orchestrate ingest → refocus → decode for the notebook flow; a full nanochat engine is planned | Input: streaming tokens; Output: next-token logits, telemetry |
| CLI tools | `tools/` | Command-line helpers for dataset prep, logging, evaluation | Input: CLI args/config; Output: reports, artifacts |
| Evaluation/tests | `tests/` mirrored per module | Validate substitutability, focus policy, end-to-end behavior | Input: synthetic + real traces |

> **Implementation note:** All `src/megacontext/...` modules are stopgaps used by the research notebook. The nanochat-based counterparts will replace them during the migration described in [[MegaContext PRD Index]] and [[Migration Plan - Nanochat Integration]].

```mermaid
graph LR
    subgraph Storage & Compression
        GistNet[GistNet<br/>src/megacontext/gistnet]
        MCT[MegaContext Tree<br/>(design)]
    end
    subgraph Focus Control
        LensNet[LensNet<br/>(design)]
        FA[Focus Allocator<br/>(design)]
    end
    WC[Working Context<br/>src/megacontext/runtime]
    Runtime[Runtime Loop / Base LLM]

    GistNet --> MCT
    MCT --> WC
    WC --> LensNet
    LensNet --> FA
    FA --> WC
    WC --> Runtime
    Runtime -->|ΔNLL / telemetry| LensNet
```

## Framework & environment assumptions

- **Base model:** start with `HuggingFaceTB/SmolLM3-3B` (bf16) or, if compute is tighter, `Qwen/Qwen3-1.7B`. Both run comfortably on a single 24–48 GB GPU.
- **Runtime stack:** PyTorch ≥ 2.2 with FlashAttention 2, Hugging Face `transformers`, `accelerate`, and `datasets`.
- **Environment bootstrap:** prefer [`uv`](https://github.com/astral-sh/uv) for reproducible installs: `uv venv`, `uv pip install -r requirements.txt`, then `uv run python -m pip install -e .` for editable modules if needed.
- **Logging:** use [Weights & Biases](https://wandb.ai) for metrics and counterfactual ΔNLL traces; keep raw gists in memory for the POC.
- **Precision:** bf16 for model forward/backward; fp16 for gist snapshots if you need serialization.
- **Configuration:** place experiment configs under `configs/` (YAML) documenting block size `K`, horizon `H`, ΔNLL sampling strategy, and thresholds (`τ_expand`, `τ_collapse`).
- **Dataset staging:** tokenize corpora into contiguous 32-token blocks and store them as `.arrow` shards under `data/<dataset>/<split>.arrow`; provide `uv run python -m tools.prepare_dataset --config configs/<experiment>.yaml` (e.g., `configs/Gutenberg_SmolLM3.yaml`) to regenerate them. Set `MEGACONTEXT_DATA_ROOT=/path/to/storage` (e.g., a mounted NFS directory) to redirect outputs to persistent storage.
- **GistNet training:** orchestrate runs via `megacontext.gistnet.lightning.build_gistnet_experiment` (see `notebooks/megacontext.ipynb` for a ready-made Jupyter workflow) instead of the deprecated CLI script.
- **Storage layout:** persist [[MegaContext Tree]] memory as `{LOD0,LOD1,LOD2}.ctx` binary files with a fixed header plus packed data (see below). Fixed block sizes make byte offsets deterministic, so no external index is required.

## Binary storage layout (`{LOD0,LOD1,LOD2}.ctx`)

Each file begins with a 64-byte header followed by tightly packed payloads. The header uses little-endian encoding and the following fields:

| Offset | Field | Type | Meaning |
|--------|-------|------|---------|
| 0 | `magic` | `uint32` | Constant `0x4D434354` (`MCCT`) to detect corruption. |
| 4 | `version` | `uint16` | Format revision (start at `1`). |
| 6 | `level` | `uint16` | 0, 1, or 2 indicating `LOD0`, `LOD1`, or `LOD2`. |
| 8 | `block_size` | `uint16` | Number of LOD0 tokens per gist (default 32). |
| 10 | `embedding_dim` | `uint16` | Width `d` of gist vectors (for `LOD1`/`LOD2`). |
| 12 | `dtype_code` | `uint16` | 0=`uint32`, 1=`fp16`, 2=`bf16`. |
| 14 | `model_name` | `char[32]` | UTF-8 null-terminated identifier of the base model (e.g., `SmolLM3-3B`). |
| 46 | `reserved` | 18 bytes | Zeroed; available for future metadata (checksum, flags). |

Payload layout per level:
- **LOD0 (`dtype_code=0`):** contiguous `uint32` token ids matching the base tokenizer vocabulary. Each block stores exactly `block_size` entries.
- **LOD1/LOD2 (`dtype_code=1`):** contiguous `fp16` vectors of shape `[num_nodes, embedding_dim]`. Gists inherit the same orientation as the base embedding matrix, so random access is `offset = header_size + index * embedding_dim * 2`.

Per-node metadata (`span_id`, `start_token`, `level`, parent/child pointers) stays in the [[MegaContext Tree]]'s in-memory index; because the binary payloads are fixed-width, offsets can always be recomputed on the fly.

## Sample run config (`configs/Gutenberg_SmolLM3.yaml`)

```yaml
name: Gutenberg_SmolLM3
dataset:
  dataset_name: gutenberg_sample
  tokenizer: HuggingFaceTB/SmolLM2-360M-Instruct
  block_size: 32
  context_tokens: 512
  horizon: 32
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
  training:
    batch_size: 8
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

Refer back to this configuration when wiring up the [[Runtime Loop]] or when validating scope boundaries in [[POC Scope]].
