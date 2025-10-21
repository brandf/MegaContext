# GistNet Training Notes

This document summarizes the minimal data and training pipeline for the Phase 2
gist compressor.

## Dataset Fields

`tools/prepare_dataset.py` now emits Arrow shards with the following columns:

- `input_ids` — L0 token ids for the block being compressed (`block_size` tokens).
- `attention_mask` — mask for the block (currently all ones).
- `context_input_ids` — flattened horizon window (`horizon` tokens) used when the
  teacher model produced cached embeddings.
- `context_attention_mask` — mask for the horizon window.
- `teacher_hidden` — cached teacher embeddings with shape
  `[block_size, teacher_hidden_size]` stored as float32 lists.
- `gist_target` — pooled target vector (mean of the teacher hidden states) with
  length `teacher_hidden_size`.

The metadata written to `data/<dataset>/metadata.yaml` records the tokenizer,
block size, horizon, teacher model, and per-split statistics.

## Preparing the Sample Dataset

```
uv run python -m tools.prepare_dataset --config configs/data/sample_text.yaml
```

The sample configuration uses `sshleifer/tiny-gpt2` as the teacher and writes a
single shard to `data/sample_text/train.arrow`. Adjust `teacher_device` or the
horizon in the config to match your setup.

## Training Scaffold

`tools/train_gistnet.py` provides a thin training loop that:

1. Loads a single Arrow shard into memory.
2. Instantiates `GistNet` from a YAML/JSON config block.
3. Runs an MSE reconstruction objective between the predicted gist and the cached
   target vector.
4. Saves a checkpoint to `artifacts/gistnet/gistnet.pt`.

Example config snippet (`configs/runs/gistnet_example.yaml` to be added later):

```yaml
model:
  hidden_size: 2048
  block_size: 32
  num_heads: 8
  mlp_ratio: 4.0
training:
  batch_size: 8
  lr: 1.0e-3
  max_steps: 500
  device: cuda:0
```

Run the trainer:

```
uv run python -m tools.train_gistnet \
    --dataset data/sample_text/train.arrow \
    --config configs/runs/gistnet_example.yaml
```

The scaffold logs step-wise losses to stdout and keeps the code path entirely
tensor-first so future phases can swap in larger datasets or evaluation metrics.
