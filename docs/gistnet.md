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

### Larger corpus option

For more realistic experiments (<1 GB total), run:

```
bash tools/download_gutenberg.sh data/raw/gutenberg
uv run python -m tools.prepare_dataset --config configs/data/gutenberg_sample.yaml
```

The Gutenberg subset feeds into the same pipeline (`block_size=32`,
`horizon=64`) and produces `data/gutenberg_sample/train.arrow` for training.

## Training Scaffold

`tools/train_gistnet.py` provides a thin training loop that:

1. Loads a single Arrow shard into memory.
2. Instantiates `GistNet` from a YAML/JSON config block.
3. Runs an MSE reconstruction objective between the predicted gist and the cached
   target vector.
4. Saves a checkpoint to `artifacts/gistnet/gistnet.pt`.

Run the trainer:

```
uv run python -m tools.train_gistnet \
    --dataset data/sample_text/train.arrow \
    --config configs/runs/gistnet_example.yaml
```

The scaffold logs step-wise losses to stdout and keeps the code path entirely
tensor-first so future phases can swap in larger datasets or evaluation metrics.

The provided `configs/runs/gistnet_example.yaml` matches the Gutenberg shard
defaults (`block_size=32`, hidden size 1024) and targets the MobileLLM teacher.
Throttle `max_steps`/`batch_size` if you need a faster smoke run on smaller GPUs.

### Logging & Visualisation

- Progress defaults to a `tqdm` bar when the dependency is available (ideal for
  Colab). Disable it with `--no-tqdm`.
- Add `--metrics-path artifacts/gistnet/metrics.json` to dump raw losses for
  custom plotting/notebooks.
- Use `--save-plot artifacts/gistnet/loss.png` to emit a ready-made curve (falls
  back gracefully if `matplotlib` is missing).
- CLI runs on infra like Novita can pass `--use-wandb --wandb-project <name>` to
  stream the same metrics to Weights & Biases without touching notebook code.

When the script detects a notebook environment (e.g., Colab), it automatically
renders the loss curve inline *and* saves the PNG. Headless terminals simply
receive the file/log output unless you add `--save-plot` or `--use-wandb`.

### Colab notebook

Open the curated notebook via Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-org>/MegaContext/blob/main/notebooks/gistnet_demo.ipynb)

The notebook automates cloning, dataset download, preparation, and training with
inline visualisations.
