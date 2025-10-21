"""
Convert raw corpora into tensor-aligned Arrow shards for gist training.

Usage:
    uv run python -m tools.prepare_dataset --config configs/data/sample_text.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from megacontext.data import DatasetConfig, SplitConfig


def load_dataset_config(path: Path) -> DatasetConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return DatasetConfig.model_validate(raw)


def gather_documents(split: SplitConfig, base_dir: Path) -> list[str]:
    pattern = (base_dir / split.source).expanduser()
    matches = sorted(glob.glob(str(pattern), recursive=True))
    if not matches:
        msg = f"No files matched pattern {pattern!s}"
        raise FileNotFoundError(msg)
    documents: list[str] = []
    limit = split.max_files or len(matches)
    for file_path in matches[:limit]:
        text = Path(file_path).read_text(encoding="utf-8")
        documents.append(text)
    return documents


def chunkify(tokens: Sequence[int], block_size: int) -> Iterable[list[int]]:
    for start in range(0, len(tokens), block_size):
        block = tokens[start : start + block_size]
        if len(block) == block_size:
            yield list(block)


def tokenize_documents(
    documents: Sequence[str],
    tokenizer,
    block_size: int,
    max_tokens: int | None,
) -> list[list[list[int]]]:
    doc_blocks: list[list[list[int]]] = []
    tokens_emitted = 0
    for doc in documents:
        encoded = tokenizer(doc, add_special_tokens=False, return_attention_mask=False)
        token_ids = encoded["input_ids"]
        blocks: list[list[int]] = []
        for block in chunkify(token_ids, block_size):
            blocks.append(block)
            tokens_emitted += len(block)
            if max_tokens is not None and tokens_emitted >= max_tokens:
                doc_blocks.append(blocks)
                return doc_blocks
        doc_blocks.append(blocks)
        if max_tokens is not None and tokens_emitted >= max_tokens:
            break
    return doc_blocks


def build_horizon_examples(
    doc_blocks: Sequence[Sequence[Sequence[int]]],
    *,
    block_size: int,
    horizon: int,
) -> list[dict[str, list[int]]]:
    blocks_per_window = horizon // block_size
    examples: list[dict[str, list[int]]] = []
    for blocks in doc_blocks:
        if len(blocks) < blocks_per_window:
            continue
        for start in range(0, len(blocks) - blocks_per_window + 1):
            window = blocks[start : start + blocks_per_window]
            context_tokens = [token for block in window for token in block]
            examples.append(
                {
                    "tokens": window[0],
                    "context_tokens": context_tokens,
                }
            )
    return examples


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    if isinstance(value, str):
        if hasattr(torch, value):
            return getattr(torch, value)
        raise ValueError(f"Unsupported torch dtype {value!r}")
    raise TypeError(f"Expected dtype string or None, received {type(value)!r}")


def compute_teacher_embeddings(
    teacher_model,
    examples: Sequence[dict[str, list[int]]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    hidden_chunks: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        input_ids = torch.tensor(
            [ex["context_tokens"] for ex in batch_examples],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1][:, :block_size, :].to(torch.float32)
        hidden_chunks.append(hidden.cpu())
    if not hidden_chunks:
        return torch.empty(0, block_size, 0, dtype=torch.float32)
    return torch.cat(hidden_chunks, dim=0)


def write_arrow(output_path: Path, records: dict[str, list[Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "input_ids": pa.array(records["input_ids"], type=pa.list_(pa.int64())),
            "attention_mask": pa.array(
                records["attention_mask"], type=pa.list_(pa.int8())
            ),
            "context_input_ids": pa.array(
                records["context_input_ids"], type=pa.list_(pa.int64())
            ),
            "context_attention_mask": pa.array(
                records["context_attention_mask"], type=pa.list_(pa.int8())
            ),
            "teacher_hidden": pa.array(
                records["teacher_hidden"],
                type=pa.list_(pa.list_(pa.float32())),
            ),
            "gist_target": pa.array(
                records["gist_target"], type=pa.list_(pa.float32())
            ),
        }
    )
    with pa_ipc.new_file(output_path.open("wb"), table.schema) as writer:
        writer.write_table(table)


def update_metadata(
    meta_path: Path,
    config: DatasetConfig,
    summary: dict[str, dict[str, Any]],
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": config.dataset_name,
        "tokenizer": config.tokenizer,
        "block_size": config.block_size,
        "horizon": config.horizon,
        "teacher_model": config.teacher_model,
        "teacher_dtype": config.teacher_dtype,
        "splits": summary,
    }
    if meta_path.suffix in {".yaml", ".yml"}:
        meta_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_split(
    split_config: SplitConfig,
    tokenizer,
    config: DatasetConfig,
    *,
    base_dir: Path,
    teacher_model,
) -> dict[str, Any]:
    documents = gather_documents(split_config, base_dir=base_dir)
    doc_blocks = tokenize_documents(
        documents,
        tokenizer=tokenizer,
        block_size=config.block_size,
        max_tokens=split_config.max_tokens,
    )

    examples = build_horizon_examples(
        doc_blocks,
        block_size=config.block_size,
        horizon=config.horizon,
    )

    teacher_hidden_size = 0
    teacher_rows: list[list[list[float]]] = [[] for _ in examples]
    gist_targets: list[list[float]] = [[] for _ in examples]
    if teacher_model is not None and examples:
        device = next(teacher_model.parameters()).device
        hidden = compute_teacher_embeddings(
            teacher_model,
            examples,
            block_size=config.block_size,
            batch_size=config.teacher_batch_size,
            device=device,
        )
        teacher_hidden_size = int(hidden.shape[-1])
        tensor_rows = hidden.tolist()
        teacher_rows = tensor_rows
        gist_targets = hidden.mean(dim=1).tolist()

    records = {
        "input_ids": [],
        "attention_mask": [],
        "context_input_ids": [],
        "context_attention_mask": [],
        "teacher_hidden": [],
        "gist_target": [],
    }
    for idx, example in enumerate(examples):
        records["input_ids"].append(example["tokens"])
        records["attention_mask"].append([1] * config.block_size)
        records["context_input_ids"].append(example["context_tokens"])
        records["context_attention_mask"].append([1] * len(example["context_tokens"]))
        records["teacher_hidden"].append(teacher_rows[idx])
        records["gist_target"].append(gist_targets[idx])

    output_path = Path(split_config.output_path)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    write_arrow(output_path, records)

    blocks_processed = sum(len(blocks) for blocks in doc_blocks)
    summary: dict[str, Any] = {
        "documents": len(documents),
        "blocks": blocks_processed,
        "examples": len(examples),
        "teacher_hidden_size": teacher_hidden_size,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for MegaContext.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML config.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_dataset_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Allow arbitrarily long documents while chunking without warnings.
    tokenizer.model_max_length = max(config.horizon, config.block_size) * 1024
    teacher_model = None
    if config.teacher_model is not None:
        dtype = resolve_torch_dtype(config.teacher_dtype)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            dtype=dtype,
        )
        teacher_model.eval()
        teacher_model.to(config.teacher_device)

    base_dir = config_path.parent
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_name, split_config in tqdm(config.splits.items(), desc="Splits"):
        summary = process_split(
            split_config,
            tokenizer,
            config,
            base_dir=base_dir,
            teacher_model=teacher_model,
        )
        split_summaries[split_name] = summary
        summary_line = (
            f"[{split_name}] {summary['documents']} docs, {summary['blocks']} blocks "
            f"→ {summary['examples']} examples "
            f"(teacher dim={summary['teacher_hidden_size']})"
        )
        print(summary_line)

    metadata_path = config.metadata_path()
    if metadata_path.suffix not in {".yaml", ".yml"}:
        metadata_path = metadata_path.with_suffix(".yaml")
    update_metadata(metadata_path, config, split_summaries)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
