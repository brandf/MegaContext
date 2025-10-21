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
from tqdm.auto import tqdm
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


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "auto":
        return None
    if isinstance(value, str):
        if hasattr(torch, value):
            return getattr(torch, value)
        raise ValueError(f"Unsupported torch dtype {value!r}")
    raise TypeError(f"Expected dtype string or None, received {type(value)!r}")


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise ValueError(f"Unsupported torch dtype {dtype!r}")


def select_teacher_dtype(requested: str | None, device_str: str) -> torch.dtype:
    resolved = resolve_torch_dtype(requested)
    if resolved is not None:
        return resolved
    device = torch.device(device_str)
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        major, _ = torch.cuda.get_device_capability(index)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def torch_dtype_to_arrow(dtype: torch.dtype) -> pa.DataType:
    if dtype is torch.float16:
        return pa.float16()
    if dtype is torch.bfloat16:
        if hasattr(pa, "bfloat16"):
            return pa.bfloat16()
        raise RuntimeError(
            "pyarrow does not support bfloat16; upgrade pyarrow or choose a "
            "different dtype"
        )
    if dtype is torch.float32:
        return pa.float32()
    raise ValueError(f"Unsupported torch dtype {dtype!r} for Arrow export")


class ArrowShardWriter:
    def __init__(self, output_path: Path, *, teacher_type: pa.DataType) -> None:
        self._path = output_path
        self._file_handle = None
        self._writer: pa_ipc.RecordBatchWriter | None = None
        self._teacher_type = teacher_type

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                pa.field("context_input_ids", pa.list_(pa.int64())),
                pa.field("context_attention_mask", pa.list_(pa.int8())),
                pa.field("future_input_ids", pa.list_(pa.int64())),
                pa.field("future_attention_mask", pa.list_(pa.int8())),
                pa.field(
                    "teacher_context_hidden", pa.list_(pa.list_(self._teacher_type))
                ),
                pa.field(
                    "teacher_future_hidden", pa.list_(pa.list_(self._teacher_type))
                ),
            ]
        )
        self._file_handle = self._path.open("wb")
        self._writer = pa_ipc.new_file(self._file_handle, schema)

    def write_batch(
        self,
        *,
        context_input_ids: list[list[int]],
        context_attention_mask: list[list[int]],
        future_input_ids: list[list[int]],
        future_attention_mask: list[list[int]],
        teacher_context_hidden: list[list[list[float]]],
        teacher_future_hidden: list[list[list[float]]],
    ) -> None:
        if not context_input_ids:
            return
        self._ensure_writer()
        assert self._writer is not None
        table = pa.table(
            {
                "context_input_ids": pa.array(
                    context_input_ids, type=pa.list_(pa.int64())
                ),
                "context_attention_mask": pa.array(
                    context_attention_mask, type=pa.list_(pa.int8())
                ),
                "future_input_ids": pa.array(
                    future_input_ids, type=pa.list_(pa.int64())
                ),
                "future_attention_mask": pa.array(
                    future_attention_mask, type=pa.list_(pa.int8())
                ),
                "teacher_context_hidden": pa.array(
                    teacher_context_hidden, type=pa.list_(pa.list_(self._teacher_type))
                ),
                "teacher_future_hidden": pa.array(
                    teacher_future_hidden, type=pa.list_(pa.list_(self._teacher_type))
                ),
            }
        )
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None


def update_metadata(
    meta_path: Path,
    config: DatasetConfig,
    summary: dict[str, dict[str, Any]],
    *,
    teacher_dtype: str,
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": config.dataset_name,
        "tokenizer": config.tokenizer,
        "block_size": config.block_size,
        "context_tokens": config.context_tokens,
        "context_stride": config.context_stride or config.context_tokens,
        "horizon": config.horizon,
        "teacher_model": config.teacher_model,
        "teacher_dtype": teacher_dtype,
        "teacher_dtype_config": config.teacher_dtype,
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
    teacher_dtype: torch.dtype,
) -> dict[str, Any]:
    documents = gather_documents(split_config, base_dir=base_dir)
    context_tokens = config.context_tokens
    horizon_tokens = config.horizon
    stride_tokens = config.context_stride or context_tokens
    block_size = config.block_size
    blocks_per_context = context_tokens // block_size
    blocks_per_horizon = horizon_tokens // block_size
    blocks_per_example = blocks_per_context + blocks_per_horizon
    stride_blocks = stride_tokens // block_size

    output_path = Path(split_config.output_path)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    try:
        teacher_arrow_type = torch_dtype_to_arrow(teacher_dtype)
    except (RuntimeError, ValueError) as exc:
        print(
            f"Warning: {exc}. Falling back to float32 for teacher embeddings.",
            flush=True,
        )
        teacher_dtype = torch.float32
        teacher_arrow_type = torch_dtype_to_arrow(teacher_dtype)
    teacher_dtype_str = torch_dtype_to_str(teacher_dtype)
    writer = ArrowShardWriter(output_path, teacher_type=teacher_arrow_type)

    teacher_device = None
    if teacher_model is not None:
        teacher_device = next(teacher_model.parameters()).device

    batch_examples: list[dict[str, list[int]]] = []
    flush_threshold = config.teacher_batch_size if teacher_model is not None else 512

    contexts_processed = 0
    examples_emitted = 0
    teacher_hidden_size = 0
    tokens_consumed = 0
    documents_processed = 0

    doc_iter = tqdm(
        documents,
        desc=f"Tokenizing {split_config.name}",
        leave=False,
        position=1,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    def flush_batch() -> None:
        nonlocal batch_examples, teacher_hidden_size, examples_emitted
        if not batch_examples:
            return
        if teacher_model is not None:
            sequences = [
                ex["context_tokens"] + ex["future_tokens"] for ex in batch_examples
            ]
            input_ids = torch.tensor(sequences, dtype=torch.long, device=teacher_device)
            attention_mask = torch.ones_like(input_ids, device=teacher_device)
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = outputs.hidden_states[-1][:, : context_tokens + horizon_tokens, :]
            hidden = hidden.to(teacher_dtype)
            teacher_hidden_size = int(hidden.shape[-1])
            context_hidden = hidden[:, :context_tokens, :]
            future_hidden = hidden[:, context_tokens:, :]
            context_hidden_cpu = context_hidden.detach().cpu()
            future_hidden_cpu = future_hidden.detach().cpu()
            teacher_context_hidden = context_hidden_cpu.tolist()
            teacher_future_hidden = future_hidden_cpu.tolist()
        else:
            teacher_context_hidden = [[] for _ in batch_examples]
            teacher_future_hidden = [[] for _ in batch_examples]

        context_attention = [[1] * context_tokens for _ in batch_examples]
        future_attention = [[1] * horizon_tokens for _ in batch_examples]
        writer.write_batch(
            context_input_ids=[ex["context_tokens"] for ex in batch_examples],
            context_attention_mask=context_attention,
            future_input_ids=[ex["future_tokens"] for ex in batch_examples],
            future_attention_mask=future_attention,
            teacher_context_hidden=teacher_context_hidden,
            teacher_future_hidden=teacher_future_hidden,
        )
        examples_emitted += len(batch_examples)
        batch_examples = []

    for doc in doc_iter:
        encoded = tokenizer(doc, add_special_tokens=False, return_attention_mask=False)
        token_ids = encoded["input_ids"]
        blocks: list[list[int]] = []
        for block in chunkify(token_ids, config.block_size):
            blocks.append(block)
            tokens_consumed += len(block)
            if (
                split_config.max_tokens is not None
                and tokens_consumed >= split_config.max_tokens
            ):
                break
        documents_processed += 1
        total_blocks = len(blocks)
        if total_blocks < blocks_per_example:
            if (
                split_config.max_tokens is not None
                and tokens_consumed >= split_config.max_tokens
            ):
                break
            continue

        max_start = total_blocks - blocks_per_example + 1
        for start in range(0, max_start, stride_blocks):
            context_blocks = blocks[start : start + blocks_per_context]
            future_blocks = blocks[
                start + blocks_per_context : start + blocks_per_example
            ]
            context_flat = [token for block in context_blocks for token in block]
            future_flat = [token for block in future_blocks for token in block]
            if (
                len(context_flat) != context_tokens
                or len(future_flat) != horizon_tokens
            ):
                continue
            batch_examples.append(
                {
                    "context_tokens": context_flat,
                    "future_tokens": future_flat,
                }
            )
            contexts_processed += 1
            if (
                split_config.max_tokens is not None
                and tokens_consumed >= split_config.max_tokens
            ):
                if batch_examples:
                    flush_batch()
                writer.close()
                summary = {
                    "documents": documents_processed,
                    "contexts": contexts_processed,
                    "examples": examples_emitted,
                    "teacher_hidden_size": teacher_hidden_size,
                    "teacher_dtype": teacher_dtype_str,
                }
                return summary
            if len(batch_examples) >= flush_threshold:
                flush_batch()

        if (
            split_config.max_tokens is not None
            and tokens_consumed >= split_config.max_tokens
        ):
            break

    flush_batch()
    writer.close()

    summary = {
        "documents": documents_processed,
        "contexts": contexts_processed,
        "examples": examples_emitted,
        "teacher_hidden_size": teacher_hidden_size,
        "teacher_dtype": teacher_dtype_str,
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
    # Allow very long documents; chunking keeps blocks within block_size.
    tokenizer.model_max_length = int(1e6)
    teacher_dtype = select_teacher_dtype(config.teacher_dtype, config.teacher_device)
    teacher_model = None
    if config.teacher_model is not None:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=teacher_dtype,
            trust_remote_code=config.teacher_trust_remote_code,
        )
        teacher_model.to(config.teacher_device)
        teacher_model.eval()

    base_dir = config_path.parent
    split_summaries: dict[str, dict[str, Any]] = {}
    actual_teacher_dtype_str: str | None = None
    for split_name, split_config in tqdm(config.splits.items(), desc="Splits"):
        summary = process_split(
            split_config,
            tokenizer,
            config,
            base_dir=base_dir,
            teacher_model=teacher_model,
            teacher_dtype=teacher_dtype,
        )
        split_summaries[split_name] = summary
        actual_teacher_dtype_str = summary["teacher_dtype"]
        summary_line = (
            f"[{split_name}] {summary['documents']} docs, "
            f"{summary['contexts']} contexts → {summary['examples']} examples "
            f"(teacher dim={summary['teacher_hidden_size']} "
            f"dtype={summary['teacher_dtype']})"
        )
        print(summary_line)

    metadata_path = config.metadata_path()
    if metadata_path.suffix not in {".yaml", ".yml"}:
        metadata_path = metadata_path.with_suffix(".yaml")
    dtype_str = actual_teacher_dtype_str or torch_dtype_to_str(teacher_dtype)
    update_metadata(
        metadata_path,
        config,
        split_summaries,
        teacher_dtype=dtype_str,
    )
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
