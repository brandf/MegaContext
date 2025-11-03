"""
Convert raw corpora into tensor-aligned Arrow shards for gist training.

Usage:
    uv run python -m tools.prepare_dataset --config configs/Gutenberg_SmolLM3.yaml
    # Combined experiment configs live under configs/*.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
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
from megacontext.utils.precision import resolve_runtime_precision


def load_dataset_config(path: Path) -> DatasetConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    dataset_payload = raw.get("dataset") if isinstance(raw, dict) else None
    if dataset_payload is None:
        dataset_payload = raw
    if isinstance(raw, dict):
        base_model_payload = raw.get("base_model")
        if base_model_payload and isinstance(base_model_payload, dict):
            base_model_name = base_model_payload.get("name")
            tokenizer = dataset_payload.get("tokenizer")
            if tokenizer in (None, "auto"):
                if not base_model_name:
                    raise ValueError(
                        "Specify dataset.tokenizer when base_model.name is unset."
                    )
                dataset_payload["tokenizer"] = base_model_name
            teacher_model = dataset_payload.get("teacher_model")
            if teacher_model in (None, "auto"):
                if not base_model_name:
                    raise ValueError(
                        "Specify dataset.teacher_model when base_model.name is unset."
                    )
                dataset_payload["teacher_model"] = base_model_name
    return DatasetConfig.model_validate(dataset_payload)


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


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise ValueError(f"Unsupported torch dtype {dtype!r}")


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
        teacher_context_hidden: list[list[list[float]]] | pa.Array,
        teacher_future_hidden: list[list[list[float]]] | pa.Array,
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
                "teacher_context_hidden": (
                    teacher_context_hidden
                    if isinstance(teacher_context_hidden, pa.Array)
                    else pa.array(
                        teacher_context_hidden,
                        type=pa.list_(pa.list_(self._teacher_type)),
                    )
                ),
                "teacher_future_hidden": (
                    teacher_future_hidden
                    if isinstance(teacher_future_hidden, pa.Array)
                    else pa.array(
                        teacher_future_hidden,
                        type=pa.list_(pa.list_(self._teacher_type)),
                    )
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

    output_path_cfg = Path(split_config.output_path)
    data_root_env = os.environ.get("MEGACONTEXT_DATA_ROOT")
    if output_path_cfg.is_absolute():
        output_path = output_path_cfg
    else:
        default_output = (base_dir / output_path_cfg).resolve()
        if data_root_env:
            resolved_base = base_dir.resolve()
            repo_root = (
                resolved_base.parents[1]
                if len(resolved_base.parents) >= 2
                else resolved_base
            )
            try:
                relative = default_output.relative_to(repo_root)
            except ValueError:
                relative = default_output.name
            output_path = (
                Path(data_root_env).expanduser().resolve() / relative
            ).resolve()
        else:
            output_path = default_output
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
    context_bar = tqdm(
        desc=f"Contexts {split_config.name}",
        leave=False,
        position=2,
        dynamic_ncols=True,
        mininterval=0.2,
        total=0,
        unit="ctx",
    )
    context_bar.set_postfix(examples=0)

    def flush_batch() -> None:
        nonlocal batch_examples, teacher_hidden_size
        nonlocal examples_emitted, contexts_processed
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

            def to_fixed_size_array(
                tensor: torch.Tensor, *, list_sizes: Sequence[int]
            ) -> pa.Array:
                values = pa.array(tensor.reshape(-1).numpy(), type=teacher_arrow_type)
                array: pa.Array = values
                for size in list_sizes:
                    array = pa.FixedSizeListArray.from_arrays(array, size)
                return array

            teacher_context_hidden = to_fixed_size_array(
                context_hidden_cpu, list_sizes=[teacher_hidden_size, context_tokens]
            ).to_pylist()
            teacher_future_hidden = to_fixed_size_array(
                future_hidden_cpu, list_sizes=[teacher_hidden_size, horizon_tokens]
            ).to_pylist()
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
        produced_examples = len(batch_examples)
        examples_emitted += produced_examples
        batch_examples = []
        if (
            torch.cuda.is_available()
            and shutil.which("nvidia-smi")
            and contexts_processed > 0
            and contexts_processed % 50 == 0
        ):
            try:
                subprocess.run(["nvidia-smi"], check=False)
            except OSError:
                pass

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
        if max_start > 0 and stride_blocks > 0:
            contexts_for_doc = (max_start + stride_blocks - 1) // stride_blocks
            context_bar.total += contexts_for_doc
            context_bar.refresh()
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
            context_bar.update(1)
            if contexts_processed % 20 == 0:
                context_bar.set_postfix(examples=examples_emitted)
            if (
                split_config.max_tokens is not None
                and tokens_consumed >= split_config.max_tokens
            ):
                if batch_examples:
                    flush_batch()
                context_bar.n = contexts_processed
                context_bar.set_postfix(examples=examples_emitted)
                context_bar.total = max(context_bar.total, contexts_processed)
                context_bar.refresh()
                doc_iter.close()
                context_bar.close()
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
    context_bar.n = contexts_processed
    context_bar.set_postfix(examples=examples_emitted)
    context_bar.total = max(context_bar.total, contexts_processed)
    context_bar.refresh()
    doc_iter.close()
    context_bar.close()
    writer.close()

    summary = {
        "documents": documents_processed,
        "contexts": contexts_processed,
        "examples": examples_emitted,
        "teacher_hidden_size": teacher_hidden_size,
        "teacher_dtype": teacher_dtype_str,
    }
    return summary


def prepare_dataset_from_config(
    config_path: Path,
    *,
    log: bool = True,
) -> dict[str, Any]:
    """
    Prepare dataset shards and metadata using the provided YAML config.

    Args:
        config_path: Path to a ``DatasetConfig`` YAML file.
        log: When ``True``, emit progress summaries to stdout (default: ``True``).

    Returns:
        Dictionary containing per-split summaries and the metadata path.
    """

    resolved_path = config_path.resolve()
    config = load_dataset_config(resolved_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e6)

    teacher_device_str, teacher_dtype = resolve_runtime_precision(
        device_preference=config.teacher_device,
        dtype_preference=config.teacher_dtype,
    )
    if teacher_device_str.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    teacher_model = None
    if config.teacher_model is not None:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=teacher_dtype,
            trust_remote_code=config.teacher_trust_remote_code,
        )
        teacher_model.to(teacher_device_str)
        teacher_model.eval()

    base_dir = resolved_path.parent
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
        if log:
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
    if log:
        print(f"Wrote metadata to {metadata_path}")
    return {
        "config_path": str(resolved_path),
        "metadata_path": str(metadata_path),
        "teacher_dtype": dtype_str,
        "splits": split_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for MegaContext.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML config.",
    )
    args = parser.parse_args()

    prepare_dataset_from_config(args.config)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
