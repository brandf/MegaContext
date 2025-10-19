"""
Convert raw text corpora into 32-token `.arrow` shards for MegaContext training.

Usage:
    uv run python -m tools.prepare_dataset --config configs/data/sample_text.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

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
) -> list[list[int]]:
    blocks: list[list[int]] = []
    for doc in documents:
        encoded = tokenizer(doc, add_special_tokens=False, return_attention_mask=False)
        token_ids = encoded["input_ids"]
        for block in chunkify(token_ids, block_size):
            blocks.append(block)
            if max_tokens is not None and len(blocks) * block_size >= max_tokens:
                return blocks
    return blocks


def write_arrow(output_path: Path, blocks: list[list[int]]) -> None:
    attention_masks = [[1] * len(block) for block in blocks]
    table = pa.table(
        {
            "input_ids": pa.array(blocks, type=pa.list_(pa.int64())),
            "attention_mask": pa.array(attention_masks, type=pa.list_(pa.int8())),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa_ipc.new_file(output_path.open("wb"), table.schema) as writer:
        writer.write_table(table)


def update_metadata(
    meta_path: Path,
    config: DatasetConfig,
    summary: dict[str, dict],
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": config.dataset_name,
        "tokenizer": config.tokenizer,
        "block_size": config.block_size,
        "splits": summary,
    }
    if meta_path.suffix in {".yaml", ".yml"}:
        meta_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_split(
    split_config: SplitConfig,
    tokenizer,
    block_size: int,
    base_dir: Path,
) -> dict[str, int]:
    documents = gather_documents(split_config, base_dir=base_dir)
    blocks = tokenize_documents(
        documents,
        tokenizer=tokenizer,
        block_size=block_size,
        max_tokens=split_config.max_tokens,
    )
    write_arrow(Path(split_config.output_path), blocks)
    return {"documents": len(documents), "blocks": len(blocks)}


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

    base_dir = config_path.parent
    split_summaries: dict[str, dict[str, int]] = {}
    for split_name, split_config in tqdm(config.splits.items(), desc="Splits"):
        summary = process_split(
            split_config,
            tokenizer,
            config.block_size,
            base_dir=base_dir,
        )
        split_summaries[split_name] = summary
        print(
            f"[{split_name}] processed {summary['documents']} documents into "
            f"{summary['blocks']} blocks of size {config.block_size}"
        )

    metadata_path = config.metadata_path()
    if metadata_path.suffix not in {".yaml", ".yml"}:
        metadata_path = metadata_path.with_suffix(".yaml")
    update_metadata(metadata_path, config, split_summaries)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
