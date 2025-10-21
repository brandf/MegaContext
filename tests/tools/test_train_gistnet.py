from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from tools.train_gistnet import (
    GistArrowDataset,
    build_dataloader,
    load_train_config,
)


def test_load_train_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  hidden_size: 64\ntraining:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    config = load_train_config(config_path)
    assert config["model"]["hidden_size"] == 64
    assert config["training"]["batch_size"] == 2


def test_gist_arrow_dataset_round_trip(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.arrow"
    table = pa.table(
        {
            "input_ids": pa.array([[1, 2, 3, 4]], type=pa.list_(pa.int64())),
            "attention_mask": pa.array([[1, 1, 1, 1]], type=pa.list_(pa.int8())),
            "context_input_ids": pa.array(
                [[1, 2, 3, 4, 5, 6, 7, 8]], type=pa.list_(pa.int64())
            ),
            "context_attention_mask": pa.array([[1] * 8], type=pa.list_(pa.int8())),
            "teacher_hidden": pa.array(
                [
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [0.7, 0.8],
                    ]
                ],
                type=pa.list_(pa.list_(pa.float32())),
            ),
            "gist_target": pa.array([[0.2, 0.4]], type=pa.list_(pa.float32())),
        }
    )
    with shard_path.open("wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    dataset = GistArrowDataset(shard_path)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["tokens"].shape == (4,)
    assert sample["teacher_hidden"].shape == (4, 2)
    assert torch.allclose(sample["gist_target"], torch.tensor([0.2, 0.4]))

    loader = build_dataloader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["tokens"].shape == (1, 4)
    assert batch["teacher_hidden"].shape == (1, 4, 2)
