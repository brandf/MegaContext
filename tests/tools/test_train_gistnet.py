from pathlib import Path

import pyarrow as pa
import torch
from pyarrow import ipc as pa_ipc
from tools.train_gistnet import (
    ContextArrowDataset,
    build_dataloader,
    load_train_config,
    train_step,
)

from megacontext.gistnet import GistNet, GistNetConfig


def test_load_train_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  hidden_size: 64\ntraining:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    config = load_train_config(config_path)
    assert config["model"]["hidden_size"] == 64
    assert config["training"]["batch_size"] == 2


def test_context_arrow_dataset_round_trip(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.arrow"
    table = pa.table(
        {
            "context_input_ids": pa.array(
                [[1, 2, 3, 4, 5, 6, 7, 8]], type=pa.list_(pa.int64())
            ),
            "context_attention_mask": pa.array([[1] * 8], type=pa.list_(pa.int8())),
            "future_input_ids": pa.array([[9, 10, 11, 12]], type=pa.list_(pa.int64())),
            "future_attention_mask": pa.array([[1] * 4], type=pa.list_(pa.int8())),
            "teacher_context_hidden": pa.array(
                [
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [0.7, 0.8],
                        [0.9, 1.0],
                        [1.1, 1.2],
                        [1.3, 1.4],
                        [1.5, 1.6],
                    ]
                ],
                type=pa.list_(pa.list_(pa.float32())),
            ),
            "teacher_future_hidden": pa.array(
                [
                    [
                        [1.7, 1.8],
                        [1.9, 2.0],
                        [2.1, 2.2],
                        [2.3, 2.4],
                    ]
                ],
                type=pa.list_(pa.list_(pa.float32())),
            ),
        }
    )
    with shard_path.open("wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    dataset = ContextArrowDataset(shard_path)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["context_tokens"].shape == (8,)
    assert sample["future_tokens"].shape == (4,)
    assert sample["context_hidden"].shape == (8, 2)
    assert sample["future_hidden"].shape == (4, 2)

    loader = build_dataloader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["context_tokens"].shape == (1, 8)
    assert batch["context_hidden"].shape == (1, 8, 2)


def test_train_step_pooling_mse() -> None:
    config = GistNetConfig(hidden_size=4, block_size=4, num_heads=1, mlp_ratio=1.0)
    model = GistNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "context_tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "future_tokens": torch.tensor([[5, 6]], dtype=torch.long),
        "context_hidden": torch.randn(1, 4, 4, dtype=torch.float32),
        "future_hidden": torch.randn(1, 2, 4, dtype=torch.float32),
    }
    metrics = train_step(
        model,
        batch,
        optimizer,
        device=torch.device("cpu"),
        window_tokens=4,
        objective="pooling_mse",
    )
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_train_step_delta_requires_base_model() -> None:
    config = GistNetConfig(hidden_size=4, block_size=4, num_heads=1, mlp_ratio=1.0)
    model = GistNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "context_tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "future_tokens": torch.tensor([[5, 6]], dtype=torch.long),
        "context_hidden": torch.randn(1, 4, 4, dtype=torch.float32),
        "future_hidden": torch.randn(1, 2, 4, dtype=torch.float32),
    }
    try:
        train_step(
            model,
            batch,
            optimizer,
            device=torch.device("cpu"),
            window_tokens=4,
            objective="delta_nll",
        )
    except ValueError as exc:
        assert "base model" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("delta_nll objective should require a base model")
