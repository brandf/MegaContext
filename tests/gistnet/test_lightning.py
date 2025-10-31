from pathlib import Path

import pyarrow as pa
import torch
from pyarrow import ipc as pa_ipc
from torch.utils.data import DataLoader

from megacontext.gistnet import (
    BaseModelSettings,
    ContextArrowDataset,
    GistNetConfig,
    GistNetLightningModule,
    GistNetTrainingConfig,
    GistNetTrainingPhase,
    build_gistnet_experiment,
)


def _write_sample_arrow(path: Path) -> None:
    table = pa.table(
        {
            "context_input_ids": pa.array([[1, 2, 3, 4]], type=pa.list_(pa.int64())),
            "context_attention_mask": pa.array(
                [[1, 1, 1, 1]], type=pa.list_(pa.int8())
            ),
            "future_input_ids": pa.array([[5, 6]], type=pa.list_(pa.int64())),
            "future_attention_mask": pa.array([[1, 1]], type=pa.list_(pa.int8())),
            "teacher_context_hidden": pa.array(
                [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]],
                type=pa.list_(pa.list_(pa.float32())),
            ),
            "teacher_future_hidden": pa.array(
                [[[0.9, 1.0], [1.1, 1.2]]],
                type=pa.list_(pa.list_(pa.float32())),
            ),
        }
    )
    with path.open("wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def test_context_arrow_dataset_round_trip(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.arrow"
    _write_sample_arrow(shard_path)

    dataset = ContextArrowDataset(shard_path)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["context_tokens"].shape == (4,)
    assert sample["future_tokens"].shape == (2,)
    assert sample["context_hidden"].shape == (4, 2)
    assert sample["future_hidden"].shape == (2, 2)

    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["context_tokens"].shape == (1, 4)
    assert batch["context_hidden"].shape == (1, 4, 2)


def test_gistnet_lightning_pooling_step() -> None:
    config = GistNetConfig(hidden_size=4, block_size=4, num_heads=1, mlp_ratio=1.0)
    phase = GistNetTrainingPhase(
        objective="pooling_mse",
        max_steps=1,
        window_tokens=4,
        lr=1e-3,
        name="phase-1",
    )
    module = GistNetLightningModule(config, [phase])
    module.train()

    batch = {
        "context_tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "future_tokens": torch.tensor([[5, 6]], dtype=torch.long),
        "context_hidden": torch.randn(1, 4, 4, dtype=torch.float32),
        "future_hidden": torch.randn(1, 2, 4, dtype=torch.float32),
    }
    loss = module.training_step(batch, 0)
    assert torch.is_tensor(loss)
    assert loss.item() >= 0.0


def test_build_gistnet_experiment(tmp_path: Path) -> None:
    shard_path = tmp_path / "train.arrow"
    _write_sample_arrow(shard_path)

    config = GistNetConfig(hidden_size=2, block_size=2, num_heads=1, mlp_ratio=1.0)
    phase = GistNetTrainingPhase(
        objective="pooling_mse",
        max_steps=5,
        window_tokens=2,
        lr=1e-3,
        name="phase-1",
    )
    trainer, module, data_module = build_gistnet_experiment(
        dataset_path=shard_path,
        model_config=config,
        training=GistNetTrainingConfig(
            batch_size=1,
            phases=(phase,),
        ),
    )

    assert module.total_steps == 5
    assert trainer.max_steps == 5
    data_module.setup()
    assert len(data_module.train_dataset) == 1


def test_training_config_from_dict_merges_fields() -> None:
    cfg = {
        "batch_size": 3,
        "seed": 7,
        "phases": [
            {
                "name": "phase-1",
                "objective": "pooling_mse",
                "max_steps": 10,
                "window_tokens": 4,
                "lr": 1e-3,
            }
        ],
        "base_model": {
            "model_name": "sshleifer/tiny-gpt2",
            "torch_dtype": "float32",
            "device": "cpu",
            "run_name": "demo",
        },
    }
    training = GistNetTrainingConfig.from_dict(cfg)
    assert training.batch_size == 3
    assert training.seed == 7
    assert len(training.phases) == 1
    assert isinstance(training.base_model, BaseModelSettings)
    assert training.base_model.name == "sshleifer/tiny-gpt2"
    assert training.base_model.torch_dtype == "float32"
