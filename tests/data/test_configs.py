from pathlib import Path

import pytest
import yaml

from megacontext.data import DatasetConfig


def test_dataset_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "sample",
                "tokenizer": "gpt2",
                "block_size": 32,
                "splits": {
                    "train": {
                        "name": "train",
                        "source": "data/train.txt",
                        "output_path": "data/train.arrow",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    model = DatasetConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    assert model.block_size == 32
    assert "train" in model.splits


def test_output_path_requires_arrow(tmp_path: Path) -> None:
    payload = {
        "dataset_name": "sample",
        "tokenizer": "gpt2",
        "block_size": 32,
        "splits": {
            "train": {
                "name": "train",
                "source": "data.txt",
                "output_path": "data/train.bin",
            }
        },
    }
    with pytest.raises(ValueError):
        DatasetConfig.model_validate(payload)
