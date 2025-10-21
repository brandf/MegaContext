from types import SimpleNamespace

import pyarrow.ipc as pa_ipc
import torch
from tools.prepare_dataset import process_split
from torch import nn

from megacontext.data import DatasetConfig, SplitConfig


class DummyTokenizer:
    eos_token_id = 7
    pad_token = None

    def __call__(self, text, *, add_special_tokens=False, return_attention_mask=False):
        tokens = [idx for idx, _ in enumerate(text.split())]
        return {"input_ids": tokens}


class DummyTeacher(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        hidden_size = int(self.bias.shape[0])
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        offsets = torch.arange(hidden_size, dtype=torch.float32, device=base.device)
        hidden = base + offsets
        return SimpleNamespace(hidden_states=[hidden])


def test_process_split_writes_arrow(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample_path = docs_dir / "sample.txt"
    sample_path.write_text("a b c d e f g h", encoding="utf-8")

    split_config = SplitConfig.model_validate(
        {
            "name": "train",
            "source": "docs/*.txt",
            "output_path": str(tmp_path / "out.arrow"),
        }
    )
    config = DatasetConfig.model_validate(
        {
            "dataset_name": "demo",
            "tokenizer": "dummy",
            "block_size": 4,
            "horizon": 8,
            "teacher_batch_size": 2,
            "splits": {
                "train": {
                    "name": "train",
                    "source": "docs/*.txt",
                    "output_path": str(tmp_path / "out.arrow"),
                }
            },
        }
    )

    summary = process_split(
        split_config,
        DummyTokenizer(),
        config,
        base_dir=tmp_path,
        teacher_model=DummyTeacher(hidden_size=3),
    )
    assert summary["examples"] == 1
    assert summary["blocks"] == 2
    assert summary["teacher_hidden_size"] == 3

    with pa_ipc.open_file((tmp_path / "out.arrow").open("rb")) as reader:
        table = reader.read_all()
    assert table.num_rows == 1

    teacher_hidden = table.column("teacher_hidden")[0].as_py()
    assert len(teacher_hidden) == 4
    assert len(teacher_hidden[0]) == 3

    gist_target = table.column("gist_target")[0].as_py()
    assert len(gist_target) == 3

    context_ids = table.column("context_input_ids")[0].as_py()
    assert len(context_ids) == 8
