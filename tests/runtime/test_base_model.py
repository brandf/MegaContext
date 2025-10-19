from types import SimpleNamespace

import torch
from torch import nn

from megacontext.runtime import BaseModel


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0


class DummyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 16)
        self.proj = nn.Linear(16, 8)

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embed(input_ids)
        logits = self.proj(inputs_embeds)
        return SimpleNamespace(logits=logits)


def test_forward_with_input_ids() -> None:
    model = DummyLM()
    tokenizer = DummyTokenizer()
    base = BaseModel(model=model, tokenizer=tokenizer)  # type: ignore[arg-type]
    inputs = torch.ones((1, 4), dtype=torch.long)
    outputs = base.forward(input_ids=inputs)
    assert outputs.logits.shape == (1, 4, 8)


def test_forward_with_inputs_embeds() -> None:
    model = DummyLM()
    tokenizer = DummyTokenizer()
    base = BaseModel(model=model, tokenizer=tokenizer)  # type: ignore[arg-type]
    embeds = torch.zeros((1, 4, 16), dtype=torch.float32)
    outputs = base.forward(inputs_embeds=embeds)
    assert outputs.logits.shape == (1, 4, 8)
