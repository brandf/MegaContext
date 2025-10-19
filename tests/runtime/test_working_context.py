import torch
from torch import nn

from megacontext.runtime import WorkingContext


def test_working_context_basic_properties() -> None:
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    context = WorkingContext(token_ids=token_ids, attention_mask=attention_mask)

    entries = list(context.entries())
    assert len(entries) == token_ids.shape[-1]
    assert entries[0].level == "L0"

    tensors = context.to_tensors()
    assert torch.equal(tensors["input_ids"], token_ids)
    assert torch.equal(tensors["attention_mask"], attention_mask)


def test_materialize_embeddings() -> None:
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    context = WorkingContext(token_ids=token_ids, attention_mask=attention_mask)

    embedding = nn.Embedding(3, 4)
    embeds = context.materialize_embeddings(embedding)
    assert embeds.shape == (1, 3, 4)

    tensors = context.to_tensors(embedding)
    assert "inputs_embeds" in tensors
    assert tensors["inputs_embeds"].shape == (1, 3, 4)
