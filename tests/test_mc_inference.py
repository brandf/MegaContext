import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn

from mc.config import MCConfig
from mc.runtime import MCController


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.transformer = DummyTransformer(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", cos_sin_override=None, alibi_override=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.transformer.wte(idx)
        return self.lm_head(x)


def test_inference_facade_smoke():
    vocab_size = 32
    embed_dim = 8
    model = DummyModel(vocab_size, embed_dim)
    config = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=16,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        horizon_tokens=0,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    controller = MCController(model, config)
    session = controller.begin_inference_session(torch.randint(0, vocab_size, (1, 6)))
    assert isinstance(session, str)
    wc0 = controller.get_inference_working_context()
    assert wc0 is not None and wc0.length > 0
    controller.inference_step(torch.randint(0, vocab_size, (1, 2)))
    wc1 = controller.get_inference_working_context()
    assert wc1 is not None and wc1.length >= wc0.length


def test_mc_controller_validates_head_divisibility():
    vocab_size = 16
    embed_dim = 10
    model = DummyModel(vocab_size, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=16,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        horizon_tokens=0,
        allocator_recent_tokens=0,
        num_heads=3,
    )
    with pytest.raises(ValueError, match="embed_dim must be divisible"):
        MCController(model, cfg)
