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

    def forward(self, idx, **kwargs):
        return self.lm_head(self.transformer.wte(idx))


def test_mc_controller_rejects_lod2d_positional():
    model = DummyModel(32, 8)
    cfg = MCConfig(
        embed_dim=8,
        max_seq_len=16,
        block_size=2,
        device="cpu",
        positional_type="gaussian_lod2d",
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    with pytest.raises(ValueError, match="LOD-2D positional modes"):
        MCController(model, cfg)
