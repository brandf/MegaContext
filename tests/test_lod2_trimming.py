import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn
import torch

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


def _build_controller(block_size: int, long_multiplier: int) -> MCController:
    vocab = 32
    embed_dim = 8
    model = DummyModel(vocab, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=block_size * long_multiplier,
        block_size=block_size,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        horizon_tokens=0,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    cfg.long_horizon_multiplier = long_multiplier
    return MCController(model, cfg)


def test_lod2_loss_present_when_enough_blocks():
    block = 2
    multiplier = 2
    controller = _build_controller(block, multiplier)
    vocab = controller.embed.num_embeddings
    horizon_len = block * multiplier
    tokens = torch.randint(0, vocab, (1, horizon_len))
    logits = torch.randn(1, horizon_len, vocab)
    lod1, lod2 = controller._compute_lod_losses(tokens, logits, use_lod2=True)
    assert lod1 is not None
    assert lod2 is not None


def test_lod2_loss_none_when_insufficient_blocks():
    block = 2
    multiplier = 2
    controller = _build_controller(block, multiplier)
    vocab = controller.embed.num_embeddings
    horizon_len = block * multiplier - 1
    tokens = torch.randint(0, vocab, (1, horizon_len))
    logits = torch.randn(1, horizon_len, vocab)
    lod1, lod2 = controller._compute_lod_losses(tokens, logits, use_lod2=True)
    assert lod1 is not None
    assert lod2 is None
