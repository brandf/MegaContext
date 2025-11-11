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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", cos_sin_override=None, alibi_override=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.transformer.wte(idx)
        return self.lm_head(x)


def test_lod0_cache_policy_disables_cache_on_init():
    vocab = 32
    embed_dim = 8
    model = DummyModel(vocab, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=16,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    cfg.cache_lod0 = False
    controller = MCController(model, cfg)
    controller.begin_inference_session(torch.randint(0, vocab, (1, 6)))
    assert controller.inference_state is not None
    tree = controller.inference_state.tree
    assert tree._lod0_cache is None
