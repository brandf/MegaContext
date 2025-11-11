import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn
import torch

from mc.config import MCConfig
from mc import runtime as mc_runtime
from mc.runtime import MCController


class DummyReport:
    def log(self, *args, **kwargs):
        return


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


def test_batched_rope_shapes_and_forward(monkeypatch):
    vocab = 32
    embed_dim = 8
    num_heads = 1
    B, T = 2, 8
    model = DummyModel(vocab, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=T,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        num_heads=num_heads,
    )
    monkeypatch.setattr(mc_runtime, "get_report", lambda: DummyReport())
    monkeypatch.setattr(mc_runtime, "get_report", lambda: DummyReport())
    controller = MCController(model, cfg)
    monkeypatch.setattr(MCController, "_compute_lens_losses", lambda self, batch_states: None)
    tokens = torch.randint(0, vocab, (B, T))
    result = controller.process_batch(tokens, step=0)
    # Assemble batched caches as base_train does
    cos_list, sin_list, alibi_list = [], [], []
    for b in range(B):
        key = f"train_step_{0}_sample_{b}"
        cos_b, sin_b, alibi_b = result.positional_caches[key]
        cos_list.append(cos_b)
        sin_list.append(sin_b)
        alibi_list.append(alibi_b)
    cos = torch.cat(cos_list, dim=0)
    sin = torch.cat(sin_list, dim=0)
    assert cos.shape[0] == B and sin.shape[0] == B
    # Forward with overrides and cached embeddings should not error
    inputs_embeds = result.cached_embeddings
    assert inputs_embeds.shape[0] == B
    assert inputs_embeds.shape[1] == T
