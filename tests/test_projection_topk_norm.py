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


def test_projection_topk_renormalizes():
    vocab = 8
    embed_dim = 8
    model = DummyModel(vocab, embed_dim)
    # Make the embedding weight an identity to simplify inspection
    with torch.no_grad():
        model.transformer.wte.weight.copy_(torch.eye(vocab))
    cfg = MCConfig(
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
    cfg.loss_projection_top_k = 2
    controller = MCController(model, cfg)
    # Build logits for two positions where top-2 have probs summing to < 1
    # Position 0: p = [0.5, 0.25] + rest 0.25 spread
    # Position 1: p = [0.6, 0.2] + rest 0.2 spread
    probs = torch.zeros(1, 2, vocab)
    probs[0, 0, 0] = 0.5
    probs[0, 0, 1] = 0.25
    probs[0, 0, 2:] = 0.25 / (vocab - 2)
    probs[0, 1, 3] = 0.6
    probs[0, 1, 4] = 0.2
    probs[0, 1, [i for i in range(vocab) if i not in (3, 4)]] = 0.2 / (vocab - 2)
    logits = torch.log(probs)
    pred_embeds = controller._project_logits_to_embeddings(logits)
    # With identity embeddings and renormalized top-2, L1 norm of each row equals 1
    l1 = pred_embeds.abs().sum(dim=-1)
    assert torch.allclose(l1, torch.ones_like(l1), atol=1e-6)

