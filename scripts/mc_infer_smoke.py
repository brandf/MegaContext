"""
Lightweight smoke test for MC inference wiring.
Run with: uv run python -m scripts.mc_infer_smoke
"""
from __future__ import annotations

import torch
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


def main() -> None:
    device = "cpu"
    vocab_size = 64
    embed_dim = 16
    model = DummyModel(vocab_size, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=64,
        block_size=4,
        device=device,
        initial_working_contexts=1,
        max_counterfactuals=1,
        horizon_tokens=0,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    mc = MCController(model, cfg)
    # Begin session with a short prompt and append a few tokens
    session = mc.begin_inference_session(torch.randint(0, vocab_size, (1, 8)))
    mc.inference_step(torch.randint(0, vocab_size, (1, 2)))
    wc = mc.get_inference_working_context()
    assert wc is not None and wc.length > 0
    print(f"MC inference smoke OK (session={session}, wc_len={wc.length})")


if __name__ == "__main__":
    main()

