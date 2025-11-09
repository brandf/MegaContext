from __future__ import annotations

import torch
import torch.nn as nn

from nanochat.gpt import Block, GPTConfig

from .working_context import WorkingContext


class TransformerLensNet(nn.Module):
    """
    Transformer-based focus estimator. Runs a small stack of nanochat blocks over the
    working context and emits a single score per position (tanh-clamped to [-1, 1]).
    Positive scores request expansion, negative scores request collapse.
    """

    def __init__(
        self,
        embed_dim: int,
        max_length: int,
        layers: int = 2,
        num_heads: int = 4,
        head: str = "mlp",
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        config = GPTConfig(
            sequence_len=max_length,
            vocab_size=1,
            n_layer=layers,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=embed_dim,
        )
        self.blocks = nn.ModuleList([Block(config, layer_idx=i) for i in range(layers)])
        self.scorer = self._build_head(embed_dim, head)

    def _build_head(self, embed_dim: int, head: str) -> nn.Module:
        if head == "linear":
            return nn.Linear(embed_dim, 1)
        if head == "mlp":
            hidden = max(embed_dim // 2, 128)
            # We shrink to embed_dim//2 here (instead of ×2 like GistNet) because LensNet operates
            # over long working-context windows every focus step; halving keeps compute/memory in check.
            return nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        raise ValueError(f"Unknown LensNet head: {head}")

    def forward(self, working_context: WorkingContext) -> torch.Tensor:
        x = working_context.to_tensor()  # [B, W, D]

        cos, sin, _ = working_context.get_positional_encodings()
        cos = cos.expand(-1, -1, self.num_heads, -1).to(x.dtype)
        sin = sin.expand(-1, -1, self.num_heads, -1).to(x.dtype)
        cos_sin = (cos, sin)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache=None, alibi=None)
        scores = self.scorer(x).squeeze(-1)
        return torch.tanh(scores)


def build_lensnet(
    kind: str,
    embed_dim: int,
    max_length: int,
    num_heads: int,
    layers: int,
    head: str,
) -> TransformerLensNet:
    if kind.lower() != "transformer":
        raise ValueError(f"Unknown LensNet implementation: {kind}")
    return TransformerLensNet(
        embed_dim=embed_dim,
        max_length=max_length,
        layers=layers,
        num_heads=num_heads,
        head=head,
    )
