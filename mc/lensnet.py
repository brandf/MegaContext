from __future__ import annotations

import math

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
        block_size: int,
        layers: int = 2,
        num_heads: int = 4,
        head: str = "mlp",
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.block_size = block_size
        self._log_block = math.log(max(block_size, 1)) if block_size > 1 else 0.0
        config = GPTConfig(
            sequence_len=max_length,
            vocab_size=1,
            n_layer=layers,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=embed_dim,
        )
        self.blocks = nn.ModuleList([Block(config, layer_idx=i) for i in range(layers)])
        self.level_embed = nn.Embedding(4, embed_dim)
        self.span_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scorer = self._build_head(embed_dim * 2, head)

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
        target_dtype = self.blocks[0].attn.c_q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        cos, sin, _ = working_context.get_positional_encodings()
        cos = cos.to(target_dtype).expand(-1, -1, self.num_heads, -1)
        sin = sin.to(target_dtype).expand(-1, -1, self.num_heads, -1)
        cos_sin = (cos, sin)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache=None, alibi=None)
        levels = working_context.get_lod_tensor().to(x.device)
        positions = working_context.get_positions().to(x.device)
        span_width = torch.exp(levels.float() * self._log_block).unsqueeze(-1)
        cursor_norm = positions.float()
        denom = torch.maximum(cursor_norm.max(), cursor_norm.new_tensor(1.0))
        cursor_norm = cursor_norm / denom
        level_feat = self.level_embed(levels.long())
        span_feat = self.span_mlp(span_width)
        dist_feat = self.dist_mlp(cursor_norm.unsqueeze(-1))
        features = level_feat + span_feat + dist_feat
        cat_feat = torch.cat([x, features], dim=-1)
        scores = self.scorer(cat_feat).squeeze(-1)
        return torch.tanh(scores)


def build_lensnet(
    kind: str,
    embed_dim: int,
    max_length: int,
    block_size: int,
    num_heads: int,
    layers: int,
    head: str,
) -> TransformerLensNet:
    if kind.lower() != "transformer":
        raise ValueError(f"Unknown LensNet implementation: {kind}")
    return TransformerLensNet(
        embed_dim=embed_dim,
        max_length=max_length,
        block_size=block_size,
        layers=layers,
        num_heads=num_heads,
        head=head,
    )
