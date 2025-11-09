from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn


class GistNetBase(nn.Module):
    block_size: int

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanPooledGistNet(GistNetBase):
    """Mean pool followed by an MLP over pooled embeddings."""

    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        hidden_dims = list(hidden_dims or [embed_dim])
        assert hidden_dims[-1] == embed_dim, "Last hidden dim must equal embed_dim"
        layers = []
        in_dim = embed_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if i != len(hidden_dims) - 1:
                layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blocks: [B, block_size, D] tensor of embeddings.
        Returns:
            [B, D] pooled gist vector per block.
        """
        assert blocks.shape[1] == self.block_size
        pooled = blocks.mean(dim=1)
        return self.mlp(pooled)


class MeanPoolHead(nn.Module):
    """Head that simply mean-pools the block output."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """blocks: [B, block_size, D] -> [B, D] via mean over block axis."""
        assert blocks.shape[1] == self.block_size
        return blocks.mean(dim=1)


class MeanLinearHead(nn.Module):
    """Mean pool followed by a single projection back to embed_dim."""

    def __init__(self, embed_dim: int, block_size: int) -> None:
        super().__init__()
        self.pool = MeanPoolHead(block_size)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """blocks: [B, block_size, D] -> [B, D] mean+linear projection."""
        pooled = self.pool(blocks)
        return self.proj(pooled)


class PoolingOnlyGistNet(GistNetBase):
    """
    Baseline compressor that applies the selected pooling head directly
    without intermediate attention.
    """

    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        head: str = "mlp2",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.head = build_pooling_head(
            head=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """Directly apply the configured head to [B, block_size, D] blocks."""
        return self.head(blocks)


def build_pooling_head(
    head: str,
    embed_dim: int,
    block_size: int,
    head_hidden_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    name = head.lower()
    if name in ("mlp2", "default"):
        hidden = list(head_hidden_dims) if head_hidden_dims else [
            embed_dim * 2,
            embed_dim,
        ]
        return MeanPooledGistNet(embed_dim, block_size, hidden_dims=hidden)
    if name == "mlp":
        hidden = list(head_hidden_dims) if head_hidden_dims else [
            embed_dim,
            embed_dim,
        ]
        return MeanPooledGistNet(embed_dim, block_size, hidden_dims=hidden)
    if name in ("linear", "proj"):
        return MeanLinearHead(embed_dim, block_size)
    if name in ("mean", "pool"):
        return MeanPoolHead(block_size)
    raise ValueError(f"Unknown gist head: {head}")


class ResidualSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim or embed_dim * 2),
            nn.ReLU(),
            nn.Linear(ff_dim or embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D] with residual attention + FFN."""
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ff(x)
        return x


class SelfAttentionGistNet(GistNetBase):
    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        layers: int = 2,
        head: str = "mlp2",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.layers = nn.ModuleList(
            [ResidualSelfAttention(embed_dim) for _ in range(layers)]
        )
        self.head = build_pooling_head(
            head=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blocks: [B, block_size, D] token embeddings.
        Returns:
            [B, D] gist vector after intra-block attention + head.
        """
        x = blocks
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class SlotAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_slots: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, embed_dim))
        self.cross = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> updated slots [B, num_slots, D]."""
        B = x.size(0)
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)
        updated, _ = self.cross(slots, x, x)
        return self.norm(slots + updated)


class SlotAttentionGistNet(GistNetBase):
    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        num_slots: int = 4,
        head: str = "mlp2",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.self_attn = ResidualSelfAttention(embed_dim)
        self.slot_block = SlotAttentionBlock(embed_dim, num_slots=num_slots)
        self.cross_back = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.gist_slot = SlotAttentionBlock(embed_dim, num_slots=1)
        self.final = build_pooling_head(
            head=head,
            embed_dim=embed_dim,
            block_size=1,
            head_hidden_dims=head_hidden_dims,
        )

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blocks: [B, block_size, D] token embeddings.
        Returns:
            [B, D] gist distilled through slot attention + pooling head.
        """
        x = self.self_attn(blocks)
        slots = self.slot_block(x)
        back, _ = self.cross_back(x, slots, slots)
        back = self.self_attn(back)
        gist_slot = self.gist_slot(back)
        return self.final(gist_slot)


GistnetFactory = Callable[..., GistNetBase]


def _pooling_only_factory(head: str) -> GistnetFactory:
    def factory(
        *,
        embed_dim: int,
        block_size: int = 32,
        head_hidden_dims: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> GistNetBase:
        return PoolingOnlyGistNet(
            embed_dim=embed_dim,
            block_size=block_size,
            head=head,
            head_hidden_dims=head_hidden_dims,
        )

    return factory


GISTNET_REGISTRY: Dict[str, GistnetFactory] = {
    # Mean-only compressors (no attention)
    "mean": _pooling_only_factory("mean"),
    "mean_linear": _pooling_only_factory("linear"),
    "mean_mlp": _pooling_only_factory("mlp"),
    "mean_mlp2": _pooling_only_factory("mlp2"),
    # Self-attention compressors (increasing head complexity)
    "selfattention_mean": partial(SelfAttentionGistNet, head="mean"),
    "selfattention_linear": partial(SelfAttentionGistNet, head="linear"),
    "selfattention_mlp": partial(SelfAttentionGistNet, head="mlp"),
    "selfattention_mlp2": partial(SelfAttentionGistNet, layers=2, head="mlp2"),
    # Slot-attention compressors (matching ordering)
    "slotattention_mean": partial(SlotAttentionGistNet, head="mean"),
    "slotattention_linear": partial(SlotAttentionGistNet, head="linear"),
    "slotattention_mlp": partial(SlotAttentionGistNet, head="mlp"),
    "slotattention_mlp2": partial(SlotAttentionGistNet, head="mlp2"),
}


def build_gistnet(kind: str, embed_dim: int, **kwargs) -> GistNetBase:
    key = kind.lower()
    if key not in GISTNET_REGISTRY:
        raise ValueError(f"Unknown GistNet implementation: {kind}")
    constructor = GISTNET_REGISTRY[key]
    return constructor(embed_dim=embed_dim, **kwargs)
