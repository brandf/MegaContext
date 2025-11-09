from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Block, GPTConfig, apply_rotary_emb


class GistNetBase(nn.Module):
    block_size: int

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanLinearHead(nn.Module):
    """Mean pool followed by a single projection back to embed_dim."""

    def __init__(self, embed_dim: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        assert blocks.shape[1] == self.block_size
        pooled = blocks.mean(dim=1)
        return self.proj(pooled)


class MeanMLPHead(nn.Module):
    """Mean pool followed by a small MLP revisiting the embedding space."""

    def __init__(
        self,
        embed_dim: int,
        block_size: int,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        dims = list(hidden_dims or [embed_dim * 2, embed_dim])
        assert dims[-1] == embed_dim
        layers = []
        cur = embed_dim
        for i, h in enumerate(dims):
            layers.append(nn.Linear(cur, h))
            if i != len(dims) - 1:
                layers.append(nn.ReLU())
            cur = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        assert blocks.shape[1] == self.block_size
        pooled = blocks.mean(dim=1)
        return self.mlp(pooled)


class QueryPoolingHead(nn.Module):
    """Learned query attends to the sequence once, then projects."""

    def __init__(self, embed_dim: int, proj: str = "linear") -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        if proj == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        elif proj == "linear":
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown query pooling projection: {proj}")

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        query = self.query.to(blocks.device)
        weights = torch.softmax(blocks @ query, dim=1).unsqueeze(-1)  # [B, T, 1]
        summary = torch.sum(blocks * weights, dim=1)
        return self.proj(summary)


class CLSHead(nn.Module):
    """Read the first token (CLS) and optionally project."""

    def __init__(self, embed_dim: int, proj: str = "linear") -> None:
        super().__init__()
        if proj == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        elif proj == "linear":
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown CLS projection: {proj}")

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        cls = blocks[:, 0, :]
        return self.proj(cls)


def build_head(
    head: str,
    embed_dim: int,
    block_size: int,
    head_hidden_dims: Optional[Sequence[int]] = None,
) -> Tuple[str, nn.Module]:
    """
    Returns a tuple (summary_mode, module).
    summary_mode ∈ {"mean", "cls"} informs callers whether a CLS token is required.
    """
    name = head.lower()
    if name == "mean_linear":
        return "mean", MeanLinearHead(embed_dim, block_size)
    if name == "mean_mlp":
        return "mean", MeanMLPHead(embed_dim, block_size, head_hidden_dims)
    if name == "query_linear":
        return "mean", QueryPoolingHead(embed_dim, proj="linear")
    if name == "query_mlp":
        return "mean", QueryPoolingHead(embed_dim, proj="mlp")
    if name == "cls_linear":
        return "cls", CLSHead(embed_dim, proj="linear")
    if name == "cls_mlp":
        return "cls", CLSHead(embed_dim, proj="mlp")
    raise ValueError(f"Unknown head type: {head}")


class PoolingOnlyGistNet(GistNetBase):
    """Simple baseline that applies the configured head without attention."""

    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        head: str = "mean_linear",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        summary_mode, head_module = build_head(
            head=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )
        if summary_mode != "mean":
            raise ValueError("PoolingOnlyGistNet supports only mean-based heads")
        self.head = head_module

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        return self.head(blocks)


def build_rotary_cos_sin(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
    device = device or torch.device("cpu")
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().bfloat16().unsqueeze(0).unsqueeze(2)
    sin = freqs.sin().bfloat16().unsqueeze(0).unsqueeze(2)
    return cos, sin


class TransformerGistNet(GistNetBase):
    """Mini nanochat transformer blocks followed by a configurable head."""

    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        layers: int = 2,
        num_heads: int = 8,
        head: str = "mean_mlp",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.block_size = block_size
        config = GPTConfig(
            sequence_len=block_size + 1,  # accommodate optional CLS token
            vocab_size=1,
            n_layer=layers,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=embed_dim,
        )
        self.blocks = nn.ModuleList([Block(config, layer_idx=i) for i in range(layers)])
        self.head_dim = embed_dim // num_heads
        self.rope_base = 10000.0
        self.summary_mode, head_module = build_head(
            head=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )
        if self.summary_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)
        self.head = head_module

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        x = blocks
        if self.summary_mode == "cls":
            cls = self.cls_token.to(x.device).expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        cos, sin = build_rotary_cos_sin(
            seq_len=x.size(1),
            head_dim=self.head_dim,
            base=self.rope_base,
            device=x.device,
        )
        cos_sin = (cos, sin)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache=None, alibi=None)
        return self.head(x)


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


def _transformer_factory(layers: int, head: str) -> GistnetFactory:
    return partial(TransformerGistNet, layers=layers, head=head)


GISTNET_REGISTRY: Dict[str, GistnetFactory] = {
    # Depth-2 transformer variants (default tier)
    "transformer2_mean_mlp": _transformer_factory(2, "mean_mlp"),
    "transformer2_query_mlp": _transformer_factory(2, "query_mlp"),
    "transformer2_cls_mlp": _transformer_factory(2, "cls_mlp"),
    "transformer2_mean_linear": _transformer_factory(2, "mean_linear"),
    "transformer2_query_linear": _transformer_factory(2, "query_linear"),
    "transformer2_cls_linear": _transformer_factory(2, "cls_linear"),
    # Depth-4 transformer variants (heavier tier)
    "transformer4_mean_mlp": _transformer_factory(4, "mean_mlp"),
    "transformer4_query_mlp": _transformer_factory(4, "query_mlp"),
    "transformer4_cls_mlp": _transformer_factory(4, "cls_mlp"),
    "transformer4_mean_linear": _transformer_factory(4, "mean_linear"),
    "transformer4_query_linear": _transformer_factory(4, "query_linear"),
    "transformer4_cls_linear": _transformer_factory(4, "cls_linear"),
    # Baseline
    "mean_linear": _pooling_only_factory("mean_linear"),
}


def build_gistnet(kind: str, embed_dim: int, **kwargs) -> GistNetBase:
    key = kind.lower()
    if key not in GISTNET_REGISTRY:
        raise ValueError(f"Unknown GistNet implementation: {kind}")
    constructor = GISTNET_REGISTRY[key]
    return constructor(embed_dim=embed_dim, **kwargs)
