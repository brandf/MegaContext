from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

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
    pooling: str,
    head_type: str,
    embed_dim: int,
    block_size: int,
    head_hidden_dims: Optional[Sequence[int]] = None,
) -> Tuple[str, nn.Module]:
    """Returns pooling summary mode and the scoring module."""
    pooling = pooling.lower()
    head_type = head_type.lower()
    if pooling == "mean":
        if head_type == "linear":
            return "mean", MeanLinearHead(embed_dim, block_size)
        if head_type == "mlp":
            return "mean", MeanMLPHead(embed_dim, block_size, head_hidden_dims)
        raise ValueError(f"Unknown head type for mean pooling: {head_type}")
    if pooling == "query":
        return "mean", QueryPoolingHead(embed_dim, proj=head_type)
    if pooling == "cls":
        return "cls", CLSHead(embed_dim, proj=head_type)
    raise ValueError(f"Unknown pooling mode: {pooling}")


class PoolingOnlyGistNet(GistNetBase):
    """Simple baseline that applies the configured head without attention."""

    def __init__(
        self,
        embed_dim: int,
        block_size: int = 32,
        head: str = "linear",
        head_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        summary_mode, head_module = build_head(
            pooling="mean",
            head_type=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )
        if summary_mode != "mean":
            raise ValueError("PoolingOnlyGistNet supports only mean-based heads")
        self.head = head_module

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        target_dtype = next(self.head.parameters(), None)
        if target_dtype is not None:
            dtype = target_dtype.dtype
            blocks = blocks.to(dtype)
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
        pooling: str = "mean",
        head: str = "mlp",
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
            pooling=pooling,
            head_type=head,
            embed_dim=embed_dim,
            block_size=block_size,
            head_hidden_dims=head_hidden_dims,
        )
        if self.summary_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)
        self.head = head_module

    def forward(self, blocks: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with optional key padding mask.

        Args:
            blocks: [B, T, D] block sequences
            key_padding_mask: [B, T] boolean mask where True marks valid tokens
        """
        target_dtype = self.blocks[0].attn.c_q.weight.dtype
        x = blocks.to(target_dtype)
        mask_bool = None
        if key_padding_mask is not None:
            mask_bool = key_padding_mask.to(torch.bool)
        # If CLS is used, prepend it and extend mask accordingly
        if self.summary_mode == "cls":
            cls = self.cls_token.to(x.device).expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if mask_bool is not None:
                pad_true = torch.ones((mask_bool.size(0), 1), dtype=torch.bool, device=mask_bool.device)
                mask_bool = torch.cat([pad_true, mask_bool], dim=1)
        # Zero out padded tokens before attention
        if mask_bool is not None:
            mask_expanded = mask_bool.unsqueeze(-1).to(x.device)
            x = x.masked_fill(~mask_expanded, 0.0)
        cos, sin = build_rotary_cos_sin(
            seq_len=x.size(1),
            head_dim=self.head_dim,
            base=self.rope_base,
            device=x.device,
        )
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)
        cos_sin = (cos, sin)
        # Build attention bias from key_padding_mask if provided: mask padded keys across all queries
        alibi_bias = None
        if mask_bool is not None:
            # True = keep, False = mask; convert to bias with large negative penalty
            B, Tk = mask_bool.shape
            Tq = x.size(1)
            penalty = -1e4  # use finite value to avoid NaNs in bfloat16
            bias = (~mask_bool).to(torch.float32) * penalty  # [B, Tk]
            bias = bias.view(B, 1, 1, Tk).expand(B, 1, Tq, Tk)  # [B, 1, Tq, Tk]
            alibi_bias = bias.to(torch.bfloat16)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache=None, alibi=alibi_bias)
        return self.head(x)


def build_gistnet(
    mode: str,
    embed_dim: int,
    block_size: int,
    layers: int,
    pooling: str,
    head: str,
    num_heads: int,
) -> GistNetBase:
    mode = mode.lower()
    if mode == "mean":
        return PoolingOnlyGistNet(
            embed_dim=embed_dim,
            block_size=block_size,
            head=head,
        )
    if mode == "transformer":
        return TransformerGistNet(
            embed_dim=embed_dim,
            block_size=block_size,
            layers=layers,
            num_heads=num_heads,
            pooling=pooling,
            head=head,
        )
    raise ValueError(f"Unknown GistNet mode: {mode}")
