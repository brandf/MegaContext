"""
Building blocks for the GistNet compression model.

These utilities stay close to raw tensor representations so the higher-level model
can wrap them without introducing heavy Python object graphs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _split_heads(tensor: Tensor, num_heads: int) -> Tensor:
    batch, seq_len, hidden_size = tensor.shape
    head_dim = hidden_size // num_heads
    tensor = tensor.view(batch, seq_len, num_heads, head_dim)
    return tensor.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]


def _merge_heads(tensor: Tensor) -> Tensor:
    batch, num_heads, seq_len, head_dim = tensor.shape
    tensor = tensor.permute(
        0, 2, 1, 3
    ).contiguous()  # [batch, seq_len, heads, head_dim]
    return tensor.view(batch, seq_len, num_heads * head_dim)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding helper mirroring the structure used in modern LLMs.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(
            dtype if dtype is not None else torch.float32
        )  # [seq_len, dim]
        sin = emb.sin().to(
            dtype if dtype is not None else torch.float32
        )  # [seq_len, dim]
        return cos, sin


class SelfAttention(nn.Module):
    """
    Self-attention block with Rotary positional embeddings applied to queries/keys.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        q = _split_heads(
            self.q_proj(x), self.num_heads
        )  # [batch, heads, seq_len, head_dim]
        k = _split_heads(self.k_proj(x), self.num_heads)
        v = _split_heads(self.v_proj(x), self.num_heads)

        cos, sin = self.rotary(
            seq_len,
            device=x.device,
            dtype=x.dtype,
        )
        cos = cos.view(
            1, 1, seq_len, self.head_dim
        )  # broadcast to [1,1,seq_len,head_dim]
        sin = sin.view(1, 1, seq_len, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # [batch, heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.view(batch, 1, 1, seq_len)  # align with attn_scores
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(x.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)  # [batch, heads, seq_len, head_dim]
        context = _merge_heads(context)  # [batch, seq_len, hidden_size]
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Simple MLP block with GELU activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SlotAttention(nn.Module):
    """
    Shared query slots attending over a block of token features to produce gists.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_slots: int = 1,
        *,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_slots = num_slots

        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_size))

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len, _ = tokens.shape

        slots = self.slot_queries.unsqueeze(0).expand(
            batch, -1, -1
        )  # [batch, num_slots, hidden]
        q = _split_heads(
            self.q_proj(slots), self.num_heads
        )  # [batch, heads, num_slots, head_dim]
        k = _split_heads(
            self.k_proj(tokens), self.num_heads
        )  # [batch, heads, seq_len, head_dim]
        v = _split_heads(self.v_proj(tokens), self.num_heads)

        cos, sin = self.rotary(
            seq_len,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        cos = cos.view(1, 1, seq_len, self.head_dim)  # broadcast over slots
        sin = sin.view(1, 1, seq_len, self.head_dim)
        k = (k * cos) + (_rotate_half(k) * sin)

        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # [batch, heads, num_slots, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.view(batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(
                mask == 0, torch.finfo(tokens.dtype).min
            )

        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)  # [batch, heads, num_slots, head_dim]
        context = _merge_heads(context)  # [batch, num_slots, hidden_size]
        return self.out_proj(context)
