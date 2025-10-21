"""
Torch module performing 32→1 gist compression on token embeddings.

The implementation keeps data in contiguous tensors to align with on-disk storage and
future MegaContext containers.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .blocks import FeedForward, SelfAttention, SlotAttention


@dataclass
class GistNetConfig:
    hidden_size: int
    block_size: int = 32
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_slots: int = 1
    rope_base: float = 10000.0
    layer_norm_eps: float = 1e-5


class GistNet(nn.Module):
    """
    Compress blocks of token embeddings into a fixed set of gist vectors.

    Input tensors are shaped ``[batch, num_blocks, block_size, hidden_size]`` and the
    output tensor is ``[batch, num_blocks, num_slots, hidden_size]``.
    """

    def __init__(self, config: GistNetConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = config

        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SelfAttention(
            config.hidden_size,
            config.num_heads,
            rope_base=config.rope_base,
        )

        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = FeedForward(
            config.hidden_size,
            mlp_ratio=config.mlp_ratio,
        )

        self.slot_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.slot_attn = SlotAttention(
            config.hidden_size,
            config.num_heads,
            config.num_slots,
            rope_base=config.rope_base,
        )

    def forward(
        self,
        blocks: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if blocks.ndim != 4:
            raise ValueError("blocks must be [batch, num_blocks, block_size, hidden]")
        batch, num_blocks, block_size, hidden = blocks.shape
        if block_size != self.config.block_size:
            raise ValueError(
                f"Expected block_size {self.config.block_size}, got {block_size}"
            )
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden_size {self.config.hidden_size}, got {hidden}"
            )

        # tokens: [batch * num_blocks, block_size, hidden_size]
        tokens = blocks.view(batch * num_blocks, block_size, hidden)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, num_blocks, block_size):
                raise ValueError(
                    "attention_mask must be [batch, num_blocks, block_size]"
                )
            # mask: [batch * num_blocks, block_size]
            mask = attention_mask.view(batch * num_blocks, block_size)

        residual = tokens
        tokens = self.attn_norm(tokens)
        # tokens: [batch * num_blocks, block_size, hidden_size]
        tokens = self.attn(tokens, mask) + residual

        residual = tokens
        tokens = self.mlp(self.mlp_norm(tokens)) + residual

        gist_inputs = self.slot_norm(tokens)
        # slot_attn output: [batch * num_blocks, num_slots, hidden_size]
        gists = self.slot_attn(gist_inputs, mask)
        # reshape back to [batch, num_blocks, num_slots, hidden_size]
        gists = gists.view(batch, num_blocks, self.config.num_slots, hidden)
        return gists

    def compress(
        self,
        blocks: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Alias for ``forward`` to match prospective runtime naming."""

        return self.forward(blocks, attention_mask=attention_mask)
