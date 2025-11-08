from __future__ import annotations

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn


class PositionalEncodingBase(nn.Module):
    def forward(self, positions: torch.Tensor, lod_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SimpleLODRoPE(PositionalEncodingBase):
    """Standard RoPE using global positions; ignores LOD variance."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.inv_freq = 1.0 / (base ** (channel_range / head_dim))

    def forward(self, positions: torch.Tensor, lod_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device)
        theta = torch.einsum("bt,d->btd", positions.to(device).float(), inv_freq)
        cos = torch.cos(theta)[:, :, None, :].bfloat16()
        sin = torch.sin(theta)[:, :, None, :].bfloat16()
        return cos, sin


class GaussianRoPE(PositionalEncodingBase):
    """Gaussian rotary embeddings driven by global positions and LOD."""

    def __init__(self, head_dim: int, block_size: int, base: float = 10000.0) -> None:
        super().__init__()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.inv_freq_template = 1.0 / (base ** (channel_range / head_dim))
        self.block_size = block_size

    def _lod_to_sigma(self, lod_tensor: torch.Tensor) -> torch.Tensor:
        return (self.block_size ** lod_tensor.float()).clamp(min=1.0)

    def forward(self, positions: torch.Tensor, lod_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq_template.to(device)
        positions = positions.to(device).float()
        sigma = self._lod_to_sigma(lod_tensor.to(device))
        theta = torch.einsum("bt,d->btd", positions, inv_freq)
        decay = torch.exp(-0.5 * (sigma.unsqueeze(-1) * inv_freq) ** 2)
        cos = (torch.cos(theta) * decay)[:, :, None, :].bfloat16()
        sin = (torch.sin(theta) * decay)[:, :, None, :].bfloat16()
        return cos, sin


POSITIONAL_REGISTRY: Dict[str, Type[PositionalEncodingBase]] = {
    "simple": SimpleLODRoPE,
    "gaussian": GaussianRoPE,
}


def build_positional(kind: str, head_dim: int, block_size: int) -> PositionalEncodingBase:
    key = kind.lower()
    if key not in POSITIONAL_REGISTRY:
        raise ValueError(f"Unknown positional encoding: {kind}")
    cls = POSITIONAL_REGISTRY[key]
    if key == "gaussian":
        return cls(head_dim=head_dim, block_size=block_size)
    return cls(head_dim=head_dim)
