from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GaussianRoPEConfig:
    use_sigma_decay: bool = True
    use_lod_axis: bool = False
    use_alibi: bool = False
    base: float = 10000.0


class GaussianRoPE(nn.Module):
    """Configurable Gaussian RoPE supporting sigma decay, LOD axis, and ALiBi."""

    def __init__(
        self,
        head_dim: int,
        block_size: int,
        num_heads: int,
        config: GaussianRoPEConfig,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.config = config
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.inv_freq_template = 1.0 / (config.base ** (channel_range / head_dim))
        lod_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.lod_freq_template = 1.0 / (config.base ** (lod_range / head_dim))
        if config.use_alibi:
            slopes = torch.pow(2, torch.linspace(0, -(num_heads - 1), num_heads))
            self.register_buffer("alibi_slopes", slopes.view(num_heads, 1, 1), persistent=False)
        else:
            self.alibi_slopes = None

    def _lod_to_sigma(self, lod_tensor: torch.Tensor) -> torch.Tensor:
        return (self.block_size ** lod_tensor.float()).clamp(min=1.0)

    def forward(
        self, positions: torch.Tensor, lod_tensor: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inv_freq = self.inv_freq_template.to(device)
        pos = positions.to(device).float()
        theta = torch.einsum("bt,d->btd", pos, inv_freq)
        decay = 1.0
        if self.config.use_sigma_decay:
            sigma = self._lod_to_sigma(lod_tensor.to(device))
            decay = torch.exp(-0.5 * (sigma.unsqueeze(-1) * inv_freq) ** 2)
        cos = (torch.cos(theta) * decay)[:, :, None, :].bfloat16()
        sin = (torch.sin(theta) * decay)[:, :, None, :].bfloat16()

        if self.config.use_lod_axis:
            lod_freq = self.lod_freq_template.to(device)
            lod_phases = torch.einsum("bt,d->btd", lod_tensor.to(device).float(), lod_freq)
            cos_lod = torch.cos(lod_phases)[:, :, None, :].bfloat16()
            sin_lod = torch.sin(lod_phases)[:, :, None, :].bfloat16()
            cos = torch.cat([cos, cos_lod], dim=-1)
            sin = torch.cat([sin, sin_lod], dim=-1)

        alibi = None
        if self.config.use_alibi and self.alibi_slopes is not None:
            alibi = self.alibi_slopes.to(device).bfloat16()
        return cos, sin, alibi


POSITIONAL_REGISTRY: Dict[str, GaussianRoPEConfig] = {
    "simple": GaussianRoPEConfig(use_sigma_decay=False, use_lod_axis=False, use_alibi=False),
    "gaussian": GaussianRoPEConfig(use_sigma_decay=True, use_lod_axis=False, use_alibi=False),
    "gaussian_lod2d": GaussianRoPEConfig(use_sigma_decay=True, use_lod_axis=True, use_alibi=False),
    "gaussian_alibi": GaussianRoPEConfig(use_sigma_decay=True, use_lod_axis=False, use_alibi=True),
    "gaussian_lod2d_alibi": GaussianRoPEConfig(use_sigma_decay=True, use_lod_axis=True, use_alibi=True),
}


def build_positional(kind: str, head_dim: int, block_size: int, num_heads: int) -> GaussianRoPE:
    key = kind.lower()
    if key not in POSITIONAL_REGISTRY:
        raise ValueError(f"Unknown positional encoding: {kind}")
    return GaussianRoPE(head_dim, block_size, num_heads, POSITIONAL_REGISTRY[key])
