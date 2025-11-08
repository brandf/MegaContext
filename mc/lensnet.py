from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn


class LensNetBase(nn.Module):
    def forward(self, wc_embeddings: torch.Tensor, lod_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimpleLensNet(LensNetBase):
    """Minimal focus scoring network."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, wc_embeddings: torch.Tensor, lod_tensor: torch.Tensor) -> torch.Tensor:
        lod_tensor = lod_tensor.to(wc_embeddings.device).float().unsqueeze(-1)
        features = torch.cat([wc_embeddings, lod_tensor], dim=-1)
        return self.mlp(features)


LENSNET_REGISTRY: Dict[str, Type[LensNetBase]] = {
    "simple": SimpleLensNet,
}


def build_lensnet(kind: str, embed_dim: int, **kwargs) -> LensNetBase:
    key = kind.lower()
    if key not in LENSNET_REGISTRY:
        raise ValueError(f"Unknown LensNet implementation: {kind}")
    return LENSNET_REGISTRY[key](embed_dim=embed_dim, **kwargs)
