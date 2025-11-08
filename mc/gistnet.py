from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class GistNetBase(nn.Module):
    def encode_block(self, block_embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimpleGistNet(GistNetBase):
    """Simple feed-forward compressor placeholder for GistNet."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj_in(embeddings))
        return self.proj_out(x)

    def encode_block(self, block_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = block_embeddings.mean(dim=1)
        return self.forward(pooled)


GISTNET_REGISTRY: Dict[str, Type[GistNetBase]] = {
    "simple": SimpleGistNet,
}


def build_gistnet(kind: str, embed_dim: int, **kwargs) -> GistNetBase:
    key = kind.lower()
    if key not in GISTNET_REGISTRY:
        raise ValueError(f"Unknown GistNet implementation: {kind}")
    return GISTNET_REGISTRY[key](embed_dim=embed_dim, **kwargs)
