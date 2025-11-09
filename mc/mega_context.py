from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, TYPE_CHECKING

import torch

from .config import MegaContextConfig

if TYPE_CHECKING:
    from .gistnet import GistNetBase


class MegaContextTree:
    """
    Tensor-first representation of the MegaContext hierarchy.

    Levels are stored as dense tensors shaped (batch, num_nodes, embed_dim)
    so downstream code can slice directly without additional Python structures.
    """

    def __init__(self, config: MegaContextConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.levels: Dict[int, torch.Tensor] = {}
        self.positions: Dict[int, torch.Tensor] = {}

    @classmethod
    def from_embeddings(
        cls,
        embeddings: torch.Tensor,
        config: MegaContextConfig,
        gistnet: Optional["GistNetBase"] = None,
    ) -> "MegaContextTree":
        """
        Build a tree from token embeddings of shape (batch, seq_len, embed_dim).
        """
        tree = cls(config, embeddings.device)
        tree._build_levels(embeddings, gistnet=gistnet)
        return tree

    def _build_levels(
        self, embeddings: torch.Tensor, gistnet: Optional["GistNetBase"] = None
    ) -> None:
        batch, seq_len, embed_dim = embeddings.shape
        level_data = embeddings
        level_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(
            batch, 1
        )
        self.levels[0] = level_data
        self.positions[0] = level_positions

        for lod in range(1, self.config.max_lod + 1):
            prev_len = level_data.shape[1]
            if prev_len < self.config.block_size:
                break
            grouped, new_len = self._reshape_for_pool(level_data, self.config.block_size)
            if gistnet is not None:
                grouped = grouped.contiguous()
                flattened = grouped.view(
                    batch * new_len, self.config.block_size, embed_dim
                )
                pooled = gistnet(flattened)
                level_data = pooled.view(batch, new_len, embed_dim)
            else:
                level_data = grouped.mean(dim=2)
            grouped_pos, _ = self._reshape_for_pool(
                level_positions, self.config.block_size
            )
            level_positions = grouped_pos[:, :, 0]
            self.levels[lod] = level_data
            self.positions[lod] = level_positions

    @staticmethod
    def _reshape_for_pool(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
        batch, length, dim = x.shape
        new_len = math.ceil(length / block_size)
        pad = new_len * block_size - length
        if pad > 0:
            pad_tensor = torch.zeros(batch, pad, dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
        return x.view(batch, new_len, block_size, dim), new_len

    def slice(self, lod: int, start: int, end: int) -> torch.Tensor:
        data = self.levels.get(lod)
        if data is None:
            raise ValueError(f"LOD {lod} not present in tree")
        return data[:, start:end]

    def append(self, embedding: torch.Tensor, lod: int = 0) -> None:
        """
        Append a single token/gist embedding and rebuild affected parents.
        """
        if lod != 0:
            raise NotImplementedError("Appending higher LOD nodes not yet supported")
        if 0 not in self.levels:
            self.levels[0] = embedding.unsqueeze(1)
            self.positions[0] = torch.tensor(
                [[0]], device=self.device, dtype=torch.long
            )
        else:
            self.levels[0] = torch.cat([self.levels[0], embedding.unsqueeze(1)], dim=1)
            next_pos = self.positions[0][:, -1:] + 1
            self.positions[0] = torch.cat([self.positions[0], next_pos], dim=1)
        base = self.levels[0]
        base_pos = self.positions[0]
        self._build_levels(base)
        self.levels[0] = base
        self.positions[0] = base_pos

    def get_level(self, lod: int) -> torch.Tensor:
        if lod not in self.levels:
            raise ValueError(f"LOD {lod} not available")
        return self.levels[lod]

    def num_levels(self) -> int:
        return len(self.levels)

    def summary(self) -> Dict[int, Tuple[int, int]]:
        """
        Return a quick summary useful for telemetry: {lod: (nodes, embed_dim)}.
        """
        return {lod: tensor.shape[1:] for lod, tensor in self.levels.items()}
