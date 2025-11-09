from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import WorkingContextConfig


@dataclass
class ReplacementPlan:
    start: int
    count: int
    replacements: torch.Tensor
    lod: int
    global_start: int


class WorkingContext:
    """
    Represents the small on-device window (analogous to an L1 cache).
    Stores contiguous embeddings plus CPU-side metadata describing the
    origin LOD and global position of each element.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        config: WorkingContextConfig,
        lod_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        batch, seq_len, dim = embeddings.shape  # embeddings: [B, T, D]
        window = min(seq_len, config.max_length)
        self.tensor = embeddings[:, -window:].clone()  # [B, W, D]
        if lod_tensor is None:
            self.lod_tensor = torch.zeros(
                batch, window, dtype=torch.long, device=self.tensor.device
            )
        else:
            self.lod_tensor = lod_tensor[:, -window:].to(self.tensor.device)
        self.positions = positions[:, -window:].to(self.tensor.device)  # [B, W]

    def to_tensor(self) -> torch.Tensor:
        """Return current working context embeddings [B, W, D] on device."""
        return self.tensor

    def get_lod_tensor(self) -> torch.Tensor:
        """Return LOD annotations [B, W] colocated with embeddings."""
        return self.lod_tensor

    def get_positions(self) -> torch.Tensor:
        """Return global positions [B, W] colocated with embeddings."""
        return self.positions


    def append(self, embedding: torch.Tensor, lod: int, global_position: int) -> None:
        embedding = embedding.to(self.tensor.device)
        self.tensor = torch.cat([self.tensor, embedding.unsqueeze(1)], dim=1)  # extend sequence axis
        self.lod_tensor = torch.cat(
            [self.lod_tensor, torch.full_like(self.lod_tensor[:, :1], lod)], dim=1
        )
        self.positions = torch.cat(
            [self.positions, torch.full_like(self.positions[:, :1], global_position)],
            dim=1,
        )
        self._trim()

    def replace(self, plan: ReplacementPlan) -> None:
        start, count = plan.start, plan.count
        if count <= 0:
            return
        end = start + count
        replacements = plan.replacements.to(self.tensor.device)
        self.tensor = torch.cat(
            [self.tensor[:, :start], replacements, self.tensor[:, end:]], dim=1
        )
        lod_column = torch.full(
            (self.lod_tensor.shape[0], replacements.shape[1]),
            plan.lod,
            dtype=self.lod_tensor.dtype,
        )
        self.lod_tensor = torch.cat(
            [self.lod_tensor[:, :start], lod_column, self.lod_tensor[:, end:]], dim=1
        )
        position_col = torch.arange(
            plan.global_start,
            plan.global_start + replacements.shape[1],
            dtype=self.positions.dtype,
            device=self.positions.device,
        ).unsqueeze(0)
        self.positions = torch.cat(
            [self.positions[:, :start], position_col, self.positions[:, end:]], dim=1
        )
        self._trim()

    def _trim(self) -> None:
        window = self.config.max_length
        if self.tensor.shape[1] <= window:
            return
        excess = self.tensor.shape[1] - window
        self.tensor = self.tensor[:, excess:]
        self.lod_tensor = self.lod_tensor[:, excess:]
        self.positions = self.positions[:, excess:]
