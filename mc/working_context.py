from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import WorkingContextConfig
from .gaussian_rope import build_positional, GaussianRoPE


@dataclass
class WorkingContextEdit:
    wc_start: int
    replacements: torch.Tensor
    lod: int
    mc_start_position: int
    stride: int = 1


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
        self._positional_encoder: Optional[GaussianRoPE] = None
        self._positional_spec: Optional[Tuple[str, int, int, int]] = None
        self._positional_cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None

    def to_tensor(self) -> torch.Tensor:
        """Return current working context embeddings [B, W, D] on device."""
        return self.tensor

    @property
    def length(self) -> int:
        return int(self.tensor.shape[1])

    def get_lod_tensor(self) -> torch.Tensor:
        """Return LOD annotations [B, W] colocated with embeddings."""
        return self.lod_tensor

    def get_positions(self) -> torch.Tensor:
        """Return global positions [B, W] colocated with embeddings."""
        return self.positions

    def load_from_level(self, embeddings: torch.Tensor, positions: torch.Tensor, lod: int) -> None:
        window = min(embeddings.shape[1], self.config.max_length)
        self.tensor = embeddings[:, -window:].clone()
        self.positions = positions[:, -window:].to(self.tensor.device).long()
        self.lod_tensor = torch.full(
            (embeddings.shape[0], window),
            lod,
            dtype=torch.long,
            device=self.tensor.device,
        )
        self._positional_cache = None


    def append(self, embedding: torch.Tensor, lod: int, global_position: int) -> None:
        embedding = embedding.to(self.tensor.device)
        self.tensor = torch.cat([self.tensor, embedding.unsqueeze(1)], dim=1)  # extend sequence axis
        lod_col = torch.full(
            (self.lod_tensor.shape[0], 1),
            lod,
            dtype=self.lod_tensor.dtype,
            device=self.lod_tensor.device,
        )
        self.lod_tensor = torch.cat([self.lod_tensor, lod_col], dim=1)
        pos_col = torch.full(
            (self.positions.shape[0], 1),
            global_position,
            dtype=self.positions.dtype,
            device=self.positions.device,
        )
        self.positions = torch.cat([self.positions, pos_col], dim=1)
        self._positional_cache = None
        self._trim()

    def replace(self, edit: WorkingContextEdit) -> None:
        start = edit.wc_start
        replacements = edit.replacements.to(self.tensor.device)
        count = replacements.shape[1]
        if count == 0:
            return
        end = start + count
        self.tensor = torch.cat(
            [self.tensor[:, :start], replacements, self.tensor[:, end:]], dim=1
        )
        lod_column = torch.full(
            (self.lod_tensor.shape[0], count),
            edit.lod,
            dtype=self.lod_tensor.dtype,
            device=self.lod_tensor.device,
        )
        self.lod_tensor = torch.cat(
            [self.lod_tensor[:, :start], lod_column, self.lod_tensor[:, end:]], dim=1
        )
        stride = max(int(edit.stride), 1)
        position_col = torch.arange(
            edit.mc_start_position,
            edit.mc_start_position + stride * count,
            stride,
            dtype=self.positions.dtype,
            device=self.positions.device,
        ).unsqueeze(0)
        if position_col.shape[1] != count:
            pad_len = count - position_col.shape[1]
            tail = position_col[:, -1:].repeat(1, pad_len)
            position_col = torch.cat([position_col, tail], dim=1)
        self.positions = torch.cat(
            [self.positions[:, :start], position_col, self.positions[:, end:]], dim=1
        )
        self._positional_cache = None
        self._trim()

    def _trim(self) -> None:
        window = self.config.max_length
        if self.tensor.shape[1] <= window:
            return
        excess = self.tensor.shape[1] - window
        self.tensor = self.tensor[:, excess:]
        self.lod_tensor = self.lod_tensor[:, excess:]
        self.positions = self.positions[:, excess:]
        self._positional_cache = None

    def set_positional_spec(
        self,
        positional_type: str,
        head_dim: int,
        num_heads: int,
        block_size: Optional[int] = None,
    ) -> None:
        spec = (
            positional_type,
            head_dim,
            num_heads,
            block_size or self.config.max_length,
        )
        if spec == self._positional_spec:
            return
        self._positional_encoder = build_positional(
            positional_type,
            head_dim=head_dim,
            block_size=spec[3],
            num_heads=num_heads,
        )
        self._positional_spec = spec
        self._positional_cache = None

    def get_positional_encodings(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self._positional_encoder is None:
            raise ValueError("WorkingContext positional encoder not configured")
        if self._positional_cache is None:
            self._positional_cache = self._positional_encoder(
                self.positions,
                self.lod_tensor,
                device=self.tensor.device,
            )
        return self._positional_cache
