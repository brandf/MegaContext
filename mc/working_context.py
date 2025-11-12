from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    # Optional annotation for telemetry/debugging: 'expand' or 'collapse'
    action: str = ""
    # Optional: how many WC elements this edit replaces (for telemetry)
    old_count: int = 0


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
        recent_tokens: int = 0,
    ) -> None:
        self.config = config
        self.recent_tokens = int(max(0, recent_tokens))
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
        self._event_log: List[Dict[str, int]] = []
        self._enforce_recent_lod0()

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
        self._event_log.append({"event": "load", "lod": lod, "length": window})
        self._enforce_recent_lod0()


    def append(self, embedding: torch.Tensor, lod: int, global_position: int) -> None:
        """
        Append a single timestep embedding `[B, D]` with its metadata.
        """
        if embedding.dim() != 2:
            raise ValueError("WorkingContext.append expects an embedding shaped [B, D]")
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
        self._event_log.append({"event": "append", "lod": lod, "position": global_position})
        self._enforce_recent_lod0()

    def replace(self, edit: WorkingContextEdit) -> None:
        total_len = self.tensor.shape[1]
        start = max(0, min(edit.wc_start, total_len))
        replacements = edit.replacements.to(self.tensor.device)
        count = replacements.shape[1]
        if count == 0:
            return
        remove_count = int(edit.old_count) if getattr(edit, "old_count", 0) > 0 else count
        end = max(start, min(total_len, start + remove_count))
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
        if position_col.shape[0] != self.positions.shape[0]:
            position_col = position_col.repeat(self.positions.shape[0], 1)
        if position_col.shape[1] != count:
            pad_len = count - position_col.shape[1]
            tail = position_col[:, -1:].repeat(1, pad_len)
            position_col = torch.cat([position_col, tail], dim=1)
        self.positions = torch.cat(
            [self.positions[:, :start], position_col, self.positions[:, end:]], dim=1
        )
        self._positional_cache = None
        self._trim()
        self._event_log.append(
            {
                "event": "replace",
                "start": start,
                "count": count,
                "new_lod": edit.lod,
                "mc_start": edit.mc_start_position,
            }
        )
        self._enforce_recent_lod0()

    def _trim(self) -> None:
        window = self.config.max_length
        if self.tensor.shape[1] <= window:
            return
        excess = self.tensor.shape[1] - window
        self.tensor = self.tensor[:, excess:]
        self.lod_tensor = self.lod_tensor[:, excess:]
        self.positions = self.positions[:, excess:]
        self._positional_cache = None
        self._enforce_recent_lod0()

    def _enforce_recent_lod0(self) -> None:
        if self.recent_tokens <= 0:
            return
        if self.lod_tensor.shape[1] == 0:
            return
        tail = min(self.recent_tokens, self.lod_tensor.shape[1])
        if torch.any(self.lod_tensor[:, -tail:] != 0):
            raise RuntimeError(
                "[WorkingContext] Recent tokens invariant violated: "
                f"expected last {tail} entries to be LOD0"
            )

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

    def drain_events(self) -> List[Dict[str, int]]:
        events = self._event_log
        self._event_log = []
        return events
