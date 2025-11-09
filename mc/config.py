from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MegaContextConfig:
    embed_dim: int
    block_size: int = 32
    max_lod: int = 3
    device: str = "cuda"


@dataclass
class WorkingContextConfig:
    embed_dim: int
    max_length: int
    device: str = "cuda"


@dataclass
class MCConfig:
    """
    Aggregate configuration passed to the MCController.
    """

    embed_dim: int
    max_seq_len: int
    block_size: int = 32
    max_lod: int = 3
    device: str = "cuda"
    enable_gaussian_rope: bool = True
    telemetry_interval: int = 100
    gistnet_type: str = "transformer2_mean_mlp"
    lensnet_type: str = "simple"
    allocator_type: str = "simple"
    num_heads: int = 1
    positional_type: Optional[str] = "gaussian"

    tree_config: MegaContextConfig = field(init=False)
    wc_config: WorkingContextConfig = field(init=False)

    def __post_init__(self) -> None:
        self.tree_config = MegaContextConfig(
            embed_dim=self.embed_dim,
            block_size=self.block_size,
            max_lod=self.max_lod,
            device=self.device,
        )
        self.wc_config = WorkingContextConfig(
            embed_dim=self.embed_dim,
            max_length=self.max_seq_len,
            device=self.device,
        )
