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
    gistnet_type: str = "transformer"  # transformer | mean
    gistnet_layers: int = 2
    gistnet_pooling: str = "mean"  # mean | query | cls
    gistnet_head: str = "mlp"  # mlp | linear
    lensnet_type: str = "transformer"
    lensnet_layers: int = 2
    lensnet_head: str = "mlp"  # mlp | linear
    allocator_type: str = "greedy"
    mc_tree_type: str = "ram"
    initial_working_contexts: int = 4
    max_counterfactuals: int = 8
    horizon_tokens: int = 32
    long_horizon_multiplier: int = 32
    token_loss_weight: float = 1.0
    lod1_loss_weight: float = 0.1
    lod2_loss_weight: float = 0.05
    lens_loss_weight: float = 0.1
    soft_max_length: Optional[int] = None
    allocator_recent_tokens: int = 128
    allocator_expand_threshold: float = 0.1
    allocator_collapse_threshold: float = 0.1
    allocator_max_replacements: int = 4
    allocator_iterations: int = 2
    num_heads: int = 1
    positional_type: Optional[str] = "gaussian"
    random_seed: Optional[int] = None
    loss_projection_top_k: int = 64
    build_workers: int = 1

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
        if self.soft_max_length is None:
            self.soft_max_length = self.wc_config.max_length
        self.initial_working_contexts = max(1, self.initial_working_contexts)
        self.max_counterfactuals = max(self.initial_working_contexts, self.max_counterfactuals)
        self.horizon_tokens = max(1, self.horizon_tokens)
