from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MegaContextConfig:
    embed_dim: int
    block_size: int = 32
    max_lod: int = 2
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
    max_lod: int = 2
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
    max_counterfactuals: int = 4
    token_loss_weight: float = 1.0
    lens_loss_weight: float = 0.1
    soft_max_length: Optional[int] = None
    allocator_recent_tokens: int = 128
    allocator_expand_threshold: float = 0.1
    allocator_collapse_threshold: float = 0.1
    allocator_max_replacements: int = 4
    allocator_iterations: int = 1
    allocator_sample_top_k: int = 4
    allocator_sample_temperature: float = 1.0
    num_heads: int = 1
    positional_type: Optional[str] = "gaussian"
    random_seed: Optional[int] = None
    build_workers: int = 1  # TODO(mc): wire up parallel tree builds or remove this knob
    cache_lod0: bool = True
    auxiliary_dtype: str = "auto"  # auto | fp32 | bf16

    tree_config: MegaContextConfig = field(init=False)
    wc_config: WorkingContextConfig = field(init=False)
    eval_soft_max_length: Optional[int] = None
    infer_allocator_max_replacements: Optional[int] = None
    infer_allocator_iterations: Optional[int] = None
    infer_refocus_interval: int = 32
    infer_rebuild_max_replacements: Optional[int] = None
    infer_rebuild_iterations: Optional[int] = None
    collect_debug_metrics: bool = False
    log_lod_ascii_train: bool = False
    log_lod_ascii_val: bool = False
    log_lens_debug: bool = False
    lens_rank_weight: float = 0.5
    lens_budget_weight: float = 0.1
    lens_margin: float = 0.1
    disable_validation: bool = False
    lens_collapse_weight: float = 1.0
    lens_temperature: float = 1.0
    lens_adv_norm_beta: float = 0.9
    lens_kl_weight: float = 0.0
    lens_budget_smooth_weight: float = 0.0
    lens_budget_smooth_beta: float = 0.9
    lens_hard_negative_ratio: float = 1.0
    total_train_steps: int = 1
    train_wc_length: Optional[int] = None
    num_random_variants: int = 4
    random_variant_iterations: int = 4
    max_lens_pairs: int = 8

    def __post_init__(self) -> None:
        self.mc_tree_type = self.mc_tree_type.lower()
        if self.mc_tree_type != "ram":
            raise ValueError("Only mc_tree_type='ram' is supported in the current release.")
        self.auxiliary_dtype = (self.auxiliary_dtype or "auto").lower()
        if self.auxiliary_dtype not in {"auto", "fp32", "bf16"}:
            raise ValueError("auxiliary_dtype must be one of {'auto', 'fp32', 'bf16'}")
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
        self.max_counterfactuals = max(1, int(self.max_counterfactuals))
        if self.eval_soft_max_length is None:
            self.eval_soft_max_length = self.wc_config.max_length
        self.infer_refocus_interval = max(1, int(self.infer_refocus_interval))
        if self.train_wc_length is None:
            default_len = int(self.wc_config.max_length * 0.2)
            self.train_wc_length = max(1, min(self.wc_config.max_length, default_len))
        self.num_random_variants = max(0, int(self.num_random_variants))
        self.random_variant_iterations = max(1, int(self.random_variant_iterations))
        self.max_lens_pairs = max(1, int(self.max_lens_pairs))
        if self.lens_temperature <= 0:
            raise ValueError("lens_temperature must be > 0")
        if not (0.0 < self.lens_adv_norm_beta < 1.0):
            raise ValueError("lens_adv_norm_beta must be in (0, 1)")
        if not (0.0 < self.lens_budget_smooth_beta < 1.0):
            raise ValueError("lens_budget_smooth_beta must be in (0, 1)")
        self.lens_hard_negative_ratio = float(self.lens_hard_negative_ratio)
        if self.lens_hard_negative_ratio <= 0 or self.lens_hard_negative_ratio > 1.0:
            raise ValueError("lens_hard_negative_ratio must be in (0, 1]")
        self.total_train_steps = max(1, int(self.total_train_steps))
