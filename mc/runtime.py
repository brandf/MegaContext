from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch

from nanochat.report import get_report

from .config import MCConfig
from .gistnet import build_gistnet
from .lensnet import build_lensnet
from .focus_allocator import (
    build_focus_allocator,
    FocusAllocatorBase,
    FocusAllocatorConfig,
)
from .mega_context import MegaContextTree, build_mega_context
from .working_context import WorkingContext
from .gaussian_rope import build_positional, GaussianRoPE


class MCTelemetry:
    def __init__(self, interval: int = 100) -> None:
        self.interval = interval
        self.report = get_report()

    def log_tree(self, step: int, tree: MegaContextTree) -> None:
        if step % self.interval != 0:
            return
        summary = tree.summary()
        data = {f"lod_{lod}_nodes": int(shape[0]) for lod, shape in summary.items()}
        self.report.log(section="MegaContext Tree", data=data)

    def log_focus(self, step: int, plans: int) -> None:
        if step % self.interval != 0:
            return
        self.report.log(
            section="Focus Allocator",
            data={"edits_per_step": plans},
        )


class MCController:
    """
    Bridges the nanochat training loop with the MegaContext components.
    Safe to instantiate even when the MC path is disabled—callers should
    guard process_batch() with the `mc_enabled` flag.
    """

    def __init__(self, model: torch.nn.Module, config: MCConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.embed = self._resolve_embedding_layer(model)
        embed_dim = config.embed_dim
        self.gistnet = build_gistnet(
            config.gistnet_type,
            embed_dim,
            block_size=config.block_size,
            layers=config.gistnet_layers,
            pooling=config.gistnet_pooling,
            head=config.gistnet_head,
            num_heads=config.num_heads,
        ).to(self.device)
        self.lensnet = build_lensnet(
            config.lensnet_type,
            embed_dim,
            max_length=config.wc_config.max_length,
            num_heads=config.num_heads,
            layers=config.lensnet_layers,
            head=config.lensnet_head,
        ).to(self.device)
        self.focus_allocator: Optional[FocusAllocatorBase] = None
        self.telemetry = MCTelemetry(interval=config.telemetry_interval)
        self.positional_encoder: Optional[GaussianRoPE] = None
        if config.positional_type:
            self.positional_encoder = build_positional(
                config.positional_type,
                head_dim=embed_dim // config.num_heads,
                block_size=config.block_size,
                num_heads=config.num_heads,
            )

        for module in (self.gistnet, self.lensnet):
            module.eval()

    @staticmethod
    def _resolve_embedding_layer(model: torch.nn.Module):
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        raise ValueError("Unable to locate embedding layer on model")

    def process_batch(
        self,
        tokens: torch.Tensor,
        step: int,
        context: str = "train",
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            tokens: [B, T] token ids from nanochat loader.
        Returns:
            Optional positional cache tuple (cos, sin, alibi) each shaped per head.
        """
        with torch.no_grad():
            tokens_device = tokens.to(self.device)
            tree = build_mega_context(
                self.config.mc_tree_type,
                tokens_device,
                self.embed,
                self.config.tree_config,
                gistnet=self.gistnet,
            )
            level0, positions = tree.get_level_metadata(0)
            wc = WorkingContext(
                level0,
                positions,
                self.config.wc_config,
            )
            wc.set_positional_spec(
                self.config.positional_type or "gaussian",
                self.config.embed_dim // self.config.num_heads,
                self.config.num_heads,
                self.config.wc_config.max_length,
            )
            tree.release_lod0_cache(disable_future_cache=True)
            allocator_cfg = FocusAllocatorConfig(
                block_size=self.config.block_size,
                max_lod=self.config.max_lod,
                soft_max_length=self.config.soft_max_length,
                recent_tokens=self.config.allocator_recent_tokens,
                expand_threshold=self.config.allocator_expand_threshold,
                collapse_threshold=self.config.allocator_collapse_threshold,
            )
            self.focus_allocator = build_focus_allocator(
                self.config.allocator_type,
                tree=tree,
                working_context=wc,
                lensnet=self.lensnet,
                config=allocator_cfg,
            )
            edits = self.focus_allocator.rebuild(
                max_replacements_per_iteration=self.config.allocator_max_replacements,
                num_iterations=self.config.allocator_iterations,
            )
            self.telemetry.log_tree(step, tree)
            self.telemetry.log_focus(step, edits)
            positional = None
            if self.positional_encoder is not None:
                cos, sin, alibi_slopes = self.positional_encoder(
                    wc.get_positions(),  # [B, W]
                    wc.get_lod_tensor(),  # [B, W]
                    device=self.device,
                )
                alibi_bias = None
                if alibi_slopes is not None:
                    positions = wc.get_positions().float()  # already on device
                    rel = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, W, W]
                    slopes = alibi_slopes.to(self.device).view(1, self.config.num_heads, 1, 1)
                    alibi_bias = (slopes * rel.unsqueeze(1)).bfloat16()  # [1, H, W, W]
                positional = (cos, sin, alibi_bias)
            return positional
