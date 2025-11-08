from __future__ import annotations

import contextlib
from typing import Optional

import torch

from nanochat.report import get_report

from .config import MCConfig
from .gistnet import build_gistnet, GistNetBase
from .lensnet import build_lensnet, LensNetBase
from .focus_allocator import build_focus_allocator, FocusAllocatorBase
from .mega_context import MegaContextTree
from .working_context import WorkingContext
from .positional import build_positional, PositionalEncodingBase


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
        self.gistnet = build_gistnet(config.gistnet_type, embed_dim).to(self.device)
        self.lensnet = build_lensnet(config.lensnet_type, embed_dim).to(self.device)
        self.focus_allocator: FocusAllocatorBase = build_focus_allocator(
            config.allocator_type
        )
        self.telemetry = MCTelemetry(interval=config.telemetry_interval)
        self.positional_encoder: Optional[PositionalEncodingBase] = None
        if config.positional_type:
            self.positional_encoder = build_positional(
                config.positional_type,
                head_dim=embed_dim // config.num_heads,
                block_size=config.block_size,
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
    ) -> None:
        with torch.no_grad():
            emb = self.embed(tokens.to(self.device))
            tree = MegaContextTree.from_embeddings(
                emb, self.config.tree_config, gistnet=self.gistnet
            )
            level0 = tree.get_level(0)
            positions = tree.positions[0]
            wc = WorkingContext(
                level0,
                positions,
                self.config.wc_config,
            )
            logits = self.lensnet(wc.to_tensor(), wc.get_lod_tensor())
            plans = self.focus_allocator.build_plan(tree, logits, wc)
            for plan in plans:
                wc.replace(plan)
            self.telemetry.log_tree(step, tree)
            self.telemetry.log_focus(step, len(plans))
            if self.gaussian_rope is not None:
                cos, sin = self.gaussian_rope.build(
                    wc.get_positions(),
                    wc.get_lod_tensor(),
                    device=self.device,
                )
                self.telemetry.report.log(
                    section="Gaussian RoPE",
                    data={"cos_std": float(cos.std().cpu()), "sin_std": float(sin.std().cpu())},
                )
