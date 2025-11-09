from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class WorkingContextVariant:
    working_context: WorkingContext
    source: str
    lod_hint: int
    edits_applied: int = 0
    allocator: Optional[FocusAllocatorBase] = None


@dataclass
class SampleContext:
    tree: MegaContextTree
    variants: List[WorkingContextVariant]


@dataclass
class InferenceState:
    tree: MegaContextTree
    working_context: WorkingContext
    allocator: FocusAllocatorBase


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
        self._head_dim = embed_dim // config.num_heads
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
                head_dim=self._head_dim,
                block_size=config.block_size,
                num_heads=config.num_heads,
            )

        for module in (self.gistnet, self.lensnet):
            module.eval()
        self.current_batch_states: List[SampleContext] = []
        self.inference_state: Optional[InferenceState] = None

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
        batch_states: List[SampleContext] = []
        total_edits = 0
        tokens_device = tokens.to(self.device)
        with torch.no_grad():
            for idx in range(tokens_device.size(0)):
                seq = tokens_device[idx : idx + 1]
                tree = build_mega_context(
                    self.config.mc_tree_type,
                    seq,
                    self.embed,
                    self.config.tree_config,
                    gistnet=self.gistnet,
                )
                sample_state = self._build_sample_context(tree)
                batch_states.append(sample_state)
                for variant in sample_state.variants:
                    total_edits += max(0, variant.edits_applied)
                self.telemetry.log_tree(step, tree)
                tree.release_lod0_cache(disable_future_cache=True)
        self.current_batch_states = batch_states
        self.telemetry.log_focus(step, total_edits)
        primary_wc = self._select_primary_variant(batch_states)
        if primary_wc is None or self.positional_encoder is None:
            return None
        cos, sin, alibi_slopes = self.positional_encoder(
            primary_wc.get_positions(),
            primary_wc.get_lod_tensor(),
            device=self.device,
        )
        alibi_bias = None
        if alibi_slopes is not None:
            positions = primary_wc.get_positions().float()
            rel = positions.unsqueeze(2) - positions.unsqueeze(1)
            slopes = alibi_slopes.to(self.device).view(1, self.config.num_heads, 1, 1)
            alibi_bias = (slopes * rel.unsqueeze(1)).bfloat16()
        return cos, sin, alibi_bias

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def _build_sample_context(self, tree: MegaContextTree) -> SampleContext:
        variants = self._sample_initial_wcs(tree)
        refined = self._refine_variants(tree, variants)
        limited = refined[: self.config.max_counterfactuals]
        return SampleContext(tree=tree, variants=limited)

    def _sample_initial_wcs(self, tree: MegaContextTree) -> List[WorkingContextVariant]:
        variants: List[WorkingContextVariant] = []
        baseline = self._build_recency_variant(tree)
        if baseline is not None:
            variants.append(baseline)
        target = self.config.initial_working_contexts
        for lod in range(1, self.config.max_lod + 1):
            if len(variants) >= target:
                break
            variant = self._build_lod_variant(tree, lod)
            if variant is not None:
                variants.append(variant)
        while len(variants) < target:
            variant = self._build_random_span_variant(tree)
            if variant is None:
                break
            variants.append(variant)
        return variants

    def _build_recency_variant(self, tree: MegaContextTree) -> Optional[WorkingContextVariant]:
        try:
            embeddings, positions = tree.get_level_metadata(0)
        except ValueError:
            return None
        window = self.config.wc_config.max_length
        if embeddings.shape[1] > window:
            embeddings = embeddings[:, -window:]
            positions = positions[:, -window:]
        return self._create_variant(embeddings, positions, lod=0, source="recency_baseline")

    def _build_lod_variant(self, tree: MegaContextTree, lod: int) -> Optional[WorkingContextVariant]:
        try:
            embeddings, positions = tree.get_level_metadata(lod)
        except ValueError:
            return None
        if embeddings.shape[1] == 0:
            return None
        return self._create_variant(embeddings, positions, lod=lod, source=f"lod_{lod}")

    def _build_random_span_variant(self, tree: MegaContextTree) -> Optional[WorkingContextVariant]:
        try:
            embeddings, positions = tree.get_level_metadata(0)
        except ValueError:
            return None
        total = embeddings.shape[1]
        window = self.config.wc_config.max_length
        if total <= window:
            return None
        start = random.randint(0, max(0, total - window))
        end = start + window
        span_embeddings = embeddings[:, start:end]
        span_positions = positions[:, start:end]
        return self._create_variant(span_embeddings, span_positions, lod=0, source=f"random_span_{start}")

    def _create_variant(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        lod: int,
        source: str,
    ) -> WorkingContextVariant:
        wc = WorkingContext(embeddings, positions, self.config.wc_config)
        self._configure_wc_positional(wc)
        return WorkingContextVariant(working_context=wc, source=source, lod_hint=lod)

    def _configure_wc_positional(self, wc: WorkingContext) -> None:
        wc.set_positional_spec(
            self.config.positional_type or "gaussian",
            self._head_dim,
            self.config.num_heads,
            self.config.wc_config.max_length,
        )

    def _refine_variants(
        self,
        tree: MegaContextTree,
        variants: List[WorkingContextVariant],
    ) -> List[WorkingContextVariant]:
        refined: List[WorkingContextVariant] = []
        for variant in variants:
            allocator = self._build_allocator(tree, variant.working_context)
            variant.allocator = allocator
            variant.edits_applied = allocator.update_focus(
                max_replacements_per_iteration=self.config.allocator_max_replacements,
                num_iterations=self.config.allocator_iterations,
            )
            refined.append(variant)
            siblings = self._generate_sibling_variants(tree, variant)
            for sibling in siblings:
                refined.append(sibling)
                if len(refined) >= self.config.max_counterfactuals:
                    return refined
        return refined

    def _generate_sibling_variants(
        self,
        tree: MegaContextTree,
        variant: WorkingContextVariant,
    ) -> List[WorkingContextVariant]:
        # Placeholder for ΔNLL-based counterfactuals. Future work will perturb
        # spans (e.g., forced expand/collapse) to create counterfactual labels.
        return []

    def _build_allocator(
        self,
        tree: MegaContextTree,
        working_context: WorkingContext,
    ) -> FocusAllocatorBase:
        allocator_cfg = FocusAllocatorConfig(
            block_size=self.config.block_size,
            max_lod=self.config.max_lod,
            soft_max_length=self.config.soft_max_length,
            recent_tokens=self.config.allocator_recent_tokens,
            expand_threshold=self.config.allocator_expand_threshold,
            collapse_threshold=self.config.allocator_collapse_threshold,
        )
        return build_focus_allocator(
            self.config.allocator_type,
            tree=tree,
            working_context=working_context,
            lensnet=self.lensnet,
            config=allocator_cfg,
        )

    def _select_primary_variant(
        self, batch_states: List[SampleContext]
    ) -> Optional[WorkingContext]:
        for sample in batch_states:
            if sample.variants:
                return sample.variants[0].working_context
        return None

    # ------------------------------------------------------------------ #
    # Inference facade
    # ------------------------------------------------------------------ #
    def begin_inference_session(self, initial_tokens: torch.Tensor) -> None:
        """
        Initialize a persistent MegaContext for inference/autoregressive decoding.
        """
        if initial_tokens.dim() == 1:
            initial_tokens = initial_tokens.unsqueeze(0)
        tokens = initial_tokens.to(self.device)
        tree = build_mega_context(
            self.config.mc_tree_type,
            tokens,
            self.embed,
            self.config.tree_config,
            gistnet=self.gistnet,
        )
        recency_variant = self._build_recency_variant(tree)
        if recency_variant is None:
            raise ValueError("Unable to build initial working context for inference")
        allocator = self._build_allocator(tree, recency_variant.working_context)
        allocator.rebuild(
            max_replacements_per_iteration=self.config.allocator_max_replacements,
            num_iterations=self.config.allocator_iterations,
        )
        self.inference_state = InferenceState(
            tree=tree,
            working_context=recency_variant.working_context,
            allocator=allocator,
        )

    def inference_step(self, new_tokens: torch.Tensor) -> None:
        """
        Append freshly generated tokens and refocus the Working Context.
        """
        if self.inference_state is None:
            raise RuntimeError("Inference session not initialized. Call begin_inference_session() first.")
        if new_tokens.dim() == 1:
            new_tokens = new_tokens.unsqueeze(0)
        tokens = new_tokens.to(self.device)
        embeddings = self.embed(tokens)
        self.inference_state.allocator.append(tokens, embeddings)
        self.inference_state.allocator.update_focus(
            max_replacements_per_iteration=self.config.allocator_max_replacements,
            num_iterations=self.config.allocator_iterations,
        )

    def get_inference_working_context(self) -> Optional[WorkingContext]:
        if self.inference_state is None:
            return None
        return self.inference_state.working_context
