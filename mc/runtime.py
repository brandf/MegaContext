from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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
from .working_context import WorkingContext, WorkingContextEdit
from .gaussian_rope import build_positional, GaussianRoPE


@dataclass
class WorkingContextVariant:
    working_context: WorkingContext
    source: str
    lod_hint: int
    edits_applied: int = 0
    allocator: Optional[FocusAllocatorBase] = None
    lens_scores: Optional[torch.Tensor] = None
    token_loss_value: Optional[torch.Tensor] = None
    lod1_loss_value: Optional[torch.Tensor] = None


@dataclass
class SampleContext:
    tree: MegaContextTree
    variants: List[WorkingContextVariant]


@dataclass
class InferenceState:
    tree: MegaContextTree
    working_context: WorkingContext
    allocator: FocusAllocatorBase


@dataclass
class MCBatchResult:
    positional_cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    token_loss: Optional[torch.Tensor]
    lod1_loss: Optional[torch.Tensor]
    lens_loss: Optional[torch.Tensor]


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
            module.train()
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
            Batch result containing positional cache and auxiliary losses.
        """
        batch_states: List[SampleContext] = []
        total_edits = 0
        tokens_device = tokens.to(self.device)
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
        self.current_batch_states = batch_states
        self.telemetry.log_focus(step, total_edits)
        positional_cache = self._build_primary_positional(batch_states)
        token_loss, lod1_loss = self._aggregate_horizon_losses(batch_states)
        lens_loss = self._compute_lens_losses(batch_states)
        result = MCBatchResult(
            positional_cache=positional_cache,
            token_loss=token_loss,
            lod1_loss=lod1_loss,
            lens_loss=lens_loss,
        )
        self.current_batch_states = []
        return result

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def _build_sample_context(self, tree: MegaContextTree) -> SampleContext:
        variants = self._sample_initial_wcs(tree)
        refined = self._refine_variants(tree, variants)
        limited = refined[: self.config.max_counterfactuals]
        return SampleContext(tree=tree, variants=limited)

    def _build_primary_positional(
        self, batch_states: List[SampleContext]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        wc = self._select_primary_variant(batch_states)
        if wc is None or self.positional_encoder is None:
            return None
        cos, sin, alibi_slopes = wc.get_positional_encodings()
        alibi_bias = None
        if alibi_slopes is not None:
            positions = wc.get_positions().float()
            rel = positions.unsqueeze(2) - positions.unsqueeze(1)
            slopes = alibi_slopes.to(self.device).view(1, self.config.num_heads, 1, 1)
            alibi_bias = (slopes * rel.unsqueeze(1)).bfloat16()
        return cos, sin, alibi_bias

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
            scores = self.lensnet(variant.working_context)
            variant.lens_scores = scores.squeeze(-1)
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
        if variant.lens_scores is None:
            return []
        siblings: List[WorkingContextVariant] = []
        scores = variant.lens_scores
        if scores.dim() > 1:
            scores = scores.squeeze(0)
        # Force an expand on the strongest positive score.
        expand_indices = torch.argsort(scores, descending=True).tolist()
        for idx in expand_indices:
            sibling = self._force_expand_variant(tree, variant, idx)
            if sibling is not None:
                siblings.append(sibling)
                break
        # Force a collapse on the strongest negative score.
        collapse_indices = torch.argsort(scores, descending=False).tolist()
        for idx in collapse_indices:
            sibling = self._force_collapse_variant(tree, variant, idx)
            if sibling is not None:
                siblings.append(sibling)
                break
        return siblings

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

    def _clone_working_context(self, wc: WorkingContext) -> WorkingContext:
        embeddings = wc.to_tensor().clone()
        positions = wc.get_positions().clone()
        lods = wc.get_lod_tensor().clone()
        clone = WorkingContext(
            embeddings,
            positions,
            wc.config,
            lod_tensor=lods,
        )
        self._configure_wc_positional(clone)
        return clone

    def _aggregate_horizon_losses(
        self, batch_states: List[SampleContext]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_losses = []
        lod1_losses = []
        for sample in batch_states:
            token_loss, lod1_loss = self._compute_horizon_losses_for_sample(sample)
            if token_loss is not None:
                token_losses.append(token_loss)
            if lod1_loss is not None:
                lod1_losses.append(lod1_loss)
        agg_token = torch.stack(token_losses).mean() if token_losses else None
        agg_lod1 = torch.stack(lod1_losses).mean() if lod1_losses else None
        return agg_token, agg_lod1

    def _compute_horizon_losses_for_sample(
        self, sample: SampleContext
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_losses = []
        lod1_losses = []
        for variant in sample.variants:
            token_loss, lod1_loss = self._evaluate_variant_horizon(sample, variant)
            if token_loss is not None:
                variant.token_loss_value = token_loss
                token_losses.append(token_loss)
            if lod1_loss is not None:
                variant.lod1_loss_value = lod1_loss
                lod1_losses.append(lod1_loss)
        token_loss = torch.stack(token_losses).mean() if token_losses else None
        lod1_loss = torch.stack(lod1_losses).mean() if lod1_losses else None
        return token_loss, lod1_loss

    def _evaluate_variant_horizon(
        self,
        sample: SampleContext,
        variant: WorkingContextVariant,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        horizon = self.config.horizon_tokens
        if horizon <= 0 or sample.tree.tokens is None:
            return None, None
        wc = variant.working_context
        wc_embeddings = wc.to_tensor()
        wc_positions = wc.get_positions()
        last_pos = int(wc_positions[0, -1].item())
        horizon_tokens = self._slice_tokens(sample.tree, last_pos + 1, horizon)
        if horizon_tokens is None:
            return None, None
        horizon_embeddings = self._embed_with_padding(horizon_tokens)
        combined = torch.cat([wc_embeddings, horizon_embeddings], dim=1)
        cos_sin = self._compose_positional_overrides(wc, last_pos, horizon_tokens.shape[1])
        dummy_idx = torch.zeros(
            (1, combined.shape[1]), dtype=torch.long, device=self.device
        )
        logits = self.model(
            dummy_idx,
            targets=None,
            cos_sin_override=cos_sin,
            inputs_embeds=combined,
        )
        horizon_logits = logits[:, -horizon_tokens.shape[1]:, :]
        token_loss = F.cross_entropy(
            horizon_logits.reshape(-1, horizon_logits.size(-1)),
            horizon_tokens.view(-1),
            ignore_index=-1,
        )
        lod1_loss = self._compute_lod1_loss(horizon_tokens, horizon_logits)
        return token_loss, lod1_loss

    def _compose_positional_overrides(
        self,
        wc: WorkingContext,
        last_pos: int,
        horizon_len: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.positional_encoder is None:
            return None
        wc_cos, wc_sin, _ = wc.get_positional_encodings()
        horizon_positions = torch.arange(
            last_pos + 1,
            last_pos + 1 + horizon_len,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        lod_tensor = torch.zeros_like(horizon_positions)
        hor_cos, hor_sin, _ = self.positional_encoder(
            horizon_positions,
            lod_tensor,
            device=self.device,
        )
        cos = torch.cat([wc_cos, hor_cos], dim=1)
        sin = torch.cat([wc_sin, hor_sin], dim=1)
        return cos, sin

    def _embed_with_padding(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != -1
        safe_tokens = tokens.clone()
        safe_tokens[~mask] = 0
        embeddings = self.embed(safe_tokens)
        embeddings = embeddings * mask.unsqueeze(-1)
        return embeddings

    def _slice_tokens(
        self, tree: MegaContextTree, start: int, length: int
    ) -> Optional[torch.Tensor]:
        if tree.tokens is None or length <= 0:
            return None
        total = tree.tokens.shape[1]
        if start >= total:
            return None
        end = min(total, start + length)
        slice_tokens = tree.tokens[:, start:end]
        if slice_tokens.shape[1] < length:
            pad = torch.full(
                (slice_tokens.shape[0], length - slice_tokens.shape[1]),
                -1,
                dtype=slice_tokens.dtype,
                device=slice_tokens.device,
            )
            slice_tokens = torch.cat([slice_tokens, pad], dim=1)
        return slice_tokens

    def _compute_lod1_loss(
        self, tokens: torch.Tensor, horizon_logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        block = self.config.block_size
        if block <= 0:
            return None
        horizon_len = tokens.shape[1]
        num_blocks = horizon_len // block
        if num_blocks == 0:
            return None
        trim = num_blocks * block
        valid_tokens = tokens[:, :trim]
        mask = valid_tokens != -1
        if not mask.any():
            return None
        safe_tokens = valid_tokens.clone()
        safe_tokens[~mask] = 0
        gt_embeddings = self.embed(safe_tokens) * mask.unsqueeze(-1)
        gt_blocks = gt_embeddings.view(num_blocks, block, -1)
        pred_logits = horizon_logits[:, :trim, :]
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_embeds = torch.matmul(pred_probs, self.embed.weight)
        pred_blocks = pred_embeds.view(num_blocks, block, -1)
        gt_gists = self.gistnet(gt_blocks)
        pred_gists = self.gistnet(pred_blocks)
        loss = 1 - F.cosine_similarity(pred_gists, gt_gists, dim=-1)
        return loss.mean()

    def _compute_lens_losses(
        self, batch_states: List[SampleContext]
    ) -> Optional[torch.Tensor]:
        losses = []
        for sample in batch_states:
            best = self._select_best_variant(sample.variants)
            if best is None or best.token_loss_value is None:
                continue
            best_map = self._build_lod_lookup(best.working_context)
            best_loss = best.token_loss_value
            for variant in sample.variants:
                if variant.lens_scores is None:
                    continue
                scores = variant.lens_scores
                if scores.dim() > 1:
                    scores = scores.squeeze(0)
                targets = self._build_lens_targets(variant, best_map, scores)
                base_loss = F.mse_loss(scores, targets, reduction='mean')
                if variant.token_loss_value is not None:
                    weight = 1.0 + (variant.token_loss_value - best_loss).clamp(min=0)
                    base_loss = base_loss * weight
                losses.append(base_loss)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _select_best_variant(
        self, variants: List[WorkingContextVariant]
    ) -> Optional[WorkingContextVariant]:
        best = None
        best_val = float("inf")
        for variant in variants:
            if variant.token_loss_value is None:
                continue
            val_item = float(variant.token_loss_value.detach())
            if val_item < best_val:
                best = variant
                best_val = val_item
        return best

    def _build_lod_lookup(self, wc: WorkingContext) -> Dict[int, int]:
        positions = wc.get_positions()[0]
        lods = wc.get_lod_tensor()[0]
        return {int(pos.item()): int(lod.item()) for pos, lod in zip(positions, lods)}

    def _build_lens_targets(
        self,
        variant: WorkingContextVariant,
        best_map: Dict[int, int],
        score_template: torch.Tensor,
    ) -> torch.Tensor:
        positions = variant.working_context.get_positions()[0]
        lods = variant.working_context.get_lod_tensor()[0]
        targets = torch.zeros_like(score_template)
        for idx, (pos, lod) in enumerate(zip(positions, lods)):
            pos_int = int(pos.item())
            desired_lod = best_map.get(pos_int, lod.item() + 1)
            if desired_lod < lod.item():
                targets[idx] = 1.0  # expand
            elif desired_lod > lod.item():
                targets[idx] = -1.0  # collapse
            else:
                targets[idx] = 0.0
        return targets

    def _force_expand_variant(
        self,
        tree: MegaContextTree,
        variant: WorkingContextVariant,
        idx: int,
    ) -> Optional[WorkingContextVariant]:
        wc = variant.working_context
        lods = wc.get_lod_tensor()[0]
        if idx >= lods.shape[0]:
            return None
        lod = int(lods[idx].item())
        if lod <= 0:
            return None
        positions = wc.get_positions()[0]
        global_pos = int(positions[idx].item())
        try:
            children = tree.get_children_embeddings(lod, global_pos)
        except ValueError:
            return None
        if children.shape[1] == 0:
            return None
        edit = WorkingContextEdit(
            wc_start=idx,
            replacements=children,
            lod=lod - 1,
            mc_start_position=global_pos,
            stride=tree.tokens_per_entry(lod - 1),
        )
        sibling_wc = self._clone_working_context(wc)
        sibling_wc.replace(edit)
        return WorkingContextVariant(
            working_context=sibling_wc,
            source=f"{variant.source}+expand",
            lod_hint=lod - 1,
        )

    def _force_collapse_variant(
        self,
        tree: MegaContextTree,
        variant: WorkingContextVariant,
        idx: int,
    ) -> Optional[WorkingContextVariant]:
        wc = variant.working_context
        tensor = wc.to_tensor()
        length = tensor.shape[1]
        block = self.config.block_size
        if idx + block > length:
            return None
        lods = wc.get_lod_tensor()[0]
        positions = wc.get_positions()[0]
        lod = int(lods[idx].item())
        if lod >= self.config.max_lod:
            return None
        block_lods = lods[idx : idx + block]
        if torch.any(block_lods != lod):
            return None
        stride = tree.tokens_per_entry(lod)
        global_pos = int(positions[idx].item())
        expected = torch.arange(
            global_pos,
            global_pos + stride * block,
            stride,
            device=positions.device,
            dtype=positions.dtype,
        )
        if torch.any(positions[idx : idx + block] != expected):
            return None
        try:
            parent = tree.get_parent_embedding(lod, global_pos)
        except ValueError:
            return None
        edit = WorkingContextEdit(
            wc_start=idx,
            replacements=parent,
            lod=lod + 1,
            mc_start_position=global_pos,
            stride=tree.tokens_per_entry(lod + 1),
        )
        sibling_wc = self._clone_working_context(wc)
        sibling_wc.replace(edit)
        return WorkingContextVariant(
            working_context=sibling_wc,
            source=f"{variant.source}+collapse",
            lod_hint=lod + 1,
        )

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
        with torch.no_grad():
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
        with torch.no_grad():
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
