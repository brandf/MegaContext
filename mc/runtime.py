from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid

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
from .telemetry import TelemetryEvent, TelemetryProvider, NoOpTelemetryProvider


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
    session_id: str
    tree: MegaContextTree
    variants: List[WorkingContextVariant]


@dataclass
class InferenceState:
    session_id: str
    tree: MegaContextTree
    working_context: WorkingContext
    allocator: FocusAllocatorBase


@dataclass
class MCBatchResult:
    positional_cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    token_loss: Optional[torch.Tensor]
    lod1_loss: Optional[torch.Tensor]
    lod2_loss: Optional[torch.Tensor]
    lens_loss: Optional[torch.Tensor]
    cached_embeddings: Optional[torch.Tensor]
    positional_caches: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = field(default_factory=dict)


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

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCConfig,
        telemetry_provider: Optional[TelemetryProvider] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self._rng = random.Random(config.random_seed)
        self._batch_counters = {
            "horizon_triggers": 0,
            "sibling_expands": 0,
            "sibling_collapses": 0,
            "allocator_edits": 0,
        }
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
        self.telemetry_provider = telemetry_provider or NoOpTelemetryProvider()
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
        self._reset_batch_counters()
        cached_embeddings: List[torch.Tensor] = []
        workers = max(1, self.config.build_workers)
        if workers > 1 and tokens_device.size(0) > 1:
            tasks = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx in range(tokens_device.size(0)):
                    seq = tokens_device[idx : idx + 1]
                    session_id = f"train_step_{step}_sample_{idx}"
                    tasks.append(
                        (session_id, pool.submit(self._build_tree_sample, seq, session_id))
                    )
                for session_id, task in tasks:
                    tree, sample_state, seq_embeds, sample_edits = task.result()
                    batch_states.append(sample_state)
                    total_edits += sample_edits
                    cached_embeddings.append(seq_embeds)
                    self._log_tree_snapshot(session_id, tree, tag="training_build")
                    self.telemetry.log_tree(step, tree)
        else:
            for idx in range(tokens_device.size(0)):
                seq = tokens_device[idx : idx + 1]
                session_id = f"train_step_{step}_sample_{idx}"
                tree, sample_state, seq_embeds, sample_edits = self._build_tree_sample(seq, session_id)
                batch_states.append(sample_state)
                total_edits += sample_edits
                cached_embeddings.append(seq_embeds)
                self._log_tree_snapshot(session_id, tree, tag="training_build")
                self.telemetry.log_tree(step, tree)
        self.current_batch_states = batch_states
        self.telemetry.log_focus(step, total_edits)
        positional_cache = self._build_primary_positional(batch_states) if len(batch_states) == 1 else None
        positional_cache_map = self._build_session_positional(batch_states)
        token_loss, lod1_loss, lod2_loss = self._aggregate_horizon_losses(batch_states)
        lens_loss = self._compute_lens_losses(batch_states)
        result = MCBatchResult(
            positional_cache=positional_cache,
            token_loss=token_loss,
            lod1_loss=lod1_loss,
            lod2_loss=lod2_loss,
            lens_loss=lens_loss,
            cached_embeddings=torch.cat(cached_embeddings, dim=0) if cached_embeddings else None,
            positional_caches=positional_cache_map,
        )
        self.current_batch_states = []
        self._emit_batch_counters(step)
        return result

    def _build_tree_sample(
        self,
        seq: torch.Tensor,
        session_id: str,
    ) -> Tuple[MegaContextTree, SampleContext, torch.Tensor, int]:
        seq_embeds = self.embed(seq)
        tree = build_mega_context(
            self.config.mc_tree_type,
            seq,
            self.embed,
            self.config.tree_config,
            gistnet=self.gistnet,
            precomputed_embeddings=seq_embeds,
        )
        sample_state = self._build_sample_context(tree, session_id=session_id)
        sample_edits = 0
        for variant in sample_state.variants:
            edits = max(0, variant.edits_applied)
            sample_edits += edits
            if edits:
                self._increment_counter("allocator_edits", edits)
        return tree, sample_state, seq_embeds, sample_edits

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def _build_sample_context(self, tree: MegaContextTree, session_id: str) -> SampleContext:
        variants = self._sample_initial_wcs(tree, session_id=session_id)
        refined = self._refine_variants(tree, variants, session_id=session_id)
        limited = refined[: self.config.max_counterfactuals]
        return SampleContext(session_id=session_id, tree=tree, variants=limited)

    def _build_primary_positional(
        self, batch_states: List[SampleContext]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        wc = self._select_primary_variant(batch_states)
        if wc is None or self.positional_encoder is None:
            return None
        return self._build_wc_positional(wc)

    def _build_session_positional(
        self, batch_states: List[SampleContext]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        if self.positional_encoder is None:
            return {}
        positional_map: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
        for sample in batch_states:
            if not sample.variants:
                continue
            wc = sample.variants[0].working_context
            positional_map[sample.session_id] = self._build_wc_positional(wc)
        return positional_map

    def _build_wc_positional(
        self, wc: WorkingContext
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cos, sin, alibi_slopes = wc.get_positional_encodings()
        alibi_bias = None
        if alibi_slopes is not None:
            positions = wc.get_positions().float()
            rel = positions.unsqueeze(2) - positions.unsqueeze(1)
            slopes = alibi_slopes.to(self.device).view(1, self.config.num_heads, 1, 1)
            alibi_bias = (slopes * rel.unsqueeze(1)).bfloat16()
        return cos.to(self.device), sin.to(self.device), alibi_bias

    def _sample_initial_wcs(self, tree: MegaContextTree, session_id: str) -> List[WorkingContextVariant]:
        variants: List[WorkingContextVariant] = []
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        baseline = self._build_recency_variant(tree, level_cache)
        if baseline is not None:
            variants.append(baseline)
        target = self.config.initial_working_contexts
        for lod in range(1, self.config.max_lod + 1):
            if len(variants) >= target:
                break
            variant = self._build_lod_variant(tree, lod, level_cache)
            if variant is not None:
                variants.append(variant)
        self._ensure_highest_lod_coverage(tree, variants, target, level_cache)
        needed = target - len(variants)
        if needed > 0:
            starts = self._sample_random_span_starts(tree, level_cache, needed)
            for start in starts:
                variant = self._build_span_variant(tree, start, level_cache)
                if variant is None:
                    continue
                variants.append(variant)
                if len(variants) >= target:
                    break
        return variants

    def _ensure_highest_lod_coverage(
        self,
        tree: MegaContextTree,
        variants: List[WorkingContextVariant],
        target: int,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        available = [
            lod
            for lod in tree.levels.keys()
            if lod > 0 and tree.levels[lod].shape[1] > 0
        ]
        if not available:
            return
        highest = max(available)
        if any(v.lod_hint == highest for v in variants):
            return
        variant = self._build_lod_variant(tree, highest, level_cache)
        if variant is None:
            return
        if len(variants) >= target:
            variants.pop()
        variants.append(variant)

    def _build_recency_variant(
        self,
        tree: MegaContextTree,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[WorkingContextVariant]:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None:
            return None
        embeddings, positions = metadata
        window = self.config.wc_config.max_length
        if embeddings.shape[1] > window:
            embeddings = embeddings[:, -window:]
            positions = positions[:, -window:]
        return self._create_variant(embeddings, positions, lod=0, source="recency_baseline")

    def _build_lod_variant(
        self,
        tree: MegaContextTree,
        lod: int,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[WorkingContextVariant]:
        metadata = self._get_level_metadata_cached(tree, lod, level_cache)
        if metadata is None:
            return None
        embeddings, positions = metadata
        if embeddings.shape[1] == 0:
            return None
        return self._create_variant(embeddings, positions, lod=lod, source=f"lod_{lod}")

    def _build_random_span_variant(
        self,
        tree: MegaContextTree,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[WorkingContextVariant]:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None:
            return None
        embeddings, positions = metadata
        total = embeddings.shape[1]
        window = self.config.wc_config.max_length
        if total <= window:
            return None
        start = self._rng.randint(0, max(0, total - window))
        return self._build_span_variant_from_metadata(embeddings, positions, start, window)

    def _build_span_variant(
        self,
        tree: MegaContextTree,
        start: int,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[WorkingContextVariant]:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None:
            return None
        embeddings, positions = metadata
        window = self.config.wc_config.max_length
        return self._build_span_variant_from_metadata(embeddings, positions, start, window)

    def _build_span_variant_from_metadata(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        start: int,
        window: int,
    ) -> Optional[WorkingContextVariant]:
        total = embeddings.shape[1]
        if total <= window:
            return None
        max_start = max(0, total - window)
        clamped_start = max(0, min(start, max_start))
        end = clamped_start + window
        span_embeddings = embeddings[:, clamped_start:end]
        span_positions = positions[:, clamped_start:end]
        return self._create_variant(
            span_embeddings,
            span_positions,
            lod=0,
            source=f"random_span_{clamped_start}",
        )

    def _sample_random_span_starts(
        self,
        tree: MegaContextTree,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        count: int,
    ) -> List[int]:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None or count <= 0:
            return []
        embeddings, _ = metadata
        total = embeddings.shape[1]
        window = self.config.wc_config.max_length
        if total <= window:
            return []
        max_start = max(0, total - window)
        if max_start == 0:
            return [0] * count
        gen_seed = self._rng.randint(0, 2**31 - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(gen_seed)
        starts = torch.randint(
            0,
            max_start + 1,
            (count,),
            generator=generator,
            device=self.device,
        )
        return [int(s.item()) for s in starts]

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

    def _get_level_metadata_cached(
        self,
        tree: MegaContextTree,
        lod: int,
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if lod in cache:
            return cache[lod]
        try:
            metadata = tree.get_level_metadata(lod)
        except ValueError:
            return None
        cache[lod] = metadata
        return metadata

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
        session_id: str,
    ) -> List[WorkingContextVariant]:
        refined: List[WorkingContextVariant] = []
        for variant in variants:
            allocator = self._build_allocator(tree, variant.working_context)
            variant.allocator = allocator
            scores = self.lensnet(variant.working_context)
            scores_detached = scores.detach()
            variant.lens_scores = scores_detached.clone()
            variant.edits_applied = allocator.update_focus(
                max_replacements_per_iteration=self.config.allocator_max_replacements,
                num_iterations=self.config.allocator_iterations,
                scores=scores_detached,
            )
            refined.append(variant)
            self._log_wc_snapshot(session_id, variant.working_context, variant.source)
            self._log_focus_stats(session_id, variant, f"{variant.source}_focus")
            siblings = self._generate_sibling_variants(tree, variant, session_id)
            for sibling in siblings:
                refined.append(sibling)
                self._log_wc_snapshot(session_id, sibling.working_context, sibling.source)
                self._log_focus_stats(session_id, sibling, f"{sibling.source}_focus")
                if len(refined) >= self.config.max_counterfactuals:
                    return refined
        return refined

    def _generate_sibling_variants(
        self,
        tree: MegaContextTree,
        variant: WorkingContextVariant,
        session_id: str,
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
                self._increment_counter("sibling_expands")
                break
        # Force a collapse on the strongest negative score.
        collapse_indices = torch.argsort(scores, descending=False).tolist()
        for idx in collapse_indices:
            sibling = self._force_collapse_variant(tree, variant, idx)
            if sibling is not None:
                siblings.append(sibling)
                self._increment_counter("sibling_collapses")
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

    def _log_event(self, session_id: str, event_type: str, payload: Dict[str, any]) -> None:
        event = TelemetryEvent(session_id=session_id, event_type=event_type, payload=payload)
        self.telemetry_provider.log_event(event)

    def _reset_batch_counters(self) -> None:
        for key in self._batch_counters:
            self._batch_counters[key] = 0

    def _increment_counter(self, name: str, value: int = 1) -> None:
        self._batch_counters[name] = self._batch_counters.get(name, 0) + int(value)

    def _emit_batch_counters(self, step: int) -> None:
        payload = {"step": int(step)}
        payload.update({k: int(v) for k, v in self._batch_counters.items()})
        self._log_event(f"train_step_{step}", "mc_batch_counters", payload)

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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_losses = []
        lod1_losses = []
        lod2_losses = []
        for sample in batch_states:
            token_loss, lod1_loss, lod2_loss = self._compute_horizon_losses_for_sample(sample)
            if token_loss is not None:
                token_losses.append(token_loss)
            if lod1_loss is not None:
                lod1_losses.append(lod1_loss)
            if lod2_loss is not None:
                lod2_losses.append(lod2_loss)
        agg_token = torch.stack(token_losses).mean() if token_losses else None
        agg_lod1 = torch.stack(lod1_losses).mean() if lod1_losses else None
        agg_lod2 = torch.stack(lod2_losses).mean() if lod2_losses else None
        return agg_token, agg_lod1, agg_lod2

    def _compute_horizon_losses_for_sample(
        self, sample: SampleContext
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_losses = []
        lod1_losses = []
        lod2_losses = []
        for variant in sample.variants:
            token_loss, lod1_loss, lod2_loss = self._evaluate_variant_horizon(sample, variant)
            if token_loss is not None:
                variant.token_loss_value = token_loss
                token_losses.append(token_loss)
            if lod1_loss is not None:
                variant.lod1_loss_value = lod1_loss
                lod1_losses.append(lod1_loss)
            if lod2_loss is not None:
                lod2_losses.append(lod2_loss)
        token_loss = torch.stack(token_losses).mean() if token_losses else None
        lod1_loss = torch.stack(lod1_losses).mean() if lod1_losses else None
        lod2_loss = torch.stack(lod2_losses).mean() if lod2_losses else None
        return token_loss, lod1_loss, lod2_loss

    def _evaluate_variant_horizon(
        self,
        sample: SampleContext,
        variant: WorkingContextVariant,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        base_horizon = self.config.horizon_tokens
        if base_horizon <= 0 or sample.tree.tokens is None:
            return None, None, None
        wc = variant.working_context
        wc_embeddings = wc.to_tensor()
        wc_positions = wc.get_positions()
        last_pos = int(wc_positions[0, -1].item())
        available = sample.tree.tokens.shape[1] - (last_pos + 1)
        lod2_horizon = self.config.block_size * self.config.long_horizon_multiplier
        use_lod2 = available >= lod2_horizon >= base_horizon
        horizon = lod2_horizon if use_lod2 else base_horizon
        horizon_tokens = self._slice_tokens(sample.tree, last_pos + 1, horizon)
        if horizon_tokens is None:
            return None, None, None
        if use_lod2:
            self._increment_counter("horizon_triggers")
            self._log_event(
                sample.session_id,
                "horizon_trigger",
                {"variant": variant.source, "lod": 2, "length": horizon},
            )
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
        lod1_loss, lod2_loss = self._compute_lod_losses(horizon_tokens, horizon_logits, use_lod2=use_lod2)
        return token_loss, lod1_loss, lod2_loss

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

    def _compute_lod_losses(
        self,
        tokens: torch.Tensor,
        horizon_logits: torch.Tensor,
        use_lod2: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        block = self.config.block_size
        if block <= 0:
            return None, None
        horizon_len = tokens.shape[1]
        num_blocks = horizon_len // block
        if num_blocks == 0:
            return None, None
        trim = num_blocks * block
        valid_tokens = tokens[:, :trim]
        mask = valid_tokens != -1
        if not mask.any():
            return None, None
        safe_tokens = valid_tokens.clone()
        safe_tokens[~mask] = 0
        gt_embeddings = self.embed(safe_tokens) * mask.unsqueeze(-1)
        gt_blocks = gt_embeddings.view(num_blocks, block, -1)
        pred_logits = horizon_logits[:, :trim, :]
        top_k = min(self.config.loss_projection_top_k, pred_logits.size(-1))
        probs = F.softmax(pred_logits, dim=-1)
        if top_k < pred_logits.size(-1):
            top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
            embed_weights = self.embed.weight[top_idx]
            pred_embeds = (top_vals.unsqueeze(-1) * embed_weights).sum(dim=-2)
        else:
            pred_embeds = torch.matmul(probs, self.embed.weight)
        pred_blocks = pred_embeds.view(num_blocks, block, -1)
        gt_gists = self.gistnet(gt_blocks)
        pred_gists = self.gistnet(pred_blocks)
        lod1_loss = 1 - F.cosine_similarity(pred_gists, gt_gists, dim=-1)
        lod1_mean = lod1_loss.mean()
        lod2_mean = None
        if use_lod2 and pred_gists.numel() > 0:
            num_lod1 = pred_gists.shape[0]
            lod2_block = self.config.long_horizon_multiplier
            if num_lod1 >= lod2_block:
                trim = (num_lod1 // lod2_block) * lod2_block
                if trim > 0:
                    pred_lod2 = self.gistnet(pred_gists[:trim].view(1, trim, -1)).view(-1, self.config.embed_dim)
                    gt_lod2 = self.gistnet(gt_gists[:trim].view(1, trim, -1)).view(-1, self.config.embed_dim)
                    lod2 = 1 - F.cosine_similarity(pred_lod2, gt_lod2, dim=-1)
                    lod2_mean = lod2.mean()
        return lod1_mean, lod2_mean

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

    # ------------------------------------------------------------------ #
    # Telemetry helpers
    # ------------------------------------------------------------------ #
    def _log_tree_snapshot(self, session_id: str, tree: MegaContextTree, tag: str) -> None:
        payload = {
            "tag": tag,
            "summary": {str(k): [int(v[0]), int(v[1])] for k, v in tree.summary().items()},
            "total_tokens": tree.num_tokens(),
            "max_lod": self.config.max_lod,
            "access": tree.get_access_stats(),
        }
        self._log_event(session_id, "mc_tree_snapshot", payload)

    def _log_wc_snapshot(self, session_id: str, wc: WorkingContext, tag: str) -> None:
        positions = wc.get_positions()[0].tolist()
        lods = wc.get_lod_tensor()[0].tolist()
        payload = {
            "tag": tag,
            "length": wc.length,
            "lods": lods,
            "positions": positions,
            "events": wc.drain_events(),
        }
        self._log_event(session_id, "working_context_snapshot", payload)

    def _log_focus_stats(
        self,
        session_id: str,
        variant: WorkingContextVariant,
        tag: str,
    ) -> None:
        scores = variant.lens_scores
        score_payload = {}
        if scores is not None:
            if scores.dim() > 1:
                scores = scores.squeeze(0)
            score_payload = {
                "score_mean": float(scores.mean().item()),
                "score_std": float(scores.std(unbiased=False).item()),
                "score_max": float(scores.max().item()),
                "score_min": float(scores.min().item()),
            }
        payload = {
            "tag": tag,
            "source": variant.source,
            "lod_hint": variant.lod_hint,
            "edits": variant.edits_applied,
        }
        payload.update(score_payload)
        self._log_event(session_id, "focus_allocator", payload)

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
    def begin_inference_session(
        self,
        initial_tokens: torch.Tensor,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Initialize a persistent MegaContext for inference/autoregressive decoding.
        """
        if initial_tokens.dim() == 1:
            initial_tokens = initial_tokens.unsqueeze(0)
        tokens = initial_tokens.to(self.device)
        session = session_id or f"infer_{uuid.uuid4().hex}"
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
        self._log_tree_snapshot(session, tree, tag="inference_init")
        self._log_wc_snapshot(session, recency_variant.working_context, recency_variant.source)
        self.inference_state = InferenceState(
            session_id=session,
            tree=tree,
            working_context=recency_variant.working_context,
            allocator=allocator,
        )
        return session

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
        self._log_wc_snapshot(self.inference_state.session_id, self.inference_state.working_context, tag="inference_update")
        self._log_focus_stats(
            self.inference_state.session_id,
            WorkingContextVariant(
                working_context=self.inference_state.working_context,
                source="inference",
                lod_hint=0,
                edits_applied=0,
                lens_scores=None,
            ),
            tag="inference_focus",
        )

    def get_inference_working_context(self) -> Optional[WorkingContext]:
        if self.inference_state is None:
            return None
        return self.inference_state.working_context
