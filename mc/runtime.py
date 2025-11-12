from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch
import torch.nn.functional as F

from nanochat.report import get_report

from .config import MCConfig, WorkingContextConfig
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
    batch_index: int = 0
    is_baseline: bool = False


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
    allocator: Optional[FocusAllocatorBase]
    rebuild_max_replacements: int
    rebuild_iterations: int
    refocus_max_replacements: int
    refocus_iterations: int
    refocus_interval: int
    soft_max_length: int
    original_seq_len: int = 0
    prefill_iterations: int = 0
    prefill_replacements: int = 0
    refocus_updates: int = 0
    refocus_iterations_accum: int = 0
    refocus_replacements: int = 0
    steps_since_refocus: int = 0


@dataclass
class MCBatchResult:
    positional_cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    variant_loss: Optional[torch.Tensor]
    lod0_loss: Optional[float]
    lens_loss: Optional[torch.Tensor]
    cached_embeddings: Optional[torch.Tensor]
    positional_caches: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = field(default_factory=dict)
    variants: List["WorkingContextVariant"] = field(default_factory=list)
    delta_mean: Optional[float] = None
    delta_p95: Optional[float] = None
    lod_metrics: Dict[int, float] = field(default_factory=dict)
    lod_counts: Dict[int, int] = field(default_factory=dict)


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
        try:
            self._target_dtype = next(model.parameters()).dtype
        except StopIteration:
            self._target_dtype = torch.float32
        self._rng = random.Random(config.random_seed)
        self._batch_counters = {
            "sibling_expands": 0,
            "sibling_collapses": 0,
            "allocator_edits": 0,
        }
        self._debug_flags: Dict[str, bool] = {}
        self.last_batch_profile: Optional[Dict[str, int]] = None
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for positional encodings")
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
        self._aux_dtype = self._resolve_aux_dtype()
        self.gistnet.to(dtype=self._aux_dtype)
        self.lensnet.to(dtype=self._aux_dtype)
        self.focus_allocator: Optional[FocusAllocatorBase] = None
        self.telemetry = MCTelemetry(interval=config.telemetry_interval)
        self.telemetry_provider = telemetry_provider or NoOpTelemetryProvider()
        self.positional_encoder: Optional[GaussianRoPE] = None
        self._mem_debug = os.getenv("MC_MEMORY_DEBUG", "0").lower() not in {"", "0", "false", "no"}
        self._eval_wc_config = WorkingContextConfig(
            embed_dim=config.embed_dim,
            max_length=config.eval_soft_max_length or config.wc_config.max_length,
            device=config.device,
        )
        self.last_inference_report: Optional[Dict[str, Any]] = None
        self.last_train_report: Optional[Dict[str, Any]] = None
        if config.positional_type:
            if config.positional_type in {"gaussian_lod2d", "gaussian_lod2d_alibi"}:
                raise ValueError(
                    "LOD-2D positional modes require a GPT rotary kernel that supports 2D; "
                    "select one of {simple, gaussian, gaussian_alibi}."
                )
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
        self.last_timings: Dict[str, float] = {}
        self._focus_time_accum: float = 0.0
        self.debug_metrics = config.collect_debug_metrics
        self._timing_device = torch.device(config.device)

    @staticmethod
    def _resolve_embedding_layer(model: torch.nn.Module):
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        raise ValueError("Unable to locate embedding layer on model")

    def _resolve_aux_dtype(self) -> torch.dtype:
        choice = self.config.auxiliary_dtype
        if choice == "fp32":
            return torch.float32
        if choice == "bf16":
            if self.device.type == "cuda" and torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                print("[MegaContext] auxiliary_dtype=bf16 requested but device lacks bf16 support; falling back to fp32", flush=True)
            return torch.float32
        # auto
        if self.device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def process_batch(
        self,
        tokens: torch.Tensor,
        step: int,
        context: str = "train",
    ) -> Optional["MCBatchResult"]:
        """
        Args:
            tokens: [B, T] token ids from nanochat loader.
        Returns:
            Batch result containing positional cache and auxiliary losses.
        """
        self.last_batch_profile = None
        self.last_timings = {}
        self._focus_time_accum = 0.0
        batch_states: List[SampleContext] = []
        total_edits = 0
        tokens_device = tokens.to(self.device)
        self._reset_batch_counters()
        cached_embeddings: List[torch.Tensor] = []
        train_variants: List[WorkingContextVariant] = []
        variant_counts: List[int] = []
        # Build trees sequentially to avoid GPU stream contention and keep allocator edits deterministic across runs
        t_total0 = time.time()
        t_build0 = time.time()
        for idx in range(tokens_device.size(0)):
            seq = tokens_device[idx : idx + 1]
            session_id = f"train_step_{step}_sample_{idx}"
            tree, sample_state, seq_embeds, sample_edits = self._build_tree_sample(seq, session_id)
            for variant in sample_state.variants:
                variant.batch_index = idx
            batch_states.append(sample_state)
            total_edits += sample_edits
            cached_embeddings.append(self._to_model_dtype(seq_embeds))
            variant_counts.append(len(sample_state.variants))
            if context == "train":
                train_variants.extend(sample_state.variants)
            if self.debug_metrics:
                self._log_tree_snapshot(session_id, tree, tag="training_build")
                self.telemetry.log_tree(step, tree)
        t_build1 = time.time()
        self.current_batch_states = batch_states
        if self.debug_metrics:
            self.telemetry.log_focus(step, total_edits)
        t_pos0 = time.time()
        positional_cache = self._build_primary_positional(batch_states) if len(batch_states) == 1 else None
        positional_cache_map = self._build_session_positional(batch_states)
        t_pos1 = time.time()
        if context != "train":
            result = MCBatchResult(
                positional_cache=positional_cache,
                variant_loss=None,
                lod0_loss=None,
                lens_loss=None,
                cached_embeddings=torch.cat(cached_embeddings, dim=0) if cached_embeddings else None,
                positional_caches=positional_cache_map,
                variants=[],
            )
            self.current_batch_states = []
            self._emit_batch_counters(step)
            self.last_train_report = None
            total_ms = (time.time() - t_total0) * 1000.0
            self.last_timings = {
                "build_ms": (t_build1 - t_build0) * 1000.0,
                "positional_ms": (t_pos1 - t_pos0) * 1000.0,
                "variant_ms": 0.0,
                "lens_ms": 0.0,
                "focus_ms": self._focus_time_accum * 1000.0,
                "total_ms": total_ms,
            }
            return result
        t_var0 = time.time()
        variant_loss, lod0_loss, delta_mean, delta_p95, lod_metrics, lod_counts = self._compute_variant_losses(batch_states, tokens_device)
        t_var1 = time.time()
        t_lens0 = time.time()
        lens_loss = self._compute_lens_losses(batch_states)
        t_lens1 = time.time()
        result = MCBatchResult(
            positional_cache=positional_cache,
            variant_loss=variant_loss,
            lod0_loss=lod0_loss,
            lens_loss=lens_loss,
            cached_embeddings=torch.cat(cached_embeddings, dim=0) if cached_embeddings else None,
            positional_caches=positional_cache_map,
            variants=train_variants,
            delta_mean=delta_mean,
            delta_p95=delta_p95,
            lod_metrics=lod_metrics,
            lod_counts=lod_counts,
        )
        self.current_batch_states = []
        self._emit_batch_counters(step)
        self._refresh_train_report(batch_states)
        total_variants = sum(variant_counts) if variant_counts else 0
        self.last_batch_profile = {
            "variant_counts": variant_counts,
            "total_variants": total_variants,
        }
        if delta_mean is not None:
            self._log_event(
                f"train_step_{step}",
                "delta_nll",
                {"mean": float(delta_mean), "p95": float(delta_p95 or delta_mean)},
            )
        if lod_metrics:
            self._log_event(
                f"train_step_{step}",
                "lod_metrics",
                {f"lod_{lod}_loss": float(val) for lod, val in lod_metrics.items()},
            )
        if lod_counts:
            self._log_event(
                f"train_step_{step}",
                "lod_counts",
                {f"lod_{lod}_count": int(val) for lod, val in lod_counts.items()},
            )
        self._log_event(
            f"train_step_{step}",
            "mc_batch_stats",
            {
                "variants_total": total_variants,
                "variants_mean": (total_variants / max(1, len(variant_counts))),
                "variants_max": max(variant_counts) if variant_counts else 0,
                "variants_min": min(variant_counts) if variant_counts else 0,
            },
        )
        total_ms = (time.time() - t_total0) * 1000.0
        self.last_timings = {
            "build_ms": (t_build1 - t_build0) * 1000.0,
            "positional_ms": (t_pos1 - t_pos0) * 1000.0,
            "variant_ms": (t_var1 - t_var0) * 1000.0,
            "lens_ms": (t_lens1 - t_lens0) * 1000.0,
            "focus_ms": self._focus_time_accum * 1000.0,
            "total_ms": total_ms,
        }
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
        if not self.config.cache_lod0:
            tree.release_lod0_cache(disable_future_cache=True)
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
        focus_start = time.time()
        refined = self._refine_variants(tree, variants, session_id=session_id)
        self._focus_time_accum += time.time() - focus_start
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
            # Prefer recency-baseline variant when available; fallback to first
            wc_choice = sample.variants[0]
            for v in sample.variants:
                if v.source.startswith("recency_baseline") or v.lod_hint == 0:
                    wc_choice = v
                    break
            wc = wc_choice.working_context
            if wc_choice.lod_hint == 0:
                self._assert_recent_lod0(wc, f"train_session:{sample.session_id}")
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
        wc_config: Optional[WorkingContextConfig] = None,
    ) -> Optional[WorkingContextVariant]:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None:
            return None
        embeddings, positions = metadata
        window = wc_config.max_length if wc_config is not None else self.config.wc_config.max_length
        if embeddings.shape[1] > window:
            embeddings = embeddings[:, -window:]
            positions = positions[:, -window:]
        variant = self._create_variant(embeddings, positions, lod=0, source="recency_baseline", wc_config=wc_config)
        variant.is_baseline = True
        return variant

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
        # Generate on CPU to avoid device syncs per .item()
        generator = torch.Generator()
        generator.manual_seed(gen_seed)
        starts_cpu = torch.randint(0, max_start + 1, (count,), generator=generator, device=torch.device("cpu"))
        return [int(s) for s in starts_cpu.tolist()]

    def _create_variant(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        lod: int,
        source: str,
        wc_config: Optional[WorkingContextConfig] = None,
    ) -> WorkingContextVariant:
        config = wc_config or self.config.wc_config
        lod_tensor = torch.full(
            (embeddings.shape[0], embeddings.shape[1]),
            lod,
            dtype=torch.long,
            device=embeddings.device,
        )
        wc = WorkingContext(embeddings, positions, config, lod_tensor=lod_tensor)
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
            if variant.is_baseline:
                refined.append(variant)
                self._log_wc_snapshot(session_id, variant.working_context, variant.source)
                continue
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
        *,
        soft_max_length: Optional[int] = None,
        recent_tokens: Optional[int] = None,
    ) -> FocusAllocatorBase:
        soft_length = soft_max_length or self.config.soft_max_length
        recent = self.config.allocator_recent_tokens if recent_tokens is None else recent_tokens
        allocator_cfg = FocusAllocatorConfig(
            block_size=self.config.block_size,
            max_lod=self.config.max_lod,
            soft_max_length=soft_length,
            recent_tokens=recent,
            expand_threshold=self.config.allocator_expand_threshold,
            collapse_threshold=self.config.allocator_collapse_threshold,
            sample_top_k=self.config.allocator_sample_top_k,
            sample_temperature=self.config.allocator_sample_temperature,
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

    def _log_event(self, session_id: str, event_type: str, payload: Dict[str, any], *, force: bool = False) -> None:
        if not (self.debug_metrics or force):
            return
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
        self._log_event(f"train_step_{step}", "mc_batch_counters", payload, force=True)

    def _log_memory(self, tag: str) -> None:
        if not self._mem_debug:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            print(f"[MegaContext][mem] {tag}: device={self.device} (non-CUDA)", flush=True)
            return
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_alloc = torch.cuda.max_memory_allocated(self.device)
        print(
            f"[MegaContext][mem] {tag}: alloc={allocated / 1e9:.2f}GB "
            f"reserved={reserved / 1e9:.2f}GB max_alloc={max_alloc / 1e9:.2f}GB",
            flush=True,
        )

    def _timing_sync(self) -> None:
        if self._timing_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._timing_device)

    def _assert_recent_lod0(self, wc: WorkingContext, tag: str) -> None:
        recent = int(self.config.allocator_recent_tokens)
        if recent <= 0 or wc.length == 0:
            return
        window = min(recent, wc.length)
        lods = wc.get_lod_tensor()
        if lods.shape[0] == 0:
            return
        tail = lods[0, -window:]
        if torch.any(tail != 0):
            raise RuntimeError(
                "[MegaContext] Recent tokens must remain LOD0 "
                f"(context={tag}, tail={window}). Observed LODs: {tail.tolist()}"
            )

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

    def _compute_variant_losses(
        self,
        batch_states: List[SampleContext],
        original_tokens: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[float], Optional[float], Dict[int, float], Dict[int, int]]:
        entries = self._prepare_variant_entries(batch_states, original_tokens)
        if not entries:
            return None, None, None, None, {}, {}
        losses = []
        for seq_len, group in entries.items():
            losses.extend(self._run_variant_batch(group, seq_len))
        if not losses:
            return None, None, None, None, {}, {}
        loss_tensor = torch.stack(losses)
        lod0_loss = None
        delta_values: List[float] = []
        lod_buckets: Dict[int, List[float]] = {}
        for sample in batch_states:
            baseline = None
            for variant in sample.variants:
                if variant.lod_hint == 0 and variant.token_loss_value is not None:
                    baseline = float(variant.token_loss_value.detach())
                    lod0_loss = baseline if lod0_loss is None else lod0_loss
                    break
            for variant in sample.variants:
                if variant.token_loss_value is None:
                    continue
                val = float(variant.token_loss_value.detach())
                lod_buckets.setdefault(variant.lod_hint, []).append(val)
                if baseline is not None and variant is not None and variant.lod_hint != 0:
                    delta_values.append(val - baseline)
        delta_mean = float(torch.tensor(delta_values).mean().item()) if delta_values else None
        delta_p95 = (
            float(torch.quantile(torch.tensor(delta_values), 0.95).item())
            if delta_values
            else None
        )
        lod_metrics = {
            lod: float(torch.tensor(vals).mean().item())
            for lod, vals in lod_buckets.items()
            if vals
        }
        lod_counts = {lod: len(vals) for lod, vals in lod_buckets.items()}
        return loss_tensor.mean(), lod0_loss, delta_mean, delta_p95, lod_metrics, lod_counts

    def _prepare_variant_entries(
        self,
        batch_states: List[SampleContext],
        original_tokens: torch.Tensor,
    ) -> Dict[int, List[Dict[str, Any]]]:
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for sample in batch_states:
            for variant in sample.variants:
                wc = variant.working_context
                embeddings = self._to_model_dtype(wc.to_tensor())
                seq_len = embeddings.shape[1]
                cos, sin, alibi = wc.get_positional_encodings()
                entry = {
                    "variant": variant,
                    "embeddings": embeddings,
                    "cos": self._to_model_dtype(cos),
                    "sin": self._to_model_dtype(sin),
                    "alibi": self._to_model_dtype(alibi),
                    "seq_len": seq_len,
                    "batch_idx": getattr(variant, "batch_index", 0),
                    "original_tokens": original_tokens,
                }
                groups.setdefault(seq_len, []).append(entry)
        return groups

    def _run_variant_batch(
        self,
        group: List[Dict[str, Any]],
        seq_len: int,
    ) -> List[torch.Tensor]:
        batch_size = len(group)
        embeddings = torch.cat([entry["embeddings"] for entry in group], dim=0)
        cos = None
        sin = None
        alibi = None
        if all(entry["cos"] is not None for entry in group):
            cos = torch.cat([entry["cos"] for entry in group], dim=0)
        if all(entry["sin"] is not None for entry in group):
            sin = torch.cat([entry["sin"] for entry in group], dim=0)
        if any(entry["alibi"] is not None for entry in group):
            alibi_list = [
                entry["alibi"]
                if entry["alibi"] is not None
                else torch.zeros((1, self.config.num_heads, seq_len, seq_len), dtype=self._target_dtype, device=self.device)
                for entry in group
            ]
            alibi = torch.cat(alibi_list, dim=0)
        token_batch = torch.cat(
            [
                self._align_tokens_to_embeddings(
                    entry["original_tokens"][entry["batch_idx"] : entry["batch_idx"] + 1].to(torch.long),
                    seq_len,
                )
                for entry in group
            ],
            dim=0,
        )
        dummy_idx = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        autocast_ctx = nullcontext()
        if self.device.type == "cuda" and self._target_dtype in (torch.bfloat16, torch.float16):
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=self._target_dtype)
        with autocast_ctx:
            loss2d = self.model(
                dummy_idx,
                token_batch,
                loss_reduction="none",
                cos_sin_override=(cos, sin) if cos is not None and sin is not None else None,
                alibi_override=alibi,
                inputs_embeds=embeddings,
            )
        if loss2d.dim() > 2:
            logits = loss2d
            logits = logits.view(batch_size, seq_len, -1)
            tokens_flat = token_batch.view(batch_size, seq_len)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens_flat.view(-1),
                ignore_index=-1,
                reduction="none",
            )
            loss2d = loss_flat.view(batch_size, seq_len)
        else:
            loss2d = loss2d.view(batch_size, -1)
        valid = (token_batch.view(batch_size, -1) >= 0).to(loss2d.dtype)
        sample_losses = (loss2d * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        outputs = []
        for entry, loss in zip(group, sample_losses):
            entry["variant"].token_loss_value = loss.detach()
            outputs.append(loss)
        return outputs

    def _run_variant_forward(
        self,
        variant: WorkingContextVariant,
        original_tokens: torch.Tensor,
    ) -> torch.Tensor:
        wc = variant.working_context
        embeddings = self._to_model_dtype(wc.to_tensor())
        seq_len = embeddings.shape[1]
        cos, sin, alibi = wc.get_positional_encodings()
        cos = self._to_model_dtype(cos)
        sin = self._to_model_dtype(sin)
        alibi = self._to_model_dtype(alibi)
        batch_idx = getattr(variant, "batch_index", 0)
        token_slice = original_tokens[batch_idx : batch_idx + 1].to(torch.long)
        token_slice = self._align_tokens_to_embeddings(token_slice, seq_len)
        dummy_idx = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
        autocast_ctx = nullcontext()
        if self.device.type == "cuda" and self._target_dtype in (torch.bfloat16, torch.float16):
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=self._target_dtype)
        with autocast_ctx:
            return self.model(
                dummy_idx,
                token_slice,
                cos_sin_override=(cos, sin),
                alibi_override=alibi,
                inputs_embeds=embeddings,
            )

    def _align_tokens_to_embeddings(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] >= target_len:
            return tokens[:, -target_len:]
        pad = torch.full(
            (tokens.shape[0], target_len - tokens.shape[1]),
            -1,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([pad, tokens], dim=1)

    def _to_model_dtype(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.to(device=self.device, dtype=self._target_dtype)

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
                # Recompute scores here to allow gradients to flow into LensNet
                scores_live = self.lensnet(variant.working_context)
                if scores_live.dim() > 1:
                    scores_1d = scores_live.squeeze(0)
                else:
                    scores_1d = scores_live
                targets = self._build_lens_targets(variant, best_map, scores_1d)
                base_loss = F.mse_loss(scores_1d, targets, reduction='mean')
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
        if not self.debug_metrics:
            return
        payload = {
            "tag": tag,
            "summary": {str(k): [int(v[0]), int(v[1])] for k, v in tree.summary().items()},
            "total_tokens": tree.num_tokens(),
            "max_lod": self.config.max_lod,
            "access": tree.get_access_stats(),
        }
        self._log_event(session_id, "mc_tree_snapshot", payload)

    def _log_wc_snapshot(self, session_id: str, wc: WorkingContext, tag: str) -> None:
        if not self.debug_metrics:
            return
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
        *,
        force: bool = False,
    ) -> None:
        if not self.debug_metrics and not force:
            return
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
        # Attach swap-rate and token-budget utilization if allocator exposed stats
        stats: Dict[str, float] = {}
        if variant.allocator is not None and hasattr(variant.allocator, "_last_edit_stats"):
            stats = getattr(variant.allocator, "_last_edit_stats") or {}
        wc_len = max(1, int(stats.get("wc_length", variant.working_context.length)))
        swap_rate = float(stats.get("total", 0)) / float(wc_len)
        payload.update({
            "swap_rate": swap_rate,
            "num_expand": int(stats.get("expand", 0)),
            "num_collapse": int(stats.get("collapse", 0)),
            "wc_length": int(wc_len),
            "utilization": float(variant.working_context.length) / float(self.config.wc_config.max_length),
            "residency_mean": float(stats.get("residency_mean", 0.0)),
            "residency_p95": float(stats.get("residency_p95", 0.0)),
        })
        payload.update(score_payload)
        self._log_event(session_id, "focus_allocator", payload, force=force)

    def _refresh_inference_report(self) -> None:
        if self.inference_state is None:
            self.last_inference_report = None
            return
        wc = self.inference_state.working_context
        lod_hist = self._lod_histogram(wc)
        coverage = self._wc_token_coverage(wc, self.inference_state.tree)
        hist_equiv = self._lod_equivalent_tokens_from_hist(lod_hist)
        if coverage != hist_equiv:
            raise RuntimeError(
                "[MegaContext] Inference LOD coverage mismatch: "
                f"hist_equiv={hist_equiv}, coverage={coverage}"
            )
        report = {
            "original_length": int(self.inference_state.original_seq_len),
            "wc_length": int(wc.length),
            "lod_counts": lod_hist,
            "prefill_iterations": int(self.inference_state.prefill_iterations),
            "prefill_replacements": int(self.inference_state.prefill_replacements),
            "refocus_updates": int(self.inference_state.refocus_updates),
            "refocus_iterations": int(self.inference_state.refocus_iterations_accum),
            "refocus_replacements": int(self.inference_state.refocus_replacements),
            "coverage_tokens": int(coverage),
        }
        self.last_inference_report = report

    def _refresh_train_report(self, batch_states: List[SampleContext]) -> None:
        if not batch_states:
            self.last_train_report = None
            return
        primary_variant: Optional[WorkingContextVariant] = None
        primary_tree_tokens = 0
        primary_tree: Optional[MegaContextTree] = None
        # Prefer the variant with the highest available LOD across all samples.
        best_lod = -1
        for sample in batch_states:
            for variant in sample.variants:
                if variant.is_baseline:
                    continue
                lod_hist = self._lod_histogram(variant.working_context)
                highest_variant_lod = max(lod_hist.keys()) if lod_hist else variant.lod_hint
                if highest_variant_lod > best_lod:
                    best_lod = highest_variant_lod
                    primary_variant = variant
                    primary_tree_tokens = sample.tree.num_tokens()
                    primary_tree = sample.tree
            if best_lod == self.config.max_lod:
                break
        if primary_variant is None:
            sample = batch_states[0]
            for variant in sample.variants:
                if not variant.is_baseline:
                    primary_variant = variant
                    primary_tree_tokens = sample.tree.num_tokens()
                    primary_tree = sample.tree
                    break
            if primary_variant is None and sample.variants:
                primary_variant = sample.variants[0]
                primary_tree_tokens = sample.tree.num_tokens()
                primary_tree = sample.tree
        aggregate_counts: Dict[int, int] = {}
        aggregate_lengths = 0
        aggregate_variants = 0
        aggregate_hist_equiv = 0
        aggregate_coverage = 0
        aggregate_expected = 0
        for sample in batch_states:
            expected = min(sample.tree.num_tokens(), self.config.wc_config.max_length)
            for variant in sample.variants:
                lod_hist = self._lod_histogram(variant.working_context)
                for lod, count in lod_hist.items():
                    aggregate_counts[lod] = aggregate_counts.get(lod, 0) + count
                aggregate_lengths += variant.working_context.length
                aggregate_variants += 1
                hist_equiv = self._lod_equivalent_tokens_from_hist(lod_hist)
                coverage = self._wc_token_coverage(variant.working_context, sample.tree)
                if hist_equiv != coverage:
                    raise RuntimeError(
                        "[MegaContext] Aggregate LOD coverage mismatch: "
                        f"hist_equiv={hist_equiv}, coverage={coverage}"
                    )
                aggregate_hist_equiv += hist_equiv
                aggregate_coverage += coverage
                aggregate_expected += expected
        primary_report = None
        if primary_variant is not None and primary_tree is not None:
            wc = primary_variant.working_context
            stats = getattr(primary_variant.allocator, "_last_edit_stats", {}) or {}
            lod_hist = self._lod_histogram(wc)
            primary_cov = self._wc_token_coverage(wc, primary_tree)
            primary_hist = self._lod_equivalent_tokens_from_hist(lod_hist)
            if primary_hist != primary_cov:
                raise RuntimeError(
                    "[MegaContext] Primary LOD coverage mismatch: "
                    f"hist_equiv={primary_hist}, coverage={primary_cov}"
                )
            primary_report = {
                "original_length": int(primary_tree_tokens),
                "wc_length": int(wc.length),
                "lod_counts": lod_hist,
                "focus_iterations": int(stats.get("iterations", 0)),
                "focus_replacements": int(stats.get("total", 0)),
                "expected_tokens": min(primary_tree_tokens, self.config.wc_config.max_length),
                "coverage_tokens": primary_hist,
            }
        aggregate_report = {
            "variants": int(aggregate_variants),
            "avg_wc_length": float(aggregate_lengths / aggregate_variants) if aggregate_variants else 0.0,
            "lod_counts": aggregate_counts,
            "expected_tokens": int(aggregate_expected),
            "coverage_tokens": int(aggregate_hist_equiv),
        }
        self.last_train_report = {
            "primary": primary_report,
            "aggregate": aggregate_report,
        }

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

    def _lod_histogram(self, wc: WorkingContext) -> Dict[int, int]:
        lods = wc.get_lod_tensor()[0]
        values, counts = torch.unique(lods, return_counts=True)
        return {int(v.item()): int(c.item()) for v, c in zip(values, counts)}

    def _wc_token_coverage(self, wc: WorkingContext, tree: MegaContextTree) -> int:
        positions = wc.get_positions()[0]
        lods = wc.get_lod_tensor()[0]
        total_tokens = tree.num_tokens()
        if total_tokens <= 0:
            return 0
        covered = torch.zeros(total_tokens, dtype=torch.bool)
        for pos_tensor, lod_tensor in zip(positions, lods):
            start = int(pos_tensor.item())
            lod = int(lod_tensor.item())
            span = max(1, tree.tokens_per_entry(lod))
            if start >= total_tokens:
                continue
            end = min(total_tokens, start + span)
            covered[start:end] = True
        return int(covered.sum().item())

    def _lod_equivalent_tokens_from_hist(self, hist: Dict[int, int]) -> int:
        total = 0
        for lod, count in hist.items():
            total += int(count) * (self.config.block_size ** int(lod))
        return total

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
            old_count=1,
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
            old_count=block,
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
        rebuild: bool = True,
    ) -> str:
        """
        Initialize a persistent MegaContext for inference/autoregressive decoding.
        """
        t_total0 = time.time()
        timings: Dict[str, float] = {}
        if initial_tokens.dim() == 1:
            initial_tokens = initial_tokens.unsqueeze(0)
        tokens = initial_tokens.to(self.device)
        original_len = int(tokens.shape[1])
        eval_soft_max = self.config.eval_soft_max_length or self.config.wc_config.max_length
        fits_soft_max = original_len <= eval_soft_max
        session = session_id or f"infer_{uuid.uuid4().hex}"
        self._timing_sync()
        t_tree0 = time.time()
        with torch.no_grad():
            tree = build_mega_context(
                self.config.mc_tree_type,
                tokens,
                self.embed,
                self.config.tree_config,
                gistnet=self.gistnet,
                lazy_levels=fits_soft_max,
            )
        self._timing_sync()
        timings["tree_build_ms"] = (time.time() - t_tree0) * 1000.0
        if not self.config.cache_lod0:
            tree.release_lod0_cache(disable_future_cache=True)
        # Build a recency-based working context with a local level cache,
        # mirroring the training path.
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._timing_sync()
        t_variant0 = time.time()
        recency_variant = self._build_recency_variant(tree, level_cache, wc_config=self._eval_wc_config)
        self._timing_sync()
        timings["recency_variant_ms"] = (time.time() - t_variant0) * 1000.0
        if recency_variant is None:
            raise ValueError("Unable to build initial working context for inference")
        fits_soft_max = fits_soft_max and (recency_variant.working_context.length <= eval_soft_max)
        allocator: Optional[FocusAllocatorBase] = None
        alloc_init_ms = 0.0
        if not fits_soft_max:
            self._timing_sync()
            t_alloc0 = time.time()
            allocator = self._build_allocator(
                tree,
                recency_variant.working_context,
                soft_max_length=eval_soft_max,
            )
            self._timing_sync()
            alloc_init_ms = (time.time() - t_alloc0) * 1000.0
        timings["allocator_init_ms"] = alloc_init_ms
        rebuild_repl = self.config.infer_rebuild_max_replacements
        if rebuild_repl is None:
            rebuild_repl = self.config.infer_allocator_max_replacements
        if rebuild_repl is None:
            rebuild_repl = self.config.allocator_max_replacements
        rebuild_repl = int(max(0, rebuild_repl or 0))
        rebuild_iters = self.config.infer_rebuild_iterations
        if rebuild_iters is None:
            rebuild_iters = self.config.infer_allocator_iterations
        if rebuild_iters is None:
            rebuild_iters = self.config.allocator_iterations
        rebuild_iters = int(max(0, rebuild_iters or 0))
        refocus_repl = self.config.infer_allocator_max_replacements
        if refocus_repl is None:
            refocus_repl = self.config.allocator_max_replacements
        refocus_repl = int(max(0, refocus_repl or 0))
        refocus_iters = self.config.infer_allocator_iterations
        if refocus_iters is None:
            refocus_iters = self.config.allocator_iterations
        refocus_iters = int(max(0, refocus_iters or 0))
        refocus_interval = self.config.infer_refocus_interval
        prefill_iterations = 0
        prefill_replacements = 0
        telemetry_time = 0.0
        if allocator is not None and rebuild:
            self._timing_sync()
            t_rebuild0 = time.time()
            allocator.rebuild(
                max_replacements_per_iteration=rebuild_repl,
                num_iterations=rebuild_iters,
            )
            stats = getattr(allocator, "_last_edit_stats", {}) or {}
            prefill_iterations = int(stats.get("iterations", 0))
            prefill_replacements = int(stats.get("total", 0))
            self._timing_sync()
            timings["allocator_rebuild_ms"] = (time.time() - t_rebuild0) * 1000.0
        else:
            timings["allocator_rebuild_ms"] = 0.0
        if self.debug_metrics:
            t_tel = time.time()
            self._log_tree_snapshot(session, tree, tag="inference_init")
            self._log_wc_snapshot(session, recency_variant.working_context, recency_variant.source)
            self._timing_sync()
            telemetry_time += time.time() - t_tel
        self._timing_sync()
        t_state0 = time.time()
        self.inference_state = InferenceState(
            session_id=session,
            tree=tree,
            working_context=recency_variant.working_context,
            allocator=allocator,
            rebuild_max_replacements=rebuild_repl,
            rebuild_iterations=rebuild_iters,
            refocus_max_replacements=refocus_repl,
            refocus_iterations=refocus_iters,
            refocus_interval=refocus_interval,
            soft_max_length=eval_soft_max,
            original_seq_len=original_len,
            prefill_iterations=prefill_iterations,
            prefill_replacements=prefill_replacements,
        )
        self._timing_sync()
        timings["state_init_ms"] = (time.time() - t_state0) * 1000.0
        self._refresh_inference_report()
        self._timing_sync()
        total_ms = (time.time() - t_total0) * 1000.0
        timings["telemetry_ms"] = telemetry_time * 1000.0
        timings["total_ms"] = total_ms
        timings["early_exit"] = 1.0 if fits_soft_max else 0.0
        self.last_timings = timings
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
        state = self.inference_state
        with torch.no_grad():
            embeddings = self.embed(tokens)
            if state.allocator is None:
                self._inference_append_without_allocator(state, tokens, embeddings)
            else:
                state.allocator.append(tokens, embeddings)
                state.steps_since_refocus += tokens.shape[1]
                if (
                    state.refocus_max_replacements > 0
                    and state.refocus_iterations > 0
                    and state.steps_since_refocus >= state.refocus_interval
                ):
                    state.allocator.update_focus(
                        max_replacements_per_iteration=state.refocus_max_replacements,
                        num_iterations=state.refocus_iterations,
                    )
                    stats = getattr(state.allocator, "_last_edit_stats", {}) or {}
                    state.refocus_updates += 1
                    state.refocus_iterations_accum += int(stats.get("iterations", 0))
                    state.refocus_replacements += int(stats.get("total", 0))
                    state.steps_since_refocus = 0
        self._log_wc_snapshot(state.session_id, state.working_context, tag="inference_update")
        self._log_focus_stats(
            state.session_id,
            WorkingContextVariant(
                working_context=state.working_context,
                source="inference",
                lod_hint=0,
                edits_applied=0,
                lens_scores=None,
                allocator=state.allocator,
            ),
            tag="inference_focus",
            force=True,
        )
        self._refresh_inference_report()

    def _inference_append_without_allocator(
        self,
        state: InferenceState,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        append_len = embeddings.shape[1]
        state.tree.append_with_embeddings(tokens, embeddings)
        start_pos = max(0, state.tree.num_tokens() - append_len)
        for offset in range(append_len):
            slice_embed = embeddings[:, offset, :].contiguous()
            state.working_context.append(
                slice_embed,
                lod=0,
                global_position=start_pos + offset,
            )
        state.steps_since_refocus += append_len
        if state.working_context.length > state.soft_max_length:
            self._bootstrap_inference_allocator(state)

    def _bootstrap_inference_allocator(self, state: InferenceState) -> None:
        if state.allocator is not None:
            return
        allocator = self._build_allocator(
            state.tree,
            state.working_context,
            soft_max_length=state.soft_max_length,
        )
        state.allocator = allocator
        if state.rebuild_iterations > 0 and state.rebuild_max_replacements > 0:
            allocator.rebuild(
                max_replacements_per_iteration=state.rebuild_max_replacements,
                num_iterations=state.rebuild_iterations,
            )
            stats = getattr(allocator, "_last_edit_stats", {}) or {}
            state.prefill_iterations = int(stats.get("iterations", 0))
            state.prefill_replacements = int(stats.get("total", 0))
        else:
            state.prefill_iterations = 0
            state.prefill_replacements = 0
        state.steps_since_refocus = 0
        self._refresh_inference_report()

    def get_inference_working_context(self) -> Optional[WorkingContext]:
        if self.inference_state is None:
            return None
        wc = self.inference_state.working_context
        self._assert_recent_lod0(wc, "inference")
        return wc

    def get_inference_report(self) -> Optional[Dict[str, Any]]:
        return self.last_inference_report

    def get_training_report(self) -> Optional[Dict[str, Any]]:
        return self.last_train_report
