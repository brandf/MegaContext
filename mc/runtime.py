from __future__ import annotations

import math
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
from .working_context import WorkingContext
from .gaussian_rope import build_positional, GaussianRoPE
from .telemetry import TelemetryEvent, TelemetryProvider, NoOpTelemetryProvider


@dataclass
class WorkingContextVariant:
    working_context: WorkingContext
    source: str
    lod_hint: int
    edits_applied: int = 0
    allocator: Optional[FocusAllocatorBase] = None
    policy_scores: Optional[torch.Tensor] = None
    token_loss_value: Optional[torch.Tensor] = None
    lod1_loss_value: Optional[torch.Tensor] = None
    batch_index: int = 0
    is_baseline: bool = False
    adv_delta: float = 0.0
    norm_adv_delta: float = 0.0


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
    adv_delta_mean: Optional[float] = None
    adv_delta_p95: Optional[float] = None
    lod_metrics: Dict[int, float] = field(default_factory=dict)
    lod_counts: Dict[int, int] = field(default_factory=dict)
    preference_corr_mean: Optional[float] = None
    preference_corr_max: Optional[float] = None
    preference_corr_min: Optional[float] = None
    preference_corr_mean_valid: bool = False
    preference_corr_max_valid: bool = False
    preference_corr_min_valid: bool = False
    preference_pair_count: int = 0
    adv_delta_std: Optional[float] = None
    preference_agreement: Optional[float] = None
    policy_score_abs_mean: Optional[float] = None
    policy_score_std_mean: Optional[float] = None


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

    _LOD_CHAR_MAP = {
        0: ".",
        1: "─",
        2: "═",
        3: "█",
    }
    _LOD_PARTIAL_CHAR = ","

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
        rank = os.getenv("RANK")
        local_rank = os.getenv("LOCAL_RANK")
        self._is_rank0 = True
        if rank not in (None, "", "0"):
            self._is_rank0 = False
        if local_rank not in (None, "", "0"):
            self._is_rank0 = False
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
            block_size=config.block_size,
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
        self._last_preference_corr: Dict[str, float] = {}
        self._last_preference_corr_valid: Dict[str, bool] = {}
        self._last_preference_pair_count: int = 0
        self._last_policy_stats: Dict[str, float] = {}
        self._last_preference_agreement: Optional[float] = None
        self._policy_history: Dict[int, torch.Tensor] = {}
        self._adv_norm_mean: float = 0.0
        self._adv_norm_var: float = 1.0
        self._adv_norm_initialized: bool = False
        self._budget_mass_ema: float = 0.0
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
        self.debug_metrics = config.collect_debug_metrics
        self._timing_device = torch.device(config.device)
        self.total_train_steps = max(1, int(config.total_train_steps))
        self._train_progress = 0.0

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
        self._update_train_progress(step, context)
        self.last_batch_profile = None
        self.last_timings = {}
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
                "focus_ms": 0.0,
                "total_ms": total_ms,
            }
            return result
        t_var0 = time.time()
        (
            variant_loss,
            lod0_loss,
            adv_delta_mean,
            adv_delta_p95,
            adv_delta_std,
            lod_metrics,
            lod_counts,
        ) = self._compute_variant_losses(batch_states, tokens_device)
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
            adv_delta_mean=adv_delta_mean,
            adv_delta_p95=adv_delta_p95,
            adv_delta_std=adv_delta_std,
            lod_metrics=lod_metrics,
            lod_counts=lod_counts,
        )
        if getattr(self, "_last_preference_corr", None):
            result.preference_corr_mean = self._last_preference_corr.get("preference_corr_mean")
            result.preference_corr_max = self._last_preference_corr.get("preference_corr_max")
            result.preference_corr_min = self._last_preference_corr.get("preference_corr_min")
            valid_map = getattr(self, "_last_preference_corr_valid", {})
            result.preference_corr_mean_valid = bool(valid_map.get("preference_corr_mean_valid", False))
            result.preference_corr_max_valid = bool(valid_map.get("preference_corr_max_valid", False))
            result.preference_corr_min_valid = bool(valid_map.get("preference_corr_min_valid", False))
            result.preference_pair_count = int(getattr(self, "_last_preference_pair_count", 0))
            def _fmt_corr(val: Optional[float], valid: bool) -> str:
                if not valid:
                    return "n/a"
                return f"{val:.3f}" if val is not None else "n/a"

            print(
                "[MegaContext][PreferenceSummary] pairs=%d corr_mean=%s corr_max=%s corr_min=%s"
                % (
                    result.preference_pair_count,
                    _fmt_corr(result.preference_corr_mean, result.preference_corr_mean_valid),
                    _fmt_corr(result.preference_corr_max, result.preference_corr_max_valid),
                    _fmt_corr(result.preference_corr_min, result.preference_corr_min_valid),
                ),
                flush=True,
            )
        else:
            result.preference_pair_count = int(getattr(self, "_last_preference_pair_count", 0))
        result.preference_agreement = self._last_preference_agreement
        if getattr(self, "_last_policy_stats", None):
            result.policy_score_abs_mean = self._last_policy_stats.get("score_abs_mean")
            result.policy_score_std_mean = self._last_policy_stats.get("score_std_mean")
        self.current_batch_states = []
        self._emit_batch_counters(step)
        self._refresh_train_report(batch_states)
        self._log_train_lod_ascii(step, batch_states)
        self._log_preference_debug(batch_states)
        total_variants = sum(variant_counts) if variant_counts else 0
        self.last_batch_profile = {
            "variant_counts": variant_counts,
            "total_variants": total_variants,
        }
        if adv_delta_mean is not None:
            self._log_event(
                f"train_step_{step}",
                "delta_nll",
                {"mean": float(adv_delta_mean), "p95": float(adv_delta_p95 or adv_delta_mean)},
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
            "focus_ms": 0.0,
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
        random_variants = self._build_random_variant_set(tree, session_id=session_id)
        return SampleContext(session_id=session_id, tree=tree, variants=random_variants)

    def _build_random_variant_set(self, tree: MegaContextTree, session_id: str) -> List[WorkingContextVariant]:
        variants: List[WorkingContextVariant] = []
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        planned_target_len = min(self._current_target_wc_length(), tree.num_tokens())
        baseline = self._build_trimmed_baseline_variant(tree, level_cache, planned_target_len)
        if baseline is not None:
            baseline.source = "lod_0_baseline"
            if tree.num_tokens() > planned_target_len:
                self._normalize_wc_length(baseline.working_context, tree, planned_target_len)
            variants.append(baseline)
        seed = baseline
        if seed is None:
            return variants
        target_len = baseline.working_context.length
        if tree.num_tokens() <= target_len:
            return variants[:1]
        limit = max(1, self.config.max_counterfactuals)
        seed_len = seed.working_context.length
        seen_signatures: set[Tuple[Tuple[int, int], ...]] = set()
        seen_signatures.add(self._variant_signature(seed.working_context))
        for idx in range(self.config.num_random_variants):
            variant_target = self._sample_variant_target_length(seed_len, target_len)
            variant = self._build_random_variant(tree, seed, variant_target, idx)
            if variant is None:
                continue
            self._normalize_wc_length(variant.working_context, tree, target_len)
            signature = self._variant_signature(variant.working_context)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            variants.append(variant)
            if len(variants) >= limit:
                break
        if len(variants) < limit and self.config.num_random_variants > 0:
            aggressive_target = max(self.config.block_size, target_len // 2)
            extra_variant = self._build_random_variant(
                tree,
                seed,
                aggressive_target,
                self.config.num_random_variants,
            )
            if extra_variant is not None:
                signature = self._variant_signature(extra_variant.working_context)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    variants.append(extra_variant)
        return variants[:limit]

    def _build_random_variant(
        self,
        tree: MegaContextTree,
        seed: WorkingContextVariant,
        target_len: int,
        index: int,
    ) -> Optional[WorkingContextVariant]:
        if seed.working_context.length == 0:
            return None
        wc = self._clone_working_context(seed.working_context)
        allocator = self._build_allocator(
            tree,
            wc,
            soft_max_length=target_len,
        )
        tensor_device = wc.to_tensor().device
        generator = self._make_torch_generator(tensor_device)
        max_iterations = max(1, int(self.config.random_variant_iterations))
        total_edits = 0
        for _ in range(max_iterations):
            length = wc.length
            if length == 0:
                break
            scores = torch.randn(
                1,
                length,
                device=tensor_device,
                generator=generator,
            )
            edits = allocator.update_focus(
                max_replacements_per_iteration=self.config.allocator_max_replacements,
                num_iterations=1,
                scores=scores,
            )
            total_edits += edits
            if wc.length <= target_len:
                break
        lod_hint = self._infer_variant_lod_hint(wc)
        return WorkingContextVariant(
            working_context=wc,
            source=f"random_variant_{index}",
            lod_hint=lod_hint,
            edits_applied=total_edits,
            allocator=allocator,
        )

    def _sample_variant_target_length(self, seed_len: int, curriculum_target: int) -> int:
        if seed_len <= self.config.block_size:
            return seed_len
        upper_bound = min(seed_len - self.config.block_size, curriculum_target)
        if upper_bound <= self.config.block_size:
            return max(self.config.block_size, upper_bound)
        min_ratio = 0.25
        max_ratio = 0.9
        frac = self._rng.uniform(min_ratio, max_ratio)
        sampled = int(max(self.config.block_size, seed_len * frac))
        sampled = min(sampled, upper_bound)
        return max(self.config.block_size, sampled)

    def _variant_signature(self, wc: WorkingContext) -> Tuple[Tuple[int, int], ...]:
        positions = wc.get_positions()[0].tolist()
        lods = wc.get_lod_tensor()[0].tolist()
        return tuple(zip(positions, lods))

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
                if v.source.startswith("lod_0_baseline") or v.lod_hint == 0:
                    wc_choice = v
                    break
            wc = wc_choice.working_context
            if wc_choice.lod_hint == 0:
                self._assert_recent_lod0(wc, sample.tree, f"train_session:{sample.session_id}")
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

    def _build_trimmed_baseline_variant(
        self,
        tree: MegaContextTree,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        target_len: int,
    ) -> Optional[WorkingContextVariant]:
        total_tokens = tree.num_tokens()
        if total_tokens <= 0:
            return None
        target_len = max(1, min(target_len, total_tokens))
        tail_tokens = min(total_tokens, int(self.config.allocator_recent_tokens))
        tail_tokens = min(tail_tokens, target_len)
        if tail_tokens <= 0 and target_len > 0:
            tail_tokens = min(target_len, self.config.block_size)
        tail_tokens = min(tail_tokens, total_tokens)

        head_tokens = max(0, total_tokens - tail_tokens)
        head_budget = max(0, target_len - tail_tokens)

        segments_embeddings: List[torch.Tensor] = []
        segments_positions: List[torch.Tensor] = []
        segments_lods: List[torch.Tensor] = []

        def append_segment(tensor: torch.Tensor, positions: torch.Tensor, lod: int) -> None:
            segments_embeddings.append(tensor)
            segments_positions.append(positions.long())
            lod_tensor = torch.full(
                (tensor.shape[0], tensor.shape[1]),
                lod,
                dtype=torch.long,
                device=tensor.device,
            )
            segments_lods.append(lod_tensor)

        def append_lod_range(start: int, end: int, lod: int) -> None:
            if end <= start:
                return
            if lod == 0:
                slice_embeddings = tree.get_lod0_slice(start, end)
                slice_positions = tree.get_positions_for_lod(0)[:, start:end]
            else:
                metadata = self._get_level_metadata_cached(tree, lod, level_cache)
                if metadata is None or metadata[0].shape[1] == 0:
                    return
                slice_embeddings, slice_positions = metadata
                stride = max(1, tree.tokens_per_entry(lod))
                max_entries = math.ceil((end - start) / stride)
                slice_embeddings = slice_embeddings[:, :max_entries, :]
                slice_positions = slice_positions[:, :max_entries]
            append_segment(slice_embeddings, slice_positions, lod)

        if head_tokens > 0 and head_budget > 0:
            # Mix high LOD head with LOD0 to fill the budget deterministically.
            lod = self.config.max_lod
            while head_budget > 0 and lod > 0:
                append_lod_range(0, min(head_tokens, head_budget * tree.tokens_per_entry(lod)), lod)
                head_budget -= tree.tokens_per_entry(lod) * segments_embeddings[-1].shape[1]
                lod -= 1
            if head_budget > 0:
                append_lod_range(0, head_tokens, 0)

        if tail_tokens > 0:
            tail_start = max(0, total_tokens - tail_tokens)
            append_lod_range(tail_start, total_tokens, 0)

        if not segments_embeddings:
            append_lod_range(0, total_tokens, 0)

        head_embeddings = torch.cat(segments_embeddings, dim=1)
        head_positions = torch.cat(segments_positions, dim=1)
        head_lods = torch.cat(segments_lods, dim=1)
        wc_config = WorkingContextConfig(
            embed_dim=self.config.embed_dim,
            max_length=target_len,
            device=self.config.device,
        )
        variant = self._create_variant(
            tree,
            head_embeddings,
            head_positions,
            lod=0,
            source="lod_0_baseline",
            wc_config=wc_config,
            tail_tokens_override=tail_tokens,
        )
        variant.is_baseline = True
        return variant

    def _build_inference_recency_variant(
        self,
        tree: MegaContextTree,
        level_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> WorkingContextVariant:
        metadata = self._get_level_metadata_cached(tree, 0, level_cache)
        if metadata is None:
            raise RuntimeError("Unable to build inference baseline: missing LOD0 metadata")
        embeddings, positions = metadata
        window = self._eval_wc_config.max_length
        if embeddings.shape[1] > window:
            embeddings = embeddings[:, -window:]
            positions = positions[:, -window:]
        variant = self._create_variant(
            tree,
            embeddings,
            positions,
            lod=0,
            source="inference_baseline",
            wc_config=self._eval_wc_config,
        )
        variant.is_baseline = True
        return variant

    def _reinforce_recent_tail(self, wc: WorkingContext, tree: MegaContextTree) -> None:
        recent = int(self.config.allocator_recent_tokens)
        if recent <= 0 or wc.length == 0:
            return
        total_tokens = tree.num_tokens()
        if total_tokens <= 0:
            return
        tail_start = max(0, total_tokens - recent)
        positions = wc.get_positions()
        lods = wc.get_lod_tensor()
        tensor = wc.to_tensor()
        mask = positions >= tail_start
        if not torch.any(mask):
            return
        tail_embeddings = tree.get_lod0_slice(tail_start, total_tokens)
        tail_len = tail_embeddings.shape[1]
        for b in range(tensor.shape[0]):
            mask_b = mask[b]
            if not torch.any(mask_b):
                continue
            idx = (positions[b, mask_b] - tail_start).long()
            idx = torch.clamp(idx, 0, tail_len - 1)
            tensor[b, mask_b, :] = tail_embeddings[b, idx, :]
            lods[b, mask_b] = 0

    def _create_variant(
        self,
        tree: MegaContextTree,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        lod: int,
        source: str,
        wc_config: Optional[WorkingContextConfig] = None,
        tail_tokens_override: Optional[int] = None,
    ) -> WorkingContextVariant:
        config = wc_config or self.config.wc_config
        tree_tokens = tree.num_tokens()
        if tree_tokens <= 0:
            raise RuntimeError("[MegaContext] Cannot build working context with zero tree tokens")
        if lod == 0 and embeddings.shape[1] > config.max_length:
            raise RuntimeError(
                "[MegaContext] Pure LOD0 variant exceeds working-context capacity: "
                f"variant_len={embeddings.shape[1]}, max_length={config.max_length}"
            )
        block = max(1, self.config.block_size)
        tail_tokens = min(tree_tokens, int(self.config.allocator_recent_tokens))
        if tail_tokens_override is not None:
            tail_tokens = min(tree_tokens, max(0, int(tail_tokens_override)))
        head_limit = max(0, tree_tokens - tail_tokens)
        remainder = head_limit % block
        if remainder != 0:
            # Shift any partial block into the recent tail so head stays block-aligned.
            extra_tail = min(tree_tokens - tail_tokens, block - remainder)
            tail_tokens += extra_tail
            head_limit = max(0, tree_tokens - tail_tokens)
        tail_tokens = min(tail_tokens, tree_tokens)
        positions_lod0 = tree.get_positions_for_lod(0)

        segments: List[torch.Tensor] = []
        pos_segments: List[torch.Tensor] = []
        lod_segments: List[torch.Tensor] = []

        def append_segment(
            tensor: torch.Tensor,
            pos: torch.Tensor,
            lod_values: torch.Tensor,
        ) -> None:
            if tensor.shape[1] == 0:
                return
            segments.append(tensor)
            pos_segments.append(pos.long())
            lod_segments.append(lod_values.long())

        def append_lod0_range(start: int, end: int) -> None:
            if end <= start:
                return
            slice_embeddings = tree.get_lod0_slice(start, end)
            slice_positions = positions_lod0[:, start:end]
            lod_tensor = torch.zeros(
                (slice_embeddings.shape[0], slice_embeddings.shape[1]),
                dtype=torch.long,
                device=slice_embeddings.device,
            )
            append_segment(slice_embeddings, slice_positions, lod_tensor)

        head_cover = 0
        if head_limit > 0:
            if lod > 0:
                stride = max(1, tree.tokens_per_entry(lod))
                max_candidates = min(head_limit // stride, embeddings.shape[1])
                usable = 0
                while usable < max_candidates:
                    pos_val = int(positions[0, usable].item())
                    if pos_val != head_cover:
                        break
                    if pos_val + stride > head_limit:
                        break
                    usable += 1
                    head_cover = pos_val + stride
                if usable > 0:
                    head_embeddings = embeddings[:, :usable, :]
                    head_positions = positions[:, :usable]
                    lod_tensor = torch.full(
                        (head_embeddings.shape[0], head_embeddings.shape[1]),
                        lod,
                        dtype=torch.long,
                        device=head_embeddings.device,
                    )
                    append_segment(head_embeddings, head_positions, lod_tensor)
            if lod == 0 or head_cover < head_limit:
                append_lod0_range(head_cover, head_limit)
                head_cover = head_limit

        if tail_tokens > 0:
            tail_start = max(0, tree_tokens - tail_tokens)
            append_lod0_range(tail_start, tree_tokens)

        if not segments:
            append_lod0_range(0, tree_tokens)

        embeddings_combined = torch.cat(segments, dim=1)
        positions_combined = torch.cat(pos_segments, dim=1)
        lod_tensor_combined = torch.cat(lod_segments, dim=1)

        wc = WorkingContext(
            embeddings_combined,
            positions_combined,
            config,
            lod_tensor=lod_tensor_combined,
        )
        self._reinforce_recent_tail(wc, tree)
        self._assert_recent_lod0(wc, tree, source)
        expected_coverage = tree_tokens
        coverage = self._wc_token_coverage(wc, tree)
        if coverage != expected_coverage:
            raise RuntimeError(
                f"[MegaContext] Working context coverage mismatch ({source}): "
                f"expected {expected_coverage}, got {coverage}"
            )
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

    def _update_train_progress(self, step: int, context: str) -> None:
        if context != "train":
            return
        denom = max(1, self.total_train_steps - 1)
        self._train_progress = min(1.0, max(0.0, step / denom))

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

    def _make_torch_generator(self, device: torch.device) -> torch.Generator:
        target_device = device
        if target_device.type == "meta":
            target_device = torch.device("cpu")
        generator = torch.Generator(device=target_device)
        generator.manual_seed(self._rng.randint(0, 2**31 - 1))
        return generator

    def _timing_sync(self) -> None:
        if self._timing_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._timing_device)

    def _assert_recent_lod0(self, wc: WorkingContext, tree: MegaContextTree, tag: str) -> None:
        recent = int(self.config.allocator_recent_tokens)
        if recent <= 0 or wc.length == 0:
            return
        total_tokens = tree.num_tokens()
        if total_tokens <= 0:
            return
        tail_start = max(0, total_tokens - recent)
        positions = wc.get_positions()
        lods = wc.get_lod_tensor()
        mask = positions >= tail_start
        if not torch.any(mask):
            return
        offending = lods[mask]
        if offending.numel() > 0 and torch.any(offending != 0):
            raise RuntimeError(
                "[MegaContext] Recent tokens must remain LOD0 "
                f"(context={tag}). Observed LODs: {offending.tolist()}"
            )

    def _lod_char_for_block(self, lod: int, is_last: bool, remainder: int) -> str:
        if lod < 0:
            return "?"
        capped = min(lod, 3)
        if capped == 0 and is_last and remainder > 0:
            return self._LOD_PARTIAL_CHAR
        return self._LOD_CHAR_MAP.get(capped, self._LOD_CHAR_MAP[3])

    def _render_lod_ascii_lines(self, wc: WorkingContext, total_tokens: int) -> List[str]:
        if total_tokens <= 0 or wc.length == 0:
            return []
        block_size = max(1, self.config.block_size)
        num_blocks = max(1, math.ceil(total_tokens / block_size))
        remainder = total_tokens % block_size
        positions = wc.get_positions()
        lods = wc.get_lod_tensor()
        rows: List[str] = []
        for b in range(positions.shape[0]):
            block_levels = [-1] * num_blocks
            for idx in range(wc.length):
                pos = int(positions[b, idx].item())
                lod = int(lods[b, idx].item())
                span = max(1, block_size ** max(lod, 0))
                start = pos
                end = min(total_tokens, pos + span)
                if end <= start:
                    continue
                block_idx = max(0, start // block_size)
                while block_idx < num_blocks:
                    block_start = block_idx * block_size
                    block_end = min(total_tokens, block_start + block_size)
                    if block_start >= end:
                        break
                    block_levels[block_idx] = lod
                    block_idx += 1
            chars = [
                self._lod_char_for_block(level, idx == num_blocks - 1, remainder)
                for idx, level in enumerate(block_levels)
            ]
            self._highlight_high_lod_segments(chars, block_levels)
            rows.append("".join(chars))
        return rows

    def _highlight_high_lod_segments(self, chars: List[str], block_levels: List[int], threshold: int = 2) -> None:
        num_blocks = len(chars)
        idx = 0
        while idx < num_blocks:
            level = block_levels[idx]
            if level < threshold:
                idx += 1
                continue
            start = idx
            while idx < num_blocks and block_levels[idx] >= threshold:
                idx += 1
            end = idx - 1
            segment_levels = block_levels[start : end + 1]
            standout_level = max(segment_levels) if segment_levels else level
            for pos in range(start, end + 1):
                chars[pos] = " "
            left = start
            right = end
            chars[left] = "|"
            chars[right] = "|" if right >= left else "|"
            length = right - left + 1
            if length <= 1:
                chars[left] = self._LOD_CHAR_MAP.get(min(max(standout_level, 0), 3), self._LOD_CHAR_MAP[3])
                continue
            center = left + length // 2
            if center == left:
                center = min(left + 1, right)
            if center == right:
                center = max(right - 1, left)
            chars[center] = self._LOD_CHAR_MAP.get(
                min(max(standout_level, 0), 3),
                self._LOD_CHAR_MAP[3],
            )

    def _log_train_lod_ascii(self, step: int, batch_states: List[SampleContext]) -> None:
        if not (self.config.log_lod_ascii_train and self._is_rank0):
            return
        lines: List[str] = []
        for sample_idx, sample in enumerate(batch_states):
            tree_tokens = sample.tree.num_tokens()
            for variant_idx, variant in enumerate(sample.variants):
                ascii_rows = self._render_lod_ascii_lines(variant.working_context, tree_tokens)
                if not ascii_rows:
                    continue
                joined = " | ".join(ascii_rows)
                lines.append(
                    f"  sample{sample_idx:02d}/var{variant_idx:02d} {joined} [{variant.source}] ({variant.working_context.length})"
                )
        if lines:
            print(f"[MegaContext][LOD ASCII][train step {step}]", flush=True)
            for line in lines:
                print(line, flush=True)

    def log_inference_lod_ascii(self, label: str) -> None:
        if not (self.config.log_lod_ascii_val and self._is_rank0):
            return
        state = self.inference_state
        if state is None:
            return
        ascii_rows = self._render_lod_ascii_lines(state.working_context, state.tree.num_tokens())
        if not ascii_rows:
            return
        print(f"[MegaContext][LOD ASCII][{label}]", flush=True)
        for idx, row in enumerate(ascii_rows):
            print(f"  seq{idx:02d}: {row} ({state.working_context.length})", flush=True)

    def _ensure_policy_scores_logged(
        self,
        variants: List[WorkingContextVariant],
        score_cache: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        with torch.no_grad():
            for variant in variants:
                if variant.policy_scores is not None:
                    continue
                if score_cache is not None:
                    cached = score_cache.get(id(variant))
                    if cached is not None:
                        variant.policy_scores = cached.detach()
                        continue
                preview = self.lensnet(variant.working_context)
                variant.policy_scores = preview.detach()

    def _log_preference_debug(self, batch_states: List[SampleContext]) -> None:
        if not (self.config.log_lens_debug and self._is_rank0):
            return
        rows: List[Tuple[str, float, float, float, float, float]] = []
        mean_scores: List[float] = []
        adv_values: List[float] = []
        max_scores: List[float] = []
        min_scores: List[float] = []
        total_variants = 0
        for sample in batch_states:
            total_variants += len(sample.variants)
            for variant in sample.variants:
                scores = variant.policy_scores
                adv = getattr(variant, "adv_delta", None)
                if scores is None or adv is None:
                    continue
                scores_flat = scores.float().view(-1)
                score_mean = float(scores_flat.mean().item())
                score_std = float(scores_flat.std(unbiased=False).item())
                score_max = float(scores_flat.max().item())
                score_min = float(scores_flat.min().item())
                rows.append((variant.source, score_mean, score_std, score_max, score_min, float(adv)))
                mean_scores.append(score_mean)
                adv_values.append(float(adv))
                max_scores.append(score_max)
                min_scores.append(score_min)
        if not rows:
            print(
                "[MegaContext][PrefDebug] no preference data (variants=%d) — Δloss probably zero or LensNet disabled"
                % total_variants,
                flush=True,
            )
            self._last_preference_corr = {}
            self._last_preference_corr_valid = {}
            self._last_preference_pair_count = 0
            return
        def _corr(xs: List[float], ys: List[float]) -> Optional[float]:
            n = len(xs)
            if n < 2:
                return None
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den_x = sum((x - mean_x) ** 2 for x in xs)
            den_y = sum((y - mean_y) ** 2 for y in ys)
            if den_x <= 0 or den_y <= 0:
                return None
            return num / (den_x ** 0.5 * den_y ** 0.5)
        corr_mean = _corr(mean_scores, adv_values)
        corr_max = _corr(max_scores, adv_values)
        corr_min = _corr(min_scores, adv_values)
        corr_mean_valid = corr_mean is not None
        corr_max_valid = corr_max is not None
        corr_min_valid = corr_min is not None
        corr_mean_val = float(corr_mean) if corr_mean is not None else 0.0
        corr_max_val = float(corr_max) if corr_max is not None else 0.0
        corr_min_val = float(corr_min) if corr_min is not None else 0.0
        self._last_preference_corr = {
            "preference_corr_mean": corr_mean_val,
            "preference_corr_max": corr_max_val,
            "preference_corr_min": corr_min_val,
        }
        self._last_preference_corr_valid = {
            "preference_corr_mean_valid": corr_mean_valid,
            "preference_corr_max_valid": corr_max_valid,
            "preference_corr_min_valid": corr_min_valid,
        }
        self._last_preference_pair_count = len(rows)

        def _fmt(val: Optional[float]) -> str:
            return f"{val:.3f}" if val is not None else "n/a"

        agreement = self._last_preference_agreement
        agreement_str = _fmt(agreement)
        print(
            "[MegaContext][PrefDebug] pairs=%d corr_mean=%s corr_max=%s corr_min=%s agreement=%s"
            % (
                len(rows),
                _fmt(corr_mean if corr_mean_valid else None),
                _fmt(corr_max if corr_max_valid else None),
                _fmt(corr_min if corr_min_valid else None),
                agreement_str,
            ),
            flush=True,
        )
        for entry in rows[:5]:
            source, mean_val, std_val, max_val, min_val, adv = entry
            print(
                f"   {source:>16} score_mean={mean_val:.3f} std={std_val:.3f} "
                f"max={max_val:.3f} min={min_val:.3f} adv={adv:.3f}",
                flush=True,
            )

    def _ensure_wc_full_coverage(self, wc: WorkingContext, tree: MegaContextTree, tag: str) -> WorkingContext:
        expected = tree.num_tokens()
        if expected <= 0:
            return wc
        coverage = self._wc_token_coverage(wc, tree)
        if coverage == expected:
            return wc
        repaired = self._rebuild_wc_with_lod0(wc, tree)
        self._reinforce_recent_tail(repaired, tree)
        self._assert_recent_lod0(repaired, tree, tag)
        repaired_coverage = self._wc_token_coverage(repaired, tree)
        if repaired_coverage != expected:
            raise RuntimeError(
                f"[MegaContext] Unable to restore coverage ({tag}): "
                f"expected {expected}, got {repaired_coverage}"
            )
        self._configure_wc_positional(repaired)
        return repaired

    def _rebuild_wc_with_lod0(self, wc: WorkingContext, tree: MegaContextTree) -> WorkingContext:
        positions = wc.get_positions()
        lods = wc.get_lod_tensor()
        tensor = wc.to_tensor()
        total_tokens = tree.num_tokens()
        positions_lod0 = tree.get_positions_for_lod(0)
        cursor = 0
        seg_embeddings: List[torch.Tensor] = []
        seg_positions: List[torch.Tensor] = []
        seg_lods: List[torch.Tensor] = []

        def append_lod0_span(start: int, end: int) -> None:
            if end <= start:
                return
            lod0_embed = tree.get_lod0_slice(start, end)
            lod0_pos = positions_lod0[:, start:end]
            lod0_tensor = torch.zeros(
                (lod0_embed.shape[0], lod0_embed.shape[1]),
                dtype=torch.long,
                device=lod0_embed.device,
            )
            seg_embeddings.append(lod0_embed)
            seg_positions.append(lod0_pos)
            seg_lods.append(lod0_tensor)

        for idx in range(wc.length):
            if cursor >= total_tokens:
                break
            pos = int(positions[0, idx].item())
            lod = int(lods[0, idx].item())
            span = max(1, tree.tokens_per_entry(lod))
            if pos < cursor:
                next_pos = pos + span
                if next_pos <= cursor:
                    continue
                append_lod0_span(cursor, min(next_pos, total_tokens))
                cursor = min(next_pos, total_tokens)
                continue
            if pos > cursor:
                append_lod0_span(cursor, pos)
                cursor = pos
            seg_embeddings.append(tensor[:, idx : idx + 1, :])
            seg_positions.append(positions[:, idx : idx + 1])
            seg_lods.append(lods[:, idx : idx + 1])
            cursor = min(total_tokens, pos + span)

        if cursor < total_tokens:
            append_lod0_span(cursor, total_tokens)

        new_embeddings = torch.cat(seg_embeddings, dim=1)
        new_positions = torch.cat(seg_positions, dim=1)
        new_lods = torch.cat(seg_lods, dim=1)
        rebuilt = WorkingContext(
            new_embeddings,
            new_positions,
            wc.config,
            lod_tensor=new_lods,
        )
        return rebuilt

    def _clone_working_context(self, wc: WorkingContext) -> WorkingContext:
        embeddings = wc.to_tensor().clone()
        positions = wc.get_positions().clone()
        lods = wc.get_lod_tensor().clone()
        clone = WorkingContext(
            embeddings,
            positions,
            wc.config,
            lod_tensor=lods,
            recent_tokens=self.config.allocator_recent_tokens,
        )
        self._configure_wc_positional(clone)
        return clone

    def _infer_variant_lod_hint(self, wc: WorkingContext) -> int:
        lods = wc.get_lod_tensor()
        if lods.numel() == 0:
            return 0
        return int(lods.max().item())

    def _normalize_wc_length(self, wc: WorkingContext, tree: MegaContextTree, target_len: int) -> None:
        target_len = int(max(1, min(target_len, tree.num_tokens())))
        if wc.length == target_len:
            return
        allocator = self._build_allocator(tree, wc, soft_max_length=target_len)
        max_attempts = abs(wc.length - target_len) * 4 + 10
        device = wc.to_tensor().device
        while wc.length != target_len and max_attempts > 0:
            prefer_expand = wc.length < target_len
            scores = torch.ones(1, wc.length, device=device)
            if not prefer_expand:
                scores = -scores
            edits = allocator.update_focus(
                max_replacements_per_iteration=1,
                num_iterations=1,
                scores=scores,
            )
            if edits == 0:
                break
            max_attempts -= 1
        if wc.length != target_len:
            raise RuntimeError(
                f"[MegaContext] Unable to normalize WC length to {target_len} (final={wc.length})"
            )

    def _current_target_wc_length(self) -> int:
        max_len = self.config.wc_config.max_length
        start_len = int(0.8 * max_len)
        end_len = int(min(max_len, self.config.train_wc_length or max_len))
        progress = self._train_progress
        target = int(round(start_len + (end_len - start_len) * progress))
        return max(1, min(max_len, target))

    def _compute_variant_losses(
        self,
        batch_states: List[SampleContext],
        original_tokens: torch.Tensor,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Dict[int, float],
        Dict[int, int],
    ]:
        entries = self._prepare_variant_entries(batch_states, original_tokens)
        if not entries:
            return None, None, None, None, None, {}, {}
        losses = []
        losses.extend(self._run_variant_batch(entries))
        if not losses:
            return None, None, None, None, None, {}, {}
        loss_tensor = torch.stack(losses)
        lod0_loss = None
        adv_values: List[float] = []
        all_advantages: List[float] = []
        lod_sums: Dict[int, float] = {}
        lod_weights: Dict[int, float] = {}
        lod_token_counts: Dict[int, int] = {}
        for sample in batch_states:
            for variant in sample.variants:
                variant.adv_delta = None  # type: ignore[attr-defined]
            baseline = None
            for variant in sample.variants:
                if variant.lod_hint == 0 and variant.token_loss_value is not None:
                    baseline = float(variant.token_loss_value.detach())
                    lod0_loss = baseline if lod0_loss is None else lod0_loss
                    variant.adv_delta = 0.0
                    break
            for variant in sample.variants:
                if variant.token_loss_value is None:
                    continue
                val = float(variant.token_loss_value.detach())
                if baseline is not None:
                    variant.adv_delta = val - baseline
                hist = self._lod_histogram(variant.working_context)
                total_bucket_tokens = sum(hist.values()) or 1
                for lod, count in hist.items():
                    weight = count / total_bucket_tokens
                    lod_sums[lod] = lod_sums.get(lod, 0.0) + val * weight
                    lod_weights[lod] = lod_weights.get(lod, 0.0) + weight
                    lod_token_counts[lod] = lod_token_counts.get(lod, 0) + int(count)
                if baseline is not None and variant is not None and variant.lod_hint != 0:
                    adv_values.append(val - baseline)
                if variant.adv_delta is not None:
                    all_advantages.append(variant.adv_delta)
        adv_delta_mean = float(torch.tensor(adv_values).mean().item()) if adv_values else None
        adv_delta_p95 = (
            float(torch.quantile(torch.tensor(adv_values), 0.95).item())
            if adv_values
            else None
        )
        adv_delta_std = float(torch.tensor(adv_values).std(unbiased=False).item()) if adv_values else None

        lod_metrics = {
            lod: (lod_sums[lod] / max(lod_weights[lod], 1e-6))
            for lod in lod_sums.keys()
            if lod_weights.get(lod)
        }
        lod_counts = lod_token_counts

        if all_advantages:
            adv_tensor = torch.tensor(all_advantages, dtype=torch.float32)
            batch_mean = float(adv_tensor.mean().item())
            batch_var = float(adv_tensor.var(unbiased=False).item()) if adv_tensor.numel() > 1 else 0.0
            if not self._adv_norm_initialized:
                self._adv_norm_mean = batch_mean
                self._adv_norm_var = max(batch_var, 1e-6)
                self._adv_norm_initialized = True
            else:
                beta = float(self.config.lens_adv_norm_beta)
                self._adv_norm_mean = beta * self._adv_norm_mean + (1.0 - beta) * batch_mean
                self._adv_norm_var = beta * self._adv_norm_var + (1.0 - beta) * max(batch_var, 1e-6)
            std = math.sqrt(max(self._adv_norm_var, 1e-6))
            mean = self._adv_norm_mean
            for sample in batch_states:
                for variant in sample.variants:
                    if variant.adv_delta is None:
                        continue
                    variant.norm_adv_delta = (variant.adv_delta - mean) / std

        return loss_tensor.mean(), lod0_loss, adv_delta_mean, adv_delta_p95, adv_delta_std, lod_metrics, lod_counts

    def _prepare_variant_entries(
        self,
        batch_states: List[SampleContext],
        original_tokens: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        target_len = min(self.config.wc_config.max_length, original_tokens.shape[1])
        if self.config.train_wc_length is not None:
            target_len = min(target_len, int(self.config.train_wc_length))
        target_len = max(1, target_len)
        for sample in batch_states:
            for variant in sample.variants:
                wc = variant.working_context
                embeddings = self._to_model_dtype(wc.to_tensor())
                seq_len = int(embeddings.shape[1])
                cos, sin, alibi = wc.get_positional_encodings()
                entry = {
                    "variant": variant,
                    "embeddings": embeddings,
                    "cos": self._to_model_dtype(cos),
                    "sin": self._to_model_dtype(sin),
                    "alibi": self._to_model_dtype(alibi),
                    "seq_len": seq_len,
                    "target_len": target_len,
                    "batch_idx": getattr(variant, "batch_index", 0),
                    "original_tokens": original_tokens,
                }
                entries.append(entry)
        return entries

    def _run_variant_batch(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        if not entries:
            return []
        batch_size = len(entries)
        target_len = min(self.config.wc_config.max_length, entries[0]["target_len"] if entries else self.config.wc_config.max_length)
        max_len = target_len
        embed_dim = entries[0]["embeddings"].shape[-1]
        embeddings = torch.zeros((batch_size, max_len, embed_dim), dtype=self._target_dtype, device=self.device)

        cos_tensor: Optional[torch.Tensor] = None
        sin_tensor: Optional[torch.Tensor] = None
        if all(entry["cos"] is not None for entry in entries):
            extra_shape = entries[0]["cos"].shape[2:]
            cos_tensor = torch.zeros((batch_size, max_len) + extra_shape, dtype=self._target_dtype, device=self.device)
        if all(entry["sin"] is not None for entry in entries):
            extra_shape = entries[0]["sin"].shape[2:]
            sin_tensor = torch.zeros((batch_size, max_len) + extra_shape, dtype=self._target_dtype, device=self.device)

        need_alibi = any(entry["alibi"] is not None for entry in entries)
        alibi_tensor: Optional[torch.Tensor] = None
        if need_alibi:
            alibi_tensor = torch.zeros(
                (batch_size, self.config.num_heads, max_len, max_len),
                dtype=self._target_dtype,
                device=self.device,
            )

        token_batch = torch.full((batch_size, max_len), -1, dtype=torch.long, device=self.device)
        for idx, entry in enumerate(entries):
            seq_len = int(entry["seq_len"])
            embed_slice = entry["embeddings"]
            if embed_slice.dim() == 3:
                embed_slice = embed_slice.squeeze(0)
            if seq_len >= max_len:
                start_idx = seq_len - max_len
                embed_slice = embed_slice[start_idx:, :]
                seq_len = max_len
                start = 0
            else:
                start = max_len - seq_len
            embeddings[idx, start:, :] = embed_slice
            if cos_tensor is not None and entry["cos"] is not None:
                cos_slice = entry["cos"].squeeze(0)
                if cos_slice.shape[0] >= max_len:
                    cos_slice = cos_slice[-max_len:, ...]
                cos_tensor[idx, start:, ...] = cos_slice
            if sin_tensor is not None and entry["sin"] is not None:
                sin_slice = entry["sin"].squeeze(0)
                if sin_slice.shape[0] >= max_len:
                    sin_slice = sin_slice[-max_len:, ...]
                sin_tensor[idx, start:, ...] = sin_slice
            if alibi_tensor is not None and entry["alibi"] is not None:
                alibi_slice = entry["alibi"]
                if alibi_slice.dim() == 4:
                    alibi_slice = alibi_slice.squeeze(0)
                if alibi_slice.shape[-1] > max_len:
                    alibi_slice = alibi_slice[:, :, -max_len:, -max_len:]
                alibi_tensor[idx, :, start:, start:] = alibi_slice
            token_slice = self._align_tokens_to_embeddings(
                entry["original_tokens"][entry["batch_idx"] : entry["batch_idx"] + 1].to(torch.long),
                seq_len,
            )
            token_batch[idx, start:] = token_slice[0]

        assert embeddings.shape[1] == max_len, "Embeddings must align to shared sequence length"
        assert token_batch.shape[1] == max_len, "Token batch must align to shared sequence length"

        dummy_idx = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        cos = cos_tensor
        sin = sin_tensor
        alibi = alibi_tensor
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
            logits = logits.view(batch_size, max_len, -1)
            tokens_flat = token_batch.view(batch_size, max_len)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens_flat.view(-1),
                ignore_index=-1,
                reduction="none",
            )
            loss2d = loss_flat.view(batch_size, max_len)
        else:
            loss2d = loss2d.view(batch_size, -1)
        valid = (token_batch.view(batch_size, -1) >= 0).to(loss2d.dtype)
        sample_losses = (loss2d * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        outputs = []
        for entry, loss in zip(entries, sample_losses):
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
        if self.config.num_random_variants <= 0:
            return None
        losses = []
        rank_weight = float(self.config.lens_rank_weight)
        budget_weight = float(self.config.lens_budget_weight)
        margin = float(self.config.lens_margin)
        collapse_weight = float(self.config.lens_collapse_weight)
        temperature = max(1e-6, float(self.config.lens_temperature))
        kl_weight = float(self.config.lens_kl_weight)
        budget_smooth_weight = float(self.config.lens_budget_smooth_weight)
        budget_smooth_beta = float(self.config.lens_budget_smooth_beta)
        agreement_hits = 0
        agreement_total = 0
        score_abs_means: List[float] = []
        score_std_vals: List[float] = []
        self._last_preference_agreement = None
        self._last_policy_stats = {}
        for sample in batch_states:
            preference_pairs = self._build_preference_pairs(sample.variants)
            if not preference_pairs:
                self._ensure_policy_scores_logged(sample.variants)
                for variant in sample.variants:
                    scores = variant.policy_scores
                    if scores is None:
                        continue
                    stats_tensor = scores.float().view(-1)
                    if stats_tensor.numel() == 0:
                        continue
                    score_abs_means.append(float(stats_tensor.abs().mean().item()))
                    score_std_vals.append(float(stats_tensor.std(unbiased=False).item()))
                continue
            map_cache: Dict[int, Dict[int, int]] = {}
            score_cache: Dict[int, torch.Tensor] = {}

            def _get_scores(variant: WorkingContextVariant) -> torch.Tensor:
                key = id(variant)
                cached = score_cache.get(key)
                if cached is None:
                    cached = self.lensnet(variant.working_context)
                    score_cache[key] = cached
                    variant.policy_scores = cached.detach()
                return cached

            for better, worse, strength in preference_pairs:
                scores_live = _get_scores(worse)
                if scores_live.dim() > 1:
                    scores_1d = scores_live.squeeze(0)
                else:
                    scores_1d = scores_live
                ref_key = id(better.working_context)
                ref_map = map_cache.get(ref_key)
                if ref_map is None:
                    ref_map = self._build_lod_lookup(better.working_context)
                    map_cache[ref_key] = ref_map
                targets, mask, span_tokens = self._build_pairwise_targets(
                    worse,
                    ref_map,
                    scores_1d,
                    strength,
                )
                if not torch.any(mask):
                    continue
                masked_scores = scores_1d[mask]
                masked_targets = targets[mask]
                sample_weights = torch.ones_like(masked_targets)
                if collapse_weight != 1.0:
                    sample_weights[masked_targets < 0] = collapse_weight
                target_sign = torch.sign(masked_targets)
                target_strength = torch.abs(masked_targets).clamp_min(1e-6)
                pair_scale = max(1.0, float(strength)) / temperature
                scaled_scores = masked_scores * pair_scale
                logit = target_sign * scaled_scores
                pref_loss = F.softplus(-logit)
                pref_loss = pref_loss * target_strength
                pref_loss = (pref_loss * sample_weights).mean()
                rank_loss = masked_scores.new_tensor(0.0)
                budget_loss = masked_scores.new_tensor(0.0)
                if rank_weight > 0.0:
                    pos_mask = masked_targets > 0
                    neg_mask = masked_targets < 0
                    if torch.any(pos_mask) and torch.any(neg_mask):
                        pos_mean = masked_scores[pos_mask].mean()
                        neg_mean = masked_scores[neg_mask].mean()
                        rank_loss = F.relu(margin - (pos_mean - neg_mean))
                if budget_weight > 0.0:
                    span_vals = span_tokens[mask].to(masked_scores.dtype)
                    pos_mass = (span_vals * torch.relu(masked_scores)).sum()
                    neg_mass = (span_vals * torch.relu(-masked_scores)).sum()
                    denom = pos_mass + neg_mass + 1e-6
                    budget_loss = ((pos_mass - neg_mass) / denom) ** 2
                smooth_loss = masked_scores.new_tensor(0.0)
                if budget_smooth_weight > 0.0:
                    current_diff = ((pos_mass - neg_mass) / (pos_mass + neg_mass + 1e-6))
                    self._budget_mass_ema = budget_smooth_beta * self._budget_mass_ema + (1.0 - budget_smooth_beta) * float(current_diff.detach())
                    target = current_diff - self._budget_mass_ema
                    smooth_loss = target * target

                kl_loss = masked_scores.new_tensor(0.0)
                history_key = id(worse.working_context)
                if kl_weight > 0.0:
                    prev_scores = self._policy_history.get(history_key)
                    if prev_scores is not None and prev_scores.shape == scores_1d.shape:
                        prev = prev_scores.to(scores_1d.device)
                        p_curr = torch.sigmoid(scores_1d / temperature).clamp(1e-6, 1 - 1e-6)
                        p_prev = torch.sigmoid(prev / temperature).clamp(1e-6, 1 - 1e-6)
                        kl_forward = F.kl_div(p_curr.log(), p_prev, reduction="batchmean")
                        kl_backward = F.kl_div(p_prev.log(), p_curr, reduction="batchmean")
                        kl_loss = 0.5 * (kl_forward + kl_backward)
                self._policy_history[history_key] = scores_1d.detach().to("cpu")

                total_loss = (
                    pref_loss
                    + rank_weight * rank_loss
                    + budget_weight * budget_loss
                    + budget_smooth_weight * smooth_loss
                    + kl_weight * kl_loss
                )
                signed_alignment = (masked_scores * target_sign).mean().item()
                if math.isfinite(signed_alignment):
                    agreement_total += 1
                    if signed_alignment > 0:
                        agreement_hits += 1
                losses.append(total_loss)
            self._ensure_policy_scores_logged(sample.variants, score_cache)
            for variant in sample.variants:
                scores = variant.policy_scores
                if scores is None:
                    continue
                stats_tensor = scores.float().view(-1)
                if stats_tensor.numel() == 0:
                    continue
                score_abs_means.append(float(stats_tensor.abs().mean().item()))
                score_std_vals.append(float(stats_tensor.std(unbiased=False).item()))
        if not losses:
            return None
        if score_abs_means:
            score_abs_avg = sum(score_abs_means) / len(score_abs_means)
            score_std_avg = sum(score_std_vals) / len(score_std_vals) if score_std_vals else 0.0
            self._last_policy_stats = {
                "score_abs_mean": float(score_abs_avg),
                "score_std_mean": float(score_std_avg),
            }
        else:
            self._last_policy_stats = {}
        if agreement_total > 0:
            self._last_preference_agreement = agreement_hits / agreement_total
        else:
            self._last_preference_agreement = None
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
        scores = variant.policy_scores
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
        # Prefer variants that actually ran focus (edits > 0). Tie-break by highest LOD.
        def _select_variant(candidates: List[WorkingContextVariant]) -> Optional[WorkingContextVariant]:
            best = None
            best_lod_local = -1
            best_edits_local = -1
            for variant in candidates:
                if variant.is_baseline:
                    continue
                lod_hist = self._lod_histogram(variant.working_context)
                highest_variant_lod = max(lod_hist.keys()) if lod_hist else variant.lod_hint
                if (
                    highest_variant_lod > best_lod_local
                    or (
                        highest_variant_lod == best_lod_local
                        and variant.edits_applied > best_edits_local
                    )
                ):
                    best = variant
                    best_lod_local = highest_variant_lod
                    best_edits_local = variant.edits_applied
            return best

        focused_candidates = []
        unfocused_candidates = []
        for sample in batch_states:
            for variant in sample.variants:
                if variant.is_baseline:
                    continue
                if variant.edits_applied > 0:
                    focused_candidates.append((sample, variant))
                else:
                    unfocused_candidates.append((sample, variant))

        chosen = _select_variant([variant for _, variant in focused_candidates])
        if chosen is None:
            chosen = _select_variant([variant for _, variant in unfocused_candidates])

        if chosen is not None:
            for sample in batch_states:
                if chosen in sample.variants:
                    primary_variant = chosen
                    primary_tree_tokens = sample.tree.num_tokens()
                    primary_tree = sample.tree
                    break
        else:
            sample = batch_states[0]
            if sample.variants:
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
            expected = sample.tree.num_tokens()
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
                "expected_tokens": int(primary_tree_tokens),
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

    def _build_preference_pairs(
        self, variants: List[WorkingContextVariant]
    ) -> List[Tuple[WorkingContextVariant, WorkingContextVariant, float]]:
        scored: List[Tuple[float, WorkingContextVariant]] = []
        for variant in variants:
            if variant.token_loss_value is None:
                continue
            scored.append((float(variant.token_loss_value.detach()), variant))
        if len(scored) < 2:
            return []
        scored.sort(key=lambda item: item[0])
        best_loss, best_variant = scored[0]
        primary_pairs: List[Tuple[WorkingContextVariant, WorkingContextVariant, float]] = []
        seen_pairs: set[Tuple[int, int]] = set()
        for loss_val, variant in scored[1:]:
            delta = float(loss_val - best_loss)
            if delta <= 1e-6:
                continue
            primary_pairs.append((best_variant, variant, delta))
            seen_pairs.add((id(best_variant), id(variant)))
        additional_pairs: List[Tuple[WorkingContextVariant, WorkingContextVariant, float]] = []
        for i in range(len(scored)):
            loss_i, var_i = scored[i]
            for j in range(i + 1, len(scored)):
                loss_j, var_j = scored[j]
                delta = abs(loss_i - loss_j)
                if delta <= 1e-6:
                    continue
                if loss_i <= loss_j:
                    better, worse = var_i, var_j
                else:
                    better, worse = var_j, var_i
                key = (id(better), id(worse))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                additional_pairs.append((better, worse, float(delta)))
        additional_pairs.sort(key=lambda item: item[2], reverse=True)
        ratio = float(self.config.lens_hard_negative_ratio)
        if additional_pairs and ratio < 1.0:
            keep = max(1, int(math.ceil(len(additional_pairs) * ratio)))
            additional_pairs = additional_pairs[:keep]
        max_pairs = max(1, self.config.max_lens_pairs)
        if not primary_pairs and not additional_pairs:
            return []
        primary_pairs.sort(key=lambda item: item[2], reverse=True)
        kept: List[Tuple[WorkingContextVariant, WorkingContextVariant, float]] = []
        if primary_pairs:
            kept.extend(primary_pairs[:max_pairs])
        remaining = max_pairs - len(kept)
        if remaining > 0 and additional_pairs:
            kept.extend(additional_pairs[:remaining])
        if not kept:
            return []
        self._rng.shuffle(kept)
        return kept

    def _build_pairwise_targets(
        self,
        variant: WorkingContextVariant,
        best_map: Dict[int, int],
        score_template: torch.Tensor,
        delta_vs_best: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = variant.working_context.get_positions()[0]
        lods = variant.working_context.get_lod_tensor()[0]
        targets = torch.zeros_like(score_template)
        mask = torch.zeros_like(score_template, dtype=torch.bool)
        span_tokens = torch.ones_like(score_template)
        max_lod = self.config.max_lod
        block_size = self.config.block_size
        strength = math.tanh(abs(delta_vs_best))
        if strength <= 0.0:
            return targets, mask, span_tokens
        for idx, (pos, lod_tensor) in enumerate(zip(positions, lods)):
            lod = int(lod_tensor.item())
            pos_int = int(pos.item())
            desired_lod = best_map.get(pos_int, lod)
            span_tokens[idx] = float(block_size ** max(lod, 0))
            if desired_lod == lod:
                continue
            if desired_lod < lod:
                if lod <= 0:
                    continue
                targets[idx] = strength
                mask[idx] = True
            elif desired_lod > lod:
                if lod >= max_lod:
                    continue
                parent_span = block_size ** (lod + 1)
                parent_start = (pos_int // parent_span) * parent_span
                parent_end = parent_start + parent_span
                collapse_val = -strength
                for jdx, (pos_j, lod_j) in enumerate(zip(positions, lods)):
                    lod_j_val = int(lod_j.item())
                    pos_j_int = int(pos_j.item())
                    if lod_j_val != lod:
                        continue
                    if parent_start <= pos_j_int < parent_end:
                        targets[jdx] = collapse_val
                        mask[jdx] = True
        return targets, mask, span_tokens

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
        recency_variant = self._build_inference_recency_variant(tree, level_cache)
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
            self._reinforce_recent_tail(recency_variant.working_context, tree)
            self._timing_sync()
            timings["allocator_rebuild_ms"] = (time.time() - t_rebuild0) * 1000.0
        else:
            timings["allocator_rebuild_ms"] = 0.0
        recency_variant.working_context = self._ensure_wc_full_coverage(
            recency_variant.working_context,
            tree,
            "inference_recency",
        )
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
                self._reinforce_recent_tail(state.working_context, state.tree)
        self._log_wc_snapshot(state.session_id, state.working_context, tag="inference_update")
        self._log_focus_stats(
            state.session_id,
            WorkingContextVariant(
                working_context=state.working_context,
                source="inference",
                lod_hint=0,
                edits_applied=0,
                policy_scores=None,
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
        self._reinforce_recent_tail(state.working_context, state.tree)
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
        self._assert_recent_lod0(wc, self.inference_state.tree, "inference")
        return wc

    def get_inference_report(self) -> Optional[Dict[str, Any]]:
        return self.last_inference_report

    def get_training_report(self) -> Optional[Dict[str, Any]]:
        return self.last_train_report
