from __future__ import annotations

import random
from typing import Optional

import torch

from .mega_context import MegaContextTree
from .working_context import WorkingContext, WorkingContextEdit


class TrainingWCVariationAllocator:
    """
    Deterministic training-time variant builder that never consults LensNet.
    It repeatedly collapses random spans until the working context reaches
    the requested target length while respecting the recent-token tail.
    """

    def __init__(
        self,
        *,
        tree: MegaContextTree,
        block_size: int,
        max_lod: int,
        lod2_probability: float,
        rng: random.Random,
    ) -> None:
        self.tree = tree
        self.block_size = max(1, int(block_size))
        self.max_lod = max(1, int(max_lod))
        self.lod2_probability = max(0.0, min(1.0, float(lod2_probability)))
        self.rng = rng

    def collapse_to_target(
        self,
        wc: WorkingContext,
        target_len: int,
        *,
        recent_tokens: int,
        prefer_head: bool = False,
    ) -> None:
        target_len = max(1, int(target_len))
        if wc.length < target_len:
            raise RuntimeError(
                f"[MegaContext] Training WC length {wc.length} shorter than target={target_len}; cannot expand without LensNet"
            )
        max_attempts = max(1, wc.length * 8)
        while wc.length > target_len and max_attempts > 0:
            max_attempts -= 1
            if self._collapse_step(wc, recent_tokens, prefer_head=prefer_head):
                continue
            if self._expand_random_span(wc, recent_tokens, prefer_head=prefer_head):
                continue
            break
        if wc.length != target_len:
            raise RuntimeError(
                f"[MegaContext] Unable to reach training target length {target_len} (final={wc.length})"
            )

    def force_head_collapse(self, wc: WorkingContext) -> bool:
        if self.block_size <= 1:
            return False
        return self._collapse_span(
            wc,
            target_lod=0,
            recent_tokens=0,
            prefer_head=True,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _collapse_step(self, wc: WorkingContext, recent_tokens: int, *, prefer_head: bool) -> bool:
        if self.block_size <= 1:
            return False
        lod_order = [0]
        allow_lod2 = self.max_lod >= 2
        if allow_lod2:
            if self.lod2_probability > 0.0 and self.rng.random() < self.lod2_probability:
                lod_order.insert(0, 1)
            else:
                lod_order.append(1)
        for lod in lod_order:
            if lod >= self.max_lod:
                continue
            if self._collapse_span(wc, target_lod=lod, recent_tokens=recent_tokens, prefer_head=prefer_head):
                return True
        return False

    def _collapse_span(
        self,
        wc: WorkingContext,
        *,
        target_lod: int,
        recent_tokens: int,
        prefer_head: bool = False,
    ) -> bool:
        idx = self._pick_collapse_index(
            wc,
            target_lod=target_lod,
            recent_tokens=recent_tokens,
            prefer_head=prefer_head,
        )
        if idx is None:
            return False
        edit = self._build_collapse_edit(wc, idx, target_lod)
        if edit is None:
            return False
        wc.replace(edit)
        return True

    def _pick_collapse_index(
        self,
        wc: WorkingContext,
        *,
        target_lod: int,
        recent_tokens: int,
        prefer_head: bool,
    ) -> Optional[int]:
        if target_lod < 0 or target_lod >= self.max_lod:
            return None
        block = self.block_size
        tensor = wc.to_tensor()
        length = tensor.shape[1]
        if length < block:
            return None
        positions = wc.get_positions()[0]
        lods = wc.get_lod_tensor()[0]
        total_tokens = self.tree.num_tokens()
        span = max(1, self.tree.tokens_per_entry(target_lod))
        protected_tokens = min(int(recent_tokens), max(0, total_tokens - block))
        coverage_limit = max(0, total_tokens - protected_tokens)
        valid_len = length - block + 1
        lod_window = torch.nn.functional.avg_pool1d(
            lods.eq(target_lod).float().unsqueeze(0).unsqueeze(0),
            kernel_size=block,
            stride=1,
        ).squeeze(0).squeeze(0)
        lod_mask = lod_window == 1.0
        if block == 1:
            contig_mask = torch.ones(valid_len, dtype=torch.bool, device=lod_mask.device)
        else:
            diffs = positions[1:] - positions[:-1]
            contig_window = torch.nn.functional.avg_pool1d(
                (diffs == span).float().unsqueeze(0).unsqueeze(0),
                kernel_size=block - 1,
                stride=1,
            ).squeeze(0).squeeze(0)
            contig_mask = contig_window == 1.0
        start_positions = positions[:valid_len]
        coverage_mask = (start_positions + span * block) <= coverage_limit
        combined = lod_mask[:valid_len] & contig_mask & coverage_mask
        candidate_idx = torch.nonzero(combined, as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            return None
        if prefer_head:
            return int(candidate_idx[0].item())
        choice = self.rng.randrange(0, candidate_idx.numel())
        return int(candidate_idx[choice].item())

    def _build_collapse_edit(
        self,
        wc: WorkingContext,
        wc_index: int,
        lod: int,
    ) -> Optional[WorkingContextEdit]:
        required = self.block_size
        tensor = wc.to_tensor()
        length = tensor.shape[1]
        if wc_index + required > length:
            return None
        lods = wc.get_lod_tensor()[0]
        positions = wc.get_positions()[0]
        if not torch.all(lods[wc_index : wc_index + required] == lod):
            return None
        stride = max(1, self.tree.tokens_per_entry(lod))
        global_position = int(positions[wc_index].item())
        expected = torch.arange(
            global_position,
            global_position + stride * required,
            stride,
            device=positions.device,
            dtype=positions.dtype,
        )
        if not torch.all(positions[wc_index : wc_index + required] == expected):
            return None
        parent = self.tree.get_parent_embedding(lod, global_position)
        parent_stride = self.tree.tokens_per_entry(lod + 1)
        return WorkingContextEdit(
            wc_start=wc_index,
            replacements=parent,
            lod=lod + 1,
            mc_start_position=global_position,
            stride=parent_stride,
            action="collapse",
            old_count=required,
        )

    def _expand_random_span(self, wc: WorkingContext, recent_tokens: int, *, prefer_head: bool) -> bool:
        positions = wc.get_positions()[0]
        lods = wc.get_lod_tensor()[0]
        candidates = []
        total_tokens = self.tree.num_tokens()
        head_limit = max(0, total_tokens - max(0, recent_tokens))
        for idx in range(wc.length):
            lod_val = int(lods[idx].item())
            if lod_val <= 0:
                continue
            pos = int(positions[idx].item())
            if pos >= head_limit:
                continue
            candidates.append((idx, lod_val))
        if not candidates:
            return False
        if prefer_head:
            wc_index, lod = candidates[0]
        else:
            wc_index, lod = candidates[self.rng.randrange(0, len(candidates))]
        edit = self._build_expand_edit(wc, wc_index, lod)
        if edit is None:
            return False
        wc.replace(edit)
        return True

    def _build_expand_edit(
        self,
        wc: WorkingContext,
        wc_index: int,
        lod: int,
    ) -> Optional[WorkingContextEdit]:
        if lod <= 0:
            return None
        global_position = int(wc.get_positions()[0, wc_index].item())
        children = self.tree.get_children_embeddings(lod, global_position)
        if children.shape[1] == 0:
            return None
        stride = self.tree.tokens_per_entry(lod - 1)
        return WorkingContextEdit(
            wc_start=wc_index,
            replacements=children,
            lod=lod - 1,
            mc_start_position=global_position,
            stride=stride,
            action="expand",
            old_count=1,
        )
