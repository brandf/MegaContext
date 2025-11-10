from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from .mega_context import MegaContextTree
from .working_context import WorkingContext, WorkingContextEdit


@dataclass
class FocusAllocatorConfig:
    block_size: int
    max_lod: int
    soft_max_length: int
    recent_tokens: int = 0
    expand_threshold: float = 0.1
    collapse_threshold: float = 0.1


class FocusAllocatorBase:
    """
    Handles MegaContext/WorkingContext bookkeeping; subclasses only decide which
    edits to apply given LensNet scores and the current constraints.
    """

    def __init__(
        self,
        *,
        tree: MegaContextTree,
        working_context: WorkingContext,
        lensnet: torch.nn.Module,
        config: FocusAllocatorConfig,
    ) -> None:
        self.tree = tree
        self.working_context = working_context
        self.lensnet = lensnet
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    def append(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> None:
        start_pos = self.tree.num_tokens()
        self.tree.append(tokens)
        if embeddings.dim() == 2:
            seq_len = 1
        elif embeddings.dim() == 3:
            seq_len = embeddings.shape[1]
        else:
            raise ValueError("FocusAllocatorBase.append expects embeddings shaped [B, D] or [B, T, D]")
        if tokens.dim() == 2:
            tokens_to_add = tokens.shape[1]
        elif tokens.dim() == 1:
            tokens_to_add = tokens.shape[0]
        else:
            tokens_to_add = 1
        if tokens_to_add != seq_len:
            raise ValueError("Token and embedding lengths must match in FocusAllocatorBase.append")
        for offset in range(seq_len):
            step_embedding = embeddings if embeddings.dim() == 2 else embeddings[:, offset, :]
            self.working_context.append(
                step_embedding,
                lod=0,
                global_position=start_pos + offset,
            )

    def rebuild(self, max_replacements_per_iteration: int, num_iterations: int) -> int:
        target_lod = self._choose_rebuild_lod()
        embeddings, positions = self.tree.get_level_metadata(target_lod)
        self.working_context.load_from_level(embeddings, positions, lod=target_lod)
        self._reinforce_recent_tokens()
        if max_replacements_per_iteration <= 0 or num_iterations <= 0:
            return 0
        return self.update_focus(max_replacements_per_iteration, num_iterations)

    def update_focus(
        self,
        max_replacements_per_iteration: int,
        num_iterations: int,
        scores: Optional[torch.Tensor] = None,
    ) -> int:
        if max_replacements_per_iteration <= 0 or num_iterations <= 0:
            return 0
        total_edits = 0
        for _ in range(num_iterations):
            lens_scores = scores
            if lens_scores is None:
                lens_scores = self.lensnet(self.working_context)
            lens_scores = lens_scores.detach()
            if lens_scores.dim() == 2:
                # Average across batch; allocator operates on shared WC indices.
                scores_1d = lens_scores.mean(dim=0)
            else:
                scores_1d = lens_scores.squeeze(0)
            prefer_collapse = self.working_context.length > self.cfg.soft_max_length
            edits = self._select_edits(scores_1d, max_replacements_per_iteration, prefer_collapse)
            if not edits:
                break
            for edit in edits:
                self.working_context.replace(edit)
                total_edits += 1
        return total_edits

    def _select_edits(
        self,
        scores: torch.Tensor,
        max_replacements: int,
        prefer_collapse: bool,
    ) -> List[WorkingContextEdit]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Edit selection helpers
    # ------------------------------------------------------------------ #
class GreedyFocusAllocator(FocusAllocatorBase):
    """
    Thresholded greedy policy: sort by |score|, then expand/collapse nodes
    while respecting recent-token protection and soft length constraints.
    """

    def _select_edits(
        self,
        scores: torch.Tensor,
        max_replacements: int,
        prefer_collapse: bool,
    ) -> List[WorkingContextEdit]:
        edits: List[WorkingContextEdit] = []
        lods = self.working_context.get_lod_tensor()[0]
        positions = self.working_context.get_positions()[0]
        length = scores.numel()
        protected_start = max(0, length - self.cfg.recent_tokens)
        order = torch.argsort(torch.abs(scores), descending=True)
        for idx_tensor in order:
            idx = int(idx_tensor.item())
            if idx >= length or len(edits) >= max_replacements:
                break
            if idx >= protected_start:
                continue
            score = scores[idx].item()
            lod = int(lods[idx].item())
            global_pos = int(positions[idx].item())
            edit: Optional[WorkingContextEdit] = None
            if score >= self.cfg.expand_threshold and lod > 0 and not prefer_collapse:
                edit = self._build_expand_edit(idx, lod, global_pos)
            elif score <= -self.cfg.collapse_threshold and lod < self.cfg.max_lod:
                edit = self._build_collapse_edit(idx, lod, global_pos)
            if edit is not None:
                edits.append(edit)
        return edits

    def _build_expand_edit(self, wc_index: int, lod: int, global_position: int) -> Optional[WorkingContextEdit]:
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
        )

    def _build_collapse_edit(self, wc_index: int, lod: int, global_position: int) -> Optional[WorkingContextEdit]:
        required = self.cfg.block_size
        tensor = self.working_context.to_tensor()
        length = tensor.shape[1]
        if wc_index + required > length:
            return None
        lods = self.working_context.get_lod_tensor()[0]
        positions = self.working_context.get_positions()[0]
        if not torch.all(lods[wc_index : wc_index + required] == lod):
            return None
        stride = self.tree.tokens_per_entry(lod)
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
        )

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #
    def _choose_rebuild_lod(self) -> int:
        total_tokens = self.tree.num_tokens()
        for lod in range(self.cfg.max_lod, -1, -1):
            entries = math.ceil(total_tokens / self.tree.tokens_per_entry(lod))
            if entries <= self.cfg.soft_max_length:
                return lod
        return 0

    def _reinforce_recent_tokens(self) -> None:
        if self.cfg.recent_tokens <= 0:
            return
        available = self.tree.num_tokens()
        if available == 0:
            return
        tail = min(self.cfg.recent_tokens, available, self.working_context.length)
        if tail == 0:
            return
        start = self.working_context.length - tail
        slice_start = max(0, available - tail)
        slice_end = available
        replacements = self.tree.get_lod0_slice(slice_start, slice_end)
        pos0 = self.tree.get_positions_for_lod(0)
        mc_start = int(pos0[0, slice_start].item())
        edit = WorkingContextEdit(
            wc_start=start,
            replacements=replacements,
            lod=0,
            mc_start_position=mc_start,
            stride=1,
        )
        self.working_context.replace(edit)


def build_focus_allocator(
    kind: str,
    *,
    tree: MegaContextTree,
    working_context: WorkingContext,
    lensnet: torch.nn.Module,
    config: FocusAllocatorConfig,
) -> FocusAllocatorBase:
    key = kind.lower()
    if key not in {"transformer", "simple", "greedy"}:
        raise ValueError(f"Unknown FocusAllocator implementation: {kind}")
    return GreedyFocusAllocator(
        tree=tree,
        working_context=working_context,
        lensnet=lensnet,
        config=config,
    )
