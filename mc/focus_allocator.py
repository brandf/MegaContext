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
    # Stochastic-greedy options (optional)
    sample_top_k: int = 4
    sample_temperature: float = 1.0


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
        # Initialize residency ages for each WC element
        self._residency = torch.zeros(self.working_context.length, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    def append(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> None:
        start_pos = self.tree.num_tokens()
        # Avoid redundant embedding runs by supplying precomputed embeddings
        self.tree.append_with_embeddings(tokens, embeddings)
        # Advance residency once per appended token
        if embeddings.dim() == 2:
            steps = 1
        elif embeddings.dim() == 3:
            steps = embeddings.shape[1]
        else:
            raise ValueError("FocusAllocatorBase.append expects embeddings shaped [B, D] or [B, T, D]")
        for _ in range(steps):
            if self._residency.numel() > 0:
                self._residency += 1
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
            # New element at tail has age 0
            self._residency = torch.cat([self._residency, torch.zeros(1, dtype=torch.long)])

    def rebuild(self, max_replacements_per_iteration: int, num_iterations: int) -> int:
        target_lod = self._choose_rebuild_lod()
        embeddings, positions = self.tree.get_level_metadata(target_lod)
        self.working_context.load_from_level(embeddings, positions, lod=target_lod)
        self._residency = torch.zeros(self.working_context.length, dtype=torch.long)
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
        # Track last edit stats for telemetry
        self._last_edit_stats = {
            "expand": 0,
            "collapse": 0,
            "total": 0,
            "wc_length": int(self.working_context.length),
            "iterations": 0,
        }
        for _ in range(num_iterations):
            self._last_edit_stats["iterations"] += 1
            # One residency tick per allocator iteration
            if self._residency.numel() > 0:
                self._residency += 1
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
                # Update residency ages across replaced span
                old_count = edit.old_count
                if old_count <= 0:
                    if getattr(edit, "action", "") == "expand":
                        old_count = 1
                    elif getattr(edit, "action", "") == "collapse":
                        old_count = self.cfg.block_size
                    else:
                        old_count = edit.replacements.shape[1]
                new_count = edit.replacements.shape[1]
                start = int(edit.wc_start)
                start = max(0, min(start, int(self._residency.numel())))
                end = min(int(self._residency.numel()), start + max(0, int(old_count)))
                left = self._residency[:start]
                right = self._residency[end:]
                middle = torch.zeros(max(0, int(new_count)), dtype=torch.long)
                self._residency = torch.cat([left, middle, right])
                total_edits += 1
                if hasattr(edit, "action") and edit.action:
                    if edit.action == "expand":
                        self._last_edit_stats["expand"] += 1
                    elif edit.action == "collapse":
                        self._last_edit_stats["collapse"] += 1
                self._last_edit_stats["total"] += 1
        # Summarize residency after updates
        if self._residency.numel() > 0:
            ages = self._residency.to(torch.float32)
            self._last_edit_stats["residency_mean"] = float(ages.mean().item())
            try:
                self._last_edit_stats["residency_p95"] = float(torch.quantile(ages, 0.95).item())
            except Exception:
                self._last_edit_stats["residency_p95"] = float(ages.max().item())
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
            action="expand",
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
            action="collapse",
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
        wc_batch = self.working_context.to_tensor().shape[0]
        if replacements.shape[0] == 1 and wc_batch > 1:
            replacements = replacements.expand(wc_batch, -1, -1).contiguous()
        elif replacements.shape[0] != wc_batch:
            raise ValueError(
                f"Mismatched batch sizes between tree ({replacements.shape[0]}) and working context ({wc_batch})"
            )
        pos0 = self.tree.get_positions_for_lod(0)
        pos_slice = pos0[:, slice_start:slice_end]
        if pos_slice.shape[0] == 1 and wc_batch > 1:
            pos_slice = pos_slice.expand(wc_batch, -1).contiguous()
        elif pos_slice.shape[0] != wc_batch:
            raise ValueError(
                f"Mismatched position batch sizes between tree ({pos_slice.shape[0]}) and working context ({wc_batch})"
            )
        mc_start = int(pos_slice[0, 0].item())
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
    if key == "greedy":
        return GreedyFocusAllocator(
            tree=tree,
            working_context=working_context,
            lensnet=lensnet,
            config=config,
        )
    if key == "stochastic_greedy":
        return StochasticGreedyFocusAllocator(
            tree=tree,
            working_context=working_context,
            lensnet=lensnet,
            config=config,
        )
    raise ValueError(f"Unsupported FocusAllocator implementation: {kind}")


class StochasticGreedyFocusAllocator(GreedyFocusAllocator):
    """
    Stochastic variant: sample among top-k |score| candidates using a softmax over
    magnitudes with temperature, while preserving protections and thresholds.
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
        # Candidate indices sorted by |score|
        order = torch.argsort(torch.abs(scores), descending=True)
        pool: List[int] = [int(i.item()) for i in order if int(i.item()) < protected_start]
        # Sampling loop
        while pool and len(edits) < max_replacements:
            top_k = min(self.cfg.sample_top_k, len(pool))
            cand = pool[:top_k]
            mags = torch.tensor([abs(float(scores[i])) for i in cand], dtype=torch.float32)
            if mags.sum().item() == 0:
                # nothing to do
                break
            probs = torch.softmax(mags / max(self.cfg.sample_temperature, 1e-6), dim=0)
            choice_idx = int(torch.multinomial(probs, num_samples=1).item())
            idx = cand[choice_idx]
            score = float(scores[idx].item())
            lod = int(lods[idx].item())
            global_pos = int(positions[idx].item())
            edit: Optional[WorkingContextEdit] = None
            # Match greedy thresholds; when over soft length, favor collapse by disabling expand
            if score >= self.cfg.expand_threshold and lod > 0 and not prefer_collapse:
                edit = self._build_expand_edit(idx, lod, global_pos)
            elif score <= -self.cfg.collapse_threshold and lod < self.cfg.max_lod:
                edit = self._build_collapse_edit(idx, lod, global_pos)
            if edit is not None:
                edits.append(edit)
            # Remove chosen index from pool regardless to avoid repeats
            pool = [p for p in pool if p != idx]
        return edits
