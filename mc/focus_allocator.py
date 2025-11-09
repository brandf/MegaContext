from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type

import torch

from .working_context import ReplacementPlan, WorkingContext


@dataclass
class FocusDecision:
    start: int
    count: int
    lod: int
    score: float


class FocusAllocatorBase:
    def build_plan(
        self,
        tree,
        lens_logits: torch.Tensor,
        working_context: WorkingContext,
    ) -> List[ReplacementPlan]:
        """
        Args:
            lens_logits: [B, W, 2] expand/collapse scores from LensNet.
        Returns:
            List of replacement plans describing WC edits.
        """
        raise NotImplementedError


class SimpleFocusAllocator(FocusAllocatorBase):
    def __init__(self, max_edits: int = 2, threshold: float = 0.75) -> None:
        self.max_edits = max_edits
        self.threshold = threshold

    def build_plan(
        self,
        tree,
        lens_logits: torch.Tensor,
        working_context: WorkingContext,
    ) -> List[ReplacementPlan]:
        _ = tree  # placeholder for future implementations
        probs = torch.softmax(lens_logits, dim=-1)  # [B, W, 2]
        expand_conf = probs[..., 0]  # [B, W]
        topk = torch.topk(
            expand_conf.mean(dim=0),
            k=min(self.max_edits, expand_conf.shape[1]),
        )
        plans: List[ReplacementPlan] = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            if score < self.threshold:
                continue
            start = max(0, idx - 1)
            count = min(4, working_context.to_tensor().shape[1] - start)
            slice_tensor = working_context.to_tensor()[:, start : start + count]  # [B, count, D]
            replacements = slice_tensor.mean(dim=1, keepdim=True)  # [B, 1, D]
            lod = 1
            global_pos = int(working_context.get_positions()[0, start].item())
            plans.append(
                ReplacementPlan(
                    start=start,
                    count=count,
                    replacements=replacements,
                    lod=lod,
                    global_start=global_pos,
                )
            )
        return plans


ALLOCATOR_REGISTRY: Dict[str, Type[FocusAllocatorBase]] = {
    "simple": SimpleFocusAllocator,
}


def build_focus_allocator(kind: str, **kwargs) -> FocusAllocatorBase:
    key = kind.lower()
    if key not in ALLOCATOR_REGISTRY:
        raise ValueError(f"Unknown FocusAllocator implementation: {kind}")
    return ALLOCATOR_REGISTRY[key](**kwargs)
