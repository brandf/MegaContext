import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC unit tests")
import torch
import torch.nn as nn

from mc.config import MegaContextConfig, WorkingContextConfig
from mc.focus_allocator import FocusAllocatorConfig, build_focus_allocator
from mc.mega_context import MegaContextTree
from mc.working_context import WorkingContext


class ZeroLensNet(nn.Module):
    def forward(self, working_context: WorkingContext):  # type: ignore[override]
        length = working_context.length
        return working_context.to_tensor().new_zeros((1, length, 1))


def _covered_tokens(wc: WorkingContext, block_size: int) -> int:
    lods = wc.get_lod_tensor()[0]
    total = 0
    for lod_value in torch.unique(lods):
        count = int((lods == lod_value).sum().item())
        total += count * (block_size ** int(lod_value.item()))
    return total


@pytest.mark.parametrize(
    "total_tokens,soft_max,recent_tokens,block_size,max_lod",
    [
        (64, 32, 0, 2, 3),
        (64, 32, 8, 2, 3),
        (128, 64, 16, 4, 2),
        (50, 80, 20, 2, 4),
    ],
)
def test_focus_allocator_rebuild_covers_exact_context(
    total_tokens: int,
    soft_max: int,
    recent_tokens: int,
    block_size: int,
    max_lod: int,
):
    embed_dim = 8
    tree_config = MegaContextConfig(
        embed_dim=embed_dim,
        block_size=block_size,
        max_lod=max_lod,
        device="cpu",
    )
    embedder = nn.Embedding(256, embed_dim)
    tokens = torch.arange(total_tokens).view(1, total_tokens)
    tree = MegaContextTree.from_tokens(tokens, embedder, tree_config)
    wc_config = WorkingContextConfig(embed_dim=embed_dim, max_length=soft_max, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_config)
    allocator_cfg = FocusAllocatorConfig(
        block_size=block_size,
        max_lod=max_lod,
        soft_max_length=soft_max,
        recent_tokens=recent_tokens,
        expand_threshold=0.5,
        collapse_threshold=0.5,
    )
    allocator = build_focus_allocator(
        "greedy",
        tree=tree,
        working_context=wc,
        lensnet=ZeroLensNet(),
        config=allocator_cfg,
    )
    allocator.rebuild(max_replacements_per_iteration=0, num_iterations=0)
    covered = _covered_tokens(allocator.working_context, block_size)
    expected = min(total_tokens, soft_max)
    assert covered == expected, f"expected {expected} tokens, got {covered}"
