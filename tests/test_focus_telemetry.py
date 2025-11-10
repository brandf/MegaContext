import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC unit tests")
import torch.nn as nn

from mc.config import MegaContextConfig, WorkingContextConfig
from mc.mega_context import MegaContextTree
from mc.working_context import WorkingContext
from mc.focus_allocator import build_focus_allocator, FocusAllocatorConfig


class ZeroLensNet(nn.Module):
    def forward(self, wc: WorkingContext):
        return wc.to_tensor().new_zeros((1, wc.length))


def test_focus_allocator_emits_edit_stats_structure():
    cfg = MegaContextConfig(embed_dim=4, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(16, cfg.embed_dim)
    tokens = torch.arange(0, 4).view(1, 4)
    tree = MegaContextTree.from_tokens(tokens, embedder, cfg)
    wc_cfg = WorkingContextConfig(embed_dim=cfg.embed_dim, max_length=8, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_cfg)
    facfg = FocusAllocatorConfig(block_size=cfg.block_size, max_lod=cfg.max_lod, soft_max_length=8)
    alloc = build_focus_allocator("greedy", tree=tree, working_context=wc, lensnet=ZeroLensNet(), config=facfg)
    edits = alloc.update_focus(max_replacements_per_iteration=1, num_iterations=1)
    assert edits == 0
    stats = getattr(alloc, "_last_edit_stats", {})
    for key in ["expand", "collapse", "total", "wc_length", "residency_mean", "residency_p95"]:
        assert key in stats

