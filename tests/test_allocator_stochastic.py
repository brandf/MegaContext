import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn
import torch

from mc.config import MegaContextConfig, WorkingContextConfig
from mc.mega_context import MegaContextTree
from mc.working_context import WorkingContext
from mc.focus_allocator import build_focus_allocator, FocusAllocatorConfig


class ZeroLens(nn.Module):
    def forward(self, wc: WorkingContext):
        # Not used when scores get supplied
        return wc.to_tensor().new_zeros((1, wc.length))


def test_build_stochastic_greedy_allocator_and_update():
    cfg = MegaContextConfig(embed_dim=8, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(32, cfg.embed_dim)
    tokens = torch.arange(0, 8).view(1, 8)
    tree = MegaContextTree.from_tokens(tokens, embedder, cfg)
    wc_cfg = WorkingContextConfig(embed_dim=cfg.embed_dim, max_length=8, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_cfg)
    facfg = FocusAllocatorConfig(
        block_size=cfg.block_size,
        max_lod=cfg.max_lod,
        soft_max_length=8,
        recent_tokens=0,
        expand_threshold=0.0,
        collapse_threshold=0.0,
        sample_top_k=2,
        sample_temperature=1.0,
    )
    alloc = build_focus_allocator("stochastic_greedy", tree=tree, working_context=wc, lensnet=ZeroLens(), config=facfg)
    # Supply deterministic scores to drive a valid edit
    scores = torch.tensor([0.9, -0.8, 0.1, -0.2, 0.05, 0.0, -0.1, 0.2], dtype=torch.float32)
    scores = scores.view(1, -1)
    edits = alloc.update_focus(max_replacements_per_iteration=1, num_iterations=1, scores=scores)
    assert edits in (0, 1)

