import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC unit tests")
import torch.nn as nn

from mc.config import MegaContextConfig
from mc.mega_context import MegaContextTree


def test_tree_append_with_embeddings_preserves_lod0_cache():
    cfg = MegaContextConfig(embed_dim=8, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(32, cfg.embed_dim)
    tokens_a = torch.tensor([[1, 2, 3]])
    tree = MegaContextTree.from_tokens(tokens_a, embedder, cfg)
    # Append two tokens with precomputed embeddings
    tokens_b = torch.tensor([[4, 5]])
    embeds_b = embedder(tokens_b)
    tree.append_with_embeddings(tokens_b, embeds_b)
    # LOD0 slice for the appended region should match supplied embeddings
    start = tokens_a.shape[1]
    end = start + tokens_b.shape[1]
    lod0_slice = tree.get_lod0_slice(start, end)
    assert torch.allclose(lod0_slice, embeds_b, atol=0, rtol=0)

