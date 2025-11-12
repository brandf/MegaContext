import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn

from mc.config import MCConfig, WorkingContextConfig
from mc.runtime import MCController, SampleContext, WorkingContextVariant
from mc.working_context import WorkingContext


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.transformer = DummyTransformer(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", cos_sin_override=None, alibi_override=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.transformer.wte(idx)
        return self.lm_head(x)


def test_session_positional_prefers_lod0_baseline():
    embed_dim = 8
    model = DummyModel(vocab_size=64, embed_dim=embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=16,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        num_heads=1,
    )
    controller = MCController(model, cfg)
    # Build two WCs of different lengths
    def make_wc(length: int, lod: int, source: str) -> WorkingContextVariant:
        embeddings = torch.randn(1, length, embed_dim)
        positions = torch.arange(0, length).view(1, -1)
        wc = WorkingContext(embeddings, positions, WorkingContextConfig(embed_dim=embed_dim, max_length=64, device="cpu"))
        controller._configure_wc_positional(wc)
        return WorkingContextVariant(working_context=wc, source=source, lod_hint=lod)

    v1 = make_wc(5, 1, "lod_1")
    v0 = make_wc(3, 0, "lod_0_baseline")
    sample = SampleContext(session_id="s1", tree=None, variants=[v1, v0])  # type: ignore[arg-type]
    pos = controller._build_session_positional([sample])
    (cos, sin, _alibi) = pos["s1"]
    assert cos.shape[1] == 3  # should prefer recency WC length
