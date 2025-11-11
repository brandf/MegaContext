import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn

from mc.config import MCConfig
from mc.runtime import MCController
from mc.telemetry import TelemetryEvent, TelemetryProvider


class RecordingTelemetry(TelemetryProvider):
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event: TelemetryEvent) -> None:  # type: ignore[override]
        self.events.append(event)


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


def test_inference_emits_focus_telemetry_with_residency():
    embed_dim = 8
    vocab = 32
    telemetry = RecordingTelemetry()
    model = DummyModel(vocab, embed_dim)
    cfg = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=32,
        block_size=2,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        allocator_iterations=1,  # ensure one update tick
        num_heads=1,
    )
    controller = MCController(model, cfg, telemetry_provider=telemetry)
    controller.begin_inference_session(torch.randint(0, vocab, (1, 6)))
    controller.inference_step(torch.randint(0, vocab, (1, 2)))
    # Find a focus_allocator event
    fa = [e for e in telemetry.events if e.event_type == "focus_allocator"]
    assert len(fa) > 0
    payload = fa[-1].payload
    # Check newly added telemetry fields
    for key in ["swap_rate", "num_expand", "num_collapse", "wc_length", "utilization", "residency_mean", "residency_p95"]:
        assert key in payload
