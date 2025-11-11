import math
from typing import List, Optional

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC unit tests")
import torch.nn as nn
import torch.nn.functional as F

from mc.config import MegaContextConfig, WorkingContextConfig, MCConfig
from mc.focus_allocator import build_focus_allocator, FocusAllocatorConfig
from mc.gistnet import GistNetBase
from mc.mega_context import MegaContextTree
from mc.runtime import MCController
import mc.runtime as mc_runtime
from mc.telemetry import NoOpTelemetryProvider, TelemetryEvent, TelemetryProvider
from mc.working_context import WorkingContext, WorkingContextEdit


class FirstTokenGistNet(GistNetBase):
    """Summarize blocks by copying the first token, highlighting hierarchy bugs."""

    def __init__(self, embed_dim: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return blocks[:, 0, :]


class ZeroLensNet(nn.Module):
    def forward(self, working_context: WorkingContext) -> torch.Tensor:  # type: ignore[override]
        length = working_context.length
        return working_context.to_tensor().new_zeros((1, length, 1))


class DummyReport:
    def log(self, *args, **kwargs) -> None:
        return


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)


class RecordingTelemetryProvider(TelemetryProvider):
    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []

    def log_event(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.transformer = DummyTransformer(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        cos_sin_override=None,
        alibi_override=None,
        inputs_embeds=None,
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.transformer.wte(idx)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )


def _build_mc_controller(
    monkeypatch,
    embed_dim: int = 8,
    block_size: int = 2,
    random_seed: Optional[int] = None,
    telemetry_provider: Optional[TelemetryProvider] = None,
) -> MCController:
    vocab_size = 32
    monkeypatch.setattr(mc_runtime, "get_report", lambda: DummyReport())
    model = DummyModel(vocab_size=vocab_size, embed_dim=embed_dim)
    config = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=16,
        block_size=block_size,
        device="cpu",
        initial_working_contexts=1,
        max_counterfactuals=1,
        allocator_recent_tokens=0,
        random_seed=random_seed,
    )
    provider = telemetry_provider or NoOpTelemetryProvider()
    return MCController(model, config, telemetry_provider=provider)


def test_mega_context_levels_shrink_progressively():
    block_size = 2
    config = MegaContextConfig(embed_dim=4, block_size=block_size, max_lod=2, device="cpu")
    embedder = nn.Embedding(32, config.embed_dim)
    tokens = torch.arange(8).view(1, 8)
    gist = FirstTokenGistNet(config.embed_dim, block_size)
    tree = MegaContextTree.from_tokens(tokens, embedder, config, gistnet=gist)

    lod1_len = tree.levels[1].shape[1]
    lod2_len = tree.levels[2].shape[1]
    assert lod2_len == math.ceil(lod1_len / block_size)


def test_focus_allocator_append_tracks_positions():
    tree_config = MegaContextConfig(embed_dim=8, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(16, tree_config.embed_dim)
    tokens = torch.tensor([[1, 2, 3]])
    tree = MegaContextTree.from_tokens(tokens, embedder, tree_config)

    wc_config = WorkingContextConfig(embed_dim=tree_config.embed_dim, max_length=16, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_config)
    allocator_cfg = FocusAllocatorConfig(
        block_size=tree_config.block_size,
        max_lod=tree_config.max_lod,
        soft_max_length=16,
        recent_tokens=0,
        expand_threshold=1.0,
        collapse_threshold=1.0,
    )
    allocator = build_focus_allocator(
        "greedy",
        tree=tree,
        working_context=wc,
        lensnet=ZeroLensNet(),
        config=allocator_cfg,
    )

    new_tokens = torch.tensor([[4, 5]])
    allocator.append(new_tokens, embedder(new_tokens))
    positions = allocator.working_context.get_positions()[0]
    assert positions[-2:].tolist() == [3, 4]


def test_focus_allocator_expand_and_collapse():
    block_size = 2
    tree_config = MegaContextConfig(embed_dim=4, block_size=block_size, max_lod=2, device="cpu")
    embedder = nn.Embedding(32, tree_config.embed_dim)
    tokens = torch.arange(8).view(1, 8)
    tree = MegaContextTree.from_tokens(tokens, embedder, tree_config)
    wc_config = WorkingContextConfig(embed_dim=tree_config.embed_dim, max_length=16, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_config)
    lod_embeddings, lod_positions = tree.get_level_metadata(1)
    wc.load_from_level(lod_embeddings, lod_positions, lod=1)

    allocator_cfg = FocusAllocatorConfig(
        block_size=block_size,
        max_lod=2,
        soft_max_length=16,
        recent_tokens=0,
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

    expand_scores = torch.zeros(1, allocator.working_context.length)
    expand_scores[0, 0] = 1.0
    edits_expand = allocator.update_focus(
        max_replacements_per_iteration=1,
        num_iterations=1,
        scores=expand_scores,
    )
    assert edits_expand == 1
    lods_after_expand = allocator.working_context.get_lod_tensor()[0]
    collapse_index = next(i for i, lod in enumerate(lods_after_expand.tolist()) if lod == 1)
    collapse_scores = torch.zeros(1, allocator.working_context.length)
    collapse_scores[0, collapse_index] = -1.0
    edits_collapse = allocator.update_focus(
        max_replacements_per_iteration=1,
        num_iterations=1,
        scores=collapse_scores,
    )
    assert edits_collapse == 1


def test_focus_allocator_respects_supplied_scores():
    tree_config = MegaContextConfig(embed_dim=4, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(16, tree_config.embed_dim)
    tokens = torch.arange(4).view(1, 4)
    tree = MegaContextTree.from_tokens(tokens, embedder, tree_config)
    wc_config = WorkingContextConfig(embed_dim=tree_config.embed_dim, max_length=16, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_config)

    class FailingLens(nn.Module):
        def forward(self, working_context: WorkingContext) -> torch.Tensor:  # type: ignore[override]
            raise AssertionError("LensNet should not be called when scores are supplied")

    allocator_cfg = FocusAllocatorConfig(
        block_size=tree_config.block_size,
        max_lod=tree_config.max_lod,
        soft_max_length=16,
        recent_tokens=0,
        expand_threshold=10.0,
        collapse_threshold=10.0,
    )
    allocator = build_focus_allocator(
        "greedy",
        tree=tree,
        working_context=wc,
        lensnet=FailingLens(),
        config=allocator_cfg,
    )
    supplied_scores = torch.zeros(1, wc.length)
    edits = allocator.update_focus(max_replacements_per_iteration=1, num_iterations=1, scores=supplied_scores)
    assert edits == 0


def test_build_focus_allocator_rejects_unsupported_kind():
    tree_config = MegaContextConfig(embed_dim=4, block_size=2, max_lod=1, device="cpu")
    embedder = nn.Embedding(16, tree_config.embed_dim)
    tokens = torch.arange(4).view(1, 4)
    tree = MegaContextTree.from_tokens(tokens, embedder, tree_config)
    wc_config = WorkingContextConfig(embed_dim=tree_config.embed_dim, max_length=16, device="cpu")
    wc = WorkingContext(tree.get_level(0), tree.get_positions_for_lod(0), wc_config)
    cfg = FocusAllocatorConfig(
        block_size=tree_config.block_size,
        max_lod=tree_config.max_lod,
        soft_max_length=16,
        recent_tokens=0,
        expand_threshold=1.0,
        collapse_threshold=1.0,
    )
    with pytest.raises(ValueError):
        build_focus_allocator(
            "simple",
            tree=tree,
            working_context=wc,
            lensnet=ZeroLensNet(),
            config=cfg,
        )


def test_mc_controller_positional_cache_requires_single_sample(monkeypatch):
    controller = _build_mc_controller(monkeypatch)
    tokens_multi = torch.randint(0, 16, (2, 8))
    result_multi = controller.process_batch(tokens_multi, step=0)
    assert result_multi.positional_cache is None

    tokens_single = torch.randint(0, 16, (1, 8))
    result_single = controller.process_batch(tokens_single, step=1)
    assert result_single.positional_cache is not None


def test_mc_controller_lod_losses_return_tuple(monkeypatch):
    controller = _build_mc_controller(monkeypatch, block_size=4)
    short_tokens = torch.randint(0, 16, (1, 2))  # shorter than block_size
    logits = torch.randn(1, short_tokens.shape[1], 32)
    lod1, lod2 = controller._compute_lod_losses(short_tokens, logits, use_lod2=False)
    assert lod1 is None and lod2 is None


def test_working_context_replace_preserves_positions():
    cfg = WorkingContextConfig(embed_dim=4, max_length=10, device="cpu")
    embeddings = torch.randn(1, 4, cfg.embed_dim)
    positions = torch.arange(0, 4).view(1, -1)
    wc = WorkingContext(embeddings, positions, cfg)

    replacements = torch.randn(1, 2, cfg.embed_dim)
    edit = WorkingContextEdit(
        wc_start=1,
        replacements=replacements,
        lod=1,
        mc_start_position=10,
        stride=2,
    )
    wc.replace(edit)
    new_positions = wc.get_positions()[0].tolist()
    assert new_positions[1:3] == [10, 12]


def test_mean_pooling_skips_padding():
    config = MegaContextConfig(embed_dim=2, block_size=4, max_lod=1, device="cpu")
    base = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    tree = MegaContextTree.from_embeddings(base, config)
    lod1 = tree.get_level(1)
    expected = torch.tensor([[[2.0, 2.0]]])
    assert torch.allclose(lod1, expected)


def test_random_span_sampling_uses_seed(monkeypatch):
    tokens = torch.arange(0, 64).view(1, 64)
    controller_a = _build_mc_controller(monkeypatch, random_seed=42)
    embedder_a = nn.Embedding(64, controller_a.config.embed_dim)
    tree_a = MegaContextTree.from_tokens(tokens, embedder_a, controller_a.config.tree_config)
    cache_a = {}
    variant_a = controller_a._build_random_span_variant(tree_a, cache_a)
    controller_b = _build_mc_controller(monkeypatch, random_seed=42)
    embedder_b = nn.Embedding(64, controller_b.config.embed_dim)
    tree_b = MegaContextTree.from_tokens(tokens, embedder_b, controller_b.config.tree_config)
    cache_b = {}
    variant_b = controller_b._build_random_span_variant(tree_b, cache_b)
    assert variant_a is not None and variant_b is not None
    assert variant_a.source == variant_b.source
    controller_alt = _build_mc_controller(monkeypatch, random_seed=7)
    embedder_alt = nn.Embedding(64, controller_alt.config.embed_dim)
    tree_alt = MegaContextTree.from_tokens(tokens, embedder_alt, controller_alt.config.tree_config)
    variant_c = controller_alt._build_random_span_variant(tree_alt, {})
    assert variant_c is not None
    assert variant_c.source != variant_a.source


def test_mc_controller_logs_batch_counters(monkeypatch):
    telemetry = RecordingTelemetryProvider()
    controller = _build_mc_controller(monkeypatch, telemetry_provider=telemetry)
    tokens = torch.randint(0, 16, (1, 4))
    controller.process_batch(tokens, step=0)
    assert any(event.event_type == "mc_batch_counters" for event in telemetry.events)


def test_mc_controller_returns_cached_embeddings(monkeypatch):
    controller = _build_mc_controller(monkeypatch)
    tokens = torch.randint(0, 16, (1, 4))
    result = controller.process_batch(tokens, step=0)
    assert result.cached_embeddings is not None
    direct = controller.embed(tokens.to(controller.device))
    assert torch.allclose(result.cached_embeddings, direct)


def test_mc_controller_provides_per_sample_positional(monkeypatch):
    controller = _build_mc_controller(monkeypatch)
    tokens = torch.randint(0, 16, (2, 4))
    result = controller.process_batch(tokens, step=0)
    cache_map = result.positional_caches
    assert len(cache_map) == 2
    for cache in cache_map.values():
        cos, sin, alibi = cache
        assert cos.shape[0] == 1
        assert sin.shape == cos.shape
