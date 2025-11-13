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
from mc.runtime import MCController, WorkingContextVariant
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
    **overrides,
) -> MCController:
    vocab_size = 32
    monkeypatch.setattr(mc_runtime, "get_report", lambda: DummyReport())
    model = DummyModel(vocab_size=vocab_size, embed_dim=embed_dim)
    max_seq_len = overrides.pop("max_seq_len", 512)
    config = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        block_size=block_size,
        device="cpu",
        initial_working_contexts=overrides.pop("initial_working_contexts", 1),
        max_counterfactuals=overrides.pop("max_counterfactuals", 1),
        allocator_recent_tokens=overrides.pop("allocator_recent_tokens", 0),
        random_seed=random_seed,
        **overrides,
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


def test_mc_controller_handles_short_sequences(monkeypatch):
    controller = _build_mc_controller(monkeypatch, block_size=4)
    short_tokens = torch.randint(0, 16, (1, 2))  # shorter than block_size
    result = controller.process_batch(short_tokens, step=0)
    assert result.variant_loss is not None


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


def test_mean_pooling_matches_full_block():
    config = MegaContextConfig(embed_dim=2, block_size=4, max_lod=1, device="cpu")
    base = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [5.0, 5.0]]])
    tree = MegaContextTree.from_embeddings(base, config)
    lod1 = tree.get_level(1)
    expected = torch.tensor([[[(1.0 + 2.0 + 3.0 + 5.0) / 4] * 2]])
    assert torch.allclose(lod1, expected)


def test_partial_blocks_remain_lod0():
    config = MegaContextConfig(embed_dim=2, block_size=4, max_lod=1, device="cpu")
    base = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    tree = MegaContextTree.from_embeddings(base, config)
    assert 1 not in tree.levels


def test_random_span_sampling_uses_seed(monkeypatch):
    tokens = torch.arange(0, 128).view(1, 128)
    controller_a = _build_mc_controller(monkeypatch, random_seed=42, max_seq_len=64)
    embedder = nn.Embedding(128, controller_a.config.embed_dim)
    tree = MegaContextTree.from_tokens(tokens, embedder, controller_a.config.tree_config)
    cache = {}
    starts_a = controller_a._sample_random_span_starts(tree, cache, count=4)
    controller_b = _build_mc_controller(monkeypatch, random_seed=42, max_seq_len=64)
    embedder_b = nn.Embedding(128, controller_b.config.embed_dim)
    tree_b = MegaContextTree.from_tokens(tokens, embedder_b, controller_b.config.tree_config)
    starts_b = controller_b._sample_random_span_starts(tree_b, {}, count=4)
    assert starts_a == starts_b
    controller_alt = _build_mc_controller(monkeypatch, random_seed=7, max_seq_len=64)
    embedder_alt = nn.Embedding(128, controller_alt.config.embed_dim)
    tree_alt = MegaContextTree.from_tokens(tokens, embedder_alt, controller_alt.config.tree_config)
    starts_c = controller_alt._sample_random_span_starts(tree_alt, {}, count=4)
    assert starts_c != starts_a


def test_lod_ascii_renderer_handles_partial_tail(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        block_size=4,
        max_seq_len=64,
        allocator_recent_tokens=0,
    )
    tokens = torch.arange(0, 10).view(1, 10)
    tree, sample_state, _, _ = controller._build_tree_sample(tokens, "lod_ascii")
    variant = sample_state.variants[0]
    lines = controller._render_lod_ascii_lines(variant.working_context, tree.num_tokens())
    assert lines
    assert lines[0][-1] == controller._LOD_PARTIAL_CHAR


def test_mc_controller_logs_batch_counters(monkeypatch):
    telemetry = RecordingTelemetryProvider()
    controller = _build_mc_controller(monkeypatch, telemetry_provider=telemetry)
    tokens = torch.randint(0, 16, (1, 4))
    controller.process_batch(tokens, step=0)
    assert any(event.event_type == "mc_batch_counters" for event in telemetry.events)


def test_training_variant_set_includes_pure_lod0_and_highest_lod(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=3,
        max_counterfactuals=6,
        max_lod=2,
        allocator_iterations=1,
        allocator_max_replacements=2,
    )
    tokens = (torch.arange(0, 128) % 32).view(1, 128)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "train_variants")
    lod0_variants = [v for v in sample_state.variants if v.is_baseline]
    assert lod0_variants, "expected at least one pure LOD0 variant"
    lod_tensor = lod0_variants[0].working_context.get_lod_tensor()
    assert torch.all(lod_tensor == 0), "lod_0 baseline must remain all LOD0"
    highest = controller.config.max_lod
    highest_variants = [v for v in sample_state.variants if v.lod_hint == highest]
    assert highest_variants, "expected variant sourced from highest LOD"
    hv = highest_variants[0].working_context.get_lod_tensor()
    assert torch.any(hv == highest)


def test_training_variants_include_mixed_lod_entries(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=4,
        max_counterfactuals=8,
        max_lod=2,
        allocator_iterations=1,
        allocator_max_replacements=2,
    )
    tokens = (torch.arange(0, 128) % 32).view(1, 128)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "train_variants_mixed")
    mixed = [
        v
        for v in sample_state.variants
        if not v.is_baseline and torch.any(v.working_context.get_lod_tensor()[0] > 0)
    ]
    assert mixed, "expected at least one focused variant containing higher LOD entries"


def test_lod0_baseline_skips_focus_and_stays_lod0(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=3,
        max_counterfactuals=5,
        allocator_iterations=1,
        allocator_max_replacements=2,
    )
    tokens = (torch.arange(0, 64) % 32).view(1, 64)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "baseline_focus")
    baselines = [v for v in sample_state.variants if v.is_baseline]
    assert len(baselines) == 1
    baseline = baselines[0]
    assert baseline.allocator is None
    assert baseline.edits_applied == 0
    assert torch.all(baseline.working_context.get_lod_tensor() == 0)


def test_lens_targets_mask_respects_legality(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_lod=3,
    )
    embed_dim = controller.config.embed_dim
    wc_config = controller.config.wc_config
    positions = torch.tensor([[0, controller.config.block_size, controller.config.block_size * 2, controller.config.block_size * 3]])
    lods = torch.tensor([[1, 2, 0, controller.config.max_lod]])
    embeddings = torch.randn(1, positions.shape[1], embed_dim)
    wc = WorkingContext(
        embeddings,
        positions,
        wc_config,
        lod_tensor=lods,
    )
    controller._configure_wc_positional(wc)
    variant = WorkingContextVariant(working_context=wc, source="test_variant", lod_hint=1)
    best_map = {
        int(positions[0, 0]): 0,  # desire more detail
        int(positions[0, 1]): 3,  # desire less detail (collapse)
        int(positions[0, 2]): 0,  # already finest, should stay
        int(positions[0, 3]): controller.config.max_lod,  # max detail, no collapse
    }
    scores = torch.zeros(wc.length)
    targets, mask, span_tokens = controller._build_lens_targets(variant, best_map, scores)
    # Entry 0: lod=1 -> target expand
    assert mask[0]
    assert targets[0].item() == 1.0
    # Entry 1: lod=2 -> collapse target
    assert mask[1]
    assert targets[1].item() == -1.0
    # Entry 2: lod=0 cannot expand
    assert not mask[2]
    assert targets[2].item() == 0.0
    # Entry 3: lod=max cannot collapse further
    assert not mask[3]
    assert targets[3].item() == 0.0
    assert span_tokens[0].item() == float(controller.config.block_size ** 1)
    assert span_tokens[1].item() == float(controller.config.block_size ** 2)


def test_train_report_uses_non_baseline_variant(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=3,
        max_counterfactuals=6,
        max_lod=2,
        allocator_iterations=1,
        allocator_max_replacements=1,
        max_seq_len=512,
    )
    tokens = (torch.arange(0, 512) % 32).view(1, 512)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "train_report")
    controller._refresh_train_report([sample_state])
    report = controller.get_training_report()
    assert report is not None
    primary = report["primary"]
    assert primary is not None
    lod_counts = primary["lod_counts"]
    highest_nonzero_lod = max((lod for lod, count in lod_counts.items() if count > 0), default=0)
    assert highest_nonzero_lod > 0, "primary report should include higher LOD entries"


@pytest.mark.parametrize("recent", [8, 16, 32])
def test_variants_respect_recent_tokens_tail(monkeypatch, recent):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=4,
        max_counterfactuals=8,
        max_lod=2,
        allocator_recent_tokens=recent,
        allocator_iterations=1,
        allocator_max_replacements=2,
        max_seq_len=512,
    )
    tokens = (torch.arange(0, 512) % 32).view(1, 512)
    tree, sample_state, _, _ = controller._build_tree_sample(tokens, "recent_tail")
    for variant in sample_state.variants:
        wc = variant.working_context
        lods = wc.get_lod_tensor()[0]
        positions = wc.get_positions()[0]
        tail_start = max(0, tree.num_tokens() - recent)
        mask = positions >= tail_start
        if torch.any(mask):
            assert torch.all(lods[mask] == 0), "recent tail must remain LOD0"
        hist = controller._lod_histogram(wc)
        coverage = controller._wc_token_coverage(wc, tree)
        hist_equiv = controller._lod_equivalent_tokens_from_hist(hist)
        assert hist_equiv == coverage
        expected = tree.num_tokens()
        assert coverage == expected


def test_inference_session_preserves_tail_and_coverage(monkeypatch):
    recent = 32
    controller = _build_mc_controller(
        monkeypatch,
        allocator_recent_tokens=recent,
        max_seq_len=512,
        max_lod=2,
    )
    tokens = (torch.arange(0, 512) % 32).view(1, 512)
    controller.begin_inference_session(tokens)
    state = controller.inference_state
    assert state is not None
    report = controller.get_inference_report()
    assert report is not None
    assert report["coverage_tokens"] == state.tree.num_tokens()
    wc = controller.get_inference_working_context()
    assert wc is not None
    lods = wc.get_lod_tensor()[0]
    positions = wc.get_positions()[0]
    tail_start = max(0, state.tree.num_tokens() - recent)
    mask = positions >= tail_start
    if torch.any(mask):
        assert torch.all(lods[mask] == 0)
    coverage = controller._wc_token_coverage(wc, state.tree)
    expected = state.tree.num_tokens()
    assert coverage == expected


def test_mc_controller_returns_cached_embeddings(monkeypatch):
    controller = _build_mc_controller(monkeypatch)
    tokens = torch.randint(0, 16, (1, 4))
    result = controller.process_batch(tokens, step=0)
    assert result.cached_embeddings is not None
    direct = controller.embed(tokens.to(controller.device))
    assert torch.allclose(result.cached_embeddings, direct)


def test_process_batch_enforces_variant_coverage(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        initial_working_contexts=3,
        max_counterfactuals=6,
        allocator_recent_tokens=32,
        allocator_iterations=1,
        allocator_max_replacements=2,
        max_seq_len=512,
    )
    tokens = (torch.arange(0, 512) % 32).repeat(2).view(2, 512)
    controller.process_batch(tokens, step=0, context="train")
    report = controller.get_training_report()
    assert report is not None
    aggregate = report["aggregate"]
    assert aggregate["coverage_tokens"] == aggregate["expected_tokens"]


def test_validation_smoke_runs_inference_path(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        allocator_recent_tokens=32,
        max_seq_len=256,
        max_lod=2,
    )
    tokens = (torch.arange(0, 256) % 32).view(1, 256)
    controller.begin_inference_session(tokens)
    wc = controller.get_inference_working_context()
    assert wc is not None
    lods = wc.get_lod_tensor()[0]
    assert wc.length == controller.config.eval_soft_max_length
    assert torch.sum(lods == 0) >= controller.config.allocator_recent_tokens


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
