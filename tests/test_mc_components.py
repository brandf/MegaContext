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
    overrides.pop("initial_working_contexts", None)
    config = MCConfig(
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        block_size=block_size,
        device="cpu",
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


def test_random_variant_sampler_preserves_baseline_and_compresses(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_seq_len=256,
        allocator_recent_tokens=0,
        train_wc_length=64,
        num_random_variants=3,
        max_counterfactuals=4,
        random_seed=7,
    )
    tokens = (torch.arange(0, 256) % 32).view(1, 256)
    tree, sample_state, _, _ = controller._build_tree_sample(tokens, "random_variants")
    assert len(sample_state.variants) == 4
    baseline = sample_state.variants[0]
    assert baseline.is_baseline
    lods = baseline.working_context.get_lod_tensor()[0]
    positions = baseline.working_context.get_positions()[0]
    tail_start = max(0, tree.num_tokens() - controller.config.allocator_recent_tokens)
    tail_mask = positions >= tail_start
    if torch.any(tail_mask):
        assert torch.all(lods[tail_mask] == 0)
    if torch.any(~tail_mask):
        assert torch.any(lods[~tail_mask] > 0)
    total_tokens = sample_state.tree.num_tokens()
    for variant in sample_state.variants:
        coverage = controller._wc_token_coverage(variant.working_context, sample_state.tree)
        assert coverage == total_tokens
    for variant in sample_state.variants[1:]:
        assert variant.working_context.length < total_tokens
        lods = variant.working_context.get_lod_tensor()[0]
        assert torch.any(lods > 0), "compressed variants should contain non-zero LOD entries"


def test_random_variant_sampler_is_deterministic(monkeypatch):
    overrides = dict(
        max_seq_len=192,
        allocator_recent_tokens=0,
        train_wc_length=48,
        num_random_variants=2,
        max_counterfactuals=3,
    )
    controller_a = _build_mc_controller(monkeypatch, random_seed=11, **overrides)
    controller_b = _build_mc_controller(monkeypatch, random_seed=11, **overrides)
    controller_c = _build_mc_controller(monkeypatch, random_seed=13, **overrides)
    tokens = (torch.arange(0, 192) % 32).view(1, 192)

    def _signatures(controller):
        _, sample_state, _, _ = controller._build_tree_sample(tokens, "rand_det")
        sigs = []
        for variant in sample_state.variants:
            positions = variant.working_context.get_positions()[0].tolist()
            lods = variant.working_context.get_lod_tensor()[0].tolist()
            sigs.append((variant.source, positions, lods))
        return sigs

    sig_a = _signatures(controller_a)
    sig_b = _signatures(controller_b)
    sig_c = _signatures(controller_c)
    assert sig_a == sig_b, "controllers with same seed should match"
    assert sig_a != sig_c, "different seeds should change the sampled variants"


def test_pairwise_lens_loss_backprop(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_seq_len=128,
        allocator_recent_tokens=0,
        train_wc_length=32,
        num_random_variants=2,
        max_counterfactuals=3,
        random_seed=5,
        lens_kl_weight=0.05,
        lens_budget_smooth_weight=0.05,
        lens_budget_smooth_beta=0.6,
        lens_adv_norm_beta=0.8,
    )

    class LinearLens(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(1, 1)

        def forward(self, working_context: WorkingContext) -> torch.Tensor:  # type: ignore[override]
            values = working_context.to_tensor().mean(dim=-1, keepdim=True)
            return self.proj(values)

    controller.lensnet = LinearLens()
    tokens = (torch.arange(0, 128) % 32).view(1, 128)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "lens_loss")
    assert len(sample_state.variants) >= 2
    base_loss = 0.1
    for idx, variant in enumerate(sample_state.variants):
        loss_val = base_loss + 0.05 * idx
        variant.token_loss_value = torch.tensor(loss_val, dtype=torch.float32)
    loss = controller._compute_lens_losses([sample_state])
    assert loss is not None
    controller.lensnet.proj.zero_grad()
    loss.backward()
    grad = controller.lensnet.proj.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_preference_loss_stability_terms(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_seq_len=192,
        allocator_recent_tokens=0,
        train_wc_length=64,
        num_random_variants=2,
        max_counterfactuals=3,
        random_seed=17,
        lens_kl_weight=0.1,
        lens_budget_smooth_weight=0.1,
        lens_budget_smooth_beta=0.5,
        lens_adv_norm_beta=0.7,
    )
    tokens = (torch.arange(0, 192) % 32).view(1, 192)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "stability_terms")
    controller._compute_variant_losses([sample_state], tokens)
    first = controller._compute_lens_losses([sample_state])
    controller._compute_variant_losses([sample_state], tokens)
    second = controller._compute_lens_losses([sample_state])
    assert first is not None
    assert second is not None


def test_lod0_baseline_skips_focus_and_stays_lod0(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_counterfactuals=5,
        allocator_iterations=1,
        allocator_max_replacements=2,
    )
    tokens = (torch.arange(0, 64) % 32).view(1, 64)
    tree, sample_state, _, _ = controller._build_tree_sample(tokens, "baseline_focus")
    baselines = [v for v in sample_state.variants if v.is_baseline]
    assert len(baselines) == 1
    baseline = baselines[0]
    assert baseline.allocator is None
    assert baseline.edits_applied == 0
    lods = baseline.working_context.get_lod_tensor()[0]
    positions = baseline.working_context.get_positions()[0]
    tail_start = max(0, tree.num_tokens() - controller.config.allocator_recent_tokens)
    tail_mask = positions >= tail_start
    if torch.any(tail_mask):
        assert torch.all(lods[tail_mask] == 0)


def test_random_variant_lod_hint_reflects_highest_lod(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_counterfactuals=5,
        allocator_iterations=1,
        allocator_max_replacements=2,
        num_random_variants=2,
    )
    tokens = (torch.arange(0, 96) % 32).view(1, 96)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "lod_hint_sync")
    random_variants = [v for v in sample_state.variants if v.source.startswith("random_variant_")]
    assert random_variants, "expected random variants to be generated"
    for variant in random_variants:
        hist = controller._lod_histogram(variant.working_context)
        highest = max(hist.keys()) if hist else 0
        assert variant.lod_hint == highest


def test_preference_pairs_always_include_best_variant(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        num_random_variants=3,
        random_variant_iterations=2,
        train_wc_length=64,
        max_counterfactuals=5,
        max_lens_pairs=12,
    )
    tokens = (torch.arange(0, 192) % 32).view(1, 192)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "pair_best")
    controller._compute_variant_losses([sample_state], tokens)
    variants = [v for v in sample_state.variants if v.token_loss_value is not None]
    assert len(variants) >= 2
    best = min(variants, key=lambda v: float(v.token_loss_value))  # type: ignore[arg-type]
    pairs = controller._build_preference_pairs(sample_state.variants)
    for variant in variants:
        if variant is best:
            continue
        assert any(better is best and worse is variant for better, worse, _ in pairs)


def test_preference_agreement_metric_available(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        num_random_variants=2,
        random_variant_iterations=2,
        train_wc_length=64,
        max_counterfactuals=4,
    )
    tokens = (torch.arange(0, 192) % 32).view(1, 192)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "agreement_metric")
    controller._compute_variant_losses([sample_state], tokens)
    lens_loss = controller._compute_lens_losses([sample_state])
    assert lens_loss is not None
    assert controller._last_preference_agreement is not None
    assert 0.0 <= controller._last_preference_agreement <= 1.0


def test_random_variants_are_unique(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        num_random_variants=3,
        random_variant_iterations=3,
        train_wc_length=256,
        max_counterfactuals=5,
    )
    tokens = (torch.arange(0, 320) % 32).view(1, 320)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "unique_variants")
    assert sample_state.variants
    signatures = set()
    for variant in sample_state.variants:
        signatures.add(controller._variant_signature(variant.working_context))
    assert len(signatures) == len(sample_state.variants)


def test_lod_metrics_weighted_by_histogram(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        num_random_variants=2,
        random_variant_iterations=3,
        max_counterfactuals=5,
    )
    tokens = (torch.arange(0, 320) % 32).view(1, 320)
    _, sample_state, _, _ = controller._build_tree_sample(tokens, "lod_metrics")
    result = controller._compute_variant_losses([sample_state], tokens)
    (
        _,
        _,
        _,
        _,
        _,
        lod_metrics,
        lod_counts,
    ) = result
    assert 0 in lod_counts and lod_counts[0] > 0
    assert any(lod > 0 for lod in lod_counts.keys())
    assert 0 in lod_metrics


def test_lens_targets_mask_respects_legality(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
        max_lod=3,
    )
    embed_dim = controller.config.embed_dim
    wc_config = controller.config.wc_config
    positions = torch.tensor([[0, controller.config.block_size, controller.config.block_size * 2, controller.config.block_size * 2 + 1, controller.config.block_size * 3]])
    lods = torch.tensor([[1, 2, 0, 0, controller.config.max_lod]])
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
        int(positions[0, 2]): 1,  # collapse tokens into LOD1
        int(positions[0, 3]): 1,
        int(positions[0, 4]): controller.config.max_lod,  # max detail, no collapse
    }
    scores = torch.zeros(wc.length)
    delta_vs_best = 1.0
    targets, mask, span_tokens = controller._build_pairwise_targets(
        variant,
        best_map,
        scores,
        delta_vs_best,
    )
    # Entry 0: lod=1 -> target expand
    assert mask[0]
    assert pytest.approx(targets[0].item(), abs=1e-3) == math.tanh(1.0)
    # Entry 1: lod=2 -> collapse target (should apply to this block only)
    assert mask[1]
    assert pytest.approx(targets[1].item(), abs=1e-3) == -math.tanh(1.0)
    # Entries 2 & 3: lod=0 collapse => both entries in block receive same target
    assert mask[2]
    assert mask[3]
    collapse_target = math.tanh(1.0)
    assert pytest.approx(targets[2].item(), abs=1e-3) == -collapse_target
    assert pytest.approx(targets[3].item(), abs=1e-3) == -collapse_target
    # Entry 4: lod=max cannot collapse further
    assert not mask[4]
    assert targets[4].item() == 0.0
    assert span_tokens[0].item() == float(controller.config.block_size ** 1)
    assert span_tokens[1].item() == float(controller.config.block_size ** 2)


def test_train_report_uses_non_baseline_variant(monkeypatch):
    controller = _build_mc_controller(
        monkeypatch,
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
