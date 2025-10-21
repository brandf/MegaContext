import torch

from megacontext.gistnet import GistNet, GistNetConfig


def test_gistnet_forward_shapes() -> None:
    config = GistNetConfig(hidden_size=16, block_size=4, num_heads=4, num_slots=2)
    model = GistNet(config)

    blocks = torch.randn(3, 5, 4, 16)
    mask = torch.ones(3, 5, 4, dtype=torch.long)

    outputs = model(blocks, attention_mask=mask)
    assert outputs.shape == (3, 5, 2, 16)


def test_gistnet_deterministic_forward() -> None:
    torch.manual_seed(0)
    config = GistNetConfig(hidden_size=8, block_size=2, num_heads=2, num_slots=1)
    model = GistNet(config)
    blocks = torch.randn(1, 2, 2, 8)

    out_a = model(blocks)
    out_b = model(blocks)
    assert torch.allclose(out_a, out_b)


def test_gistnet_raises_on_bad_shape() -> None:
    config = GistNetConfig(hidden_size=8, block_size=2, num_heads=2, num_slots=1)
    model = GistNet(config)
    bad_blocks = torch.randn(1, 1, 3, 8)

    try:
        model(bad_blocks)
    except ValueError as exc:
        assert "Expected block_size" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected ValueError for mismatched block size")
