import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MC tests")
import torch.nn as nn
import torch


from mc.gistnet import TransformerGistNet


def test_transformer_gistnet_masks_padded_keys():
    torch.manual_seed(0)
    B, T, D = 2, 4, 16
    block_size = T
    # Single-head, shallow stack to keep behavior stable
    net = TransformerGistNet(embed_dim=D, block_size=block_size, layers=1, num_heads=1, pooling="mean", head="linear")
    # Construct blocks and mask where last token is padding
    blocks = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -1] = False
    # For sensitivity: make padded rows large magnitude so gradients would flow without mask
    blocks[:, -1, :] = 10.0 * torch.randn(B, D)
    blocks.requires_grad_(True)
    # Backprop with mask: padded gradients should vanish
    out_masked = net(blocks, key_padding_mask=mask)
    out_masked.sum().backward(retain_graph=True)
    grad_masked_pad = blocks.grad[:, -1].abs().sum()
    grad_masked_valid = blocks.grad[:, :-1].abs().sum()
    assert grad_masked_valid > 1e-6
    assert grad_masked_pad < 1e-6
    # Reset gradients and backprop without mask: padded gradient should be non-zero
    blocks.grad.zero_()
    out_unmasked = net(blocks)
    out_unmasked.sum().backward()
    grad_unmasked_pad = blocks.grad[:, -1].abs().sum()
    assert grad_unmasked_pad > 1e-6
