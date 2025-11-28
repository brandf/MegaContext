#!/usr/bin/env python3
"""
Runs a tiny MCController training step with torch.compile enabled for GistNet/LensNet.

Usage:
    PYTHONPATH=. uv run python scripts/mc_smoke_train.py --device cuda
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from mc.config import MCConfig
from mc.runtime import MCController
from mc.telemetry import NoOpTelemetryProvider


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test MCController + torch.compile aux nets.")
    parser.add_argument("--device", default="cuda", help="torch device to run on")
    parser.add_argument("--embed-dim", type=int, default=128, help="Dummy model embedding dimension.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the dummy run.")
    parser.add_argument("--block-size", type=int, default=16, help="GistNet block size.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads for aux nets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    vocab_size = 128
    model = DummyModel(vocab_size=vocab_size, embed_dim=args.embed_dim).to(device)
    config = MCConfig(
        embed_dim=args.embed_dim,
        max_seq_len=args.seq_len,
        block_size=args.block_size,
        device=args.device,
        num_heads=args.num_heads,
        compile_gistnet=True,
        compile_lensnet=True,
        allocator_iterations=1,
        allocator_max_replacements=1,
        max_counterfactuals=2,
        num_random_variants=2,
        random_variant_iterations=1,
        auxiliary_dtype="fp32",
    )
    controller = MCController(model, config, telemetry_provider=NoOpTelemetryProvider())
    tokens = torch.randint(0, vocab_size, (args.batch_size, args.seq_len))
    mc_result = controller.process_batch(tokens, step=0, context="train")
    if mc_result is None or mc_result.variant_loss is None:
        raise RuntimeError("MCController did not produce a training loss.")
    loss = mc_result.variant_loss
    if mc_result.lens_loss is not None:
        loss = loss + mc_result.lens_loss * controller.config.lens_loss_weight
    loss.backward()
    print("MC smoke test completed (forward + backward).", flush=True)


if __name__ == "__main__":
    main()
