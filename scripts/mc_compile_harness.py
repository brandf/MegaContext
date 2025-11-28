#!/usr/bin/env python3
"""
Exercises the torch.compile paths for GistNet and LensNet.

Run from the repository root:
    python scripts/mc_compile_harness.py --device cuda --enable-compile
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mc.gaussian_rope import build_positional  # noqa: E402
from mc.gistnet import build_gistnet  # noqa: E402
from mc.lensnet import build_lensnet  # noqa: E402
from mc.working_context import WorkingContext, WorkingContextConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Torch.compile smoke test for MegaContext modules.")
    parser.add_argument("--device", default=default_device, help="torch device to run on (default: %(default)s)")
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding dimension to test.")
    parser.add_argument("--block-size", type=int, default=32, help="Gist/Lens block size.")
    parser.add_argument("--seq-len", type=int, default=128, help="Working-context sequence length.")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads for LensNet.")
    parser.add_argument("--lensnet-iters", type=int, default=3, help="Iterations per LensNet scenario.")
    parser.add_argument("--gistnet-iters", type=int, default=3, help="Iterations for GistNet.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for LensNet batched call.")
    parser.add_argument("--enable-compile", action="store_true", help="Enable torch.compile for the harness.")
    parser.add_argument("--disable-compile", action="store_false", dest="enable_compile", help="Disable torch.compile.")
    parser.set_defaults(enable_compile=False)
    return parser.parse_args()


def maybe_compile(module: torch.nn.Module, enabled: bool, label: str) -> torch.nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode="reduce-overhead")
    except Exception as exc:  # pragma: no cover - harness should surface the failure
        raise RuntimeError(f"torch.compile failed for {label}: {exc}") from exc


def maybe_mark_step() -> None:
    mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        mark_step()


def _gaussian_rope(
    positions: torch.Tensor,
    lods: torch.Tensor,
    block_size: int,
    head_dim: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder = build_positional("gaussian", head_dim=head_dim, block_size=block_size, num_heads=num_heads)
    cos, sin, _ = encoder(positions, lods, device=positions.device)
    return cos, sin


def run_gistnet_section(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    gistnet = build_gistnet(
        "transformer",
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        layers=2,
        pooling="mean",
        head="mlp",
        num_heads=args.num_heads,
    ).to(device)
    gistnet = maybe_compile(gistnet, args.enable_compile, "GistNet")
    blocks = torch.randn(8, args.block_size, args.embed_dim, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(args.gistnet_iters):
            outputs = gistnet(blocks)
            if torch.isnan(outputs).any():
                raise RuntimeError("Detected NaNs in GistNet outputs during harness run.")


def run_lensnet_batched(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, lensnet) -> None:
    head_dim = args.embed_dim // args.num_heads
    embeddings = torch.randn(args.batch_size, args.seq_len, args.embed_dim, device=device, dtype=dtype)
    positions = torch.arange(args.seq_len, device=device).repeat(args.batch_size, 1)
    lods = torch.zeros(args.batch_size, args.seq_len, dtype=torch.long, device=device)
    cos, sin = _gaussian_rope(positions, lods, args.seq_len, head_dim, args.num_heads)
    payload = {
        "embeddings": embeddings,
        "positions": positions,
        "lods": lods,
        "cos": cos,
        "sin": sin,
    }
    with torch.no_grad():
        for _ in range(args.lensnet_iters):
            maybe_mark_step()
            scores = lensnet(None, **payload)
            if torch.isnan(scores).any():
                raise RuntimeError("Detected NaNs in LensNet batched outputs.")


def run_lensnet_working_context(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, lensnet) -> None:
    cfg = WorkingContextConfig(
        embed_dim=args.embed_dim,
        max_length=args.seq_len,
        device=device.type,
    )
    embeddings = torch.randn(1, args.seq_len, args.embed_dim, device=device, dtype=dtype)
    positions = torch.arange(args.seq_len, device=device).unsqueeze(0)
    wc = WorkingContext(embeddings, positions, cfg)
    wc.set_positional_spec("gaussian", head_dim=args.embed_dim // args.num_heads, num_heads=args.num_heads, block_size=args.seq_len)
    with torch.no_grad():
        for _ in range(args.lensnet_iters):
            maybe_mark_step()
            payload = {
                "embeddings": wc.to_tensor(),
                "positions": wc.get_positions(),
                "lods": wc.get_lod_tensor(),
                "cos": wc.get_positional_encodings()[0].clone(),
                "sin": wc.get_positional_encodings()[1].clone(),
            }
            scores = lensnet(None, **payload)
            if torch.isnan(scores).any():
                raise RuntimeError("Detected NaNs in LensNet allocator outputs.")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    run_gistnet_section(args, device, dtype)
    lensnet = build_lensnet(
        "transformer",
        embed_dim=args.embed_dim,
        max_length=args.seq_len,
        block_size=args.block_size,
        num_heads=args.num_heads,
        layers=2,
        head="mlp",
    ).to(device)
    lensnet = maybe_compile(lensnet, args.enable_compile, "LensNet")
    run_lensnet_batched(args, device, dtype, lensnet)
    run_lensnet_working_context(args, device, dtype, lensnet)
    print("mc_compile_harness completed without errors.", flush=True)


if __name__ == "__main__":
    main()
