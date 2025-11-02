"""
Helpers for normalising device strings and choosing execution backends.
"""

from __future__ import annotations

import torch

_AUTO_VALUES: set[str | None] = {"auto", "", None}


def normalize_device(
    preferred: str | None,
    *,
    default_gpu: str = "cuda:0",
    fallback_cpu: str = "cpu",
    warn: bool = True,
) -> str:
    """
    Return a canonical device string, preferring GPU when available.

    Args:
        preferred: Original device setting (e.g., ``"cpu"``, ``"cuda:1"``, ``"auto"``).
        default_gpu: Device string used when `preferred` is auto-like and
            CUDA is available.
        default_gpu: Device string used when `preferred` is auto-like and CUDA
            is available.
        warn: Emit a warning when falling back from CUDA to CPU.

    Raises:
        ValueError: If ``preferred`` cannot be parsed into a valid ``torch.device``.
    """

    if preferred in _AUTO_VALUES:
        return default_gpu if torch.cuda.is_available() else fallback_cpu
    try:
        device = torch.device(preferred)
    except (TypeError, ValueError) as exc:  # pragma: no cover - validation path
        msg = f"Invalid device string {preferred!r}"
        raise ValueError(msg) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        if warn:
            print(
                (
                    "CUDA device requested "
                    f"({preferred}) but no GPU detected; falling back to CPU."
                ),
                flush=True,
            )
        return fallback_cpu
    return str(device)
