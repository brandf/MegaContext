"""
Precision utilities for resolving dtype preferences to concrete torch dtypes.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .device import normalize_device

_AUTO_VALUES: Iterable[str | None] = {"auto", "", None}


def resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    """Return a torch dtype from a string value if provided."""

    if value in _AUTO_VALUES:
        return None
    if isinstance(value, str):
        if hasattr(torch, value):
            return getattr(torch, value)
        msg = f"Unsupported torch dtype {value!r}"
        raise ValueError(msg)
    raise TypeError(f"Expected dtype string or None, received {type(value)!r}")


def normalize_dtype(
    preferred: str | None,
    device: str | torch.device,
) -> torch.dtype:
    """
    Resolve a preferred dtype into a concrete ``torch.dtype`` for the given device.

    Picks bf16/float16 for CUDA depending on capability and float32 for CPU.
    """

    resolved = resolve_torch_dtype(preferred)
    if resolved is not None:
        return resolved
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        index = (
            device_obj.index
            if device_obj.index is not None
            else torch.cuda.current_device()
        )
        major, _ = torch.cuda.get_device_capability(index)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_runtime_precision(
    *,
    device_preference: str | None,
    dtype_preference: str | None,
) -> tuple[str, torch.dtype]:
    """Resolve device + dtype preferences into runtime settings.

    Returns:
        Tuple of (device string, torch dtype).
    """

    device_str = normalize_device(device_preference)
    dtype = normalize_dtype(dtype_preference, device_str)
    return device_str, dtype
