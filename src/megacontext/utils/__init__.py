"""Shared utilities for MegaContext."""

from .device import normalize_device
from .env import in_colab_env, in_notebook_env
from .instrumentation import WANDB_ENV_FLAG, maybe_init_wandb, setup_logging
from .precision import normalize_dtype, resolve_runtime_precision, resolve_torch_dtype

__all__ = [
    "setup_logging",
    "maybe_init_wandb",
    "WANDB_ENV_FLAG",
    "in_notebook_env",
    "in_colab_env",
    "normalize_device",
    "normalize_dtype",
    "resolve_torch_dtype",
    "resolve_runtime_precision",
]
