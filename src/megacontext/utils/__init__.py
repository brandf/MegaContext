"""Shared utilities for MegaContext."""

from .env import in_colab_env, in_notebook_env
from .instrumentation import WANDB_ENV_FLAG, maybe_init_wandb, setup_logging

__all__ = [
    "setup_logging",
    "maybe_init_wandb",
    "WANDB_ENV_FLAG",
    "in_notebook_env",
    "in_colab_env",
]
