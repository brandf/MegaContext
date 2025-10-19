"""
Instrumentation helpers for logging and experiment tracking.

Phase 1 only requires lightweight plumbing: structured file logging and optional
Weights & Biases metrics guarded by an environment toggle.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "megacontext"
WANDB_ENV_FLAG = "MEGACONTEXT_ENABLE_WANDB"


def setup_logging(
    run_name: str,
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that streams to stdout and writes structured logs to a file.

    The log file path is ``artifacts/run_logs/<run_name>-<timestamp>.log`` by default.
    Subsequent calls reuse the existing logger so repeated initialisation is safe.
    """

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    log_dir = log_dir or Path("artifacts") / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{run_name}-{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Logging initialised. Writing to %s", log_path)
    return logger


def maybe_init_wandb(
    *,
    config: dict[str, Any],
    project: str = "megacontext-poc",
    run_name: str | None = None,
    entity: str | None = None,
    env_flag: str = WANDB_ENV_FLAG,
):
    """
    Lazily initialise a Weights & Biases run if ``env_flag`` is truthy.

    Returns the ``wandb`` run object when enabled; otherwise returns ``None``.
    The caller is responsible for calling ``finish()`` on the run when logging ends.
    """

    enabled = os.environ.get(env_flag, "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is part of dev extra
        raise RuntimeError(
            "wandb is not installed. Install with `uv pip install wandb`."
        ) from exc

    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=config,
    )
    return run


__all__ = ["setup_logging", "maybe_init_wandb", "WANDB_ENV_FLAG"]
