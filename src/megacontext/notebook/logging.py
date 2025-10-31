"""
Notebook-friendly helpers for configuring Lightning loggers.
"""

from __future__ import annotations

from collections.abc import Mapping

from lightning.pytorch.loggers import Logger  # type: ignore


def build_logger(
    *,
    selection: str,
    project: str | None = None,
    run_name: str | None = None,
    config: Mapping[str, object] | None = None,
) -> Logger | None:
    """
    Build a Lightning logger based on the notebook selector.

    Currently supports Weights & Biases; returns ``None`` when logging is disabled.
    """

    normalized = selection.lower()
    if normalized in {"", "none", "disabled"}:
        return None
    if normalized in {"wandb", "weights & biases"}:
        try:
            from lightning.pytorch.loggers import WandbLogger  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Weights & Biases selected but the `wandb` package is not installed. "
                "Install it with `pip install wandb`."
            ) from exc

        kwargs = {}
        if project:
            kwargs["project"] = project
        if run_name:
            kwargs["name"] = run_name
        if config is not None:
            kwargs["config"] = dict(config)
        return WandbLogger(**kwargs)

    raise ValueError(f"Unknown logger selection: {selection!r}")
