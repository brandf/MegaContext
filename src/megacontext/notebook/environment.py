"""
Environment reporting utilities for the MegaContext notebooks.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass


@dataclass(slots=True)
class EnvironmentReport:
    """Snapshot of the local runtime environment."""

    python_version: str
    torch_version: str | None
    lightning_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: list[str]
    total_disk_gb: float
    free_disk_gb: float
    ipywidgets_available: bool
    wandb_available: bool


def _get_torch_info() -> tuple[str | None, bool, int, list[str]]:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return None, False, 0, []

    version = torch.__version__
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = []
    if cuda_available:
        for idx in range(device_count):
            try:
                device_names.append(torch.cuda.get_device_name(idx))
            except Exception:  # pragma: no cover - defensive
                device_names.append(f"cuda:{idx}")
    return version, cuda_available, device_count, device_names


def _get_lightning_version() -> str | None:
    try:
        from lightning import pytorch as pl  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return pl.__version__


def _check_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def collect_environment_report() -> EnvironmentReport:
    """Collect a snapshot of the local execution environment."""

    torch_version, cuda_available, device_count, device_names = _get_torch_info()
    lightning_version = _get_lightning_version()
    total_bytes, _, free_bytes = shutil.disk_usage(".")
    total_gb = total_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)

    return EnvironmentReport(
        python_version=platform.python_version(),
        torch_version=torch_version,
        lightning_version=lightning_version,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_device_names=device_names,
        total_disk_gb=total_gb,
        free_disk_gb=free_gb,
        ipywidgets_available=_check_module("ipywidgets"),
        wandb_available=_check_module("wandb"),
    )


def render_environment_report(report: EnvironmentReport) -> str:
    """Render the environment report as Markdown."""

    lines = [
        "| Item | Value |",
        "|------|-------|",
        f"| Python | {report.python_version} |",
        f"| PyTorch | {report.torch_version or '—'} |",
        f"| Lightning | {report.lightning_version or '—'} |",
        f"| CUDA available | {'Yes' if report.cuda_available else 'No'} |",
        "| CUDA devices | "
        + (", ".join(report.cuda_device_names) if report.cuda_device_names else "—")
        + " |",
        f"| Disk (free / total) | {report.free_disk_gb:.1f} GB / "
        f"{report.total_disk_gb:.1f} GB |",
        f"| ipywidgets | "
        f"{'Installed' if report.ipywidgets_available else 'Missing'} |",
        f"| Weights & Biases | "
        f"{'Installed' if report.wandb_available else 'Missing'} |",
    ]
    notes: list[str] = []
    if not report.cuda_available:
        notes.append(
            "- ⚠️ CUDA is not available. Training will fall back to CPU; "
            "ensure your session is attached to a GPU runtime."
        )
    if not report.ipywidgets_available:
        notes.append(
            "- ⚠️ `ipywidgets` is missing. Install it (`pip install ipywidgets`) "
            "to enable the interactive controls."
        )
    if report.cuda_available and report.free_disk_gb < 5:
        notes.append(
            "- ⚠️ Free disk space is below 5 GB. Dataset preparation may fail; "
            "move or delete unused artifacts."
        )
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)
