"""
Utilities for interactive MegaContext notebooks.

These helpers keep the notebook cells focused on experiment logic while centralising
environment reporting, configuration previews, metrics tracking, and artifact export.
"""

from .callbacks import MetricsTracker
from .environment import (
    EnvironmentReport,
    collect_environment_report,
    render_environment_report,
)
from .logging import build_logger
from .summary import (
    format_config_markdown,
    format_dataset_summary,
    format_training_config,
    format_training_summary,
    iter_config_sections,
    save_experiment_summary,
)

__all__ = [
    "EnvironmentReport",
    "MetricsTracker",
    "collect_environment_report",
    "render_environment_report",
    "format_config_markdown",
    "format_dataset_summary",
    "format_training_summary",
    "format_training_config",
    "iter_config_sections",
    "save_experiment_summary",
    "build_logger",
]
