"""
Formatting helpers and artifact utilities for notebook output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

STATIC_DOC_BASE = "https://brandf.github.io/MegaContext"

DOC_LINKS = {
    "GistNet": f"{STATIC_DOC_BASE}/architecture/components/GistNet",
    "GistNet Training": f"{STATIC_DOC_BASE}/architecture/components/GistNet%20Training",
    "LensNet": f"{STATIC_DOC_BASE}/architecture/components/LensNet",
    "Focus Allocator": f"{STATIC_DOC_BASE}/architecture/components/Focus%20Allocator",
    "Alternating Optimization": f"{STATIC_DOC_BASE}/ops/Alternating%20Optimization",
    "Telemetry": f"{STATIC_DOC_BASE}/ops/Telemetry",
    "Base Runtime": f"{STATIC_DOC_BASE}/ops/Base%20Runtime",
}


def _format_heading(title: str, link_key: str | None = None) -> str:
    if link_key and link_key in DOC_LINKS:
        return f"### {title} ([docs]({DOC_LINKS[link_key]}))"
    return f"### {title}"


def format_config_markdown(config: Mapping[str, Any]) -> str:
    """Render a combined experiment config (dataset/base_model/gistnet) as Markdown."""

    parts: list[str] = []

    dataset_cfg = config.get("dataset", {})
    base_model_cfg = config.get("base_model", {})
    gistnet_cfg = config.get("gistnet", {})

    parts.append(_format_heading("Dataset", "Telemetry"))
    dataset_yaml = yaml.safe_dump(dataset_cfg, sort_keys=False)
    parts.append(f"```yaml\n{dataset_yaml}\n```")

    parts.append(_format_heading("Base Model", "Base Runtime"))
    base_yaml = yaml.safe_dump(base_model_cfg, sort_keys=False)
    parts.append(f"```yaml\n{base_yaml}\n```")

    parts.append(_format_heading("GistNet", "GistNet"))
    gist_yaml = yaml.safe_dump(gistnet_cfg, sort_keys=False)
    parts.append(f"```yaml\n{gist_yaml}\n```")

    return "\n\n".join(parts)


def format_dataset_summary(summary: Mapping[str, Any]) -> str:
    """Render dataset preparation summary as Markdown."""

    lines = [
        "| Split | Documents | Contexts | Examples | Hidden Size | Teacher DType |"
    ]
    lines.append(
        "|-------|-----------|----------|----------|-------------|--------------|"
    )
    for split_name, stats in summary.items():
        lines.append(
            (
                "| {split} | {documents} | {contexts} | {examples} | "
                "{hidden_size} | {dtype} |"
            ).format(
                split=split_name,
                documents=stats.get("documents", "—"),
                contexts=stats.get("contexts", "—"),
                examples=stats.get("examples", "—"),
                hidden_size=stats.get("teacher_hidden_size", "—"),
                dtype=stats.get("teacher_dtype", "—"),
            )
        )
    return "\n".join(lines)


def format_training_summary(metrics: Mapping[str, Any]) -> str:
    """Render final training metrics as Markdown bullet points."""

    if not metrics:
        return "No metrics recorded."
    lines = ["### Training Summary"]
    for key, value in metrics.items():
        if isinstance(value, float):
            display_value = f"{value:.6f}"
        else:
            display_value = str(value)
        lines.append(f"- **{key}**: {display_value}")
    return "\n".join(lines)


def format_training_config(
    training_config: Any, *, heading: str = "Training Config"
) -> str:
    """Render a Lightning training dataclass/dict as YAML inside Markdown."""

    if is_dataclass(training_config):
        payload = asdict(training_config)
    else:
        payload = dict(training_config)
    yaml_dump = yaml.safe_dump(payload, sort_keys=False)
    return f"### {heading}\n```yaml\n{yaml_dump}\n```"


def save_experiment_summary(
    *,
    output_dir: Path,
    config_path: Path,
    dataset_summary: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a JSON summary capturing config, dataset stats, and metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = config_path.stem
    summary_path = output_dir / f"{timestamp}_{experiment_name}.json"
    payload = {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "dataset_summary": dataset_summary,
        "training_metrics": training_metrics,
        "artifacts": artifacts or {},
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
