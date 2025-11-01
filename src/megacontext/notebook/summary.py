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


def iter_config_sections(
    config: Mapping[str, Any],
) -> list[tuple[str, Any, str | None]]:
    """Return experiment config sections as tuples of (title, payload, docs link)."""

    ordered_sections = [
        ("Dataset", config.get("dataset"), "Telemetry"),
        ("Base Model", config.get("base_model"), "Base Runtime"),
        ("GistNet", config.get("gistnet"), "GistNet"),
    ]
    seen_keys = {"dataset", "base_model", "gistnet"}

    for key, value in config.items():
        if key in seen_keys:
            continue
        if isinstance(value, Mapping) and value:
            ordered_sections.append((key.replace("_", " ").title(), value, None))

    sections: list[tuple[str, Any, str | None]] = []
    for title, payload, link_key in ordered_sections:
        if payload in (None, {}, []):
            continue
        sections.append((title, payload, DOC_LINKS.get(link_key) if link_key else None))
    return sections


def format_config_markdown(config: Mapping[str, Any]) -> str:
    """Render experiment config sections as stacked Markdown blocks."""

    sections = iter_config_sections(config)
    if not sections:
        return "No experiment settings detected."

    blocks: list[str] = []
    for title, payload, docs_link in sections:
        heading = f"### {title}"
        if docs_link:
            heading = f"{heading} ([docs]({docs_link}))"
        yaml_dump = yaml.safe_dump(payload, sort_keys=False)
        blocks.append(f"{heading}\n```yaml\n{yaml_dump}\n```")
    return "\n\n".join(blocks)


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
