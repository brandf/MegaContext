"""
Formatting helpers and artifact utilities for notebook output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from html import escape
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


def _format_heading_html(title: str, link_key: str | None = None) -> str:
    """Render an HTML heading with an optional docs link."""

    doc_href = DOC_LINKS.get(link_key) if link_key else None
    if doc_href:
        return (
            f'<h3 style="margin: 0 0 8px; font-size: 1.05rem;">{title} '
            f'<a href="{doc_href}" target="_blank" rel="noopener noreferrer" '
            f'style="font-size: 0.85rem; text-decoration: none;">(docs)</a></h3>'
        )
    return f'<h3 style="margin: 0 0 8px; font-size: 1.05rem;">{title}</h3>'


def format_config_markdown(config: Mapping[str, Any]) -> str:
    """Render experiment config sections inside a horizontal HTML table."""

    sections: list[tuple[str, Mapping[str, Any], str | None]] = [
        ("Dataset", config.get("dataset", {}), "Telemetry"),
        ("Base Model", config.get("base_model", {}), "Base Runtime"),
        ("GistNet", config.get("gistnet", {}), "GistNet"),
    ]

    # Include any other nested mapping sections so the preview stays complete.
    for key, value in config.items():
        if key in {"dataset", "base_model", "gistnet"}:
            continue
        if isinstance(value, Mapping):
            sections.append((key.replace("_", " ").title(), value, None))

    visible_sections = [
        (title, payload, link) for title, payload, link in sections if payload
    ]
    if not visible_sections:
        return "<p>No experiment settings detected.</p>"

    header_cells: list[str] = []
    body_cells: list[str] = []

    for title, payload, link_key in visible_sections:
        header_cells.append(
            '<th style="text-align: left; padding: 8px 12px; '
            "border-bottom: 1px solid var(--jp-border-color2, #d0d0d0); "
            'font-weight: 600; font-size: 0.95rem;">'
            f"{_format_heading_html(title, link_key)}</th>"
        )
        yaml_dump = yaml.safe_dump(payload, sort_keys=False).strip()
        body_cells.append(
            '<td style="vertical-align: top; padding: 12px;">'
            '<pre style="margin: 0; white-space: pre-wrap; word-break: break-word; '
            "font-family: var(--jp-code-font-family, monospace); "
            "font-size: var(--jp-code-font-size, 13px); "
            "background-color: var(--jp-layout-color2, #f7f7f9); "
            "border: 1px solid var(--jp-border-color2, #d0d0d0); "
            'border-radius: 6px; padding: 12px;">'
            f"{escape(yaml_dump)}"
            "</pre></td>"
        )

    table_html = (
        "<table style=\"width: 100%; border-collapse: separate; border-spacing: 0 0; "
        "table-layout: fixed;\">"
        f"<tr>{''.join(header_cells)}</tr>"
        f"<tr>{''.join(body_cells)}</tr>"
        "</table>"
    )
    return table_html


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
