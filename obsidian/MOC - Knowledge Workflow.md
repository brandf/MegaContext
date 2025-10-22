---
title: "Knowledge Workflow Map"
type: "moc"
status: "draft"
tags: ["moc","workflow","progressive-summarization"]
summary: "Maps the Capture → Process → Refine → Create pipeline for MegaContext documentation."
links:
  - "[[MOC - MegaContext]]"
  - "[[Training & Operations]]"
aliases: ["Knowledge Workflow"]
---

## Layer 0 · Capture Summary
- Establishes how notes move from raw capture to refined assets using Capture → Process → Refine → Create principles, progressive summarization, and metadata.

## Layer 1 · Workflow Stages
- **Capture:** inbox conventions (quick bullets, links) stored as Layer 0 summaries inside each note.
- **Process:** tag incoming notes with `status:: capture` and convert to atomic concepts.
- **Refine:** build Layer 1/Layer 2 sections with dense linking; apply metadata fields (`type`, `tags`, `summary`).
- **Create:** publish outputs (README, demos, plans) drawing from refined notes; update `status:: published`.

## Layer 2 · Implementation Notes
- Inbox captures live in the top “Layer 0” section of each note; prune once processed.
- Progressive summarization:
  - **Layer 0** — one-line summary.
  - **Layer 1** — key bullets linking to related concepts.
  - **Layer 2** — detailed sections (existing content from README migration).
- Metadata schema (YAML front matter):
  - `title` — human friendly.
  - `type` — `concept`, `moc`, `process`, `plan`.
  - `status` — `capture`, `draft`, `active`, `done`.
  - `tags` — domain/topic facets (`module/gistnet`, `workflow/capture`).
  - `summary` — short abstract used in queries.
  - `links` — expected backlinks for quick navigation.
- Graph conventions: link every concept mention inline (e.g., `[[LensNet]]`) instead of relegating to reference sections.

## Layer 3 · Queries & Automation Ideas
- Dataview snippet (Obsidian) to track pipeline:
  - `table status, summary where type="concept" sort status`
- Potential Git hooks to ensure new notes include front matter.
- Future automation: convert plan checkboxes to pipeline tags (`status:: in-progress`) for dashboards.

## Layer 4 · Change Log
- 2025-10-22: Created to codify documentation hygiene after README refactor.
