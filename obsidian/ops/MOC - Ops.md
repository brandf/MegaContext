---
title: "Ops Map"
type: "moc"
status: "active"
tags: ["moc","ops"]
summary: "Navigation map for operational notes under obsidian/ops/."
links:
  - "[[Training & Operations]]"
  - "[[Implementation Roadmap]]"
  - "[[Performance Sketch]]"
  - "[[Pruning MegaContext]]"
  - "[[Knowledge Workflow]]"
aliases: ["Ops Map"]
---

- Covers training loops, execution checklists, performance envelopes, pruning strategy, and documentation workflow.

## TL;DR
- [[Training & Operations]] — alternating optimization, telemetry, validation.
- [[Implementation Roadmap]] — sequenced build order aligned with plans.
- [[Performance Sketch]] — compute/storage budgeting.
- [[Pruning MegaContext]] — telemetry-driven memory hygiene.
- [[Knowledge Workflow]] — Capture → Process → Refine → Create pipeline.

## Details
- **Training & Operations:** Rotate through modules (GistNet, LensNet, LoRA), regenerate on-policy labels, track swap rate, ΔNLL, residency, latency.
- **Implementation Roadmap:** Quick checklist for standing up the stack (compression → tree → focus → integration → evaluation).
- **Performance Sketch:** Provides per-step cost comparison, long-term storage scenarios, and key takeaways for planning hardware/storage.
- **Pruning MegaContext:** Signals (access telemetry, ΔNLL sensitivity, query alignment), strategies (level-aware shrinkage, temporal decay), guardrails (soft delete, tagging).
- **Knowledge Workflow:** Recommended note lifecycle—capture quick bullets, process into typed notes with metadata, refine via progressive summarization, publish outputs and update status.
