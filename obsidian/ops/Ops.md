---
tags:
  - ops
summary: Navigation map for operational notes covering training loops, execution checklists, performance envelopes, and documentation workflow.
---
Covers training loops, execution checklists, performance envelopes, pruning strategy, and documentation workflow.

---

- [[Training & Operations]]
    - End-to-end training loops, telemetry, validation per [[MegaContext End-to-End Training]] using the nanochat scripts (`run10.sh`, `speedrun.sh`, `run1000.sh`).
- [[Base Runtime]]
    - Runbook for the `scripts.chat_cli` / `scripts.chat_web` demos that exercise the latest checkpoints.
- [[Lifecycle]]
    - End-to-end checklist covering setup, JT phases, telemetry maintenance, and runtime hand-off (with nanochat migration notes).
- [[Performance Sketch]]
    - Compute/storage budgeting.
    - Provides per-step cost comparison, long-term storage scenarios, and key takeaways for planning hardware/storage.
- [[Nanochat Integration Guide]]
    - Commands, CI expectations, and branching strategy for the landed nanochat fork.
