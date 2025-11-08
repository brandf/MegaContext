---
tags:
  - ops
summary: Navigation map for operational notes covering training loops, execution checklists, performance envelopes, and documentation workflow.
---
Covers training loops, execution checklists, performance envelopes, pruning strategy, and documentation workflow.

---

- [[Training & Operations]]
    - End-to-end training loops, telemetry, validation per [[MegaContext End-to-End Training]].
    - Today these loops still run inside `notebooks/megacontext.ipynb`; the nanochat trainer lives on the migration plan.
- [[Base Runtime]]
    - Decode runbook for the legacy `tools.decode_demo` command plus pointers to the planned nanochat CLI.
- [[Lifecycle]]
    - End-to-end checklist covering setup, JT phases, telemetry maintenance, and runtime hand-off (with nanochat migration notes).
- [[Performance Sketch]]
    - Compute/storage budgeting.
    - Provides per-step cost comparison, long-term storage scenarios, and key takeaways for planning hardware/storage.
- [[Nanochat Integration Guide]]
    - Planning doc for the nanochat fork, commands, CI + telemetry expectations (not yet implemented in this repo).
