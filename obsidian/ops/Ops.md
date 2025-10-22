---
tags:
  - moc
  - ops
summary: Navigation map for operational notes under obsidian/ops/.
---
Covers training loops, execution checklists, performance envelopes, pruning strategy, and documentation workflow.

---

- [[Training & Operations]]
    - Alternating optimization, telemetry, validation.
    - Rotate through modules (GistNet, LensNet, LoRA), regenerate on-policy labels, track swap rate, ΔNLL, residency, latency.
- [[Implementation Roadmap]]
    - Sequenced build order aligned with plans.
    - Quick checklist for standing up the stack (compression → tree → focus → integration → evaluation).
- [[Performance Sketch]]
    - Compute/storage budgeting.
    - Provides per-step cost comparison, long-term storage scenarios, and key takeaways for planning hardware/storage.
