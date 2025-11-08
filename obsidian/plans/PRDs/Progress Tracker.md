---
tags:
  - plans
  - prd
summary: Snapshot of each active PRD with current status, blockers, and next milestones.
---
# PRD Progress Tracker

Use this table to keep the major MegaContext PRDs in sync. Update the status column when you land changes in code or documentation; populate blockers so the next contributor can pick up immediately.

| PRD | Status | Latest updates | Next milestones |
|-----|--------|----------------|-----------------|
| [[MegaContext End-to-End Training]] | 🔄 In progress | Nanochat trainer wiring drafted; alternating references retired | Land packed-forward horizon loss in `nanochat.train`; document ΔNLL dashboards. |
| [[MegaAttention Training]] | 🔄 In progress | Hierarchical KV cache spec + positional links added | Implement pyramidal mask generator + link to FlashAttention kernels. |
| [[MegaPrediction Training]] | 🔄 In progress | Shared-head plan documented; MegaPrediction vision doc aligned | Add inference hooks + gist regression targets to nanochat decode loop. |
| [[Cognitive-Core Training]] | ⏳ Pending | Awaiting MegaAttention/MegaPrediction milestones | Define eval harness + telemetry requirements for composite WC reasoning. |
| [[Hierarchical KV Caching Strategy]] | 🔄 Drafted | Table + invalidation rules updated; FlashAttention references added | Prototype dirty-range recompute in nanochat and log cache hit ratios. |

## How to contribute

1. Update this tracker whenever you modify a PRD or land code tied to it.
2. Include CI/test output links or W&B runs in the “Latest updates” column when possible.
3. Reflect blockers discovered during migration or integration so the next assignee can act quickly.

## Related docs

- [[MegaContext PRD Index]]
- [[Migration Status]]
