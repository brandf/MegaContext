---
tags:
  - vision
summary: Telemetry-driven strategies for keeping the MegaContext gist tree compact without losing valuable detail.
---
Use telemetry (access, ΔNLL, query alignment) to collapse or retire low-value spans while preserving reversibility and provenance.

---

- **Signals:** access dwell time, ΔNLL sensitivity, semantic alignment.
- **Strategies:** level-aware shrinkage, version compaction, temporal decay.
- **Guardrails:** soft delete tiers, event tags, audit metadata.
- **Automation:** asynchronous jobs, [[LensNet]]-informed utility estimates, RL/bandit approaches.
- **Domains:** robotics, codebases, documentation each need tailored policies.

---
## Details

Even with disk-backed storage, a mature [[MegaContext Tree]] memory will accumulate redundant, outdated, or low-value spans. Pruning keeps the gist tree healthy without sacrificing recall.

### Signals to collect

- **Access telemetry:** track each span's dwell time, expansion count, and last access step. Branches that never surface in the [[Working Context]] become pruning candidates.
- **ΔNLL sensitivity:** periodically replace spans (or whole subtrees) with coarser ancestors and measure ΔNLL@`H`. Low sensitivity indicates the detail can be safely collapsed or removed.
- **Query alignment:** maintain lightweight embeddings or tags for recent tasks; spans that never semantically align with active queries can be deprioritized.

### Pruning strategy

- **Level-aware shrinkage:** drop long-idle L0 tokens first (keeping their L1 gist). If L1 gists are never expanded, collapse them into L2, and so on.
- **Version compaction:** keep current file versions in high detail; archive historical revisions as coarse gists or diffs to preserve traceability without hoarding tokens.
- **Temporal decay:** assign domain-specific half-lives (e.g., fast decay for sensor logs, slow for specs) so the tree naturally thins over time.

### Guardrails & recovery

- **Soft delete tier:** move candidates to a cold or "inactive" partition before permanent removal. If future telemetry indicates renewed interest, restore the branch.
- **Event tagging:** allow the [[Runtime Loop]] or humans to tag spans ("bugfix", "incident", "reward spike"). Tagged spans bypass automated pruning.
- **Audit metadata:** retain compact descriptors (timestamp, checksum, parent ID) so pruned content remains discoverable in logs, even if embeddings are gone.

### Automation considerations

- Run pruning jobs asynchronously with the main decode loop, using accumulated telemetry to schedule compaction during low-load windows.
- Extend [[LensNet]] labeling to estimate utility loss if an ancestor disappears, providing data-driven pruning hints.
- Explore RL/bandit policies that treat storage as a constrained resource and learn which spans to retain for maximal downstream reward.

### Domain notes

- **Robotics:** tie retention to reward signals or anomaly detectors; keep high-resolution data around events (collisions, novel observations) and aggressively compress idle periods.
- **Codebases:** maintain a graph from code spans to tests/issues/PRs so spans with active dependencies stay detailed; collapse stale modules to L2/L3 summaries.
- **Documentation:** preserve canonical specs verbatim, but decay meeting notes or superseded plans once new revisions land.

Pruning is easiest if provenance, access counts, and tagging hooks exist from day one; the POC should wire these metrics even if pruning is a future feature.
