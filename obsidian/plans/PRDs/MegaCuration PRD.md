---
tags:
  - plans
  - prd
summary: Proposed MegaCuration PRD – telemetry-driven pruning and compaction for the MegaContext Tree.
---
# MegaCuration: Telemetry-Guided Pruning & Compaction (Draft)

> **Status:** Draft concept. Use this note to crystallize requirements for a future POR; do not treat it as active work until the PRD stack opens space for pruning.

## 1. Problem Statement

The MegaContext Tree grows without bound as we ingest long-lived histories and as we start composing multiple MegaContexts together (Core Knowledge, Expert Domain packs, File System watchers, etc.). Even though disk is cheaper than GPU RAM, unpruned trees introduce operational and ecosystem risks:
- wasted storage and network bandwidth on never-accessed spans, especially when shipping marketplace MC bundles to end users
- slower ingestion/refocus due to oversized indexes
- inability to guarantee provenance/auditing if stale or low-trust spans linger indefinitely
- bloated “composite MegaContexts” that were meant to act like curated system prompts but end up carrying redundant or low-value detail

We already log rich telemetry (ΔNLL@H, swap rate, access counts) via LensNet/Focus Allocator. MegaCuration proposes to convert that telemetry into pruning/compaction actions with guardrails, keeping each MegaContext (personal or marketplace) focused on the highest-value gists. That means:
1. fully pruning distractors, duplicate spans, and low-utility regions
2. partially pruning by keeping only higher-level gists when low-level detail isn’t justified, so MC bundles can stay portable

## 2. Goal & Success Criteria

**Goal:** Maintain an effectively infinite history while keeping active storage, ingestion latency, and compliance surfaces manageable by automatically pruning or compaction-spanning low-value regions.

**Success looks like:**
- Retain ≥99% of spans that LensNet expands at least once per week (configurable window).
- Reduce raw LOD0 footprint by ≥10× over a decade-long corpus compared to naive retention.
- Never prune spans tagged as “protected” (manual tags, policy flags, or model-derived high-utility regions).
- Provide audit logs + rollback options for every pruning event.

# 3. Scope

## In-scope
- Telemetry pipelines that aggregate ΔNLL drift, access counts, swap rates, and provenance metadata per span.
- Policy engine defining how/when spans are marked for compaction, frozen, or deleted.
- Compaction routines that rebuild affected parts of `{LOD0,LOD1,LOD2}.ctx` without global re-ingest.
- Administrative APIs/commands for reviewing pending deletions, tagging spans as exempt, and restoring archived branches.

## Out-of-scope (for the first iteration)
- Real-time policy learning (reinforcement learning). Start with deterministic heuristics; expand later.
- Trust/safety labeling beyond basic metadata tags. Those will integrate once the broader governance story lands.
- UI/visualization beyond simple CLI reports. Logging hooks should be robust enough for future dashboards.

## 4. Proposed Architecture

### 4.1 Telemetry Aggregation
Inputs (per span or sibling set):
- ΔNLL@H history
- Swap/expand counts + residency share
- “Last expanded” timestamp
- Manual/provenance tags (domain, trust level)
- Size/precision (LOD, quantization)
- **Marketplace feedback:** optional anonymized usage telemetry contributed by downstream users (opt-in, aggregated) so MC authors can see which spans get expanded, tagged, or reported as stale across deployments.

Aggregation service (batch job or streaming):
- Roll up metrics per branch and per partition (e.g., domain, time window).
- Emit “candidate” records with scores (utility, staleness).

### 4.2 Policy Engine
Core decisions:
- **Keep:** spans above utility threshold or with protected tags.
- **Compact:** down-level the span (LOD0→LOD1, etc.) but retain parent gists.
- **Archive:** move to cold storage (separate tree file) with restore metadata.
- **Delete:** permanently remove after a retention period (default 0 for non-critical spans, >0 when compliance requires cooling-off windows).

Configuration knobs:
- Utility thresholds per domain (code vs. chat vs. logs).
- Time-based decay functions.
- Minimum/maximum disk quotas per partition.
- Tag-based overrides (manual “never prune” or “soft delete first”).

### 4.3 Execution Pipeline
1. **Schedule** – run nightly or when storage quota alarms trigger.
2. **Select candidates** – query aggregation data; exclude protected spans.
3. **Plan actions** – generate a patch plan (compact/archive/delete) with estimated savings.
4. **Dry run + approval** – produce a report for human review or auto-approve per policy.
5. **Apply actions** – rebuild relevant `{LOD0,LOD1,LOD2}.ctx` sections and update indexes.
6. **Archive logs** – record every action for rollback/audit (span ID, old/new state, timestamps, operator/policy).

### 4.4 Tooling
- CLI commands under `tools/`:
    - `megacuration.report --since <days>` – show candidates + metrics.
    - `megacuration.apply --plan plan.json` – execute a reviewed plan.
    - `megacuration.restore --span <id>` – restore from archive.
- Marketplace telemetry endpoints:
    - Author-facing dashboards summarizing aggregated usage stats (e.g., “top spans expanded across users,” “stale spans flagged by consumers”).
    - Privacy controls and opt-in contracts for consumers who share feedback (potential revenue share / pay-for-telemetry model).
- Monitoring:
    - W&B dashboards or Prometheus metrics for “bytes reclaimed,” “spans archived,” etc.

## 5. Telemetry Integration

Existing hooks from [[Telemetry]] and [[Training & Operations]]:
- Extend logging schemas to include `last_access_step`, `access_count`, `swap_rate`, `residency`, and manual tags.
- Add optional per-span ΔNLL sampling jobs (low-frequency) to catch drift.
- Ensure `Lifecycle` hand-offs include storage health notes (e.g., “prune job pending review”).
- For marketplace MC authors, expose aggregated metrics via an API or paid telemetry plan (e.g., “I Know Kung Fu” pack owner can see which branches customers actually expand).

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Pruning a span that becomes relevant later | Archive before delete; default to compaction rather than drop; allow manual tags to override. |
| ΔNLL drift due to stale gists after compaction | Trigger a re-gist job for affected parents; log ΔNLL before/after. |
| Policy misconfiguration leading to data loss | Require dry runs + plan review; add “paranoia mode” that only archives until human approval. |
| Performance impact during compaction | Run jobs asynchronously with throttling; chunk tree rebuilds; prioritize cold partitions. |
| Marketplace telemetry privacy concerns | Anonymize on ingest, enforce opt-in/paid tiers, allow users/tenants to disable sharing entirely. |

## 7. Open Questions
- Best signal for “utility”? Raw expand counts, ΔNLL sensitivity, manual tags, or combination?
- Should compaction operate per span or per partition? (e.g., entire domain vs. per branch).
- Where to store archived spans? Separate archive tree vs. versioned object store?
- How to expose pruning summaries to downstream users/agents? (Need to avoid surprising them.)
- What tiers of marketplace telemetry are acceptable? (Anonymized aggregates? pay-for-access feed for MC authors like “I Know Kung Fu”?)

## 8. Next Steps
1. Finalize telemetry schema updates (ΔNLL sampling, access counts) – align with [[Lifecycle#Validation & Troubleshooting]].
2. Prototype aggregation + report tooling (synthetic dataset).
3. Write the first real PRD once telemetry coverage is proven and storage pressure is measurable.
4. Decide where MegaCuration fits in the roadmap (post-nanochat migration, after MegaPrediction stabilization).

---

> Draft author’s note: this spec aims to capture the shape of a future PRD so we don’t lose the ideas documented in [[MegaCuration]]. When ready to prioritize pruning, clone this file, strip the “Draft” language, and slot it into the POR.
