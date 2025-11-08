---
tags:
  - plans
summary: Navigation map for milestone plans spanning POC, research paper, and future platform development.
---
Collects milestone plans spanning various phases of this project.

---

- [[MegaContext PRD Index]]
    - Active PRD stack (MegaContext End-to-End Training, MegaAttention, MegaPrediction, Cognitive-Core, KV caching).
    - Defines requirements for the small end-to-end models, runtime upgrades, and serving features.
- [[Research Paper Plan]]
    - Research-grade system and publication.
    - Robust compression, advanced focus modeling, storage/streaming, benchmarking, paper packaging.
- [[Future Plan]]
    - Long-horizon platform and research initiatives.
    - Platform maturation, advanced learning, application showcases, research extensions, tooling & DX.
- [[Migration Status]]
    - Running tracker for the nanochat integration phases and dependencies.

---

## Program Taxonomy

| Stream | Status | Canonical doc | Notes |
|--------|--------|---------------|-------|
| Product requirements (POR) | ✅ Active | [[MegaContext PRD Index]] | Governs current engineering scope (End-to-End Training, MegaAttention, MegaPrediction, Cognitive-Core, KV caching). |
| Nanochat migration | 🔄 Planned | [[Migration Plan - Nanochat Integration]] + [[Migration Status]] | Tracks work to replace `src/megacontext/...` with the nanochat fork; commands are illustrative until the import lands. |
| Legacy POC / Research plans | 🗂 Historical | [[Research Paper Plan]], [[POC Architecture]], `_old/` notes | Useful for background, but superseded by the PRDs—cross-check before using requirements. |

Whenever you add new plans, note which stream they belong to so contributors can quickly identify whether a document is authoritative, in-flight, or archival.
