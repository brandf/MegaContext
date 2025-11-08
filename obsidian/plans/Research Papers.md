---
tags:
  - plans
summary: Sequenced research-paper roadmap where each publication introduces a single novel MegaContext contribution.
---
# Research Papers Roadmap

> **Purpose:** Outline a series of publications that build on one another. Each paper introduces **one primary innovation** while leveraging results from prior work, ensuring clarity and narrative cohesion across the program.

| Paper | Working Title | Novel Contribution | Prerequisites |
|-------|---------------|--------------------|---------------|
| Paper 0 | *MegaContext: Virtualized Context for Frozen LLMs* | Demonstrate GistNet + LensNet + Focus Allocator + MegaContext Tree (in-memory) on a single GPU (Phase 1) | Implementation Roadmap Phase 1 |
| Paper 1 | *Gaussian RoPE for Global Position Awareness* | Introduce the Gaussian RoPE positional encoding driven by global positions/LOD, showing improved long-horizon stability | Paper 0 |
| Paper 2 | *MegaAttention: Hierarchical Attention over Working-Context Trees* | Present MegaAttention masks (pyramidal attention) plus MegaPrediction heads for gist-first decoding; evaluate in 8×H100 setting | Papers 0–1 |
| Paper 3 | *MegaCache & MegaCuration: Streaming MegaContext at Scale* | Describe MegaCache (disk↔RAM↔GPU cache) and telemetry-driven MegaCuration for long-lived deployments | Papers 0–2 |
| Paper 4 | *Cognitive Core & Composite MegaContexts* | Train a 1B-parameter Cognitive Core on curated contexts; show composite contexts, dynamic services, and marketplace integration | Papers 0–3 |

Each paper sits on top of the prior one(s) and cites them explicitly, so reviewers/readers can follow a clear progression of ideas.

---

## Paper 0 — MegaContext: Virtualized Context for Frozen LLMs
**Novelty:** Introduce MegaContext as a two-context system (MegaContext Tree + Working Context) using GistNet, LensNet, and the Focus Allocator to keep a frozen LLM within fixed compute budgets.

**Core sections:**
- Problem framing: frozen LLMs struggle with long-horizon edits.
- Components: GistNet compression (32→1/1024→1), LensNet focus scoring, Focus Allocator policies, in-memory MegaContext Tree & Working Context.
- Training recipe: run10 (`--mc`) on a single GPU with Phase 1 telemetry (ΔNLL@H, swap rate, residency, MFU).
- Results: show substitutability, focus accuracy, latency.

**Artifacts:** Reference implementation (mc/), run10 scripts, documentation in [[Training & Operations]].

---

## Paper 1 — Gaussian RoPE for Global Position Awareness
**Novelty:** Replace standard RoPE with Gaussian RoPE driven by global positions + LOD-derived variance, enabling consistent positional reasoning when swapping gists/tokens.

**Core sections:**
- Background: rotary embeddings, limitations for sparse/teleported context.
- Method: track global positions/LOD in MC; derive `(μ, σ)` for each entry; compute Gaussian-attenuated cos/sin tensors; override GPT’s rotary embeddings.
- Experiments: compare ΔNLL@H, swap-rate stability, long-context QA across simple vs Gaussian RoPE; include telemetry from `mc/positional.py`.
- Integration: maintain stock RoPE when `--mc` is disabled (compatibility story).

**Artifacts:** `mc/positional.py`, GPT `cos_sin_override`, CLI knobs (`--positional <impl>`).

---

## Paper 2 — MegaAttention: Hierarchical Attention over Working-Context Trees
**Novelty:** Introduce MegaAttention masks operating over wLOD trees and MegaPrediction heads, enabling gist-first decoding and structured attention patterns at scale.

**Core sections:**
- Motivation: multi-scale attention vs flat sequences.
- Method: pyramidal masks, GistNet-informed key sharing, MegaPrediction head predicting gist trajectories before tokens.
- Experiments: `speedrun.sh --mc` runs on 8×H100 comparing baseline vs MegaAttention; evaluate sparsity, latency, gist prediction accuracy.
- Tooling: MC/WC visualizations showing attention flows; telemetry dashboards for cache hits, gist accuracy.

**Prereqs:** Papers 0–1 infrastructure; ensures positional metadata and MC interfaces exist.

---

## Paper 3 — MegaCache & MegaCuration: Streaming MegaContext at Scale
**Novelty:** Present the MegaCache (disk↔RAM↔GPU hierarchical cache) plus MegaCuration (telemetry-driven pruning/compaction) for long-lived deployments.

**Core sections:**
- MegaCache: architecture, prefetch policies, hit/miss telemetry, streaming latencies.
- MegaCuration: lens-guided pruning, policy design, before/after tree stats.
- Experiments: long-running workloads showing storage savings, streaming latency improvements, curation effects on ΔNLL/residency.
- Ops guidance: dashboards, alerts, runbooks from Phase 3.

**Prereqs:** Papers 0–2; demonstrates the leap from in-memory prototypes to disk-backed pipelines.

---

## Paper 4 — Cognitive Core & Composite MegaContexts
**Novelty:** Train a 1B-parameter Cognitive Core over curated MegaContexts, enable LoRA-based “MC-izing” of pretrained bases, and launch the MegaContext marketplace + dynamic services.

**Core sections:**
- Cognitive Core: architecture, training on ≥1T tokens, reasoning evaluations.
- LoRA retrofit pipeline: toolchain for adding MC adapters to existing models.
- Marketplace & dynamic services: sharing/selling curated contexts, file-watcher services, etc.
- Experiments: quality gains from composite contexts, end-to-end workflow demos (marketplace, streaming, services).

**Prereqs:** Papers 0–3; pushes the system to production-scale ecosystems.

---

## Next Steps
- Use [[Implementation Roadmap]] phases to track engineering progress and know when each paper’s prerequisite is satisfied.
- When a phase completes, create dedicated paper-specific branches with reproducible configs, notebooks, and telemetry snapshots so drafting the paper becomes mechanical.
