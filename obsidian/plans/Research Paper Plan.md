---
tags:
  - plans
summary: Milestone plan for producing a research-grade MegaContext system and publication through five development phases.
---
Research milestone advancing compression, focus modeling, storage, benchmarking, and publication packaging.

---

- Phase 1 expands gist compression quality and multi-domain data foundations.
- Phase 2 upgrades [[LensNet]] focus modeling with ablations and curricula.
- Phase 3 delivers disk-backed storage, streaming, and telemetry.
- Phase 4 runs comprehensive benchmarks, baselines, and visualizations.
- Phase 5 packages the paper, reproducibility kit, and release artifacts.

---
## Details

Canonical plan for the research paper milestone.

This milestone produces a polished, reproducible MegaContext system suitable for an academic paper. The deliverable includes rigorous evaluations against baselines, ablations, and thorough documentation, while remaining within manageable compute (<8 B frozen base models).

---
## Phase 1 — Robust Gist Compression & Data Foundations
**Goal:** Mature gist training to cover multiple domains, compression ratios, and quality safeguards.
- Task 1.1: Extend dataset tooling to blend narrative (PG-19, BookSum), code (The Stack slices), dialogue (ShareGPT-style), and structured-doc transcripts (OmniDocBench text renderings) with metadata tags.
- Task 1.2: Distill LLMLingua-2 style keep/drop labels from a strong teacher (GPT-4-class or curated annotations) and persist token-importance probabilities alongside gist batches.
- Task 1.3: Support multi-scale curricula—progressively tightening gist budgets (≤5×, ≤10×, >10×) and logging compression-band performance.
- Task 1.4: Implement contrastive / auxiliary losses in `src/gistnet/trainer.py` to encourage gist diversity and faithfulness.
- Task 1.5: Automate quality checks (LLMLingua faithfulness metrics, repetition detectors) gating checkpoint promotion.
- Task 1.6: Publish reproducible configs (`configs/runs/gistnet_pretrain.yaml`) and W&B dashboards summarizing ΔNLL, compression bands, and failure modes.

**Exit criteria:** Multi-domain gist models meet target losses across compression bands with validated quality gates and reproducible training scripts.

---
## Phase 2 — Advanced LensNet & Focus Allocation
**Goal:** Upgrade focus modeling for research-grade analyses and ablations.
- Task 2.1: Prototype Perceiver-inspired latent slots within [[LensNet]]; compare against baseline encoders using counterfactual utility metrics.
- Task 2.2: Incorporate Slot-Attention-style competition plus multi-scale expansion bundles (fine spans + coarse summaries) informed by compression bands.
- Task 2.3: Add layout/context-type features (headings, tables, formulas) to working entries using structured-doc metadata captured during ingestion.
- Task 2.4: Implement `tools/train_lensnet.py` with ranking + budget losses, curriculum scheduling, and evaluation scripts measuring swap rates, regret, and ΔLoss.
- Task 2.5: Produce ablation suites toggling components (latent slots, layout features, competition, multi-scale bundles) with statistical significance reporting.

**Exit criteria:** [[LensNet]] achieves reliable focus allocation improvements, with abl
