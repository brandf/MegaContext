# Paper Plan

Canonical plan for the research paper milestone.

This milestone produces a polished, reproducible MegaContext system suitable for an academic paper. The deliverable includes rigorous evaluations against baselines, ablations, and thorough documentation, while remaining within manageable compute (<8 B frozen base models).

## Phase 1 — Robust Gist Compression & Data Foundations
**Goal:** Mature gist training to cover multiple domains, compression ratios, and quality safeguards.
- Task 1.1: Extend dataset tooling to blend narrative (PG-19, BookSum), code (The Stack slices), dialogue (ShareGPT-style), and structured-doc transcripts (OmniDocBench text renderings) with metadata tags.
- Task 1.2: Distill LLMLingua-2 style keep/drop labels from a strong teacher (GPT-4-class or curated annotations) and persist token-importance probabilities alongside gist batches.
- Task 1.3: Support multi-scale curricula—progressively tightening gist budgets (≤5×, ≤10×, >10×) and logging compression-band performance.
- Task 1.4: Implement contrastive / auxiliary losses in `src/gistnet/trainer.py` to encourage gist diversity and faithfulness.
- Task 1.5: Automate quality checks (LLMLingua faithfulness metrics, repetition detectors) gating checkpoint promotion.
- Task 1.6: Publish reproducible configs (`configs/runs/gistnet_pretrain.yaml`) and W&B dashboards summarizing ΔNLL, compression bands, and failure modes.

**Exit criteria:** Multi-domain gist models meet target losses across compression bands with validated quality gates and reproducible training scripts.

## Phase 2 — Advanced LensNet & Focus Allocation
**Goal:** Upgrade focus modeling for research-grade analyses and ablations.
- Task 2.1: Prototype Perceiver-inspired latent slots within LensNet; compare against baseline encoders using counterfactual utility metrics.
- Task 2.2: Incorporate Slot-Attention-style competition plus multi-scale expansion bundles (fine spans + coarse summaries) informed by compression bands.
- Task 2.3: Add layout/context-type features (headings, tables, formulas) to working entries using structured-doc metadata captured during ingestion.
- Task 2.4: Implement `tools/train_lensnet.py` with ranking + budget losses, curriculum scheduling, and evaluation scripts measuring swap rates, regret, and ΔLoss.
- Task 2.5: Produce ablation suites toggling components (latent slots, layout features, competition, multi-scale bundles) with statistical significance reporting.

**Exit criteria:** LensNet achieves reliable focus allocation improvements, with ablation evidence explaining contributions of each enhancement.

## Phase 3 — Storage, Streaming, and Telemetry
**Goal:** Deliver a fully persistent MegaContext system with disk-backed streaming and observability.
- Task 3.1: Extend the gist tree to support append-only disk partitions, versioning, and efficient random access (memory-mapped or chunked binary blobs).
- Task 3.2: Implement incremental update pipelines that rebuild affected subtrees on edits without reprocessing the entire corpus.
- Task 3.3: Capture provenance metadata (source IDs, timestamps, retrieval scores) and layout tags alongside gist nodes.
- Task 3.4: Add telemetry hooks logging access counts, ΔNLL sensitivity, gist variance, and slot utilisation for later pruning analysis.
- Task 3.5: Provide pruning scripts (soft delete tier) gated by utility signals and ability to restore on demand.
- Task 3.6: Document operational guidelines (`docs/storage.md`) covering ingestion, streaming, recovery, and telemetry interpretation.

**Exit criteria:** Disk-backed MegaContext runs continuously, survives restarts, streams data on demand, and records rich telemetry for research analysis.

## Phase 4 — Benchmarking, Baselines, and Visualization
**Goal:** Produce comprehensive evaluations, comparisons, and visual aids for the paper.
- Task 4.1: Build `tools/run_benchmarks.py` orchestrating evaluations on narrative QA (Natural Questions, TriviaQA), code reasoning (HumanEval-lite), and structured documents (Fox, OmniDocBench text).
- Task 4.2: Implement baselines: vanilla LLM, RAG (DPR/FiD), LLMLingua-2 compression, and ablated MegaContext variants.
- Task 4.3: Integrate metrics: Loss@H, accuracy, swap rate, latency, memory overhead, provenance fidelity.
- Task 4.4: Extend the visualization app (Phase 4 of POC) with timeline playback, MegaContext tree exploration, and overlays for focus scores and provenance.
- Task 4.5: Generate ablation plots and tables (gist compression bands, LensNet variants, pruning strategies) ready for paper figures.
- Task 4.6: Store benchmark artefacts under `artifacts/evals/<date>` with metadata, commit hashes, and environment fingerprints.

**Exit criteria:** Benchmark suite runs reproducibly, baselines are competitive, plots/tables meet publication quality, and visualizations illustrate the system clearly.

## Phase 5 — Paper Packaging & Release
**Goal:** Assemble the research paper, reproducibility kit, and supporting documentation.
- Task 5.1: Draft the paper (motivation, architecture, methodology, experiments, limitations) incorporating citations (Gist Tokens, LLMLingua-2, Perceiver, RAG, Slot Attention, DeepSeek-OCR insights).
- Task 5.2: Create `docs/paper_repro.md` detailing environment setup, dataset acquisition instructions, training schedules, benchmark commands, and expected metrics.
- Task 5.3: Bundle checkpoints (gist, lens, MegaContext storage snapshots), configs, and evaluation logs under `artifacts/paper_release/<tag>/`.
- Task 5.4: Align `README.md` and `POC_PLAN.md` references with the paper milestone; ensure instructions distinguish prototype vs research builds.
- Task 5.5: Run an external-style reproducibility audit (fresh clone, clean environment) and document results.

**Exit criteria:** Paper draft complete with supporting artefacts, reproducibility package passes audit, and repository clearly differentiates POC vs research-grade workflows.

## Links
- [[Core Components]] — reference for modules evolving through Phases 1–2.
- [[Runtime Loop]] — baseline loop instrumentation targeted by Phase 3 telemetry.
- [[Grand Vision]] — long-term themes informing visualization and packaging.
