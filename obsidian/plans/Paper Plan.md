# Paper Plan

Source: `planning/PAPER_PLAN.md`. This milestone delivers a research-ready MegaContext system with rigorous evaluations, ablations, and documentation.

## Phase 1 — Robust gist compression & data foundations
- [ ] Task 1.1 — Blend narrative, code, dialogue, and structured-doc corpora with metadata tags.
- [ ] Task 1.2 — Distill LLMLingua-style keep/drop labels and persist token importance scores.
- [ ] Task 1.3 — Support multi-scale curricula for ≤5×, ≤10×, >10× compression bands.
- [ ] Task 1.4 — Introduce contrastive/auxiliary losses in `src/gistnet/trainer.py`.
- [ ] Task 1.5 — Automate quality gates (faithfulness metrics, repetition checks) for checkpoint promotion.
- [ ] Task 1.6 — Publish reproducible configs and W&B dashboards summarizing ΔNLL and failure modes.
- **Exit criteria:** Multi-domain gists meet loss targets with validated quality checks and reproducible scripts.

## Phase 2 — Advanced LensNet & focus allocation
- [ ] Task 2.1 — Prototype Perceiver-style latent slots; compare via counterfactual utility metrics.
- [ ] Task 2.2 — Add slot-attention competition and multi-scale expansion bundles.
- [ ] Task 2.3 — Inject layout/context-type features (headings, tables, formulas) into working entries.
- [ ] Task 2.4 — Build `tools/train_lensnet.py` with ranking + budget losses, curricula, and evaluation scripts.
- [ ] Task 2.5 — Produce ablations toggling latent slots, layout features, competition, and bundles with statistical significance.
- **Exit criteria:** LensNet improvements are empirically justified with ablation evidence.

## Phase 3 — Storage, streaming, telemetry
- [ ] Task 3.1 — Extend gist tree for append-only disk partitions, versioning, efficient random access.
- [ ] Task 3.2 — Implement incremental subtree rebuilds on edits.
- [ ] Task 3.3 — Capture provenance metadata and layout tags per gist node.
- [ ] Task 3.4 — Collect telemetry (access counts, ΔNLL sensitivity, gist variance, slot utilisation).
- [ ] Task 3.5 — Provide soft-delete pruning scripts with restore-on-demand.
- [ ] Task 3.6 — Document operational guidelines in `docs/storage.md`.
- **Exit criteria:** Disk-backed MegaContext streams on demand, survives restarts, and logs rich telemetry.

## Phase 4 — Benchmarking, baselines, visualization
- [ ] Task 4.1 — Build `tools/run_benchmarks.py` covering narrative QA, code reasoning, structured docs.
- [ ] Task 4.2 — Implement baselines: vanilla LLM, RAG, LLMLingua-2 compression, ablated variants.
- [ ] Task 4.3 — Track metrics (Loss@H, accuracy, swap rate, latency, memory, provenance fidelity).
- [ ] Task 4.4 — Extend visualization app with timeline playback, tree exploration, focus overlays.
- [ ] Task 4.5 — Generate publication-quality plots/tables for compression bands, LensNet variants, pruning.
- [ ] Task 4.6 — Store benchmark artefacts under `artifacts/evals/<date>` with metadata and env fingerprints.
- **Exit criteria:** Benchmark suite runs reproducibly with competitive baselines and ready-to-publish visuals.

## Phase 5 — Paper packaging & release
- [ ] Task 5.1 — Draft the paper (motivation, architecture, experiments, limitations) with citations.
- [ ] Task 5.2 — Create `docs/paper_repro.md` for environment setup, data acquisition, training schedules, benchmarks.
- [ ] Task 5.3 — Bundle checkpoints, configs, and evaluation logs under `artifacts/paper_release/<tag>/`.
- [ ] Task 5.4 — Align `README.md` and `POC_PLAN.md` references with research milestone instructions.
- [ ] Task 5.5 — Run an external-style reproducibility audit and log results.
- **Exit criteria:** Paper draft complete, reproducibility kit passes audit, repository clarifies POC vs research workflows.

## Links
- [[Core Components]] — reference for modules evolving through Phases 1–2.
- [[Runtime Loop]] — baseline loop instrumentation targeted by Phase 3 telemetry.
- [[Grand Vision]] — long-term themes informing visualization and packaging.
