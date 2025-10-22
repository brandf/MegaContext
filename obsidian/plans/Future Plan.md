# Future Plan

Source: `planning/FUTURE_PLAN.md`. Captures post-paper ambitions spanning platformization, advanced learning, applications, and developer experience.

## Track A — Platform maturation & ecosystem
- **A.1 Multi-model support**
  - Portability tooling for new frozen bases (Qwen, LLaMA 3, Mixtral) with automated compatibility tests (`pytest -m portability`).
  - Scripted onboarding via `tools/port_model.py` plus pre-made configs.
- **A.2 Production storage & deployment**
  - Sharded, replicated storage backends with async streaming and caching.
  - Observability dashboards (Prometheus/Grafana) with alerts for memory growth, gist variance, focus anomalies.
- **A.3 API & SDK**
  - Language-agnostic SDKs (Python, TypeScript) exposing ingestion, focus control, provenance queries.
  - Hosted service templates (FastAPI/gRPC) including auth, rate limiting, billing hooks.

## Track B — Advanced learning & co-optimization
- **B.1 EM-style co-learning** — Alternating optimization across gists, lens, and LoRA adapters, exploring adaptive schedules and larger base models.
- **B.2 Cognitive core & agentic reasoning** — Train compact transformers for mixed token/gist reasoning; build agentic loops with uncertainty-aware focus triggers.
- **B.3 Training LLMs from scratch** — Jointly train base models with MegaContext-aware curricula, studying scaling laws and data filtering.
- **B.4 MegaPrediction speculative planning** — Add future partitions to the gist tree for latent CoT, hierarchical refinement, and LensNet-guided commits scored by ΔNLL/RL objectives.

## Track C — Application showcases & verticalization
- **C.1 Coding assistant** — Repo ingest pipeline, live watcher, CLI agent; benchmark on HumanEval/MBPP with MegaContext memory.
- **C.2 Knowledge workflows** — Build compliance/support MegaContexts mixing docs, specs, and logs with metadata-powered retrieval + focus.
- **C.3 Multimodal/layout use cases** — Fuse UI traces, diagrams, and optical compression insights (e.g., DeepSeek-OCR) into the hierarchy.

## Track D — Research extensions
- **D.1 Comparative studies** — Evaluate MegaContext vs RETRO, MEMGPT, etc., across new domains; publish follow-on papers on pruning, focus learning, cognitive cores.
- **D.2 Community benchmarks** — Curate open long-context leaderboards, integrate with Helm/LongEval.
- **D.3 Ethics, safety, governance** — Study provenance retention, audit trails, compliance policies for long-lived memory.

## Track E — Tooling & developer experience
- **E.1 Visualization enhancements** — Interactive explorers (web + terminal) with drill-down, playback, annotation.
- **E.2 Automation & CI** — Scripted workflows for ingestion, training, evaluation; integrate long-context regression tests in CI.
- **E.3 Documentation portal** — Launch a docs site (mkdocs or similar) aggregating architecture guides, API references, tutorials, research notes.

## Links
- [[Grand Vision]] — conceptual framing for speculative planning and long-term goals.
- [[plans/Paper Plan]] — upstream milestones feeding robust storage/telemetry and benchmarking systems.
