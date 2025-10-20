# MegaContext POC Implementation Plan

This roadmap turns the MegaContext architecture from `README.md` into an executable proof-of-concept. Each phase builds on the previous one; start a phase only after the prior phase’s exit criteria are satisfied. Tests default to `uv run pytest --maxfail=1 --disable-warnings`. Update `README.md` and relevant configs whenever an interface or workflow changes.

## Phase 1 — Base Runtime & Infrastructure
**Goal:** Stand up the frozen base-model runtime, data tooling, and observability so later modules plug into a stable loop.

- **Feature 1.1: Environment bootstrap & repository skeleton**
  - [x] Task 1.1.1: Add `pyproject.toml` targeting Python 3.11 with dependencies (`torch`, `transformers`, `accelerate`, `datasets`, `wandb`, `uv`, `ruff`, `black`, `pytest`); lock with `uv lock`.
  - [x] Task 1.1.2: Provide `tools/bootstrap_env.py` (or shell script) that runs `uv venv`, installs requirements, and prints canonical commands.
  - [x] Task 1.1.3: Configure `ruff`/`black` plus pre-commit hooks in `.pre-commit-config.yaml`; wire `uv run pre-commit install`.
  - [x] Task 1.1.4: Scaffold module directories (`src/runtime`, `src/data`, `src/utils`) with parallel `tests/` stubs and placeholder smoke tests.

- **Feature 1.2: Dataset staging pipeline**
  - [x] Task 1.2.1: Implement `tools/prepare_dataset.py` to tokenize raw text into contiguous 32-token blocks, writing `.arrow` shards under `data/<dataset>/<split>.arrow` plus metadata (tokenizer, block size).
  - [x] Task 1.2.2: Define `configs/data/<dataset>.yaml`; validate via Pydantic or `pydantic-core` schemas.
  - [x] Task 1.2.3: Add dataset unit tests covering config parsing and a tiny synthetic corpus to prove deterministic chunking.

- **Feature 1.3: Base LLM wrapper & working context I/O**
  - [x] Task 1.3.1: Implement `src/runtime/base_model.py` with `BaseModel.from_pretrained()` loading `HuggingFaceTB/SmolLM3-3B` (bf16), exposing `forward(inputs_embeds, attention_mask, position_ids=None)`.
  - [x] Task 1.3.2: Create an initial `src/runtime/working_context.py` that returns pass-through tensors from tokens only; include dataclasses matching `WorkingEntry` from the README for later replacement.
  - [x] Task 1.3.3: Provide CLI `uv run python -m tools.decode_demo --config configs/runs/base_llm.yaml` that streams a short prompt and verifies logits generation.
  - [x] Task 1.3.4: Write integration tests mocking a small tokenizer to assert logits have correct batch/sequence shapes.

- **Feature 1.4: Observability & logging**
  - [x] Task 1.4.1: Wire basic Weights & Biases logging (guarded by env flag) for losses, throughput, and config snapshots.
  - [x] Task 1.4.2: Add structured logging to `artifacts/run_logs/` for latency, memory, and token throughput.
  - [x] Task 1.4.3: Document setup + run instructions in `README.md` and `docs/runs/base_runtime.md`.

**Exit criteria:** One-command environment setup, dataset preparation for a sample corpus, base LLM decoding 32-token blocks via CLI, CI passes on lint/tests, and metrics visible in W&B.

## Phase 2 — GistNet Implementation & Training
**Goal:** Train the 32→1 hierarchical GistNet so token spans can be replaced by gists with minimal ΔNLL.

- **Feature 2.1: GistNet model implementation**
  - Task 2.1.1: Implement `src/gistnet/blocks.py` (self-attn + cross-attn modules with RoPE) and `src/gistnet/model.py` exposing `GistNet.forward(tokens: Tensor[32, d]) -> Tensor[d]`.
  - Task 2.1.2: Add shared learned slot queries `Q₁`, `Q₂`, residual MLPs, and stacking logic for two-level hierarchy (L1, L2).
  - Task 2.1.3: Ensure outputs align with the base embedding dim; register buffers for block size and positional offsets.

- **Feature 2.2: GistNet training data pipeline**
  - Task 2.2.1: Extend dataset tooling to emit paired `(full_context, gist_replaced_context)` tensors over horizons `H=64`.
  - Task 2.2.2: Cache token embeddings from the frozen base model to avoid redundant forward passes.
  - Task 2.2.3: Stage long-form corpora for training: DeepMind PG-19 (novels), AllenAI BookSum (chapter-level prose + summaries), and filtered slices of The Stack (Python/TypeScript/JavaScript) for structured code traces.
  - Task 2.2.4: Provide `configs/runs/gistnet_pretrain.yaml` capturing dataset blends (narrative vs code weights), optimizer, and logging cadence; include held-out splits from each corpus for ΔNLL evaluation.

- **Feature 2.3: Training loop & losses**
  - Task 2.3.1: Implement `src/gistnet/trainer.py` computing substitutability ΔNLL@H and optional contrastive losses; support gradient accumulation.
  - Task 2.3.2: Add CLI `uv run python -m tools.train_gistnet --config ...` with resume/checkpoint support, writing to `artifacts/checkpoints/gistnet/`.
  - Task 2.3.3: Track ΔNLL metrics in W&B, emit per-span diagnostics (e.g., boundary tokens).

- **Feature 2.4: Evaluation & tests**
  - Task 2.4.1: Add unit tests for block alignment, tensor shapes, and determinism under seeded RNG.
  - Task 2.4.2: Create a smoke eval script comparing base vs gist-replaced Loss@H on a held-out dataset; target ≤5% degradation.
  - Task 2.4.3: Update `README.md` with training results and add `docs/gistnet.md` summarizing architecture knobs.

**Exit criteria:** Trained GistNet checkpoints for both layers, ΔNLL@H dashboards, reproducible training scripts, and tests ensuring deterministic outputs.

## Phase 3 — Lifetime Tree, LensNet, and Focus Allocator
**Goal:** Build the runtime loop that ingests tokens, maintains the lifetime gist tree, scores focus, and keeps the working context within budget.

- **Feature 3.1: Lifetime gist tree & storage**
  - Task 3.1.1: Implement `src/lifetime/tree.py` with node structs (`span_id`, `start_token`, `level`, parent/child pointers) and ingest/update APIs honoring 32-token blocks.
  - Task 3.1.2: Add serialization to `{L0,L1,L2}.ctx` files per the binary layout (headers, offsets) with bf16/fp16 payload support.
  - Task 3.1.3: Provide round-trip tests ensuring deterministic offsets and lossless persistence.

- **Feature 3.2: Working context manager**
  - Task 3.2.1: Implement `src/runtime/working_context.py` to tile spans contiguously, compute token-equivalent costs, and expose the `pack()`/`to_tensors()` APIs from the README.
  - Task 3.2.2: Add `TailGistCache` utilities returning L2 + recent L1 gists for LensNet conditioning.
  - Task 3.2.3: Write tests verifying contiguity, cost accounting, and legality masks.

- **Feature 3.3: LensNet model**
  - Task 3.3.1: Implement `src/lensnet/model.py` with dual cross-attention blocks, scalar feature embeddings, legality masking, and signed score head.
  - Task 3.3.2: Build `src/lensnet/dataloader.py` replaying working-context snapshots with counterfactual utility targets.
  - Task 3.3.3: Create training script `tools/train_lensnet.py` logging regression/ranking/budget losses to W&B.

- **Feature 3.4: Focus allocator**
  - Task 3.4.1: Implement `src/runtime/focus_allocator.py` with greedy expand/collapse loop, thresholds, cooldown logic, and diff limits.
  - Task 3.4.2: Enforce contiguity, token budgets, and legality masks; integrate with `WorkingEntry` data.
  - Task 3.4.3: Provide unit tests for expand/collapse scenarios, hysteresis, and budget guardrails.

- **Feature 3.5: Integrated runtime loop**
  - Task 3.5.1: Assemble `src/runtime/engine.py` ingesting streams, updating the lifetime tree, calling LensNet/allocator, and decoding via the base model.
  - Task 3.5.2: Add CLI `uv run python -m tools.run_poc_loop --config configs/runs/poc_smollm3.yaml` processing a sample dataset and logging telemetry.
  - Task 3.5.3: Implement end-to-end tests with synthetic streams verifying token budgets, focus score signs, and decode outputs under seeded RNG.

**Exit criteria:** End-to-end runtime loop executes with mocked datasets, lifetime tree persists correctly, LensNet scores apply legal focus actions, and integration tests confirm budget invariants.

## Phase 4 — Evaluation Harness & Visual Analytics
**Goal:** Quantify MegaContext gains against baselines and deliver real-time visuals for the lifetime/working context interaction.

- **Feature 4.1: Benchmark suite**
  - Task 4.1.1: Define `configs/eval/<benchmark>.yaml` covering narrative, coding, and retrieval-heavy tasks with expected metric budgets.
  - Task 4.1.2: Implement `tools/run_benchmarks.py` orchestrating base vs MegaContext runs, collecting Loss@H, accuracy, latency, and swap metrics.
  - Task 4.1.3: Store results under `artifacts/evals/<date>` with metadata (model, config, commit hash).

- **Feature 4.2: MegaContext visualization web app**
  - Task 4.2.1: Build a backend service (e.g., FastAPI + WebSocket) streaming working-context state, focus scores, and lifetime node metadata in near real time.
  - Task 4.2.2: Implement a front-end (React/Vite or similar) rendering the working context as vertical glyphs color-coded by LOD, with click-through to lifetime tree explorers showing L0 text/gists.
  - Task 4.2.3: Add playback mode for recorded runs and ensure dashboards link to W&B metrics.
  - Task 4.2.4: Package deployment instructions (`docs/visualizer.md`) and integration tests for the streaming API.

- **Feature 4.3: Ablations & reports**
  - Task 4.3.1: Automate runs toggling LensNet, GistNet, and allocator components to measure Δ performance; capture swap rate, residency histograms, and token budgets.
  - Task 4.3.2: Generate narrative reports (`docs/reports/poc_eval_<date>.md`) summarizing benchmarks and key visuals.
  - Task 4.3.3: Update `README.md` with highlight metrics and links to the visualization app.

**Exit criteria:** Benchmark scripts run reproducibly, visualization app streams live working-context data with drill-down to lifetime nodes, ablation data stored with plots, and documentation reflects eval setup plus visualization usage.

## Phase 5 — Alternate Base Model Portability (Qwen Qwen2.5-Coder-3B-Instruct)
**Goal:** Prove the stack can adapt to a different frozen model with minimal friction and codify the portability workflow.

- **Feature 5.1: Model integration & configs**
  - Task 5.1.1: Add configuration `configs/runs/qwen25_coder3b.yaml` selecting `Qwen/Qwen2.5-Coder-3B-Instruct`, precision settings, and tokenizer overrides.
  - Task 5.1.2: Extend `BaseModel.from_pretrained()` to cover Qwen coder variants (both 3B and 32B) with appropriate attention mask handling.
  - Task 5.1.3: Provide smoke tests ensuring logits and embedding dims align with GistNet/LensNet assumptions.

- **Feature 5.2: Portability pipeline**
  - Task 5.2.1: Script `tools/port_model.py` that clones configs, verifies dataset compatibility, and scaffolds gist/lens training jobs for the new base.
  - Task 5.2.2: Run Phase 2–4 workflows on the Qwen 3B model; document deltas (training time, ΔNLL, latency).
  - Task 5.2.3: Record reusable adapter notes in `docs/portability.md` (tokenization quirks, attention mask shapes, precision caveats).

- **Feature 5.3: Regression tests & CI hooks**
  - Task 5.3.1: Add model-specific regression tests (shape checks, runtime smoke loops) gated behind `pytest -m portability`.
  - Task 5.3.2: Ensure CI can run lightweight checks for both base models (SmolLM3 + Qwen 3B) within resource limits.

**Exit criteria:** Qwen2.5-Coder-3B runs through gist/lens pipelines with documented results, portability script/templates exist, and CI covers both model families.

## Phase 6 — EM-Style Co-Learning with Prior Gist/Lens and Base LoRA
**Goal:** Starting from prior-trained checkpoints, execute alternating optimization cycles that include a lightweight LoRA on the base LLM.

- **Feature 6.1: Checkpoint ingestion & validation**
  - Task 6.1.1: Implement loaders for external `gistnet.pt` and `lensnet.pt` weights (both SmolLM3 and Qwen variants), verifying config compatibility.
  - Task 6.1.2: Add sanity-check scripts comparing ΔNLL/LensNet utilities before and after load.
  - Task 6.1.3: Document expected checkpoint provenance and SHA hashes.

- **Feature 6.2: LoRA integration**
  - Task 6.2.1: Insert LoRA adapters (rank configurable, default 8) on input embedding and selected transformer layers via `peft` or custom modules.
  - Task 6.2.2: Expose LoRA configuration in `configs/runs/em_cycle.yaml` and ensure gradients freeze non-LoRA weights by default.
  - Task 6.2.3: Add tests confirming LoRA-only parameter updates and deterministic forward outputs when adapters are disabled.

- **Feature 6.3: Alternating training scheduler**
  - Task 6.3.1: Implement `tools/run_em_cycle.py` orchestrating Phase B1/B2/B3 (Gist → Lens → LoRA) as described in the README, with configurable step counts per base model.
  - Task 6.3.2: Automate counterfactual utility regeneration after each B1 before LensNet updates.
  - Task 6.3.3: Log cycle-level metrics (ΔNLL gaps, swap rate, LoRA loss) and checkpoint after each sub-phase under `artifacts/checkpoints/em_cycle/`.

**Exit criteria:** EM cycles run end-to-end for both base models, LoRA adapters update as expected, telemetry shows converging metrics, and documentation describes cycle configuration.

## Phase 7 — Coding Assistant Showcase with MegaContext
**Goal:** Deliver a compelling coding-assistant demo using `Qwen/Qwen2.5-Coder-32B-Instruct` (when compute permits) and live repository memory.

- **Feature 7.1: Repository lifetime context tooling**
  - Task 7.1.1: Implement `tools/codebase_ingest.py` that walks a repository, chunks files, and builds an initial lifetime gist tree (L0/L1/L2) tagged with file paths and language metadata.
  - Task 7.1.2: Add filesystem watcher service (e.g., `watchdog`) to detect changes, re-gist affected spans incrementally, and update serialized `{L0,L1,L2}.ctx` artifacts.
  - Task 7.1.3: Support compositing with external language/framework knowledge bases; merge metadata into the lifetime tree with provenance tags.

- **Feature 7.2: Coding model integration**
  - Task 7.2.1: Extend base-model loader to switch between Qwen coder 3B and 32B variants; document memory requirements and precision settings.
  - Task 7.2.2: Provide configs (`configs/runs/coding_showcase.yaml`) tuning working-context budgets, LensNet thresholds, and dataset sampling for coding tasks.
  - Task 7.2.3: Add evaluation scripts running representative coding benchmarks (e.g., HumanEval-lite) with and without MegaContext memory.

- **Feature 7.3: Agentic CLI & showcase experience**
  - Task 7.3.1: Build a CLI (`uv run python -m tools.coding_agent`) that connects to the visualization app, streams working-context updates, and executes file-edit actions.
  - Task 7.3.2: Integrate repository-aware prompts, context expansion logs, and success-metric tracking (latency, completion scores).
  - Task 7.3.3: Produce demo scripts and documentation (`docs/showcase/coding_assistant.md`) highlighting the workflow and benefits.

**Exit criteria:** Coding assistant CLI operates against a live repo with incremental gist updates, MegaContext-enhanced Qwen coder produces competitive results on sample tasks, and visualization links demonstrate focus actions over code spans.

## Phase 8 — Core Knowledge Lifetime Context
**Goal:** Build a durable lifetime memory populated with curated knowledge (beyond coding repos), including metadata, pruning, and monitoring.

- **Feature 8.1: Corpus curation & metadata**
  - Task 8.1.1: Define `configs/core_knowledge/*.yaml` describing domain partitions, ordering, retention policies, and external knowledge sources.
  - Task 8.1.2: Implement ingestion scripts that tokenize, gist, and tag spans with domain, timestamps, provenance IDs, and trust scores.
  - Task 8.1.3: Add metadata indexing (Parquet/Arrow) for filtering and attach to lifetime nodes.

- **Feature 8.2: Storage management & pruning**
  - Task 8.2.1: Extend lifetime storage for append-only partitions, versioning, and pruning signals (access counts, decay timers).
  - Task 8.2.2: Provide pruning jobs moving low-utility spans to a cold tier while preserving audit metadata.
  - Task 8.2.3: Document operational playbooks for adding new knowledge slices, merging with repository memories, and rolling back bad ingests.

- **Feature 8.3: Validation & telemetry**
  - Task 8.3.1: Build scripts sampling queries across domains to verify LensNet surfaces relevant spans from the core memory.
  - Task 8.3.2: Monitor storage growth, quantization precision, and gist variance; alert when thresholds exceed budgets.
  - Task 8.3.3: Update docs with corpus composition, retention policies, and monitoring dashboards.

**Exit criteria:** Core knowledge tree populated with at least one curated domain, storage/versioning tools operational, telemetry confirms access patterns, and documentation outlines maintenance workflows.

## Phase 9 — Cognitive Core Training
**Goal:** Train a compact “cognitive core” transformer that reasons over mixed token/gist inputs using LensNet/focus allocator, leveraging the core knowledge context.

- **Feature 9.1: Small-model training harness**
  - Task 9.1.1: Configure a ≤1 B parameter transformer (distilled SmolLM variant or similar) with adapters for mixed embeddings.
  - Task 9.1.2: Implement training pipeline `tools/train_cognitive_core.py` consuming core knowledge working-context batches, using gradient checkpointing and bf16 precision.
  - Task 9.1.3: Track multi-hop reasoning tasks requiring focus reallocations; log performance vs baseline runs without MegaContext.

- **Feature 9.2: Teacher distillation & memory dependence**
  - Task 9.2.1: Integrate a teacher model capable of direct knowledge access to produce supervision signals.
  - Task 9.2.2: Add penalties/rewards based on whether the cognitive core expands relevant spans (ΔNLL with/without expansions).
  - Task 9.2.3: Produce validation suites demonstrating improved reasoning when MegaContext memory is enabled.

- **Feature 9.3: Documentation & release artifacts**
  - Task 9.3.1: Package cognitive core checkpoints, configs, and instructions for deployment.
  - Task 9.3.2: Create `docs/cognitive_core.md` detailing architecture, training schedule, and expected metrics.
  - Task 9.3.3: Record outstanding research questions and future experiments.

**Exit criteria:** Cognitive core checkpoints trained, validation tasks show reliance on MegaContext memory, and documentation explains deployment and limitations.

## Phase 10 — EM-Style Fine-Tuning of GistNet, LensNet, and Cognitive Core
**Goal:** Run a second alternating optimization cycle that jointly refines GistNet, LensNet, the cognitive core, and optional LoRA adapters for production readiness.

- **Feature 10.1: Extended alternating scheduler**
  - Task 10.1.1: Update EM tooling to include cognitive core fine-tuning stages (e.g., Core → Gist → Lens → LoRA) with configurable ordering.
  - Task 10.1.2: Introduce curriculum scheduling so each cycle covers narrative, coding, QA, and repository workloads.
  - Task 10.1.3: Automate metric gating (stop if ΔNLL or accuracy regresses beyond thresholds).

- **Feature 10.2: Shared telemetry & ablations**
  - Task 10.2.1: Capture cross-module metrics (gist variance, focus score distributions, cognitive core accuracy) per cycle.
  - Task 10.2.2: Run ablations freezing one module at a time to quantify contributions to final benchmarks.
  - Task 10.2.3: Publish final report `docs/reports/em_round2.md` summarizing improvements over Phase 9 and the coding showcase.

- **Feature 10.3: Release packaging**
  - Task 10.3.1: Bundle checkpoints, configs, visualization assets, and reproducibility scripts under `artifacts/releases/<tag>/`.
  - Task 10.3.2: Update `README.md`, `POC_PLAN.md`, and `ROADMAP.md` with completion status and next-step suggestions (e.g., RL focus allocator, async disk streaming).
  - Task 10.3.3: Provide deployment guidance for running MegaContext-enabled models in production or demos.

**Exit criteria:** Second EM cycle converges with measurable benchmark gains, artifacts packaged for sharing, visualization assets refreshed, and documentation captures outcomes plus forward-looking research directions.
