---
tags:
  - plans
  - prd
summary: Migration roadmap for re-platforming MegaContext development onto the nanochat codebase while preserving baseline functionality.
---
# Migration Plan — nanochat Integration

> **Status:** Migration program (in-flight). Use this plan alongside [[Migration Status]] to track progress; notebook instructions remain canonical until these phases complete.

Goal: move the active MegaContext implementation from notebooks + PyTorch Lightning experiments onto the [nanochat](https://github.com/karpathy/nanochat) repository, layering the new PRD-driven architecture on top while keeping the upstream baseline intact.

## Phase 0 — Due diligence & repo prep ✅
- **nanochat audit:** confirmed the training entrypoints (`train.py`, `scripts/run_*`), data pipeline (tokenizer → binarized dataset), and logging (wandb/local JSON) align with MegaContext needs. Critical hook points:
  - Model definition lives in `nanochat/model.py` — future home for [[GistNet]]/[[LensNet]] adapters.
  - Trainer loop in `nanochat/train.py` — natural place for [[MegaContext]] end-to-end loss wiring and telemetry.
  - Inference utilities (`chat.py`, `sample.py`) will host [[MegaPrediction]] multi-LOD heads.
- **Repo strategy:** fork upstream nanochat, maintain two branches:
  - `main`: mirrors upstream, used for regular rebase pulls.
  - `megacontext`: primary development branch containing MegaContext extensions.
- **Constraints captured:**
  - Preserve nanochat’s baseline scripts/CLI so vanilla runs remain reproducible.
  - Keep dependency additions modular (extra requirements via optional install).
  - Ensure checkpoints/configs stay compatible with upstream formats.

## Phase 1 — Skeleton alignment
- Reproduce the vanilla nanochat pipeline end-to-end on local hardware to confirm parity. *(Already exercised in a separate repo; plan is to bring the tuned “$5 on 5090” script into the fork once upstream code lands, then scale to $100 and $1000 runs for publication.)*
- Identify hook points for MegaContext subsystems:
  - Training loop integration for [[MegaContext]] end-to-end plan.
  - Decoder extension location for [[MegaPrediction]] multi-LOD heads.
  - Memory/model boundaries needed by the [[Cognitive-Core Training]] PRD.
- Document required dependency additions (tree storage, telemetry, notebooks migration).

## Phase 2 — Legacy snapshot & archival
- After Phase 1 parity is confirmed, move notebook/Lightning code to an `_old/` folder (or equivalent archive) so all active work happens in the nanochat fork.
- Update README/onboarding to signal the new baseline stack before integrating MegaContext code.
- Maintain read-only access to notebooks solely for historical reference; no further development expected.

## Phase 3 — MegaContext core scaffolding
- Implement the following as isolated modules so nanochat baseline remains functional:
  - [[MegaContext Tree]] construction utilities (ingest, refresh, serialization).
  - [[Working Context]] assembly with dynamic mixing of LOD entries.
  - [[GistNet]] and [[LensNet]] entry points compatible with nanochat model classes.
  - Registry/config glue for selecting between vanilla nanochat runs and MegaContext runs.
- Add end-to-end smoke tests mirroring the current notebook experiments.

## Phase 4 — PRD feature layering
- **E2E Training:** Embed the [[MegaContext End-to-End Training]] loop into nanochat’s trainer, including horizon ΔNLL evaluation and gist losses.
- **MegaPrediction:** Attach the multi-LOD decoding head defined in [[MegaPrediction Training]]; expose CLI flags for gist-first inference.
- **Cognitive Core:** Scaffold composite memory management and CK reliance checks per [[Cognitive-Core Training]].

## Phase 5 — Migration & validation
- Port critical datasets/configs from notebooks; ensure determinism with nanochat pipelines.
- Recreate telemetry for ΔNLL, swap rate, budget utilization, gist regression losses, and composite MC metrics within nanochat’s logging framework.
- Validate each subsystem in isolation before full e2e runs:
  - [[GistNet]] substitutability checks (ΔNLL@H, gist cosine).
  - [[LensNet]] focus audits (target vs. predicted utilities, budget invariants).
  - [[MegaPrediction]] decoding smoke tests (gist-first vs. token-first).
  - Composite [[Working Context]] assembly sanity (LOD mixes, budgets).
- Conduct parity runs against the legacy notebooks, monitoring loss curves, ΔNLL@H, latency, and telemetry parity.

## Phase 6 — Upstream parity & maintenance
- Keep `main` synced with upstream nanochat releases; regularly rebase the `megacontext` branch.
- Audit MegaContext-specific diffs to ensure baseline scripts remain intact.
- Establish a cadence for dependency bumps, CI validation, and telemetry regression checks.
