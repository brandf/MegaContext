---
tags:
  - ops
summary: End-to-end lifecycle checklist covering setup, training, mid-run maintenance, and runtime operations for the current notebook workflow (with notes about the upcoming nanochat migration).
---
# MegaContext Lifecycle

This note stitches together the day-to-day workflow for the proof-of-concept notebook build and highlights the planned nanochat migration so contributors know which commands are live today.

---

## 1. Environment & Dataset Prep (Today)

1. Follow `README.md` / `SETUP.md` (Python 3.11, CUDA 12.x, PyTorch 2.2+).
2. Create/activate the `uv` environment and run `uv sync`.
3. Launch `notebooks/megacontext.ipynb`.
4. Run the **Quick Start** + **0. Setup Console** cells:
   - Configure `MEGACONTEXT_DATA_ROOT`, `MEGACONTEXT_ARTIFACT_ROOT`, WANDB/HF tokens.
   - Regenerate Arrow shards with `tools/prepare_dataset` via the notebook widgets if needed.

> Planned nanochat step: dataset/tooling commands will move into the fork’s CLI once [[Migration Plan - Nanochat Integration]] Phase 3 lands. Until then, treat notebook cells as the canonical entry point.

---

## 2. Training Loop (Notebook Flow)

1. Within the notebook, cycle through the JT phases described in [[Training & Operations]]:
   - JT1 `GistNet` substitutability updates (use `megacontext.gistnet.lightning.build_gistnet_experiment`).
   - JT2 `LensNet` counterfactual labeling (still design-only; collect telemetry for future use).
   - JT3 LoRA/base-model refresh (optional, depending on experiment).
2. Record ΔNLL@H, swap rate, residency, latency in WANDB (enable `MEGACONTEXT_ENABLE_WANDB=1` before launching kernels).
3. Checkpoint after each phase under `artifacts/<run_name>/`.

> Planned nanochat step: the same JT loop will be implemented inside `nanochat.train` via the [[MegaContext End-to-End Training]] PRD. Notebook references will be replaced with CLI configs once the fork is imported.

---

## 3. Mid-Run Maintenance

- **Re-gist after model changes:** if you update `GistNet`, rerun the notebook’s ingestion cells to rebuild LOD1/LOD2 summaries before launching the next JT cycle.
- **Telemetry sanity:** review WANDB dashboards for ΔNLL drift (>0.1) or swap-rate spikes (>0.25). Investigate via the notebook telemetry widgets.
- **Dataset refresh:** rerun `tools/bootstrap_env.py` (or the notebook helper) whenever you rotate corpora or block sizes.

> Planned nanochat step: ΔNLL and swap-rate dashboards will be wired into the nanochat logging hooks per [[Telemetry]] once those modules exist.

---

## 4. Runtime / Inference (Current)

1. Provision demo assets with `uv run python tools/bootstrap_env.py`.
2. Run the legacy decode demo:
   ```bash
   uv run python -m tools.decode_demo --config configs/SampleText_TinyGPT2.yaml
   ```
3. Inspect `artifacts/run_logs/<run>-<timestamp>.log` plus WANDB (if enabled) for latency + focus statistics.

> Planned nanochat step: `uv run python -m nanochat.chat --config configs/megacontext_demo.yaml` will replace the legacy command once the migration work completes (see [[Base Runtime]] and [[Nanochat Integration Guide]]).

---

## 5. Hand-off Checklist

- Update `obsidian/ops/Training & Operations.md` with any telemetry anomalies or JT status.
- Log roadmap progress in the relevant PRD (e.g., [[MegaContext End-to-End Training]] status section).
- Note nanochat migration blockers in [[Migration Status]] if you discover gaps during the notebook flow.

---
## 6. Validation & Troubleshooting

- **Telemetry targets:** keep ΔNLL@`H` ≤ 0.10, swap rate 0.05–0.20, residency ≥ 90 %. If metrics drift, follow the remediation steps in [[Training & Operations#Troubleshooting & Telemetry Targets]].
- **Runtime sanity:** after each decode demo, confirm an entry appears under `artifacts/run_logs/` and that the log reports swap-rate + residency fields. Missing logs usually mean `MEGACONTEXT_ENABLE_WANDB` or the log writer was skipped.
- **Common fixes:**
  - *CUDA unavailable:* restart kernel, re-run Quick Start, check `nvidia-smi`.
  - *ΔNLL spikes after re-gisting:* rebuild `{LOD0,LOD1,LOD2}.ctx` before measuring; stale trees inflate loss.
  - *Focus thrash:* tighten allocator thresholds and ensure LensNet labels were regenerated after any GistNet change.
- **Escalation:** if remediation fails, capture the WANDB run link + latest checkpoints and attach them to the relevant PRD issue before hand-off.
- **Parity checks:** before switching branches or handing off, rerun the decode smoke test and paste the log snippet plus WANDB link into [[PRD Progress Tracker]] so the next contributor can compare against known-good telemetry.

---

## References
- [[Training & Operations]] — Alternating optimization details and Colab setup.
- [[Base Runtime]] — Decode demo instructions.
- [[MegaContext PRD Index]] — Source of truth for active requirements.
- [[Migration Plan - Nanochat Integration]] — Tracks when the lifecycle moves into the nanochat fork.
