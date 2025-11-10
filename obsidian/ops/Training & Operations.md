---
tags:
  - ops
summary: End-to-end operating guide for the nanochat-based training flow (tokenizer → base pretrain → mid → chat SFT) plus telemetry expectations and troubleshooting tips.
---
# Training & Operations

MegaContext now trains entirely through the nanochat pipeline. The legacy JT1/JT2/JT3 notebook loop is retired—every run goes through the tokenizer + base + mid + chat-SFT stack invoked by `run10.sh`, `speedrun.sh`, or `run1000.sh`. Treat this note as the **single source of truth** for operational details; other docs should link here instead of duplicating instructions.

## 1. Environment & Dependencies

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.11.
2. Create the venv and install GPU deps:
   ```bash
   uv venv
   uv sync --extra gpu
   source .venv/bin/activate
   ```
3. Export credentials and cache roots before training:
   ```bash
   export WANDB_API_KEY=...
   export HF_TOKEN=...
   export NANOCHAT_BASE_DIR=/fast/storage/nanochat  # defaults to ~/.cache/nanochat
   ```
4. (Optional) run `uv run pytest --maxfail=1 --disable-warnings` and `uv run ruff check` before long jobs to catch obvious regressions.

Everything else (Rust tokenizer build, dataset downloads, torchrun invocations) is handled by the scripts below.

### Command map

- **Training scripts (repo root):** `run10.sh`, `speedrun.sh`, `run1000.sh`.
- **Component entrypoints (`scripts/`):** `base_train.py`, `mid_train.py`, `chat_sft.py`, `chat_eval.py`, `base_eval.py`, `tok_train.py`, `tok_eval.py`.
- **Runtime tools (`scripts/`):** `chat_cli.py`, `chat_web.py`.
- **Legacy commands:** anything under `tools/` (e.g., `tools.decode_demo`) is historical and documented only inside [[POC Implementation Guide]] / `_old/`.

If you discover a missing script or new entrypoint, add it here and update `obsidian/TODO.md` so the rest of the docs stay consistent.

## 2. Choose a Training Recipe

| Scenario | Command | Notes |
| --- | --- | --- |
| Single rented GPU (32 GB+) | `bash run10.sh --gpu 5090 [--mc] [--block_size 8|32|128] [--gistnet_* ...] [--lensnet_* ...] [--allocator ...] [--mc_tree ram|disk] [--mc_initial_wcs N --mc_max_counterfactuals N] [--mc_horizon 32 --mc_long_horizon_multiplier 32] [--mc_token_loss_weight 1.0 --mc_lod1_loss_weight 0.1 --mc_lod2_loss_weight 0.05 --mc_lens_loss_weight 0.1]` | Depth 12, ~3.1 B tokens, fits 5090/A6000 class cards. |
| Single H100 (80 GB) | `bash run10.sh --gpu h100 [--mc] ...` | Doubles device batch size, halves iteration count for the same token budget. |
| $100 speed tier | `bash speedrun.sh [--mc] ...` | 8×H100, depth 20 (Karpathy’s “best $100” recipe). |
| $1000 tier | `bash run1000.sh [--mc] ...` | 8×H100, depth 32 with tuned accumulation. |

`--mc` toggles the Phase 1 MegaContext instrumentation (tree/working-context/controller) without altering the default nanochat flow. Leave it off to reproduce the upstream baselines. The additional knobs control MC-specific behaviors as described below.

### MegaContext CLI knobs

- **Tree / WC sampling**
  - `--mc_tree ram|disk` (disk = placeholder until MegaCache lands).
  - `--mc_initial_wcs` (N1) and `--mc_max_counterfactuals` (N2) define how many Working Contexts we evaluate per training sequence (initial samples + LensNet siblings). Raise them for richer ΔNLL supervision; lower to save compute.
- **Horizon & losses**
  - `--mc_horizon` sets the base teacher-forced horizon (LOD1).  
  - `--mc_long_horizon_multiplier` (defaults to 32) opportunistically upgrades to a LOD2 horizon (`block_size * multiplier` tokens) whenever enough context remains; the controller emits `horizon_trigger` telemetry each time this path runs.
  - `--mc_token_loss_weight`, `--mc_lod1_loss_weight`, `--mc_lod2_loss_weight`, `--mc_lens_loss_weight` scale the auxiliary losses blended into the vanilla nanochat objective.
- **Allocator**
  - `--allocator_*` thresholds (`soft_max_length`, `recent_tokens`, expand/collapse thresholds, max replacements, iterations) shape how aggressively focus edits are applied per step.

All scripts perform the following stages in order:
1. Install uv deps + Rust tokenizer (via `maturin`).
2. Download tokenizer shards and the full pretraining corpus.
3. Train the tokenizer (`scripts.tok_train` + `scripts.tok_eval`).
4. Run base pretraining (`scripts.base_train` + eval/loss).
5. Run midtraining + chat SFT + chat eval.
6. Generate `report/report.md` with metrics and sample outputs.

## 3. Operating Checklist

1. **Prep shell** (see section 1), ensure `WANDB_RUN` is set if you want consistent logging.
2. **Launch script** appropriate to your hardware.
3. **Monitor**:
   - `tail -f report/log.txt` (if running inside `screen`/`tmux`).
   - WANDB dashboard for ΔNLL@H, swap rate, residency, MFU, plus the new `mc/token_loss`, `mc/lod1_loss`, `mc/lod2_loss`, and `mc/lens_loss` curves.
   - `nvidia-smi` for memory/utilization sanity.
   - MC telemetry backend (Grafana/Kibana/etc.) for tree/WC/focus visualizations if you’ve configured a `TelemetryProvider`.
4. **Resume if interrupted**: re-run the same script with the same `WANDB_RUN`. The scripts load checkpoints from `NANOCHAT_BASE_DIR` and continue.
5. **Evaluate & demo** once the script finishes:
   ```bash
   python -m scripts.base_eval
   python -m scripts.chat_eval -- -i sft
   python -m scripts.chat_cli -p "Describe MegaContext in 2 sentences."
   python -m scripts.chat_web  # optional UI
   ```
6. **Hand-off**: capture report artifacts (`report/report.md`, WANDB link), checkpoint paths (`$NANOCHAT_BASE_DIR/base/...`), and outstanding issues in [[Lifecycle#Hand-off Checklist]] / [[PRD Progress Tracker]].

### Post-run validation

Before sharing a checkpoint or moving to downstream experiments:

1. **Loss parity** — confirm `python -m scripts.base_eval` and `python -m scripts.chat_eval -- -i sft` report ΔNLL@H within the target bands (see table below). Re-run eval if WANDB shows gaps.
2. **Report completeness** — ensure `report/report.md` includes tokenizer stats, base/mid/SFT metrics, and chat samples. Missing sections usually mean a stage exited early.
3. **Checkpoint audit** — list `$NANOCHAT_BASE_DIR/base`, `/mid`, `/chat` and record the latest `step` values. Add these paths to the hand-off note.
4. **Interactive smoke** — run `python -m scripts.chat_cli -p "Summarize MegaContext."` and paste the response (and latency) into the hand-off log. This catches config mismatches before others pull the run.

## 4. Telemetry Targets & Alerts

| Metric | Target | Investigate when… | Notes |
| --- | --- | --- | --- |
| ΔNLL@H (base + mid + SFT) | ≤ 0.10 at `W_max=8k` | Sustained increase >0.15 or monotonic drift | Indicates gist compression drift or stale tokenizer. Rebuild tokenizer (`python -m scripts.tok_train`) and rerun base. |
| Swap rate | 0.05–0.20 | >0.25 or oscillation between 0/1 | Lens/controller instability. Inspect `report/report.md` swap plots; consider re-running midtraining. |
| Residency | 90–100 % | <80 % | Allocator collapsing too aggressively; verify Lens logits and budget constraints. |
| MFU per GPU | 45–55 % | <40 % | Check `--device_batch_size`, accumulation steps, and host I/O stalls. |
| `mc/token_loss` | Should trend down, ideally ≤ baseline loss | Flat or rising | Horizon eval unstable; revisit WC sampling or loss weights. |
| `mc/lod1_loss`, `mc/lod2_loss` | ≤0.3 / ≤0.2 | Flatlines >0.5 | GistNet not learning higher LODs; ensure long horizons are triggering and look at telemetry. |
| `mc/lens_loss` | →0 | >0.1 | LensNet disagreeing with ΔNLL argmin; inspect focus telemetry. |
| WANDB heartbeat | steady | Missing for >15 min | Ensure networking available; scripts fall back to DummyWandb if `WANDB_RUN=dummy`. |

## 5. Telemetry providers

The MC controller now emits structured `TelemetryEvent`s every time a session (training or inference) touches the tree, WC, allocator, or opportunistic LOD2 horizon. Plug one of these backends into `TelemetryProvider` to capture them:

| Provider | Pros | Cons |
| --- | --- | --- |
| **OpenTelemetry → Tempo/Grafana (recommended)** | Standards-based traces, Grafana makes it easy to animate “hot” spans & WC edits, integrates with existing Metrics dashboards. | Requires running OTEL collectors + Tempo storage (but scales well). |
| **OpenSearch/Kibana** | Powerful log analytics, role-based access for multi-user inference. | Heavier cluster footprint; need custom viz for WC heatmaps. |
| **Prometheus + Grafana** | Simple numeric metrics (counts, hit rates). | Not great for structured WC snapshots; pair with a log sink. |

The training scripts automatically instantiate `OpenTelemetryProvider(service_name="megacontext-train")` when `--mc` is enabled. Configure the exporter via env vars:

```bash
export MC_OTEL_ENDPOINT="http://tempo:4318"
export MC_OTEL_INSECURE=1  # set only if you’re skipping TLS
```

> **Install the OTLP deps before enabling MC telemetry.**  
> The nanochat project treats `opentelemetry-sdk` and `opentelemetry-exporter-otlp` as optional extras, so your Python environment may not have them by default. Install the CPU wheels once per environment (e.g., `uv pip install opentelemetry-sdk opentelemetry-exporter-otlp`) before running `--mc_enabled=1`, otherwise the script falls back to `NoOpTelemetryProvider`.

If you need to disable telemetry entirely, unset `MC_OTEL_ENDPOINT` (the provider falls back to OTLP defaults) or patch the script to use `NoOpTelemetryProvider`.

Once enabled, expect event types such as `mc_tree_snapshot`, `working_context_snapshot`, `focus_allocator`, `horizon_trigger`, and `inference_update`. Use them to drive Grafana dashboards / alerts (see [[Telemetry]] for schema details).

## 6. Fair MC vs. vanilla comparisons

MC-enabled steps perform extra work per batch (multiple WC evaluations, opportunistic 1024-token horizons), so raw `step` comparisons with vanilla nanochat are misleading. When plotting/alerting:

1. Use **tokens processed** (`total_batch_size * steps`) or **total FLOPs** (`total_training_flops`) on the x-axis when overlaying MC and vanilla loss curves.
2. Track **wall-clock time** (`total_training_time`) to compare throughput regardless of mix.
3. Optionally derive an “MC effective step” (e.g., sum of WC evaluations) and plot `train/loss` vs. that counter for MC runs.

Document these choices in your run notes so reviewers know which axes are being compared.

## 7. Troubleshooting

- **`torchrun` OOM:** Lower `--device_batch_size` (edit script or run component command directly), free VRAM, or switch to the 5090 profile.
- **Tokenizer build fails (maturin):** ensure Rust toolchain installed by the script (`rustup`). Delete `rustbpe/target` and rerun.
- **Dataset download slow/stuck:** runs spawn background download (`nanochat.dataset`). Check `~/.cache/nanochat/dataset_download.log`; resume by rerunning the script (it skips completed shards).
- **Report not generated:** check for earlier script failures; rerun `python -m nanochat.report generate` once training completes.
- **CLI/eval can’t find checkpoints:** confirm `NANOCHAT_BASE_DIR` matches the training run. List directories under `$NANOCHAT_BASE_DIR` to verify.
- **Telemetry flat-lines:** open WANDB run, confirm metrics are updating; if not, ensure `WANDB_RUN` wasn’t left at `dummy` and that internet access is available.

## 6. Related References

- [[Lifecycle]] — broader project checklist (setup → train → evaluate → hand-off).
- [[Base Runtime]] — chat CLI / web demo instructions.
- [[Telemetry]] — detailed metric definitions and logging hooks.
- [[MegaContext End-to-End Training]] — PRD describing planned extensions on top of the nanochat stack.

Use this doc as the source of truth for day-to-day operations. Anything referencing JT cycles or notebooks belongs in `obsidian/_old/` now.
