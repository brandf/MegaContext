---
tags:
  - ops
summary: Runbook for the base runtime decode demo covering command execution, telemetry outputs, and expected behavior.
---
Baseline decode demo for the frozen LLM runtime. Run this after provisioning assets via `uv run python tools/bootstrap_env.py`. The nanochat bootstrap described in [[Migration Plan - Nanochat Integration]] is not yet wired into this repository—treat it as forward-looking documentation.

---

- **Command (current):** `uv run python -m tools.decode_demo --config configs/SampleText_TinyGPT2.yaml`
- **Command (planned):** `uv run python -m nanochat.chat --config configs/megacontext_demo.yaml` — enable once the nanochat fork is imported and the config lands per [[Migration Plan - Nanochat Integration]].
- **Prompt source:** uses the sample corpus installed during bootstrap; update the config for custom prompts or datasets.
- **Telemetry:** CLI prints the generated continuation and logs structured events to `artifacts/run_logs/<run>-<timestamp>.log`.
- **Weights & Biases:** export `MEGACONTEXT_ENABLE_WANDB=1` (optionally `WANDB_MODE=online`) before running to stream metrics; otherwise W&B initialization is skipped.
- **Maintenance:** refresh this note when runtime flags, configs, or expected outputs change.

---

## Expected Output
1. CLI prints the continuation to stdout (prefixed with `MegaContext>`). Example:
   ```text
   MegaContext> Once upon a time ... [generated tokens]
   ```
2. A log file `artifacts/run_logs/<config>-<timestamp>.log` is created containing:
   - prompt metadata (token count, budget)
   - swap rate / residency per allocator tick
   - ΔNLL-style telemetry if enabled
3. If WANDB is enabled, a run named `decode_demo/<timestamp>` appears with latency + allocator charts.

If any of these artifacts are missing, see the troubleshooting section below.

### Sample log excerpt
```jsonc
{
  "prompt_tokens": 512,
  "working_budget": 2048,
  "swap_rate": 0.12,
  "residency": 0.95,
  "latency_ms": 153.7,
  "notes": "JT cycle 42, configs/SampleText_TinyGPT2.yaml"
}
```

---
## Troubleshooting
- **Command fails with “config not found”:** ensure `configs/SampleText_TinyGPT2.yaml` exists (run `ls configs`). Copy it before editing; keep a known-good config for smoketests.
- **No log file produced:** verify the `artifacts/run_logs/` directory exists and that the process has write permissions. Rerun with `MEGACONTEXT_ENABLE_WANDB=0` to rule out WANDB-related hangs.
- **Model download/auth errors:** supply valid Hugging Face credentials via the notebook Setup Console or set `HF_TOKEN` in the shell before running the decode script.
- **Unstable generations / empty output:** inspect the log for residency <80 % or repeated collapse actions; rerun training JT cycles to refresh LensNet if necessary.

Escalate persistent runtime issues by attaching the failing log + config to the relevant PRD or [[Migration Status]] entry.

---
## Notes

- [[Training & Operations]] outlines shared logging conventions and acceptance criteria for runtime demos, now tied to [[MegaContext End-to-End Training]] checkpoints.
- Keep this runbook in sync with `tools/bootstrap_env.py` and `tools/decode_demo.py` whenever new flags or telemetry outputs are introduced.
- Hook the nanochat command into [[MegaPrediction Training]]'s gist-first inference path once the shared readout head lands.
