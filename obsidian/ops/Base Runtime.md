---
tags:
  - ops
summary: Runbook for the base runtime decode demo covering command execution, telemetry outputs, and expected behavior.
---
Baseline decode demo for the frozen LLM runtime. Run this after provisioning assets via `uv run python tools/bootstrap_env.py`.

---

- **Command:** `uv run python -m tools.decode_demo --config configs/runs/base_llm.yaml`.
- **Prompt source:** uses the sample corpus installed during bootstrap; update the config for custom prompts or datasets.
- **Telemetry:** CLI prints the generated continuation and logs structured events to `artifacts/run_logs/<run>-<timestamp>.log`.
- **Weights & Biases:** export `MEGACONTEXT_ENABLE_WANDB=1` (optionally `WANDB_MODE=online`) before running to stream metrics; otherwise W&B initialization is skipped.
- **Maintenance:** refresh this note when runtime flags, configs, or expected outputs change.

---

## Notes

- [[Training & Operations]] outlines shared logging conventions and acceptance criteria for runtime demos.
- Keep this runbook in sync with `tools/bootstrap_env.py` and `tools/decode_demo.py` whenever new flags or telemetry outputs are introduced.
