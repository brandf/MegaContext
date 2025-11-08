---
tags:
  - ops
summary: Lightweight flow summary pointing to the canonical runbooks for environment setup, training, validation, and runtime operations.
---
# MegaContext Lifecycle

This note provides a quick “you are here” map for the nanochat implementation. Use it to orient yourself, then jump to the detailed runbooks called out below. To avoid duplication, all commands and deep checklists live in [[Training & Operations]] and [[Base Runtime]].

---

## Flow Summary

| Stage | What to do | Canonical doc |
| --- | --- | --- |
| 1. Environment prep | Install uv/Python, activate venv, export WANDB/HF tokens, set `NANOCHAT_BASE_DIR` | [[Training & Operations#1. Environment & Dependencies]] |
| 2. Training run | Pick `run10.sh` / `speedrun.sh` / `run1000.sh`, set `WANDB_RUN`, monitor telemetry | [[Training & Operations#2. Choose a Training Recipe]] |
| 3. Mid-run hygiene | Watch ΔNLL/swap/residency, resume downloads, rebuild tokenizer if needed | [[Training & Operations#3. Mid-Run Maintenance]] |
| 4. Post-run validation | Follow the validation checklist (eval parity, report, checkpoints, chat smoke) | [[Training & Operations#Post-run validation]] |
| 5. Runtime / demos | Run `scripts.chat_cli`, `scripts.chat_eval`, `scripts.chat_web` on finished checkpoints | [[Base Runtime]] |
| 6. Hand-off | Log results, update `TODO.md`, add WANDB links/checkpoint paths to [[PRD Progress Tracker]] | [[Training & Operations#Hand-off]] |

Keep this table handy for quick orientation; follow the linked sections for the authoritative commands and troubleshooting guidance.
