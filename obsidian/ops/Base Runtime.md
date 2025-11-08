---
tags:
  - ops
summary: Runbook for the nanochat-based chat CLI / web demo, covering command execution, telemetry outputs, and expected behavior.
---
Baseline decode demo for the frozen LLM runtime. Run this after training via `run10.sh`, `speedrun.sh`, or `run1000.sh` so the `~/.cache/nanochat` directory contains checkpoints. Refer to [[Training & Operations]] if you need to re-create the environment or rerun the training scripts.

---

- **Command (CLI):** `python -m scripts.chat_cli -p "Why is the sky blue?"`
- **Command (web UI):** `python -m scripts.chat_web`
- **Batch eval:** `python -m scripts.chat_eval -- -i sft`
- **Prompt source:** CLI accepts inline prompts via `-p` or reads from stdin interactively; eval pulls standard datasets baked into nanochat.
- **Telemetry:** CLI prints responses and logs structured events through nanochat’s report generator + WANDB instrumentation.
- **Weights & Biases:** export `WANDB_RUN=<name>` or `MEGACONTEXT_ENABLE_WANDB=1` before training so later CLI/eval runs attach to the same project.
- **Maintenance:** refresh this note when runtime flags, configs, or expected outputs change.

---

## Expected Output
1. CLI prints the continuation to stdout (prefixed with `>>>`). Example:
   ```text
   >>> Why is the sky blue?
   MegaContext: Rayleigh scattering ... [generated tokens]
   ```
2. `report/report.md` is refreshed with the latest chat samples plus per-phase metrics.
3. If WANDB is enabled, a run named `<WANDB_RUN>-chat` (or similar) appears with latency + allocator charts.

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
- **`scripts.chat_cli` cannot find checkpoints:** confirm `~/.cache/nanochat/base` (or the `NANOCHAT_BASE_DIR` override) contains artifacts from a recent run. Re-run `run10.sh` if empty.
- **WANDB authentication errors:** set `WANDB_API_KEY` before launching the CLI/eval scripts or specify `WANDB_MODE=offline`.
- **Latent mismatches across runs:** ensure the CLI uses the same `--max_seq_len` and tokenizer as training (the defaults match run10/speedrun). Rebuild tokenizer via `python -m scripts.tok_train` if necessary.
- **Unstable generations / empty output:** inspect `report/report.md` for swap spikes; retrain LensNet/mid stages if residency drops below 80 %.

Escalate persistent runtime issues by attaching the failing log + config to the relevant PRD or [[Migration Status]] entry.

---
## Notes

- [[Training & Operations]] outlines shared logging conventions and acceptance criteria for runtime demos, tied to [[MegaContext End-to-End Training]] checkpoints produced by the nanochat scripts.
- Hook the nanochat chat/eval commands into [[MegaPrediction Training]]'s gist-first inference path once the shared readout head lands.
