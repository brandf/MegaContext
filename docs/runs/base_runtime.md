# Base Runtime Notes

The base runtime demo decodes short prompts with a frozen LLM. After running
`tools/bootstrap_env.py` you can test the setup with:

```bash
uv run python -m tools.decode_demo --config configs/runs/base_llm.yaml
```

Expected telemetry:

- The CLI prints the generated continuation in stdout.
- Structured logs are written to `artifacts/run_logs/<run>-<timestamp>.log`.
- Weights & Biases logging is opt-in: export `MEGACONTEXT_ENABLE_WANDB=1`
  (and optionally `WANDB_MODE=online`) before running the command to stream
  metrics; otherwise the script skips W&B initialisation.

Update this document whenever runtime flags or expected outputs change.
