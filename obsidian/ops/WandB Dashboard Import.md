---
tags:
  - ops
  - wandb
summary: Ready-made WandB report layout for MegaContext runs (tokens, losses, MC diagnostics).
---

# MegaContext WandB Dashboard Import

Use this panel configuration to spin up the standard MegaContext report in any WandB project. Paste the JSON into a new report via **Add Panel → Markdown → Convert to Panels**.

```json
{
  "panels": [
    { "type": "line", "title": "Train Loss (smoothed)", "xs": ["_step"], "ys": ["train/loss"], "smoothing": 0.9 },
    { "type": "line", "title": "Validation BPB", "xs": ["_step"], "ys": ["val/bpb"], "smoothing": 0.6 },
    { "type": "line", "title": "Total Training FLOPs (est.)", "xs": ["_step"], "ys": ["total_training_flops"] },
    { "type": "line", "title": "MC Lens Loss", "xs": ["_step"], "ys": ["mc/lens_loss"], "smoothing": 0.8 },
    { "type": "line", "title": "MC Preference Corr / Agreement", "xs": ["_step"], "ys": ["mc/preference_corr_mean","mc/preference_agreement"], "legendLabels":["corr_mean","agreement"], "smoothing": 0.7 },
    { "type": "line", "title": "MC Advantage Delta (mean / p95)", "xs": ["_step"], "ys": ["mc/adv_delta_mean","mc/adv_delta_p95"], "legendLabels":["mean","p95"], "smoothing": 0.6 },
    { "type": "line", "title": "Policy Score Range", "xs": ["_step"], "ys": ["mc/policy_score_abs_mean","mc/policy_score_std_mean"], "legendLabels":["|score| mean","score std"], "smoothing": 0.5 },
    { "type": "line", "title": "Variants per Batch", "xs": ["_step"], "ys": ["mc/variants_mean","mc/variants_total"], "legendLabels":["mean","total"], "smoothing": 0.4 },
    { "type": "line", "title": "Controller Timings", "xs": ["_step"], "ys": ["mc/time_controller_ms","time/forward_ms","time/optimizer_ms","time/dataloader_ms"], "legendLabels":["controller","forward","optimizer","loader"], "smoothing": 0.3 },
    { "type": "bar", "title": "LOD Counts (latest)", "xs": ["mc/lod_count/0","mc/lod_count/1","mc/lod_count/2"], "orientation": "horizontal" },
    { "type": "markdown", "title": "Debug Notes", "text": "Use PrefDebug logs for samples; this report tracks aggregate metrics. Set `--mc_log_lens_debug 1` for console-level insight." }
  ]
}
```

Save the report (e.g., “MegaContext Dashboard”) and pin it to the project workspace so every new run inherits the layout automatically.
