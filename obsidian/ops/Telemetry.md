---
tags:
  - ops
summary: Comprehensive telemetry infrastructure for MegaContext training, runtime monitoring, and performance profiling.
---
Instrumentation and telemetry system tracking compression quality, focus behavior, resource utilization, and performance metrics across [[Training & Operations]] and the current [[MegaContext PRD Index|PRD]] phases.

---

## Overview

MegaContext's telemetry infrastructure provides **real-time and post-hoc** analysis of system behavior during training and runtime. Comprehensive logging enables debugging focus allocation issues, validating compression quality, profiling resource usage, and ensuring acceptance criteria are met.

### Core Purposes

1. **Training feedback:** Monitor loss convergence, substitutability quality (ΔNLL@`H`), and module co-adaptation during [[MegaContext End-to-End Training]].
2. **Runtime validation:** Track focus allocator behavior (swap rate, residency, thrashing) to ensure budget-constrained inference remains stable.
3. **Performance profiling:** Measure latency, memory, and token budget utilization to maintain constant-time promises.
4. **Regression detection:** Persist structured logs for diagnosing failures (oscillations, boundary artifacts, drift).
5. **Reproducibility:** Tag runs with the unified configs under `configs/*.yaml` plus key thresholds so experiments can be replayed exactly.

---

## Tracked Metrics

### Primary Metrics

#### ΔNLL (Delta Negative Log-Likelihood)
**What:** The degradation in perplexity when gist tokens replace original context spans, measured over a future horizon `H` (typically 32–128 tokens).

**Why it matters:**
- Direct measure of **substitutability quality**—lower ΔNLL means gists preserve semantic information better.
- Gold-standard acceptance criterion: ΔNLL ≤ 0.1 at `W_max = 8k` is required for [[MegaContext End-to-End Training]] exit criteria.
- Early indicator of gist drift: increasing ΔNLL over training cycles signals encoder overfitting.

**Collection:**
```python
# Computed during JT1 (GistNet update)
loss_full = model.forward(full_context, horizon_tokens)
loss_replaced = model.forward(gist_replaced_context, horizon_tokens)
delta_nll = loss_replaced - loss_full

# Log per-block and aggregated
wandb.log({
    "train/delta_nll_mean": delta_nll.mean(),
    "train/delta_nll_p95": torch.quantile(delta_nll, 0.95),
    "train/horizon_length": H
})
```

**Validation targets:**
- **Training:** ΔNLL@64 should decrease over JT1 cycles.
- **Evaluation:** ΔNLL@128 ≤ 0.1 on held-out tasks (narrative, code) at compression ratios 32×–1024×.
- **Stress test:** ΔNLL@128 ≤ 0.2 at maximum compression (1024×) on boundary-aligned samples.

---

#### Swap Rate
**What:** The fraction of blocks that undergo expand or collapse actions per iteration.

**Why it matters:**
- **Stability indicator:** High swap rates (>0.25) indicate thrashing—the allocator repeatedly changes its mind.
- Signals threshold miscalibration or insufficient hysteresis/cooldown.
- Correlates with latency spikes (frequent context reconstruction is expensive).

**Collection:**
```python
# In FocusAllocator.step()
num_expands = len(expand_actions)
num_collapses = len(collapse_actions)
total_blocks = len(megacontext_tree.l1_nodes)
swap_rate = (num_expands + num_collapses) / total_blocks

wandb.log({
    "runtime/swap_rate": swap_rate,
    "runtime/num_expands": num_expands,
    "runtime/num_collapses": num_collapses
})
```

**Target ranges:**
- **Healthy:** ≤ 0.25 actions per block (stable focus).
- **Degraded:** 0.25–0.5 (marginal thrashing; review thresholds).
- **Critical:** > 0.5 (ping-pong behavior; halt and debug).

**Debugging:** Histogram residency times to identify oscillating blocks.

---

#### Residency Time
**What:** How many iterations a block remains at a given level-of-detail (LOD0 expanded vs LOD1 gist) before being swapped.

**Why it matters:**
- Prevents expand/collapse thrash—minimum residency (e.g., ≥3 iterations) enforced via cooldown.
- Low residency indicates unstable focus signals from [[LensNet]].
- Informs hysteresis tuning for [[Focus Allocator]].

**Collection:**
```python
# In MegaContextTree
for node in tree.l1_nodes:
    node.residency_steps += 1
    if node.was_swapped:
        wandb.log({
            f"runtime/residency/{node.span_id}": node.residency_steps
        })
        node.residency_steps = 0

# Aggregate histogram
residencies = [n.residency_steps for n in tree.l1_nodes]
wandb.log({
    "runtime/mean_residency": np.mean(residencies),
    "runtime/residency_histogram": wandb.Histogram(residencies)
})
```

**Target:** Mean residency ≥ 3 iterations per block.

---

#### Token Budget Utilization
**What:** The fraction of `W_max` (working context window size) currently occupied by active tokens.

**Why it matters:**
- Verifies allocator respects budget constraints—utilization should stay ≤ 100%.
- Under-utilization (e.g., 60%) suggests conservative thresholds; over-utilization indicates a bug.
- Informs dynamic threshold adjustments in [[Focus Allocator Strategies]].

**Collection:**
```python
# In WorkingContext
active_tokens = working_context.count_tokens()  # LOD0 + gist costs
utilization = active_tokens / W_max

wandb.log({
    "runtime/token_budget_utilization": utilization,
    "runtime/active_tokens": active_tokens,
    "runtime/budget_limit": W_max
})
```

**Target:** 80–100% during active inference (optimal use of budget).

---

#### Latency (Wall-Clock Time per Block)
**What:** Milliseconds per decode block including [[LensNet]] scoring, [[Focus Allocator]] actions, and base model forward pass.

**Why it matters:**
- Validates constant-time promise: latency should remain within +10% of frozen baseline at 8k tokens.
- Spikes indicate expensive counterfactual sampling or large `N_diff` (candidate set size).
- Critical for runtime acceptance: latency overhead ≤ 10% is a POC exit criterion.

**Collection:**
```python
import time

start = time.perf_counter()
# Run LensNet + FocusAllocator + BaseModel.forward()
latency_ms = (time.perf_counter() - start) * 1000

wandb.log({
    "runtime/latency_ms": latency_ms,
    "runtime/latency_p99": percentile(latencies, 99)
})
```

**Target:** ≤ 110% of baseline latency at 8k active tokens.

---

### Secondary Metrics

#### Loss@`H` (Task NLL at Horizon)
**What:** Next-token prediction loss over the future horizon `H` with the current [[Working Context]].

**Why it matters:**
- Primary training objective for JT3 (LoRA phase).
- Validation metric: should improve then plateau across training cycles.
- Compared against token budget to assess compression vs accuracy tradeoff.

**Collection:**
```python
# During JT3 training
task_loss = compute_nll(base_model, working_context, horizon_tokens, H)
wandb.log({
    "train/loss_at_h": task_loss,
    "train/loss_vs_budget": (task_loss, active_tokens)
})
```

---

#### GPU Memory Utilization
**What:** Peak and mean GPU memory (GB) during training and inference.

**Why it matters:**
- Ensures 24GB GPU targets are met (training) or 16GB (inference).
- Detects memory leaks or inefficient KV-cache handling.

**Collection:**
```python
import torch

mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
mem_reserved = torch.cuda.memory_reserved() / 1e9

wandb.log({
    "runtime/gpu_memory_allocated_gb": mem_allocated,
    "runtime/gpu_memory_reserved_gb": mem_reserved
})
```

---

#### Non-Causal Focus Test (C1/C2 Violations)
**What:** Number of times [[LensNet]] incorrectly attends to future tokens (violates causality).

**Why it matters:**
- Validates non-causal dual cross-attention operates only on the working context cache.
- Detects implementation bugs where future gist nodes leak into scoring.

**Collection:**
```python
# In LensNet.forward()
if attention_mask.sum(dim=-1) > working_context.size():
    wandb.log({"debug/causality_violation": 1})
```

**Target:** Zero violations in production runs.

---

## Logging Infrastructure

### Weights & Biases (W&B)

**Primary logging backend** for distributed training and experiment tracking.

**Setup:**
```python
import wandb

wandb.init(
    project="megacontext-poc",
    name=f"{dataset}_{model}_JT{phase}",
    config={
        "W_max": 8192,
        "H": 64,
        "compression_ratio": 32,
        "thresholds": {"expand": 0.3, "collapse": -0.2}
    },
    tags=["phase2", "gistnet", "smollm3"]
)
```

**Logged artifacts:**
- Scalars: losses, swap rates, latencies (every iteration).
- Histograms: residency times, ΔNLL distributions (per epoch).
- Tables: counterfactual utilities, focus actions (sampled snapshots).
- Media: attention heatmaps, working context visualizations (debug runs).

**Query examples:**
```python
# Find runs with low swap rate and acceptable ΔNLL
api = wandb.Api()
runs = api.runs("megacontext-poc", filters={
    "config.W_max": 8192,
    "summary.runtime/swap_rate": {"$lt": 0.25},
    "summary.eval/delta_nll": {"$lt": 0.1}
})
```

---

### File-Based Logging

**Persistent logs** stored in `artifacts/` for offline analysis and long-term archival.

#### Checkpoint Directory: `artifacts/checkpoints/`
```
artifacts/checkpoints/
  JT1_cycle3_step4000/
    gistnet.pt          # GistNet weights
    lensnet.pt          # LensNet weights
    lora.pt             # Base-LoRA adapters
    optimizer.pt        # Optimizer state
    config.yaml         # Hyperparameters
```

#### Delta Tables: `artifacts/deltas/`
Counterfactual utility tables (Parquet or Arrow format) for efficient slicing during [[LensNet Training]].

```python
# Write ΔNLL labels after JT1
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "block_id": block_ids,
    "action": actions,  # "expand" or "collapse"
    "delta_nll": delta_nlls,
    "timestamp": timestamps
})

pq.write_table(table, "artifacts/deltas/JT1_cycle3_labels.parquet")
```

#### Action Traces: `artifacts/traces/`
Serialized focus allocator decisions for replay and debugging.

```python
# Trace format (JSONL)
{
  "step": 1024,
  "expanded": [42, 103],
  "collapsed": [7, 88],
  "working_context": {"LOD0": [1, 2, 3], "LOD1": [42, 103]},
  "latency_ms": 12.3
}
```

---

### Configuration Mirroring: `configs/*.yaml`

**Every experiment** saves a single YAML that bundles dataset staging, base-model settings, and GistNet/LensNet training parameters.

```yaml
# configs/Gutenberg_SmolLM3.yaml
name: Gutenberg_SmolLM3
dataset:
  dataset_name: gutenberg_sample
  tokenizer: HuggingFaceTB/SmolLM2-360M-Instruct
  block_size: 32
  context_tokens: 512
  horizon: 32
  splits:
    train:
      source: ../data/raw/gutenberg/**/*.txt
      output_path: ../data/gutenberg_sample/train.arrow
base_model:
  name: HuggingFaceTB/SmolLM3-3B
  torch_dtype: bfloat16
  run_name: base_llm_demo
gistnet:
  model:
    hidden_size: auto
    block_size: 32
    num_heads: 16
  training:
    batch_size: 8
    phases:
      - name: pooling-pretrain
        objective: pooling_mse
        max_steps: 2000
```

---

## Real-Time vs Post-Hoc Analysis

### Real-Time Monitoring

**Use case:** Detect training failures, oscillations, or resource spikes during long runs.

**Tools:**
- W&B live dashboard (loss curves, swap rate, GPU memory).
- TensorBoard (optional; `torch.utils.tensorboard`).

**Critical alerts:**
- Swap rate > 0.5 → Pause and review allocator thresholds.
- ΔNLL diverging → GistNet drift; refresh labels.
- Latency > 150% baseline → Profile counterfactual sampling.

---

### Post-Hoc Analysis

**Use case:** Understand converged behavior, compare experiments, debug boundary artifacts.

**Workflows:**

#### 1. Residency Histogram Analysis
```python
import pandas as pd

# Load traces
traces = pd.read_json("artifacts/traces/run_001.jsonl", lines=True)
residencies = traces.groupby("block_id")["step"].apply(lambda x: x.diff().dropna())

# Plot
residencies.hist(bins=20)
plt.xlabel("Residency (steps)")
plt.ylabel("Count")
plt.title("Focus Stability: Most blocks stay ≥3 iterations")
```

#### 2. ΔNLL vs Compression Scatter
```python
# Compare substitutability across compression ratios
df = pd.read_parquet("artifacts/deltas/eval_sweep.parquet")
sns.scatterplot(data=df, x="compression_ratio", y="delta_nll", hue="dataset")
plt.axhline(y=0.1, color="red", linestyle="--", label="Acceptance threshold")
```

#### 3. Latency Breakdown Profiling
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Run LensNet + Allocator + BaseModel
    engine.step(tokens)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# Export: prof.export_chrome_trace("artifacts/traces/latency_profile.json")
```

---

## Debugging with Telemetry

### Scenario 1: Gist Drift
**Symptoms:** ΔNLL@`H` increases over training cycles despite loss convergence.

**Diagnosis:**
1. Check W&B: plot `train/delta_nll_mean` across JT1 phases—should decrease.
2. Load checkpoints: compare gist embeddings via cosine similarity to initial checkpoint.
3. Validate labels: ensure ΔNLL labels refreshed after each JT1 (on-policy requirement).

**Fix:** Add stability loss (LOD2 regularization) to [[GistNet]] or reduce JT1 learning rate.

---

### Scenario 2: Allocator Oscillation
**Symptoms:** Swap rate > 0.5, high latency variance, unstable residency.

**Diagnosis:**
```python
# Load traces
traces = pd.read_json("artifacts/traces/run_001.jsonl", lines=True)

# Find ping-pong blocks (expand→collapse→expand within 5 steps)
oscillators = traces.groupby("block_id").filter(
    lambda g: (g["action"].diff().abs().sum() > 3)
)
print(oscillators)
```

**Fix:** Increase cooldown steps, widen hysteresis band, or smooth [[LensNet]] scores with EMA.

---

### Scenario 3: Boundary Artifacts
**Symptoms:** ΔNLL spikes > 0.2 on samples where critical tokens align with block edges.

**Diagnosis:**
1. Run synthetic boundary tests: place function definitions at 32-token boundaries.
2. Plot ΔNLL heatmap by token position within block.

**Fix:** Add boundary auxiliary loss during JT1 (upweight edge tokens) or adjust block alignment during ingestion.

---

### Scenario 4: Latency Spikes
**Symptoms:** p99 latency > 200% baseline, irregular runtime.

**Diagnosis:**
```python
# Profile counterfactual sampling
traces["num_candidates"] = traces["expanded"].apply(len) + traces["collapsed"].apply(len)
high_latency = traces[traces["latency_ms"] > 20]
cols = ["num_candidates", "latency_ms"]
print(high_latency[cols].corr())
```

**Fix:** Reduce `N_diff` (candidate set size) or batch counterfactual forwards more efficiently.

---

## Performance Profiling

### Latency Breakdown

**Target allocation:**
- [[LensNet]] scoring: 30% of block time.
- [[Focus Allocator]] (greedy selection): 10%.
- Base model forward: 50%.
- Context reconstruction: 10%.

**Profiling command:**
```bash
uv run python -m tools.profile_runtime \
  --config configs/Gutenberg_SmolLM3.yaml \
  --profile-output artifacts/traces/runtime_profile.json
```

**Visualization:**
```python
# Chrome trace viewer: chrome://tracing
# Load artifacts/traces/runtime_profile.json
```

---

### Memory Profiling

**Tools:**
- `torch.cuda.memory_summary()`: Detailed allocator state.
- `torch.cuda.memory_snapshot()`: Persistent memory timeline.

**Example:**
```python
import torch

torch.cuda.reset_peak_memory_stats()
# Run training step
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_mem:.2f} GB")

# Dump snapshot
snapshot = torch.cuda.memory_snapshot()
torch.cuda.memory._dump_snapshot("artifacts/traces/memory_snapshot.pickle")
```

**Analysis:**
```bash
# Visualize with memory_viz (PyTorch tool)
python -m torch.cuda._memory_viz trace_plot artifacts/traces/memory_snapshot.pickle -o memory.html
```

---

## Testing & Validation Checklist

### Per-Commit Smoke Tests (CI)
- [ ] Dataset prep produces deterministic 32-token blocks.
- [ ] [[GistNet]] forward pass matches saved checkpoint outputs.
- [ ] [[Focus Allocator]] respects `W_max` budget (unit test).
- [ ] Telemetry logs parse without errors.

### Pre-Merge Integration Tests
- [ ] End-to-end loop runs 100 steps without NaN losses.
- [ ] Swap rate < 0.25 on synthetic stable corpus.
- [ ] Latency overhead < 20% vs baseline (tolerance for debug builds).

### POC Acceptance Criteria
- [ ] **ΔNLL ≤ 0.1** at `W_max = 8k` on narrative + code benchmarks.
- [ ] **Swap rate ≤ 0.25** over 1000-step eval run.
- [ ] **Latency overhead ≤ 10%** vs frozen baseline.
- [ ] **Mean residency ≥ 3** iterations per block.
- [ ] All telemetry metrics logged to W&B + archived to `artifacts/`.

---

## Examples

### Example 1: Training Run Telemetry

```python
# megacontext/gistnet/lightning.py (excerpt)
class GistNetLightningModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        phase_index = self._phase_index_for_step(int(self.global_step))
        phase = self.phases[phase_index]
        loss, metrics = self._compute_phase_loss(batch, phase)

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        for name, value in metrics.items():
            self.log(f"train/{name}", value, prog_bar=False, on_step=True)

        return loss

# In a notebook configure WandbLogger or CSVLogger, then:
trainer = pl.Trainer(
    logger=pl.loggers.WandbLogger(project="megacontext-poc"),
    max_steps=module.total_steps,
    accelerator="gpu",
    devices=1,
)
trainer.fit(module, data_module)
```

---

### Example 2: Runtime Loop Telemetry

```python
# src/runtime/engine.py (excerpt)
import time
import wandb

class MegaContextEngine:
    def step(self, new_tokens):
        start = time.perf_counter()

        # 1. Ingest tokens into MegaContext Tree
        self.tree.ingest(new_tokens)

        # 2. LensNet scores working context
        focus_scores = self.lensnet.forward(self.working_context)

        # 3. Focus Allocator applies expand/collapse
        actions = self.allocator.step(focus_scores, self.tree)

        # 4. Update working context
        self.working_context.apply_actions(actions)

        # 5. Base model decode
        logits = self.base_model.forward(self.working_context.to_embeddings())

        latency_ms = (time.perf_counter() - start) * 1000

        # Log telemetry
        wandb.log({
            "runtime/swap_rate": len(actions) / len(self.tree.l1_nodes),
            "runtime/token_budget_utilization": self.working_context.count_tokens() / self.W_max,
            "runtime/latency_ms": latency_ms,
            "runtime/num_expands": sum(1 for a in actions if a.type == "expand"),
            "runtime/num_collapses": sum(1 for a in actions if a.type == "collapse")
        })

        return logits
```

---

### Example 3: Validation Sweep

```python
# tools/eval_delta_nll.py
import wandb
import pandas as pd

results = []
for W_max in [4096, 8192, 12288, 16384]:
    for dataset in ["narrative", "code", "qa"]:
        delta_nll = evaluate_substitutability(model, dataset, W_max, H=128)
        results.append({
            "W_max": W_max,
            "dataset": dataset,
            "delta_nll": delta_nll
        })

df = pd.DataFrame(results)
df.to_parquet("artifacts/deltas/eval_sweep.parquet")

# Log to W&B
wandb.log({"eval/delta_nll_table": wandb.Table(dataframe=df)})
```

---

## Related Pages

### Training & Operations
- [[Training & Operations]] — Alternating optimization schedule and on-policy labeling
- [[MegaContext End-to-End Training]] — EM-style co-training of GistNet, LensNet, and LoRA adapters
- [[Base Runtime]] — Runtime loop and inference engine architecture
- [[Performance Sketch]] — Latency, memory, and throughput analysis

### Component Training & Metrics
- [[GistNet Training]] — Phase 1 (JT1): Compression model training with ΔNLL metrics
- [[LensNet Training]] — Phase 2 (JT2): Focus policy learning and utility prediction
- [[GistNet Architecture Details]] — Encoder/decoder architecture and loss functions
- [[LensNet Scoring]] — Utility prediction and counterfactual evaluation methodology

### Core Components
- [[GistNet]] — Compression model; ΔNLL@`H` is the primary quality metric
- [[LensNet]] — Dynamic focus scorer; utilities logged as counterfactual deltas
- [[Focus Allocator]] — Greedy selection algorithm with hysteresis and cooldown
- [[Focus Allocator Strategies]] — Swap rate, residency time, and stability metrics
- [[Working Context]] — Budget-constrained token window; utilization tracked per step
- [[Working Context Assembly]] — Construction of mixed LOD token sequences
- [[Working Context Refocusing]] — Dynamic focus shift operations
- [[MegaContext Tree]] — Hierarchical compression structure

### Implementation & Planning
- [[MegaContext End-to-End Training]] — Phase-by-phase implementation with telemetry integration points
- [[POC Implementation]] — Technical implementation details and milestones
- [[POC Architecture]] — Simplified architecture for proof-of-concept
- [[POC Scope]] — Feature scope and acceptance criteria

### System Architecture
- [[Architecture]] — Complete system overview
- [[Architecture Details]] — In-depth architectural specifications
- [[Runtime Loop]] — Per-block execution cycle with telemetry checkpoints
- [[System Properties]] — Constant-time guarantees and resource bounds
- [[Invariants]] — Critical system invariants and constraints

### Related Concepts
- [[Ops]] — Operations and training overview
- [[Components]] — Component architecture index
- [[How MegaContext Works]] — High-level system explanation
- [[Context Focus]] — Introduction and quick start guide
- [[Examples]] — Practical usage examples and demonstrations

---

## Future Enhancements

### Planned (Post-POC)
- **Distributed tracing:** Integrate OpenTelemetry for multi-node training visibility.
- **Anomaly detection:** Automate threshold alerts (e.g., email on swap rate > 0.5).
- **Comparative dashboards:** Side-by-side run comparison in W&B.
- **Explainability:** Visualize which gists [[LensNet]] attends to for specific queries.

### Research Extensions
- **Causal attribution:** Link ΔNLL degradation to specific gist blocks via gradient-based saliency.
- **Online curriculum:** Adjust compression ratios dynamically based on rolling ΔNLL metrics.
- **Multi-task telemetry:** Track per-task (narrative, code, QA) metrics separately for fine-grained analysis.

---

## Summary

MegaContext's telemetry infrastructure provides comprehensive visibility into:
1. **Compression quality** via ΔNLL@`H` (gold standard for substitutability).
2. **Focus stability** via swap rate and residency times.
3. **Resource efficiency** via latency, memory, and budget utilization.
4. **Training dynamics** via loss curves, counterfactual utilities, and checkpoint artifacts.

All metrics stream to **Weights & Biases** for real-time monitoring and persist to **`artifacts/`** for reproducibility. Post-hoc analysis tools enable debugging oscillations, boundary artifacts, and performance regressions. Acceptance criteria for [[MegaContext End-to-End Training]] depend critically on these telemetry signals.
