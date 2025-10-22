---
tags:
  - ops
summary: Envelope estimates for compute and storage when running MegaContext, including a decade-long retention case study.
---
MegaContext keeps per-step compute close to a frozen LLM while amortizing storage through hierarchical gists, quantization, and telemetry-guided pruning.

---

- **Per-step compute:** essentially base decode cost; GistNet/LensNet overhead <1%.
- **Working budget:** ~8k active tokens in the POC, scaling toward 32k for future builds.
- **Storage growth:** 32-ary tree adds only ~3.2% above leaves; precision and pruning dominate footprint.
- **Case study:** 10-year, 500 Hz robotics log compresses to ~6–14 TB with pruning + entropy coding.
- **Planning hooks:** informs cost assumptions in [[Grand Vision]] and operational budgets in [[Training & Operations]].

---
## Details

### Step-level comparison

| Setup | MegaContext tokens | Active tokens | KV-cache | Disk I/O / step | Notes |
|-------|-----------------|----------------|-----------|-----------------|-------|
| **Vanilla LLM** | 32 k | 32 k | ~2 GB | n/a | Context-limited |
| **MegaContext (POC)** | ~1 M | 8 k | ~0.5 GB | few MB | Constant compute per step |
| **MegaContext (Future)** | 1 B+ | 32 k | ~2 GB | 10–50 MB/s | Fully trained base model |

Per-step compute ≈ base decode cost; gist extraction and LensNet overhead <1%.

