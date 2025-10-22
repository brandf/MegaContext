# Training & Operations

Guidance for alternating optimization, instrumentation, and evaluation when co-training GistNet, LensNet, and lightweight base-model adapters.

## Joint training (alternating / “EM-style”)

**Goal:** Let all three modules co-adapt without full end-to-end backprop through the discrete focus allocator or long unrolls.

**Method:** Short alternating phases where some modules are frozen while others learn from on-policy signals produced by the frozen parts. Repeat for a few cycles.

### What “EM-style” means here
We alternate optimization across modules:
- **E-like step:** hold policy parts fixed to produce supervision/targets (e.g., counterfactual utilities).
- **M-like step:** update another module to better fit those targets.
It’s not exact EM; it’s an alternating optimization schedule that stabilizes joint training.

### Modules
- **GistNet** (`Gist`) — 32→1, two levels; substitutability objective.
- **LensNet** (`LensNet`) — dual cross-attn (8k→6→8k); signed focus scores.
- **Base-LoRA** (`LoRA`) — tiny adapters on the base LLM to improve gist compatibility.
- **Focus allocator** remains discrete and greedy (no relaxation).

### Phase B1 — Update GistNet (fix LensNet + LoRA)
**Fix:** `LensNet`, `LoRA`
**Update:** `Gist`

Procedure (on-policy):
1. Build/refresh MegaContext trees with current `Gist`.
2. For each training block (size `K=32`): run `LensNet` + focus allocator to pick expands/collapses; form the working context used by the base LLM.
3. Optimize **GistNet** on spans touched in this block using:
   - **Substitutability loss:** KL(full || replaced) or ΔNLL@`H` (`H=32–128`) for the gist that was actually inserted.
   - **Stability loss (optional):** L2 between current gist and previous checkpoint to avoid drift.
   - **Boundary auxiliary (optional):** upweight ΔNLL terms on edge tokens so the encoder preserves boundary semantics.

**Intuition:** With the current focusing policy fixed, make gists better drop-in replacements for exactly the places the policy cares about.

### Phase B2 — Update LensNet (fix GistNet + LoRA)
**Fix:** `Gist`, `LoRA`
**Update:** `LensNet`

Procedure:
1. Using the fixed `Gist`, generate counterfactual labels on on-policy snapshots:
   - For candidate expands/collapses in the current working context, compute ΔNLL/ΔKL (batched).
   - Convert to signed utility per token (expand positive; collapse negative).
2. Train `LensNet` with:
   - Signed regression + ranking (within snapshot).
   - Zero-sum budget regularizer (token-cost weighted).
   - Legality penalties; keep runtime masking.
   - Update-every-`K` cadence (Lens runs once per block).

**Intuition:** Given the current gists, learn a better focusing policy.

### Phase B3 — Update Base-LoRA (fix GistNet + LensNet)
**Fix:** `Gist`, `LensNet`
**Update:** `LoRA` (small ranks; keep it tiny)

Recommended LoRA placement:
- Input embedding projection.
- QKV/O of the first 2 attention blocks *or* the last 2 (pick one set; not both).

Losses:
- **Task NLL@`H`** with the discrete working context produced by `LensNet` + focus allocator.
- **Substitutability keep-alive (weak):** prevents gist semantics drifting away from what the base understands.
- (Optional) **KL to teacher** if you have a larger teacher-with-MegaContext.

**Intuition:** Slightly adapt the base to “like” gist tokens and the current working-context geometry (positional anchoring, variance, etc.).

### Schedule & hyperparameters

- **Cycle length:** B1 → B2 → B3 = one cycle. Repeat 3–5 cycles.
- **Step counts per phase (per cycle):**
  - B1 (GistNet): 2–4k steps.
  - B2 (LensNet): 2–4k steps.
  - B3 (LoRA): 1–2k steps.
- **Batching:** mixed long-context tasks; block size `K=32`; horizon `H=64`.
- **Optimizers:** AdamW (bf16), cosine LR with warmup per phase.
- **Tokens / GPU:** target ~8k effective tokens per microbatch; use grad accumulation (e.g., 2 microbatches × 4 sequences) to fit within 24 GB GPUs.
- **Tokenizer:** reuse the base model’s tokenizer and embedding matrix to avoid drift between gist vectors and token embeddings.
- **Checkpoints:** save after each phase; early-stop on validation Loss@`H` vs token budget.

### Data flow per cycle (pseudo)

1. **B1:**
   - Freeze `LensNet`, `LoRA`.
   - Decode blocks with current WC (from LensNet + focus allocator).
   - Update `Gist` using on-policy substitutability losses on the replaced spans.
2. **B2:**
   - Freeze `Gist`, `LoRA`.
   - From the same blocks, compute counterfactual utilities (expand/collapse candidates).
   - Update `LensNet` with signed utilities + budget/legality losses.
3. **B3:**
   - Freeze `Gist`, `LensNet`.
   - Run normal blocks (LensNet + focus allocator active) and update `LoRA` on Task NLL@`H` (+ weak substitutability keep-alive).

### Stability & efficiency tips

- **Warm starts:** Do a short sequential pretrain (GistNet then LensNet) before the first B1; it reduces early oscillations.
- **Small LoRA ranks:** `r=4–16`, low LR; the goal is interface alignment, not knowledge injection.
- **Hysteresis in focus allocator:** minimum residency steps prevent expand/collapse thrash during B2/B3.
- **On-policy labeling:** Always regenerate ΔNLL labels after the last B1 so LensNet trains on current gists.
- **Curriculum:** start with narrative/doc tasks; add lists/tables/code once stable.
- **Telemetry:** track (a) Loss@`H` vs budget, (b) swap rate, (c) residency time, (d) non-causal C1/C2 tests.

### When to stop

- Validation Loss@`H` vs budget improves then plateaus across cycles.
- Swap rate stabilizes; no ping-pong.
- Ablations: freezing any one of {GistNet, LensNet, LoRA} causes a measurable drop.

**Outcome:** All three modules co-learn: **GistNet** encodes what the policy needs, **LensNet** chooses expansions that actually help, and **LoRA** nudges the base LLM to be friendlier to mixed-LOD inputs—without the cost/fragility of full end-to-end training.

---

## Instrumentation & artifact handling

- **Logging:** stream metrics (losses, swap rates, residency histograms) to Weights & Biases; tag runs by dataset + thresholds.
- **Checkpoints:** save GistNet, LensNet, and LoRA weights under `artifacts/checkpoints/`. Store counterfactual utility tables under `artifacts/deltas/` (Parquet or Arrow for efficient slicing).
- **Configs:** mirror each run’s YAML under `configs/runs/` so experiments are reproducible.
- **Testing harness:** add PyTest suites under `tests/` (e.g., `tests/test_gistnet.py`, `tests/test_focus_allocator.py`). Before running tests, sync dev dependencies with `uv sync --extra dev` so `pytest`, `pydantic`, and related tooling resolve inside the project virtualenv. Then run `uv run pytest --maxfail=1 --disable-warnings --cov=src` as the canonical invocation.
- **Local tooling:** provide Python entry points under `tools/` (e.g., `python -m tools.format`, `python -m tools.lint`) that wrap `ruff` and `black` to keep contributor workflows consistent.
- **CLI scripts:** expose dataset/labeling helpers as modules (`python -m tools.ingest_data`, `python -m tools.label_dnll`) and register hydra/typer CLIs if needed; keep lightweight wrappers under `scripts/` for automation.
- **Telemetry (required):** emit per-iteration metrics (`loss_at_h`, `swap_rate`, `mean_residency`, `latency_ms`, `token_budget_utilization`) and persist them alongside allocator action traces so regressions are diagnosable.

---

## Limitations & failure modes

- **Gist drift:** substitutability degrades if GistNet overfits; monitor ΔNLL@`H` gaps and refresh ΔNLL labels after each B1 phase.
- **Allocator oscillation:** repeated expand/collapse of the same block indicates thresholds/cooldown need adjustment; histogram residency times to catch this.
- **Boundary artifacts:** compressed spans that straddle critical tokens (e.g., function definitions) may cause performance cliffs; add targeted tests for boundary cases.
- **Latency spikes:** excessive counterfactual sampling or large `N_diff` values can break constant-time promises; record per-iteration latency in telemetry.
- **Positional aliasing:** swapping gists without reusing original indices can shift RoPE phases; instrumentation should validate positional consistency.

---

## Evaluation & validation checklist

**Accuracy & compression**
- ΔNLL vs budget: sweep `W_max` (4k → 16k) using held-out long-form tasks; target ΔNLL degradation ≤ 0.1 compared to full-context baselines at equivalent token budgets.
- Compression stress: verify substitutability at 32× and 1024× compression with narrative and code samples, ensuring ΔNLL@`H` stays within 0.2 of the uncompressed control.
- Focus ablations: compare causal vs non-causal LensNet and allocator variants (with/without cooldown) to confirm the non-causal controller yields ≥3% lower ΔNLL@`H`.

**Runtime & stability**
- Resource trace: log GPU memory, wall-clock latency per block, and total expand/collapse mass; keep latency within +10% of the frozen baseline at 8k active tokens.
- Swap rate & residency: track mean residency ≥3 iterations per block and swap rate ≤0.25 actions per block to avoid thrashing.
- Boundary diagnostics: run synthetic tests where important tokens align with block edges to ensure no catastrophic degradation (>0.2 ΔNLL jump).

**Benchmarks**
- Evaluate narrative QA (LongBench `NarrativeQA`), academic QA (`Qasper`), and coding/story tasks (InfiniteBench). Report ΔNLL@`H`, latency, and swap metrics alongside baseline LLM runs.
- Optional stretch: include HELM-LC suites once the pipeline stabilizes to benchmark against summarization/RAG strategies.

**POC acceptance criteria**
- Demonstrate ΔNLL degradation ≤0.1 at `W_max = 8k` with constant-time compute (latency overhead ≤10%) and stable swap metrics on at least one narrative and one coding benchmark relative to the frozen base model.

---

## Example walkthrough (toy coding session)

1. **Setup:** Load a small TypeScript project summary into MegaContext memory (≈4k tokens) and seed the working context with the latest user/system gists.
2. **User turn:** “Add logging to the `fetchUser` helper.” Ingest tokens into the MegaContext tree (32-token blocks) and update L1 gists.
3. **LensNet pass:** Scores the new query tokens highly (`u_i ≈ +0.4`) and suggests expanding the gist that summarizes `fetchUser`.
4. **Focus allocator:** Applies one expand action (L1→32×L0) and one collapse on distant chatter (`u_i ≈ -0.3`), staying within `W_max`.
5. **Decode:** The base LLM, now seeing raw tokens for `fetchUser`, produces the patch. Newly generated code is appended to the MegaContext tree.
6. **Trace capture:** Log ΔNLL utilities, focus actions, and residency times to W&B for later analysis.

Document a similar narrative under `docs/walkthroughs/` once the POC code path is live so future contributors can replay it end to end.
