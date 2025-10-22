---
tags:
  - architecture
summary: Streaming loop that ingests tokens, updates the gist tree, refocuses, and decodes with the frozen base model.
---
The runtime loop ingests tokens into the MegaContext tree, scores the working window via [[LensNet]], applies the [[Focus Allocator]], and decodes with the frozen base LLM while logging telemetry.

---

- **Stages:** ingest → gist update → focus scoring → allocation → decode → telemetry.
- **Budget:** working context remains within `W_max` using block-aligned actions.
- **Demo goals:** `tools.run_poc_loop` showcases expansion/collapse within limits.
- **Telemetry:** swap rates, ΔNLL, latency feed pruning and training loops.

---
## Details

The streaming runtime keeps a frozen base LLM within a fixed working window while preserving the full MegaContext history.

## End-to-end flow
1. **Ingest & chunk:** Incoming text is tokenized into 32-token blocks and appended to the MegaContext gist tree (`src/megacontext/memory/tree.py`).
2. **Gist updates:** GistNet generates or refreshes hierarchical summaries so higher-level nodes stay in sync with new tokens.
3. **Focus scoring:** LensNet evaluates the working window, producing signed scores that suggest which spans deserve finer or coarser detail.
4. **Allocation step:** The focus allocator expands high-score spans to raw tokens or collapses low-score spans to gists, ensuring the window adheres to `W_max`.
5. **Decode:** The working window feeds the frozen base model (`src/runtime/base_model.py`), producing next-token predictions or downstream logits.
6. **Telemetry:** Swap events, ΔNLL comparisons, and latency stats are logged for analysis and future tuning.

## Demo targets
- POC goal: `tools.run_poc_loop` streams a synthetic session, showing expansion/collapse while maintaining budget invariants ([[plans/POC Plan]]).
- Research milestone: benchmarking harnesses compare MegaContext runs against baselines and track swap rate, loss, and latency ([[Research Paper Plan]] Phase 4).

### Focus heuristics
- **Greedy but bounded:** Hysteresis and cooldowns prevent thrashing when spans hover near the decision boundary.
- **Multi-scale awareness:** Bundles of raw tokens plus their parent gists let the allocator choose hybrid representations.
- **Telemetry-driven evolution:** Access counts and ΔNLL sensitivity inform pruning strategies explored in Track B/D of [[plans/Future Plan]].