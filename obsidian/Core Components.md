# Core Components

## [[GistNet]]
- Compresses contiguous 32-token segments into single-vector gists using RoPE-enabled self-attention and residual MLPs.
- Trained against cached teacher embeddings and ΔNLL targets to ensure substituted spans preserve predictive fidelity.
- Tooling: `tools/train_gistnet.py`, `src/gistnet/blocks.py`, `src/gistnet/model.py`, and future multi-domain curricula described in [[plans/Paper Plan]].

## MegaContext tree
- Maintains contiguous tensors for L0 tokens and higher-level gists with metadata for offsets, parent pointers, and span ids.
- Persists to `{L0,L1,L2}.ctx` files in advanced milestones and supports append-only updates plus provenance.
- Core implementation work tracked under `src/megacontext/memory/tree.py` with Phase 3 targets in [[plans/POC Plan]].

## Working context manager
- Tiles a contiguous window of tokens/gists that stays within the token-equivalent budget `W_max`.
- Provides tensor views and legality masks so downstream code can enumerate candidate windows.
- Lives in `src/runtime/working_context.py`; POC milestones finalize a token pass-through version before layering in focus dynamics.

## [[LensNet]]
- Scores each working-context entry for expansion or collapse, conditioning on the surrounding gist cache.
- Research roadmap includes Perceiver-style latent slots, slot-attention competition, and layout-aware features ([[plans/Paper Plan]]).
- Training harness `tools/train_lensnet.py` measures swap rates, regret, and ΔLoss to justify focus decisions.

## [[Focus Allocator]]
- Applies LensNet scores with greedy expand/collapse while respecting contiguity, cooldowns, and hysteresis.
- Responsible for selecting which spans upgrade to raw tokens vs summarized gists.
- Engine integration occurs in `src/runtime/focus_allocator.py` and `src/runtime/engine.py`.

## Telemetry & tooling
- ΔNLL, swap rate, latency, and provenance statistics feed pruning, benchmarking, and future research.
- Visualization efforts (timeline playback, tree explorers) appear in [[plans/Paper Plan]] Phase 4 and future Track E work in [[plans/Future Plan]].
