# Architecture Overview

MegaContext separates a model’s lifetime memory into two coupled spaces:

## MegaContext hierarchy
- **Hierarchical gists:** Streaming text is chunked into 32-token L0 blocks; every 32 L0 blocks form an L1 gist, and so on, yielding a tree that can span millions of tokens.
- **Persistent storage:** The gist tree lives on disk (RAM for the POC) and retains metadata so spans can be restored at higher fidelity when the task requires it.
- **Level-of-detail control:** Each node tracks its level, offsets, and parent pointers, enabling efficient expansion/collapse without breaking contiguity.

## Working context window
- **Fixed budget:** The runtime keeps a contiguous working window on GPU with an 8k–32k token-equivalent budget, blending raw tokens with gists.
- **Swappable representation:** A span can appear as raw tokens, an L1 gist, or higher-level summaries, depending on the allocator’s focus decisions.
- **Focus loop coupling:** LensNet scores each working entry and the focus allocator enforces budget limits while respecting cooldowns and contiguity. See [[Runtime Loop]] for execution details.

## Design principles
- **Deterministic dataflows:** Tensor-first data structures and cached teacher embeddings exist to make experiments reproducible (see `src/gistnet`, `tools/train_gistnet.py`).
- **Progressive refinement:** The system always keeps a path back to the original tokens—compression is reversible by rehydrating lower levels as needed.
- **Telemetry-aware:** Access counts, focus scores, and ΔNLL statistics feed future pruning and learning, ensuring the gist tree remains tractable at scale.

## Related notes
- [[Core Components]] — module-by-module breakdown.
- [[Runtime Loop]] — ingest, scoring, and decoding lifecycle.
- [[plans/Paper Plan]] — research-grade extensions (robust compression, storage, benchmarking).
