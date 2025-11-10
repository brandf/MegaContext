---
tags:
  - design
  - mc
summary: Proposal for a chunked (rope-like) WorkingContext backend that supports efficient mid-window expand/collapse edits.
---
# Chunked WorkingContext (Block Rope) — Design Note

## Motivation

The current `WorkingContext` stores embeddings, LODs, and positions as flat tensors and applies edits via `torch.cat`. This is conceptually simple and works well at small edit rates, but mid-window expand/collapse can cause O(n) copies per edit. A chunked, rope-like structure enables near-O(1) block inserts/erasures aligned to MC’s `block_size` while preserving the public API.

## Constraints & Operations

- Edits are block-aligned by design:
  - Expand: replace 1 parent (LOD k) at `wc_start` with `block_size` children (LOD k-1).
  - Collapse: replace `block_size` contiguous children with 1 parent.
  - Both require LOD/position alignment (already enforced in allocator/runtime).
- Append happens during autoregressive decoding (LOD0 at tail).
- No autograd requirements for WC tensors (inputs only).
- WC must present a contiguous `[B, W, D]` view to the model when needed.

## Proposed Structure

“Block Rope”: represent WC as small chunks plus a lazy-concatenated view.

- Data
  - `chunks: List[torch.Tensor]` of shape `[B, n_i, D]` on device
  - `lod_chunks: List[torch.LongTensor]` of shape `[B, n_i]`
  - `pos_chunks: List[torch.LongTensor]` of shape `[B, n_i]`
  - `prefix_lengths: List[int]` cumulative element counts for O(log C) index→(chunk,offset) mapping
  - `dirty: bool` and cached flat views of embeddings/lods/positions, rebuilt only when needed
- Parameters
  - `chunk_size` ≈ `block_size * 4` (tune between 64–256)
  - Merge adjacent chunks if either becomes too small (< chunk_size/2)

### Operations

- to_tensor()/get_positions()/get_lod_tensor():
  - If `dirty`, `torch.cat` across chunks and cache; else return cached flat tensors

- append(embedding, lod=0, position):
  - If tail chunk has capacity, append into it; else create a new chunk
  - Mark `dirty=True`

- replace(edit):  (covers expand/collapse)
  - Map `wc_start` and `old_count` to (chunk,offset) via `prefix_lengths`
  - Splice out affected span across up to two chunks
  - Insert new chunk(s) for `replacements`
  - Update `lod_chunks` and `pos_chunks` with the same splice/insert
  - Recompute `prefix_lengths`; coalesce small neighbors when appropriate
  - Mark `dirty=True`

### Complexity

- Replace becomes O(#chunks touched + #chunks inserted) instead of O(W)
- Append is amortized O(1)
- to_tensor() remains O(W) but only when the model requests a flat view; repeated allocator edits no longer copy the entire window each time

## API Compatibility

Maintain the existing `WorkingContext` interface:
  - `to_tensor()`, `get_positions()`, `get_lod_tensor()` return flat tensors
  - `append(...)` and `replace(WorkingContextEdit)` semantics unchanged
  - Positional encodings are computed on-demand from the (cached) flat views

## Telemetry

Expose chunk-level stats to telemetry (optional): number of chunks, mean chunk size, number of merges per step. Useful to track fragmentation.

## Migration Plan

1. Introduce `WorkingContextChunked` alongside the current class.
2. Add a config flag (e.g., `working_context_backend = "flat"|"chunked"`, default "flat").
3. Implement minimal parity: append/replace, lod/position plumbing, and flat views.
4. Extend unit tests to run against both backends.
5. Benchmark on CPU and single GPU with aggressive expand/collapse; compare WC edit cost.
6. If stable, make chunked backend the default for large W.

## Risks & Mitigations

- GPU fragmentation from many small tensors → keep chunk size reasonable and aggressively coalesce neighbors; cap chunk count.
- Increased code complexity → encapsulate chunk logic in a dedicated class; keep the flat backend available as a fallback.
- Latency spikes when materializing flat views → reuse cached concatenations across multiple operations; invalidate only on mutation.

## Open Questions

- Optimal `chunk_size` relative to `block_size` and typical W?
- Should we keep chunks on device or stage on CPU for very large W? (likely keep on device)
- Can we align chunk boundaries with `block_size` to always make expand/collapse single-chunk operations?

---

This design keeps the simple flat path for now and offers a clear migration to efficient mid-window edits once profiling shows WC edits becoming a bottleneck.

