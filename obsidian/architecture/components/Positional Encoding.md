---
tags:
  - components
summary: Strategy for extending positional encodings so mixed-LOD working contexts align with global MegaContext coordinates while remaining compatible with frozen base models.
---
![[PositionalEncoding.png]]

MegaContext decouples storage from attention, so positional signals must stay **global** even when the [[Working Context]] presents a narrow, mixed-resolution slice of the [[MegaContext Tree]]. This note captures the staging plan for positional encodings across the proof-of-concept LoRA retrofit and the longer-term Gaussian RoPE upgrade.

---

## Why positional encodings change

In standard transformers the context window is contiguous and capped; [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] or absolute embeddings only see local indices. MegaContext violates those assumptions:

- **Teleported L0 tokens:** Blocks can be pulled from anywhere in the tree, so indices must reference global positions, not window offsets.
- **Mixed [[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|LOD]]:** Higher-level gists represent wide spans and should advertise positional uncertainty instead of sharp indices.
- **Unbounded history:** The system must eventually disambiguate billions of tokens without wrapping or aliasing.

Therefore positional logic has to ingest span metadata from the working view, stay in sync with tree offsets ([[Node Metadata]]), and cooperate with the frozen [[Base Runtime]].

---

## Near-term retrofit (POC)

The proof-of-concept keeps the base SmolLM3-3B checkpoint frozen and injects lightweight adapters so the runtime can already reason over >100k effective positions.

### NTK-scaled RoPE

- Apply a scalar stretch `s` to the rotary angles when materializing query/key phases.
- Injected inside `src/runtime/base_model.py` before attention so the rest of the stack remains untouched.
- Target range: 80k–300k tokens per head, covering worst-case working-context spans during the demo.
- Compatible with cached KV tensors because scaling is deterministic per global index.

### LOD-aware damping

- L0 blocks keep σ=0 and use the stretched RoPE directly.
- L1/L2 entries obtain span width from [[Working Context]] metadata; use it to compute a Gaussian attenuation factor `exp(-0.5 * (ω σ)^2)` that suppresses high-frequency rotary bands.
- Implemented as a post-process on rotary embedding vectors. Does not modify logits or residual streams.
- Leaves the [[GistNet]] hierarchy untouched—the same gist vector feeds both positional variants.

### ALiBi augmentation via LoRA

- Add an [[Glossary#ALiBi|ALiBi]]-style linear bias head per attention head using a small LoRA adapter on the first block.
- Supplies a monotonic “arrow of time” derived from global token indices so distant spans remain ordered even when RoPE wraps.
- Training can run on limited synthetic data (ordering tasks, long-range copy) within the POC compute budget.

### Optional ultra-slow bands

- Reserve a handful of extra rotary channels with periods ≫ working window (e.g., 1M–1B tokens).
- Encoded as additional sinusoidal bands concatenated to the existing rotary set; requires no gradient updates if weights are frozen.
- Helps disambiguate extreme offsets before Gaussian RoPE arrives.

These adaptations operate entirely inside the runtime embedding pass so allocations, focus decisions, and [[Runtime Loop]] cadence stay unchanged.

---

## Long-term architecture (Gaussian RoPE)

Once MegaContext trains end-to-end models (see [[Future Plan#Track B — Advanced Learning & Co-Optimization|Track B]]), we **add** Gaussian RoPE on top of the stretched RoPE + ALiBi foundation:

- Each entry carries `(μ, σ)` describing the center and uncertainty of its span.
- Rotary phases become expectations under `N(μ, σ²)`, automatically blurring high frequencies for coarse gists while retaining L0 precision.
- σ derives from span width plus system-learned noise so LensNet can negotiate detail vs. uncertainty signals.
- Existing NTK scaling, ALiBi biases, and ultra-slow bands remain in place to guarantee billion-scale disambiguation; Gaussian RoPE augments them with an LOD-aware “positional dimension.”
- Because σ affects every layer, we enable this only when the base model is trained jointly with MegaContext components.

The full stack keeps the very-long-context guarantees from the hybrid strategy while removing LOD discontinuities and letting gists express temporal fuzziness natively.

---

## Runtime interface contract

To drive both paths cleanly, the working-context materializer must expose:

- **Global token index:** Absolute position counted from MegaContext ingest start.
- **Span width:** Number of source L0 tokens represented by the entry.
- **LOD tag:** Useful for debugging and instrumentation even if width is sufficient.
- **Modality metadata:** Visual entries supply 2D extents `(width, height)` and per-axis σ values as described in [[Multimodal MegaContext]] so positional adapters can mix text and image signals seamlessly.

These values already exist in [[Working Context Assembly]]; we simply route them to the positional adapter alongside embeddings.

---

## Telemetry & validation

- Extend [[Telemetry]] to track positional stretch factors and ALiBi bias magnitudes per iteration.
- Add ΔNLL smoke tests where long-range spans are shuffled but indices remain fixed, ensuring positional surgery preserves logits within tolerance.
- Future work: incorporate positional-uncertainty stats into [[Focus Allocator]] cooldown heuristics so the allocator avoids over-expanding already precise regions.

---

## Related notes

- [[Architecture Details]] — explains why the two-context split demands global indexing.
- [[Base Runtime]] — describes where positional hooks land inside the frozen decoder.
- [[Working Context Refocusing]] — ensures positional metadata stays aligned as spans expand/collapse.
- [[Training & Operations]] — hosts LoRA tuning recipes and evaluation harnesses once adapters are wired up.
