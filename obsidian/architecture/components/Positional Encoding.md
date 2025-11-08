---
tags:
  - components
summary: Two-track positional encoding plan that now prioritizes native end-to-end MegaAttention models while keeping a retrofit path for upgrading frozen bases later.
---
> _Diagram TODO:_ add positional encoding figure after the next design update.

MegaContext decouples storage from attention, so positional signals must stay **global** even when the [[Working Context]] presents a narrow, mixed-resolution slice of the [[MegaContext Tree]]. This note now tracks *two* coordinated efforts:

1. **Track A (POR default):** train compact, end-to-end MegaAttention-ready models where Gaussian RoPE is baked in from the first step.
2. **Track B (backlog):** preserve the retrofit kit for adapting existing frozen checkpoints when we eventually "upgrade" larger base models.

Both tracks share metadata contracts and telemetry so we can swap between them without rediscovering the fundamentals.

---

## POR alignment

- **Current priority:** ship small, fully trained MegaAttention models that treat positional encoding as part of the base model. Every wLODn node carries `(μ, σ)` into Gaussian RoPE so [[MegaPrediction]] can read multi-scale states directly.
- **Future compatibility:** the NTK stretch + ALiBi + damping recipe still exists for the day we retrofit longer contexts into an external checkpoint; it simply moved behind the POR-critical Track A.
- **Shared interfaces:** regardless of track, the runtime must emit global indices, span widths, modality hints, and telemetry hooks so the allocator, [[LensNet]], and inference caches agree on coordinates.

---

## Why positional encodings change

In standard transformers the context window is contiguous and capped; [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] or absolute embeddings only see local indices. MegaContext violates those assumptions:

- **Teleported LOD0 tokens:** Blocks can be pulled from anywhere in the tree, so indices must reference global positions, not window offsets.
- **Mixed [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)|LOD]]:** Higher-level gists represent wide spans and should advertise positional uncertainty instead of sharp indices.
- **Unbounded history:** The system must eventually disambiguate billions of tokens without wrapping or aliasing.

Therefore positional logic has to ingest span metadata from the working view, stay in sync with tree offsets ([[Node Metadata]]), and cooperate with both Track A training runs and Track B retrofits.

---

## Track A — Native end-to-end MegaAttention models (current priority)

### Gaussian RoPE stack

Once we control the full training loop we adopt Gaussian RoPE as the default:

- Each entry carries `(μ, σ)` describing the center and uncertainty of its span; these values derive from working-context metadata and LensNet's expansion history.
- Rotary phases become expectations under `N(μ, σ²)`, automatically blurring high frequencies for coarse gists while retaining LOD0 precision.
- σ stays learnable so the model can trade detail for uncertainty; gradients flow into both the transformer and [[GistNet]] when the latter learns to emit better span stats.
- Existing NTK scaling, ALiBi biases, and ultra-slow bands remain available as optional stabilizers, but they are *trained in* rather than surgically injected later.
- Because σ affects every layer, we only enable this stack when the base model trains jointly with MegaContext components—exactly the POR scenario.

### Training & inference implications

- [[MegaAttention Training]] layers and [[MegaPrediction Training]] read the same per-level positional tuples, so the final hidden state for each wLODn (used for multi-scale readouts) already encodes the right distance/scale semantics.
- [[Hierarchical KV Caching Strategy]] must store `(μ, σ)` alongside K/V tensors so cache hits remain valid even as spans collapse or expand.
- Multi-modal nodes (see [[Multimodal MegaContext]]) feed 2D versions of `(μ, σ)` into the same Gaussian machinery, letting us experiment with image + text mixtures before we ever attempt retrofitting.
- Telemetry focuses on σ distributions, ΔNLL sensitivity to positional blur, and how often [[LensNet]] expands spans purely to reduce positional uncertainty.

---

## Track B — Retrofit path for upgrading frozen bases (backlog)

When we eventually "upgrade" a larger semi-frozen model, we fall back to the adapter stack below. The techniques mirror the prior POC plan but are now scoped as optional backlog work.

### NTK-scaled RoPE

- Apply a scalar stretch `s` to the rotary angles when materializing query/key phases.
- Injected inside `src/runtime/base_model.py` before attention so the rest of the stack remains untouched.
- Target range: 80k–300k tokens per head, covering worst-case working-context spans during the demo.
- Compatible with cached KV tensors because scaling is deterministic per global index.

### LOD-aware damping

- LOD0 blocks keep σ=0 and use the stretched RoPE directly.
- LOD1/LOD2 entries obtain span width from [[Working Context]] metadata; use it to compute a Gaussian attenuation factor `exp(-0.5 * (ω σ)^2)` that suppresses high-frequency rotary bands.
- Implemented as a post-process on rotary embedding vectors. Does not modify logits or residual streams.
- Leaves the [[GistNet]] hierarchy untouched—the same gist vector feeds both positional variants.

### ALiBi augmentation via LoRA

- Add an [[Glossary#ALiBi|ALiBi]]-style linear bias head per attention head using a small LoRA adapter on the first block.
- Supplies a monotonic “arrow of time” derived from global token indices so distant spans remain ordered even when RoPE wraps.
- Training can run on limited synthetic data (ordering tasks, long-range copy) within the POC compute budget.

### Optional ultra-slow bands

- Reserve a handful of extra rotary channels with periods ≫ working window (e.g., 1M–1B tokens).
- Encoded as additional sinusoidal bands concatenated to the existing rotary set; requires no gradient updates if weights are frozen.
- Helps disambiguate extreme offsets before the Track A Gaussian stack is in place.

These adaptations operate entirely inside the runtime embedding pass so allocations, focus decisions, and [[Runtime Loop]] cadence stay unchanged even when we retrofit.

---

## Runtime interface contract

To drive both tracks cleanly, the working-context materializer must expose:

- **Global token index:** Absolute position counted from MegaContext ingest start.
- **Span width:** Number of source LOD0 tokens represented by the entry (becomes σ during Track A training, stays metadata for Track B damping).
- **LOD tag:** Useful for debugging and instrumentation even if width is sufficient.
- **Modality metadata:** Visual entries supply 2D extents `(width, height)` and per-axis σ values as described in [[Multimodal MegaContext]] so positional adapters can mix text and image signals seamlessly.

These values already exist in [[Working Context Assembly]]; we simply route them to the positional adapter alongside embeddings.

---

## Telemetry & validation

- Extend [[Telemetry]] to track positional stretch factors, σ histograms, and ALiBi bias magnitudes per iteration (flag which track produced each run).
- Add ΔNLL smoke tests where long-range spans are shuffled but indices remain fixed, ensuring positional surgery preserves logits within tolerance.
- For Track A, monitor how Gaussian blur impacts MegaPrediction acceptance: log `(g_pred · g_true)` against σ so [[LensNet]] can weigh expansion vs. uncertainty.
- For Track B, keep the old sanity checks (RoPE wraparounds, ALiBi slope ranges) so retrofits remain predictable.

---

## Related notes

- [[MegaAttention Training]] — consumes the Gaussian RoPE stack inside each layer.
- [[MegaPrediction Training]] — relies on wLOD `(μ, σ)` pairs for multi-scale readouts.
- [[Architecture Details]] — explains why the two-context split demands global indexing.
- [[Base Runtime]] — describes where positional hooks land inside the frozen decoder.
- [[Working Context Refocusing]] — ensures positional metadata stays aligned as spans expand/collapse.
- [[Training & Operations]] — hosts LoRA tuning recipes and evaluation harnesses once adapters are wired up.
