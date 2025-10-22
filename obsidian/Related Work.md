---
title: "Related Work"
type: "concept"
status: "reference"
tags: ["related-work","research"]
summary: "References that inform MegaContext’s design, including analogues for virtual memory, latent attention, and compression."
links:
  - "[[Comparison - MegaContext vs RAG]]"
  - "[[Grand Vision]]"
---

## Layer 0 · Capture Summary
- Quick lookup of papers and systems that inspire MegaContext’s architecture and focus mechanisms.

## Layer 1 · Key References
- **Virtualization:** MegaTexture (2007) for streaming multi-resolution assets.
- **Latent attention:** Perceiver / Perceiver IO for cross-attention conditioning.
- **Slot reasoning:** Slot Attention for GistNet’s shared queries.
- **Compressed memory:** Compressive Transformer for temporal analogues.
- **Prompt compression:** Gist tokens / LLMLingua 2.
- **Retrieval baselines:** RAG family — see [[Comparison - MegaContext vs RAG]].

## Layer 2 · Detailed Table

| Concept | Reference | Relevance |
|----------|------------|------------|
| MegaTexture (id Software, 2007) | Virtualized textures | Direct analogy for streaming multi-resolution assets. |
| Perceiver / Perceiver IO (DeepMind 2021-22) | Latent cross-attention | Inspiration for LensNet’s latent conditioning. |
| Slot Attention (Locatello 2020) | Object-like latent slots | Blueprint for shared slot queries in GistNet. |
| Compressive Transformer (Rae 2019) | Long-term compressed memory | Temporal analogue for multi-scale retention. |
| Gist tokens / LLMLingua 2 (2023-24) | Prompt compression | Demonstrates substitutable learned summaries. |
| Retrieval-Augmented Generation | Search-based retrieval | Comparable memory augmentation; see [[Comparison - MegaContext vs RAG]]. |
| MegaContext (this work) | — | Unified learned compression + focus over frozen LLMs. |

## Layer 3 · Change Log
- 2025-10-22: Added metadata, layered summaries, and cross-links to the comparison note.
