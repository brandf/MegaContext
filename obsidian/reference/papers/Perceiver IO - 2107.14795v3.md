---
tags: [papers, reference]
summary: Extends Perceiver with query-based decoding for arbitrary structured inputs/outputs, enabling multi-task learning via a unified latent core.
---

# Perceiver IO (arXiv:2107.14795v3) — Report

**PDF**: [Perceiver IO - 2107.14795v3.pdf](Perceiver%20IO%20-%202107.14795v3.pdf)

## Overview
- Extends the Perceiver architecture to support **arbitrary structured inputs and
  outputs** through a unified latent processing core.
- Introduces **query-based decoding**, where task-specific output queries attend
  to the latent array, enabling dense predictions (e.g., segmentation) and
  multi-task outputs without restructuring the network.
- Validates on diverse tasks (ImageNet, audio, language modeling, optical flow),
  showing that a single architecture can handle heterogeneous input/output
  shapes while scaling to millions of tokens.


## Core Concepts
- **Separate input and output adapters**: Inputs are encoded to tokens that feed
  the latent array via cross-attention; outputs request information by sending
  queries back into the latent space.
- **Latent memory reuse**: A fixed-size latent array processes inputs regardless
  of their length; decoding cost scales with the number of output queries rather
  than input size.
- **Task conditioning**: Output queries can embed positional information (e.g.,
  pixel coordinates) or modality tags, letting one latent state answer many types
  of questions in parallel.
- **General-purpose training**: Demonstrates joint training on multiple tasks,
  highlighting the architecture's flexibility for multi-task reasoning.

## Relevance to MegaContext
- Provides a blueprint for [[LensNet]]'s decision head: treat [[Glossary#Expand|expansion]]
  requests as queries over the latent [[Working Context]] (W_max = 8,192 tokens in
  [[POC Implementation]]), retrieving which [[Gist|gist]] blocks to expand in the next
  decode step.
- Suggests how to **decode structured artifacts** from MegaContext (e.g., span
  relevance maps, compression ratios) without rebuilding the model per task.
- Reinforces the benefit of **query-based decoding** for outputting variable-
  length expansion plans without committing to a fixed output size—the [[Focus Allocator]]
  can emit at most N_diff=4 operations per iteration (see [[POC Implementation]]).
- Echoes the MegaContext philosophy: maintain a latent core ([[MegaContext Tree]])
  that can be queried in multiple ways, whether for next-token prediction via the
  [[Glossary#Base Model|base model]], expansion decisions via [[LensNet]], or introspection APIs.

## What We Can Use
- Adopt query-based cross-attention in [[LensNet Training]]: latent states aggregate
  [[Gist|gist]] metadata, then output queries (e.g., "which spans need focus?") attend to
  the latents to produce [[Glossary#Focus Score|focus scores]] for expansion decisions.
- Use PerceiverIO's **output query design** to structure [[Focus Allocator]]
  strategies—let the allocator emit variable-length lists of node-IDs to expand
  (up to N_diff=4 per iteration).
- Explore **multi-task heads** so a single [[LensNet]] can support both focus scoring
  and auxiliary tasks (e.g., predicting span relevance or [[Glossary#Substitutability|gist quality]]).
- Incorporate their input/output adapter patterns for modularity: we can swap in
  different [[GistNet]] encoders or new task decoders without retraining the core.

## Limitations & Risks
- Query design is **task-specific**; we must carefully craft output queries that
  align with the [[Focus Allocator]]'s decision logic, which may evolve as we tune
  thresholds (τ_expand = τ_collapse = 0.2 in POC).
- PerceiverIO can be **slow on small tasks** where standard models suffice; we
  must profile to ensure the overhead pays off on long contexts (target: <5% overhead
  per [[POC Implementation]]).
- Decoding from latents adds **architectural complexity** that complicates
  integration with existing LLM pipelines; careful modularization is critical to
  maintain compatibility with frozen [[Glossary#Base Model|base models]].

## Potential Follow-Up Reading
- **Perceiver AR** for autoregressive generation with latent bottlenecks.
- **Memory-augmented Transformers** (e.g., Memorizing Transformers, Compressive
  Transformers) for alternative ways to manage latent working memory.
- **Slot Attention** (see companion report) for a related approach to iterative
  latent refinement with competitive assignment.

## Open Questions for MegaContext
- Should we treat [[Working Context]] itself as a set of PerceiverIO-style output
  queries that "sample" the [[MegaContext Tree]], or maintain it as a flat token
  buffer with mixed [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)|LOD levels]]?
- Can we implement **incremental query updates** so that repeated focus decisions
  reuse latents from prior steps, reducing redundant computation in the K=32 token
  update cycle?
- What [[Telemetry|telemetry]] should track **query effectiveness** so we know when
  to refine the [[Focus Allocator]]'s query patterns during [[Alternating Optimization]]?

## Related Pages
- [[LensNet]] — Focus scoring controller
- [[LensNet Training]] — LensNet training objectives
- [[Focus Allocator]] — Greedy expand/collapse planner
- [[Working Context]] — Fixed-size GPU window (W_max tokens)
- [[MegaContext Tree]] — Complete hierarchical gist tree
- [[GistNet]] — 32→1 compression network
- [[Runtime Loop]] — End-to-end execution flow
- [[Alternating Optimization]] — Training strategy
- [[Perceiver - 2103.03206v2]] — Original Perceiver architecture
- [[POC Implementation]] — Current parameter values
- [[Telemetry]] — Metrics and logging
