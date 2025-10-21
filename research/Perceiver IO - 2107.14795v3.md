# Perceiver IO (arXiv:2107.14795v3) — Report

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
  highlighting the architecture’s flexibility for multi-task reasoning.

## Relevance to MegaContext
- Provides a blueprint for **LensNet’s decision head**: treat expansion requests
  as queries over the latent working context, retrieving which gist blocks to
  expand in the next decode step.
- Suggests how to **decode structured artifacts** from MegaContext (e.g., span
  relevance maps, compression ratios) without rebuilding the model per task.
- Reinforces the benefit of **decoupling input history size from output plan
  size**, matching our need to read huge MegaContext trees but emit compact focus
  actions.

## What We Can Use
- Implement a **query-key-value interface** for MegaContext tools: ingest large
  context via gist embeddings, then answer targeted queries (plan, summarise,
  retrieve) by projecting queries into the latent space.
- Use their **positional encoding strategies** (Fourier features) to retain
  ordering when gist nodes come from different tree depths.
- Borrow evaluation setups where the same latent state services **multiple
  downstream tasks**, mirroring our goal of sharing MegaContext memory across
  agents (Chat loop, code assistant, planner).
- Apply their insight that output complexity, not input size, governs runtime to
  keep the focus allocator light even when MegaContext history explodes.

## Limitations & Risks
- Training stability can degrade when **output queries proliferate**; MegaContext
  must cap concurrent expansions or batch them carefully.
- The architecture still needs **strong encoding adapters**; we must ensure gist
  embeddings supply enough signal for the latent array to answer downstream
  queries.
- Without task-specific inductive biases, performance might trail bespoke
  decoders on niche workloads; hybrid designs may be necessary.

## Potential Follow-Up Reading
- **PerceiverAR** for autoregressive decoding with latent memories—relevant to
  long-context generation.
- **Flamingo / Perceiver-Resampler** for multimodal retrieval; they adapt latent
  bottlenecks to align with external memories similar to MegaContext.
- **Memory-augmented Transformers** (RETRO, kNN-LM) to compare retrieval-decoder
  patterns with our gist-to-token expansion loop.

## Open Questions for MegaContext
- How should we schedule **query dispatches** from LensNet? Per step, per block,
  or event-driven when base model uncertainty spikes?
- Can we integrate **structured output queries** for telemetry (e.g., requesting
  explanations, provenance) without disturbing the main decode loop?
- What caching strategy retains **latent states across turns** so we do not
  recompute cross-attention when the working context only changes slightly?
