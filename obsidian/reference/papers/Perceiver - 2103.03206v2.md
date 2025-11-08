---
tags: [papers, reference]
summary: Modality-agnostic Transformer that scales to large inputs via cross-attention to a fixed-size latent bottleneck array.
---

# Perceiver (arXiv:2103.03206v2) — Report

**PDF**: [Perceiver - 2103.03206v2.pdf](Perceiver%20-%202103.03206v2.pdf)

## Overview
- Proposes the **Perceiver architecture**, a modality-agnostic Transformer that
  scales to very large inputs by routing them through a fixed-size latent array.
- Uses **cross-attention** from inputs to latents (and back) to decouple input
  size from model depth, enabling processing of images, audio, video, and point
  clouds without modality-specific inductive biases.
- Demonstrates competitive performance on ImageNet, AudioSet, and ModelNet40,
  highlighting the value of iterative attention for multi-modal perception.


## Core Concepts
- **Latent bottleneck**: A small set of learnable latent vectors attends to the
  entire input, dramatically reducing the quadratic cost of self-attention.
- **Iterative processing**: Alternating cross-attention (inputs→latents) and
  latent self-attention layers allow the model to refine latent representations
  while keeping computation bounded.
- **Flexible decoding**: Outputs are produced by querying the latent space via
  task-specific heads, enabling classification or dense predictions.
- **Minimal modality assumptions**: Raw inputs are encoded to a common vector
  space, letting the Transformer operate on arbitrarily ordered tokens.

## Relevance to MegaContext
- Mirrors MegaContext's idea of **latent working memory**: our [[Working Context]]
  (W_max = 8,192 tokens in the [[POC Implementation]]) acts as a latent bottleneck
  summarizing the much larger [[MegaContext Tree]].
- Suggests architectural patterns for [[LensNet]] + [[GistNet]] training—cross-
  attention can move between token-level inputs and compact [[Glossary#Gist / Gist Embedding|gist]] slots.
- Reinforces the value of **iterative focus**: repeatedly attending between raw
  history and latent summaries parallels the dynamic [[Glossary#Expand|expand]]/
  [[Glossary#Collapse|collapse]] loop in MegaContext's [[Runtime Loop]].
- The fixed latent bottleneck concept directly informs our [[Glossary#W_max (Token Budget)|W_max]]
  constraint, which ensures constant compute per decode step regardless of
  [[MegaContext Tree]] size.

## What We Can Use
- Adopt Perceiver-style **cross-attention blocks** when implementing [[LensNet]],
  letting latent focus queries aggregate relevant [[Glossary#Gist / Gist Embedding|gist]] spans before expansion.
- Use the notion of a **fixed latent budget** as inspiration for constraining the
  [[Working Context]] token-equivalent size (W_max = 8,192 in POC, see [[POC Implementation]]).
- Explore **PerceiverIO-style decoders** (see [[reference/papers/Perceiver IO - 2107.14795v3.md|Perceiver IO]]) for mapping
  latent MegaContext states back to structured outputs such as expansion plans via the
  [[Focus Allocator]].
- Design tests that mirror Perceiver's **multi-modal benchmarks**—mix text,
  code, and metadata tokens to ensure our architecture remains modality-agnostic.

## Limitations & Risks
- Without inductive biases, training can be **data-hungry**, which may be an
  issue for our narrower demo datasets; we might need hybrid models initially.
- Latent slots may struggle with **fine-grained ordering** without positional
  cues; MegaContext must retain [[Glossary#Absolute Position Index|absolute position indices]]
  and use [[Glossary#RoPE (Rotary Position Embedding)|RoPE]] compatibility for temporal ordering.
- Architectural complexity adds **integration overhead**; careful profiling is
  needed to ensure the latent bottleneck does not become a throughput bottleneck
  (target: <5% overhead per [[POC Implementation]]).

## Potential Follow-Up Reading
- **Set Transformer (Lee et al., 2019)** and **Linformer (Wang et al., 2020)** to
  compare alternative efficient attention schemes.
- **Perceiver AR** and **Perceiver Resampler** for autoregressive and perception
  applications that may inspire future MegaContext variants.
- **Latent diffusion / latent memory** literature to investigate how latent
  bottlenecks interact with generative modeling.

## Open Questions for MegaContext
- How many latent slots does [[Working Context]] need to cover both **[[Glossary#Gist / Gist Embedding|gist]]
  embeddings and raw [[Glossary#LOD0 / LOD1 / LOD2 (Level of Detail / LOD)|LOD0]] tokens**
  without starving the [[Glossary#Base Model|base model]]? (POC uses 8,192 token budget)
- Can we interleave **Perceiver-style attention** with our hierarchical [[MegaContext Tree]]
  traversal to provide better credit assignment during [[MegaContext End-to-End Training]]?
- What logging/[[Telemetry|telemetry]] should capture **latent utilization** so the
  [[Focus Allocator]] knows when to expand or compress particular regions?

## Related Pages
- [[Working Context]] — The fixed-size GPU window (W_max tokens)
- [[MegaContext Tree]] — Complete hierarchical gist tree
- [[LensNet]] — Focus scoring controller
- [[GistNet]] — 32→1 compression network
- [[Focus Allocator]] — Greedy expand/collapse planner
- [[Runtime Loop]] — End-to-end execution flow
- [[MegaContext End-to-End Training]] — Training strategy
- [[LensNet Training]] — LensNet training objectives
- [[GistNet Training]] — GistNet training pipeline
- [[reference/papers/Perceiver IO - 2107.14795v3.md|Perceiver IO]] — Query-based decoding extension
- [[POC Implementation]] — Current parameter values
