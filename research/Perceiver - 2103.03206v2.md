# Perceiver (arXiv:2103.03206v2) — Report

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
- Mirrors MegaContext’s idea of **latent working memory**: our working context
  can be seen as a latent bottleneck summarizing the much larger MegaContext
  gist tree.
- Suggests architectural patterns for **LensNet + GistNet** training—cross-
  attention can move between token-level inputs and compact latent slots.
- Reinforces the value of **iterative focus**: repeatedly attending between raw
  history and latent summaries parallels the dynamic expansion/collapse loop in
  MegaContext’s runtime.

## What We Can Use
- Adopt Perceiver-style **cross-attention blocks** when implementing LensNet,
  letting latent focus queries aggregate relevant gist spans before expansion.
- Use the notion of a **fixed latent budget** as inspiration for constraining the
  working context token-equivalent size.
- Explore **PerceiverIO-style decoders** (see companion report) for mapping
  latent MegaContext states back to structured outputs such as expansion plans.
- Design tests that mirror Perceiver’s **multi-modal benchmarks**—mix text,
  code, and metadata tokens to ensure our architecture remains modality-agnostic.

## Limitations & Risks
- Without inductive biases, training can be **data-hungry**, which may be an
  issue for our narrower demo datasets; we might need hybrid models initially.
- Latent slots may struggle with **fine-grained ordering** without positional
  cues; MegaContext must retain temporal metadata alongside gists.
- Architectural complexity adds **integration overhead**; careful profiling is
  needed to ensure the latent bottleneck does not become a throughput bottleneck.

## Potential Follow-Up Reading
- **Set Transformer (Lee et al., 2019)** and **Linformer (Wang et al., 2020)** to
  compare alternative efficient attention schemes.
- **Perceiver AR** and **Perceiver Resampler** for autoregressive and perception
  applications that may inspire future MegaContext variants.
- **Latent diffusion / latent memory** literature to investigate how latent
  bottlenecks interact with generative modeling.

## Open Questions for MegaContext
- How many latent slots does the working context need to cover both **gist
  embeddings and raw tokens** without starving the base model?
- Can we interleave **Perceiver-style attention** with our hierarchical gist tree
  traversal to provide better credit assignment during alternating EM training?
- What logging/telemetry should capture **latent utilization** so the focus
  allocator knows when to expand or compress particular regions?
