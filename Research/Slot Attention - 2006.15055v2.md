# Slot Attention (arXiv:2006.15055v2) — Report

## Overview
- Introduces **Slot Attention**, an iterative attention module that discovers a
  set of object-centric slots from dense perceptual features without supervision.
- Each slot acts as a latent representing one scene entity; slots compete for
  features via attention with **normalised competition** and are updated through
  recurrent cross-attention and GRU refinement.
- Demonstrates unsupervised object discovery and segmentation on CLEVR, Tetris,
  and other synthetic datasets, enabling downstream tasks that rely on discrete
  object representations.

## Core Concepts
- **Slot initialisation**: A fixed number of slots are randomly initialised with
  learned means and variances, providing symmetry breaking.
- **Attention-based assignment**: An input-to-slot cross-attention step assigns
  feature vectors to slots using softmax-normalised slots (rather than features),
  ensuring slots compete for exclusive explanations.
- **Iterative updates**: Slots are refined over multiple iterations using a GRU
  and MLP, integrating assigned features while preserving slot identity.
- **Permutation invariance**: Losses treat slots as an unordered set, encouraging
  consistent object factoring across scenes.

## Relevance to MegaContext
- Offers mechanisms for **object-style partitioning**, analogous to how
  MegaContext might decompose long histories into coherent spans or "objects."
- Slot competition resembles the **focus allocator** choosing which gist blocks
  to expand; iterative updates echo LensNet’s planned multi-step refinement.
- Provides inspiration for **metadata-enriched slots**, where each gist spans a
  coherent semantic chunk tracked over time.

## What We Can Use
- Adapt slot attention’s **normalised competition** to ensure focus scores across
  gist nodes form a probability simplex, preventing over-allocation.
- Use iterative refinement with **recurrent updates** in LensNet so expansion
  decisions benefit from multiple passes over the working context.
- Explore slot-based **object permanence tracking** for MegaContext—slots could
  carry provenance IDs ensuring continuity across context updates.
- Apply slot-style **regularisers** (entropy, KL) to encourage balanced focus
  across the gist tree rather than collapsing onto a few nodes.

## Limitations & Risks
- Requires pre-specified **number of slots**; MegaContext must decide how many
  focus groups to maintain dynamically or add adaptive slot counts.
- Demonstrated mainly on **synthetic visual scenes**; transferring to textual or
  code domains needs careful feature engineering.
- Iterative refinement adds **compute overhead**; we must benchmark to ensure
  working context updates stay within latency budgets.

## Potential Follow-Up Reading
- **Object-centric learning** works such as MONet, IODINE, and Genesis for other
  approaches to unsupervised entity discovery.
- **Slot Attention-based transformers** (e.g., SAVi, Perceiver-IO slots) to see
  how slots integrate with broader architectures.
- **Neural EM** or **routing transformers** for alternative competitive
  assignment strategies between latents and inputs.

## Open Questions for MegaContext
- Can we treat **gist hierarchy levels as slots**, using competition to decide
  which level to expose to the working context?
- How do we initialise slots when context shifts abruptly (new sessions, new
  domains) without losing continuity for long-lived knowledge?
- What diagnostics should track **slot utilisation** so we can prune inactive
  focus groups or spawn new ones when memory grows?
