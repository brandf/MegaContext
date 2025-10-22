---
title: "Cognitive Core"
type: "concept"
status: "draft"
tags: ["vision","cognitive-core","strategy"]
summary: "Outlines how a compact reasoning model can depend on MegaContext for knowledge while focusing weights on abstraction."
links:
  - "[[MOC - MegaContext]]"
  - "[[Training & Operations]]"
  - "[[MegaPrediction]]"
---

## Layer 0 · Capture Summary
- A cognitive core keeps model weights small and reasoning-focused, delegating knowledge storage and focus control to MegaContext.

## Layer 1 · Key Points
- **Components:** lightweight base model + [[GistNet]] + [[LensNet]] + MegaContext tree.
- **Training loop:** alternating optimization (see [[Training & Operations]]) with strong teachers.
- **Benefits:** smaller weights, continuous knowledge updates, traceability.
- **Research:** joint training, knowledge curation, focus policy learning, safety controls.

## Layer 2 · Detailed Notes

Inspired by Andrej Karpathy’s “cognitive core” concept—a compact reasoning engine whose weights specialize in abstraction while factual knowledge lives externally. MegaContext offers a pragmatic path to this separation by treating the MegaContext memory as an extensible knowledge substrate and keeping the working context small.

### What lives in the core?

- **Base model (<1 B params):** a compact transformer trained to reason over mixed token/gist embeddings delivered by the working context.
- **GistNet + LensNet stack:** keeps knowledge substitutable and focuses detail on demand.
- **MegaContext tree:** a curated, pre-gisted corpus of “core knowledge” (10 M–1 B tokens) spanning textbooks, documentation, ontologies, code—kept current without weight changes.

### Training the cognitive core

1. **Curate & gist the knowledge base:** preprocess the corpus into block-aligned spans, compute multi-level gists with a dedicated GistNet, and store them via the `{L0,L1,L2}.ctx` format.
2. **Warm start the controllers:** pretrain LensNet using counterfactual traces from a larger teacher model so the small core inherits a strong focusing policy; refresh GistNet on spans the policy touches most.
3. **Alternating training loop:** during each batch, the base model observes an 8–32 k working context assembled by LensNet/allocator from the knowledge tree. Optimize the model on task NLL, then refresh LensNet and GistNet in alternating phases (see [[Training & Operations]]).
4. **Encourage dependence on memory:** include tasks that require multi-hop reasoning across the knowledge tree; penalize ignoring relevant spans by comparing ΔNLL with/without expansions.
5. **Distill from a teacher:** use a larger LLM with direct access to the knowledge base to produce targets, distilling reasoning strategies into the smaller model.

### Why it matters

- **Smaller weights, richer knowledge:** the base model focuses on pattern recognition, logical composition, and planning while the MegaContext tree handles fact storage and updates.
- **Continuous learning:** updating facts means re-gisting new documents, not modifying weights—ideal for domains with rapid change.
- **Composable systems:** multiple cognitive cores can share or federate lifetime memories, enabling collaborative agents without redundant retraining.
- **Traceability:** hallucinations or conflicting answers can be traced back to the specific gists and source documents surfaced in the working context, making attribution and debugging more transparent than opaque weight memorization.

### Open research directions

- **Joint training:** exploring end-to-end differentiable surrogates that allow gradients to flow through expand/collapse actions.
- **Knowledge curation:** tools for versioning, deduplicating, and auditing the MegaContext tree as it scales to billions of tokens.
- **Focus policies:** RL or bandit strategies that optimize accuracy × latency beyond the current greedy allocator.
- **Safety & alignment:** policies for moderating which knowledge segments are surfaced to the working context in sensitive domains.

### Curating the core knowledge corpus

- **Segmented ordering:** group documents by domain or task (e.g., coding, scientific literature, product docs). Within each segment, order files so high-level gists correspond to coherent themes; for code, a chain like `README → design notes → module docs → source files` gives LensNet clear zoom targets.
- **Granularity & bridges:** keep base blocks contiguous, but insert “bridge” gists when cross-document reasoning is common (API description ↔ implementation). These bridges live at higher levels (L3/L4) and help LensNet jump across related materials.
- **Metadata enrichment:** tag each span with domain, file path, language, timestamp, recency, and trust scores. Feed these as features into LensNet so focus policies can prefer fresher or context-matching knowledge.
- **Quality control:** deduplicate near-identical spans before gist extraction; monitor gist variance to detect noisy inputs. Track provenance IDs for every gist so hallucinations can be traced back to the original source and corrected.
- **Incremental updates:** append new partitions instead of reprocessing the entire tree. Because offsets are deterministic, you can rebuild affected gists in place and avoid full re-ingest. Version each partition so rollbacks or audits remain manageable.
- **Curriculum for training:** as the corpus grows, schedule tasks that encourage the base model to rely on relevant segments (e.g., code tasks sample from the “code” partition). Penalize ignoring retrieved spans by comparing ΔNLL with and without expansions during training.

## Layer 3 · Change Log
- 2025-10-22: Added metadata, layered summaries, and explicit links to training and vision notes.
