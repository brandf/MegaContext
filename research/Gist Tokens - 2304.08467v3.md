# Gist Tokens (arXiv:2304.08467v3) — Report

## Overview
- Introduces **gisting**, a training recipe that teaches an LM to compress long
  prompts into a small set of reusable "gist tokens."
- Compression is achieved by modifying attention masks during instruction
  fine-tuning so that newly introduced gist tokens attend to the prompt while
  the prompt attends back primarily within the gist region.
- Demonstrates up to 26× prompt compression (often using a single token) with
  minimal quality loss on LLaMA-7B and FLAN-T5-XXL, yielding ≈40% FLOPs and
  latency savings when caching gists.

## Core Concepts
- **Attention-masked training**: During finetuning, mask the real prompt from
  seeing the original inputs so the model is forced to represent them through
  the new gist tokens.
- **Reusable caches**: Once trained, a task-specific prompt can be compressed
  once, cached as a short gist prefix, and reused without re-encoding the full
  prompt each time.
- **Quality preservation**: Automatic and human evaluations (ROUGE, GPT-4
  judgements) show small degradation; failures relate to repetitive or
  overfitted gists when capacity is too high.
- **Compute/storage efficiency**: Gains come both from shorter sequences and
  from the ability to reuse gist caches across queries.

## Relevance to MegaContext
- Aligns with our **GistNet** goals in `README.md`: compress spans while
  preserving substitutability. Their attention-masking trick is an empirical
  recipe we can mirror when teaching our gist hierarchy.
- Suggests a practical path for **prompt macro caching** inside MegaContext’s
  working context—store high-value task presets as single-token gists that can
  be injected alongside retrieved spans.
- Offers evidence that **aggressive compression (1 token)** can work, bolstering
  confidence in Phase 2’s target of 32→1 gist compression.
- Highlights the need for **quality guards** (detecting degenerate gists), which
  maps to our planned telemetry/pruning hooks for MegaContext memory quality.

## What We Can Use
- Implement a masked-attention curriculum when training GistNet so gist slots
  must reconstruct local token blocks without peeking at ground-truth tokens.
- Borrow their **gist caching** evaluation to benchmark FLOPs/latency impact of
  MegaContext’s gist replacement pipeline (useful for Phase 3 success criteria).
- Use their observation that too many gist tokens overfit as a constraint when
  we decide how many gist slots to allocate per block.
- Extend their **logit comparison** idea: measure ΔNLL when swapping token spans
  with gists and feed those scores into LensNet’s focus allocator.

## Limitations & Risks
- Training still relies on **instruction-tuned data**; domain shift (e.g., code
  contexts) may weaken compression quality—relevant for our code-heavy demos.
- Some gists degenerate into **repetitive boilerplate**, implying we need
  regularizers or post-filters (entropy checks, perplexity monitors).
- Compression is **task-specific**; cross-domain prompts may need multiple gists
  or on-the-fly recompression, so MegaContext must manage gist provenance.

## Potential Follow-Up Reading
- **LLMLingua / LLMLingua-2** for alternative prompt compression metrics rooted
  in token importance prediction (complements gisting’s generative approach).
- **Prompt Caching & Reuse** work from OpenAI/Anthropic (e.g., RePrompting) for
  operational patterns when many prompts share structure.
- **Long-context distillation** papers (e.g., LATS, LongLoRA) to understand how
  compression interacts with retrieval and adaptive context windows.

## Open Questions for MegaContext
- How to blend learned gist tokens with **hierarchical gist tree nodes**—do we
  treat cached gists as special L1 nodes or as annotations inside metadata?
- Can we **precompute gist tokens** for frequent tool instructions and ship them
  with MegaContext runtimes, reducing cold-start latency?
- What telemetry should we log to detect gist drift when the downstream base
  model is updated or replaced?
