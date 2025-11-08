---
tags:
  - reference
summary: Master index of all research papers and external work referenced throughout MegaContext documentation.
---

This page serves as the unified bibliography for all external research cited in MegaContext documentation. Each entry links to a detailed paper analysis in the `reference/papers/` directory.

Papers are referenced using **local numbering** within each document (e.g., [1], [2], [3]) rather than global numbers. This makes maintenance easier when adding new papers.

---

## Core Inspiration

- **MegaTexture** (Carmack, 2007) — [[reference/papers/MegaTexture.md|Analysis]] — Virtual texturing system that inspired the core hierarchical streaming architecture
- **Perceiver** (Jaegle et al., 2021) — [[reference/papers/Perceiver - 2103.03206v2.md|Analysis]] — Latent cross-attention bottleneck architecture
- **Perceiver IO** (Jaegle et al., 2021) — [[reference/papers/Perceiver IO - 2107.14795v3.md|Analysis]] — Query-based decoding for arbitrary structured outputs
- **Slot Attention** (Locatello et al., 2020) — [[reference/papers/Slot Attention - 2006.15055v2.md|Analysis]] — Object-centric iterative attention mechanism

## Compression & Summarization

- **Gist Tokens** (Mu et al., 2023) — [[reference/papers/Gist Tokens - 2304.08467v3.md|Analysis]] — Learned prompt compression via attention masking
- **LLMLingua-2** (Pan et al., 2024) — [[reference/papers/LLMLingua-2 - 2403.12968v2.md|Analysis]] — Task-agnostic prompt compression via token classification
- **Compressive Transformer** (Rae et al., 2019) — [[reference/papers/Compressive Transformer.md|Analysis]] — Long-term compressed memory for transformers
- **DeepSeek OCR** (DeepSeek AI, 2024) — [[reference/papers/DeepSeek_OCR_paper.md|Analysis]] — 10× optical compression via rasterization

## Retrieval & Memory Augmentation

- **RAG** (Lewis et al., 2020) — [[reference/papers/RAG - 2005.11401v4.md|Analysis]] — Retrieval-augmented generation baseline
- **RETRO** (Borgeaud et al., 2022) — [[reference/papers/RETRO.md|Analysis]] — Retrieval-enhanced autoregressive transformers
- **Memorizing Transformers** (Wu et al., 2022) — [[reference/papers/Memorizing Transformers.md|Analysis]] — kNN-augmented approximate retrieval

## Long Context Methods

- **Transformer-XL** (Dai et al., 2019) — [[reference/papers/Transformer-XL.md|Analysis]] — Segment-level recurrence and relative positional encoding
- **LongLoRA** (Chen et al., 2023) — [[reference/papers/LongLoRA.md|Analysis]] — Efficient finetuning for extended context windows
- **RoPE** (Su et al., 2021) — [[reference/papers/RoPE.md|Analysis]] — Rotary position embeddings used throughout MegaContext

## Attention Mechanisms

- **Sparse Transformers** (Child et al., 2019) — [[reference/papers/Sparse Transformers.md|Analysis]] — Factorized sparse attention patterns
- **Reformer** (Kitaev et al., 2020) — [[reference/papers/Reformer.md|Analysis]] — LSH attention and reversible layers
- **Flash Attention** (Dao et al., 2022) — [[reference/papers/Flash Attention.md|Analysis]] — IO-aware exact attention algorithm

## Focus & Selection

- **Neural Turing Machines** (Graves et al., 2014) — [[reference/papers/Neural Turing Machines.md|Analysis]] — Content-based addressing and memory controllers
- **Differentiable Neural Computer** (Graves et al., 2016) — [[reference/papers/DNC.md|Analysis]] — Learned memory allocation and routing

## Training Methods

- **LoRA** (Hu et al., 2021) — [[reference/papers/LoRA.md|Analysis]] — Low-rank adaptation used in GistNet/LensNet training
- **Knowledge Distillation** (Hinton et al., 2015) — [[reference/papers/Knowledge Distillation.md|Analysis]] — Teacher-student framework for GistNet training

---

## Papers by MegaContext Component

### [[GistNet]]
- Gist Tokens
- LLMLingua-2
- Compressive Transformer
- DeepSeek OCR
- Knowledge Distillation

### [[LensNet]]
- Perceiver
- Perceiver IO
- Slot Attention
- Neural Turing Machines
- Differentiable Neural Computer

### [[Focus Allocator]]
- Slot Attention
- Neural Turing Machines
- Differentiable Neural Computer

### [[MegaContext Tree]]
- MegaTexture
- Compressive Transformer
- Transformer-XL

### [[Working Context]]
- RoPE
- Sparse Transformers
- Reformer
- Flash Attention

### Training & Operations
- LoRA
- Knowledge Distillation
- Alternating optimization (see GAN training literature)

---

## Related Comparisons

See [[Comparisons]] for detailed comparisons between MegaContext and:
- RAG systems ([[MegaContext & RAG]])
- Standard long-context LLMs
- Sparse attention methods
- Compressive transformers

---

## Contributing References

When adding a new reference to the documentation:

1. **Add inline citation** in the source document using square brackets: `[1]`, `[2]`, etc. with **local numbering** specific to that document
2. **Add to References section** at bottom of that document:
   ```markdown
   ## References

   1. Short Title (Authors, Year) — add `reference/papers/<Paper Title>.md` and link it from the source doc.
   2. Another Paper (Authors, Year) — same pattern for the new citation.
   ```
3. **Add to this page** in the appropriate category (without numbers)
4. **Create analysis page** at `obsidian/reference/papers/Paper Name.md` following the template established in existing analyses

See any existing paper analysis (e.g., [[reference/papers/Gist Tokens - 2304.08467v3.md|Gist Tokens]]) for the standard format.

---

## Paper Analysis Status

### Complete Analyses (7 papers with PDFs)
- Gist Tokens
- LLMLingua-2
- Perceiver
- Perceiver IO
- RAG
- Slot Attention
- DeepSeek OCR

### Placeholder Analyses (14 papers awaiting PDFs)
- MegaTexture
- Compressive Transformer
- RETRO
- Memorizing Transformers
- Transformer-XL
- LongLoRA
- RoPE
- Sparse Transformers
- Reformer
- Flash Attention
- Neural Turing Machines
- DNC
- LoRA
- Knowledge Distillation
