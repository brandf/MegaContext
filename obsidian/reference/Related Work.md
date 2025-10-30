---
tags:
  - reference
summary: Master index of all research papers and external work referenced throughout MegaContext documentation.
---

This page serves as the unified bibliography for all external research cited in MegaContext documentation. Each entry links to a detailed paper analysis in the [[papers/|papers]] directory.

Papers are referenced using **local numbering** within each document (e.g., [1], [2], [3]) rather than global numbers. This makes maintenance easier when adding new papers.

---

## Core Inspiration

- **MegaTexture** (Carmack, 2007) — [[papers/MegaTexture|Analysis]] — Virtual texturing system that inspired the core hierarchical streaming architecture
- **Perceiver** (Jaegle et al., 2021) — [[papers/Perceiver - 2103.03206v2|Analysis]] — Latent cross-attention bottleneck architecture
- **Perceiver IO** (Jaegle et al., 2021) — [[papers/Perceiver IO - 2107.14795v3|Analysis]] — Query-based decoding for arbitrary structured outputs
- **Slot Attention** (Locatello et al., 2020) — [[papers/Slot Attention - 2006.15055v2|Analysis]] — Object-centric iterative attention mechanism

## Compression & Summarization

- **Gist Tokens** (Mu et al., 2023) — [[papers/Gist Tokens - 2304.08467v3|Analysis]] — Learned prompt compression via attention masking
- **LLMLingua-2** (Pan et al., 2024) — [[papers/LLMLingua-2 - 2403.12968v2|Analysis]] — Task-agnostic prompt compression via token classification
- **Compressive Transformer** (Rae et al., 2019) — [[papers/Compressive Transformer|Analysis]] — Long-term compressed memory for transformers
- **DeepSeek OCR** (DeepSeek AI, 2024) — [[papers/DeepSeek_OCR_paper|Analysis]] — 10× optical compression via rasterization

## Retrieval & Memory Augmentation

- **RAG** (Lewis et al., 2020) — [[papers/RAG - 2005.11401v4|Analysis]] — Retrieval-augmented generation baseline
- **RETRO** (Borgeaud et al., 2022) — [[papers/RETRO|Analysis]] — Retrieval-enhanced autoregressive transformers
- **Memorizing Transformers** (Wu et al., 2022) — [[papers/Memorizing Transformers|Analysis]] — kNN-augmented approximate retrieval

## Long Context Methods

- **Transformer-XL** (Dai et al., 2019) — [[papers/Transformer-XL|Analysis]] — Segment-level recurrence and relative positional encoding
- **LongLoRA** (Chen et al., 2023) — [[papers/LongLoRA|Analysis]] — Efficient finetuning for extended context windows
- **RoPE** (Su et al., 2021) — [[papers/RoPE|Analysis]] — Rotary position embeddings used throughout MegaContext

## Attention Mechanisms

- **Sparse Transformers** (Child et al., 2019) — [[papers/Sparse Transformers|Analysis]] — Factorized sparse attention patterns
- **Reformer** (Kitaev et al., 2020) — [[papers/Reformer|Analysis]] — LSH attention and reversible layers
- **Flash Attention** (Dao et al., 2022) — [[papers/Flash Attention|Analysis]] — IO-aware exact attention algorithm

## Focus & Selection

- **Neural Turing Machines** (Graves et al., 2014) — [[papers/Neural Turing Machines|Analysis]] — Content-based addressing and memory controllers
- **Differentiable Neural Computer** (Graves et al., 2016) — [[papers/DNC|Analysis]] — Learned memory allocation and routing

## Training Methods

- **LoRA** (Hu et al., 2021) — [[papers/LoRA|Analysis]] — Low-rank adaptation used in GistNet/LensNet training
- **Knowledge Distillation** (Hinton et al., 2015) — [[papers/Knowledge Distillation|Analysis]] — Teacher-student framework for GistNet training

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

   1. Short Title (Authors, Year) — [[papers/Paper Name|Analysis]]
   2. Another Paper (Authors, Year) — [[papers/Another Paper|Analysis]]
   ```
3. **Add to this page** in the appropriate category (without numbers)
4. **Create analysis page** at `obsidian/reference/papers/Paper Name.md` following the template established in existing analyses

See any existing paper analysis for the standard format.

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
