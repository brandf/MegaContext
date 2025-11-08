---
tags: [papers, reference]
summary: Unified encoder-decoder VLM achieving 10× optical compression by rasterizing text to images and decoding back to text with 97% accuracy.
---

# DeepSeek-OCR — Report

**PDF**: [DeepSeek_OCR_paper.pdf](DeepSeek_OCR_paper.pdf)

## Overview
- Proposes **DeepSeek-OCR**, a unified encoder–decoder VLM that compresses long textual contexts by rasterizing them into images and decoding back to text.
- Combines a bespoke **DeepEncoder** (≈380 M params) with a **DeepSeek-3B MoE decoder** (≈570 M active params) to translate high-resolution document images into language tokens.
- Achieves **≈10× optical compression** with ~97 % accuracy on Fox benchmark OCR tasks, and remains ≈60 % accurate at 20× compression; outperforms heavier end-to-end OCR/VLM baselines on OmniDocBench while using <800 vision tokens per page.
- Designed for production throughput (200 k+ pages/day on a single A100 40 GB) and multi-resolution deployment, hinting at practical pathways for long-context archival and targeted "memory forget" mechanisms.

## Core Concepts
- **DeepEncoder architecture**: cascades SAM-base (window attention) with CLIP-large (global attention) bridged by a 16× convolutional downsampler, reducing 4 096 patch tokens (1024² image, 16×16 patches) to 256 vision tokens while controlling activation memory.
- **Multi-resolution modes**: native presets (Tiny 512²/64 tokens, Small 640²/100 tokens, Base 1024²/256 tokens, Large 1280²/400 tokens) plus **dynamic Gundam modes** mixing tiled local crops with a global view to handle ultra-wide documents without exploding token counts.
- **MoE decoder**: DeepSeek-3B-MoE activates 6/64 routed experts per step, mapping compressed vision tokens to text sequences; inference behaves like a lightweight 500 M model with 3 B expressivity.
- **Data engine**: staged OCR1.0 (30 M PDFs with coarse/fine labels), OCR2.0 structured assets (charts, formulas, geometry), general vision corpora, and text-only pretraining; labels sourced via layout detectors (PP-DocLayout), OCR ensembles (MinerU, GOT-OCR2.0), and curated prompts.
- **Training pipeline**: joint training across resolution modes (native + Gundam), followed by continued training for extreme modes (Gundam-master) to balance load; prompts toggle between detection-inclusive and detection-free outputs.
- **Evaluation results**: state-of-the-art edit-distance performance on OmniDocBench with drastically fewer tokens than MinerU2.0 or GOT-OCR2.0; per-category analysis shows slides/books compress well with 64–100 tokens, while newspapers demand dynamic tiling.

## Relevance to MegaContext
- Demonstrates a viable **optical compression tier** that maps text to compact vision tokens, aligning with MegaContext's goal of hierarchical context compression (e.g., for archival branches or cold storage).
- Offers design cues for **multi-resolution gist representations**—tiled local views plus global context mirror how MegaContext might mix fine-grained tokens with coarse gists.
- Suggests a pathway for **memory pruning/forgetting**: documents beyond the active window could be snapshot into visual form, storing a high-density representation that is selectively reconstructed when needed.
- Highlights the value of **MoE decoders** for reconstructing compressed contexts, complementing our focus on focus allocators and gist substitution.

## What We Can Use
- Prototype a **vision-backed compression stage** for MegaContext's cold tier: store rarely accessed spans as rendered images with DeepEncoder-like token budgets, rehydrate via OCR on demand.
- Borrow the **multi-resolution scheduling** (Tiny→Gundam) to drive [[Focus Allocator]] policies—[[LensNet]] could request higher "optical" resolutions only when token counts exceed thresholds.
- Leverage their **data-engine blueprint** to create MegaContext training corpora that pair text spans with layout metadata, enriching [[Node Metadata|provenance tracking]] and [[LensNet]] features.
- Use DeepSeek-OCR outputs to **bootstrap labeled datasets** (layout boxes, transcriptions) for evaluating gist fidelity and for generating synthetic long-context test cases.

## Limitations & Risks
- Optical compression depends on **high-quality rendering**; lossy rasterization or handwriting could erode the decoder's ability to reconstruct, risking information loss in MegaContext archives.
- Model is tuned for **document OCR**; adapting to code diff views, chat transcripts, or UI logs may require additional domain-specific training.
- Multi-resolution tiling introduces **positional alignment complexity**; MegaContext must track tile-level provenance to avoid misplacing retrieved content.
- Reconstruction involves a relatively large MoE; integrating into tight latency loops may be impractical without distillation or on-device accelerators.

## Potential Follow-Up Reading
- **InternVL** and **Qwen-VL** series for alternative dynamic-resolution VLM encoders.
- **GOT-OCR2.0** and **MinerU2.0** to compare data flywheel strategies and detection–recognition coupling.
- **Vision-token optimization** works (e.g., Token Merging, Vary) for further reducing vision-token counts before decoding.
- **Compression via rasterization** literature (Neural Document Compression, LayoutLMv3) for theoretical foundations on multimodal token economies.

## Open Questions for MegaContext
- Should we treat **optical compression** as a distinct gist tier separate from [[GistNet]]'s learned embeddings, or fuse them in a unified hierarchy?
- Can we mix **vision tokens and text tokens** in [[Working Context]], letting [[LensNet]] choose whether to expand visually or textually?
- What heuristics distinguish when to archive via optical vs semantic gisting—layout complexity, access frequency, semantic coherence?
- How do we **version-control** optical snapshots so reconstructed spans remain aligned with the original [[MegaContext Tree]] topology?

## Related Pages
- [[MegaContext Tree]]
- [[GistNet]]
- [[LensNet]]
- [[Focus Allocator]]
- [[Working Context]]
- [[Node Metadata]]
- [[GistNet Training]]
- [[Telemetry]]
