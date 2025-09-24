# Mega Context

Virtualized, hierarchical lifetime memory for transformers. Inspired by id Software’s (MegaTexture)[https://www.youtube.com/watch?v=BiQCz2NjPR8], Mega Context treats “knowledge” like a clipmapped/mipmapped texture you can zoom in/out of under a fixed working‑set budget. Instead of stuffing facts into weights, we externalize them into a lifetime latent vault and let a lean reasoning core focus compute where it matters.

Status: early research prototype and design doc. Contributions welcome.

Overview
- Inspiration (MegaTexture): id Tech’s virtual texturing streams just the visible, necessary mip tiles into GPU memory, guaranteeing coarse coverage of the whole world and paging in detail on demand. Mega Context applies the same idea to transformer context.
- Core idea: store a lifetime of tokens as a hierarchical LOD tree of latents (LOD0 ≈ tokens; higher LODs are summaries). At inference, maintain a mixed‑LOD working context with strict coverage guarantees (parents present if children present). A learned Lens chooses where to “focus” (expand to lower LOD) and where to “defocus” (collapse to higher LOD) under a fixed budget (e.g., 128k latents).
- Why this matters: sub‑linear access to “infinite” context, smaller models that devote parameters to reasoning/planning instead of memorization, continual learning without catastrophic forgetting, privacy‑preserving personal vaults, and source‑traceable outputs.

Key benefits
- Nearly infinite virtualized context: lifetime memory grows without growing per‑step compute linearly.
- Smaller, smarter models: shift parameter budget from storing facts to reasoning, abstraction, planning.
- Better latency/computational control: value‑of‑information driven zoom‑in where detail pays off.
- Personalization and privacy: on‑device lifetime latent vaults with provenance and forgetfulness.
- Interpretability and governance: answers cite the memory nodes used; content‑addressing enables audits.

Architecture at a glance
- Lifetime Context (persistent): a content‑addressed LOD tree built bottom‑up from tokens/events; parents summarize children; optional residual/delta coding; provenance and versioning.
- Working Context (ephemeral): a fixed‑size set of mixed‑LOD latents that cover the entire lifetime; updated asynchronously every decode step by the Lens and Streaming Latent Manager.
- Reasoning Core: a transformer with heads for next‑token, latent‑level planning/prediction, and the Lens policy; reads from the working context via lightweight adapters (e.g., memory‑as‑prefix or Perceiver‑style cross‑attention).

Major components

1) Lifetime Context (LOD tree)
- LOD0: leaves representing raw tokens/chunks or multimodal leaf latents.
- Higher LODs (LOD1…LOD5): summaries of contiguous child spans (mipmap/clipmap style).
- Parent invariants: parent = learned aggregation of children (attention pool + MLP); parents must be present when any child is present. Optionally store children as residuals relative to parents.
- Incremental LOD tree updates (async).  Learned storage compaction (compression and forgetting).
- Storage/indexing: simple file storage with fixed sized tokens/latents provides fast batched access to lifetime context. focus/de-focus process eliminates the vector-database style queries found in RAG.

2) Working Context (mixed‑LOD)
- Fixed budget (e.g., 128k latents), always maintains full coverage at some LOD (e.g., LOD5) while selectively paging in finer LODs where useful.
- Parent‑present invariant: if a child is in the working set, its parent is also present (enables residual/delta decoding and coherent reasoning across scales).
- Representations: each latent is same dimension/footprint (e.g., 1k‑D float16), regardless of LOD, so the transformer sees a uniform token type with LOD metadata.

3) Lens (focus/defocus policy)
- Purpose: score every node in the working context each step in [-1, 1], where >0 requests expansion (bring children into the working set) and <0 requests defocus (evict children, keep parent).
- Training: start with heuristics (recency, similarity to query state, uncertainty proxy, LOD cost). Upgrade to counterfactual supervision (predict delta‑loss from expanding/collapsing a node) and cost‑aware RL under a fixed budget.
- Asynchronous control: expansion requests stream in; decoding does not block—in the meantime, the planner operates on coarser latents and refines when detail arrives.

4) Summarization and decoding
- Bottom‑up build: ingest leaf tokens, compute LOD0 latents, aggregate upward to build parents. Periodically re‑summarize for drift and after deletions/edits.
- Residual/delta coding: store children as residuals w.r.t. parents to reduce redundancy and enforce consistency; in the working context, child = parent_projection + residual.
- Un‑summarization decoder (optional): a small model that can “elaborate” LODx to speculative LOD(x−1) latents when ground‑truth children aren’t fetched yet; speculative tags ensure later reconciliation.

5) Streaming Latent Manager
- Responsibilities: paging (disk/NVMe → RAM → GPU), prefetch (based on Lens scores and predicted attention shifts), eviction (Belady‑like with Lens guidance), enforcing invariants, and throttling I/O.
- Async pipeline: expansions/collapses apply between decode steps; large prefetches are batched; hot nodes stay uncompressed in RAM; cold nodes use PQ/RQ codes on disk.

6) Reasoning Core and adapters
- Core: a modest transformer (e.g., 7–30B in the long run; much smaller for prototypes) with three heads: next‑token, latent planner, and Lens.
- Read path options:
  - Memory‑as‑prefix: project latents into the LM’s embedding space and prepend as pseudo tokens (quickest to prototype).
  - Perceiver‑style cross‑attention: a small set of state tokens reads from the large working set each step (scales linearly in working‑set size).
  - Hierarchical/sparse attention: ancestor‑first, then descend into focused branches (tree‑aware sparsity).

7) Governance, safety, and UX
- Source‑traceable answers: cite memory nodes used; show why specific nodes were focused.
- Memory hygiene: quarantine untrusted content, signatures, trust scores, canary nodes, and easy “forget this everywhere” operations.
- Personalization: per‑user vaults; shared org‑level lattices; differential privacy/federated updates for retrieval/summarization policies.

Workflow (inference)
1) Initialize working context with top‑level LOD covering the entire lifetime.
2) Adapter feeds mixed‑LOD latents to the core; the Lens scores nodes.
3) Streaming Latent Manager expands/collapses per Lens under budget, asynchronously.
4) Planner/head reasons at coarse LODs while fine details stream in; next‑token head outputs user‑visible tokens (LOD0).
5) New session tokens get written to LOD0; summaries update in the background.

What Mega Context is not
- Not traditional RAG that flat‑retrieves top‑k similar chunks. You “dig down” along the right branches and zoom out when appropriate, with coverage guarantees and multi‑scale coherence.
- Not a replacement for encoders or tools. It complements tool use, retrieval, and planning with a principled memory substrate.

Project roadmap (suggested)
- v0: Synthetic text dataset with planted multi‑scale structure; lifetime vault (LOD0–LOD3); heuristic Lens; memory‑as‑prefix adapter; baselines vs no‑memory and flat RAG.
- v1: Learned Lens (counterfactual deltas), residual/delta coding, provenance and forgetting, Perceiver read.
- v2: PQ‑compressed storage, HNSW for anchor selection, speculative children decoder, basic multimodal leaves (images/frames via CLIP/SigLIP).
- v3: Personal vault UX (explanations, memory inspector), safety mechanisms, federated policy updates.

Related work and background

| Topic | Work | Venue/Year | Link | Why relevant |
|---|---|---:|---|---|
| Retrieval-augmented LMs | RETRO (Borgeaud et al.) | 2022 | https://arxiv.org/abs/2112.04426 | External memory can substitute for model scale |
| Retrieval-augmented gen | RAG (Lewis et al.) | 2020 | https://arxiv.org/abs/2005.11401 | Classic retrieve-then-read baseline |
| Nearest-neighbor LMs | kNN-LM (Khandelwal et al.) | 2020 | https://arxiv.org/abs/1911.00172 | Improves generalization with datastore lookups |
| Learn when/what to retrieve | Self-RAG (Asai et al.) | 2023 | https://arxiv.org/abs/2310.11511 | Policy for retrieval decisions |
| Graph-structured retrieval | GraphRAG (MSR) | 2024 | https://microsoft.github.io/graphrag/ | Structured knowledge for retrieval |
| Compressed long memory | Compressive Transformer (Rae et al.) | 2019 | https://arxiv.org/abs/1911.05507 | Store compressed summaries of past activations |
| Long dependencies | Transformer-XL (Dai et al.) | 2019 | https://arxiv.org/abs/1901.02860 | Segment-level recurrence for long context |
| Explicit memory slots | Memorizing Transformer (Wu et al.) | 2022 | https://arxiv.org/abs/2203.08913 | Augments attention with key–value memory |
| Latent bottleneck | Perceiver IO (Jaegle et al.) | 2021 | https://arxiv.org/abs/2107.14795 | Read huge inputs via cross-attention into a small state |
| Token selection/merging | TokenLearner (Ryoo et al.) | 2021 | https://arxiv.org/abs/2106.11297 | Learn to keep informative tokens |
| Token merging | ToMe (Bolya et al.) | 2022 | https://arxiv.org/abs/2210.09461 | Merge similar tokens to save compute |
| Sparse/efficient attention | Reformer (Kitaev et al.) | 2020 | https://arxiv.org/abs/2001.04451 | LSH attention for long sequences |
| Clustered attention | Routing Transformer (Roy et al.) | 2021 | https://arxiv.org/abs/2003.05997 | Attend within clusters |
| Long context | Longformer (Beltagy et al.) | 2020 | https://arxiv.org/abs/2004.05150 | Sliding-window attention |
| Long context | BigBird (Zaheer et al.) | 2020 | https://arxiv.org/abs/2007.14062 | Block-sparse attention |
| State-space models | Mamba (Gu & Dao) | 2023 | https://arxiv.org/abs/2312.00752 | Linear-time sequence modeling |
| State-space models | S4 (Gu et al.) | 2021 | https://arxiv.org/abs/2111.00396 | Long-range structured state space |
| External memory | DNC/NTM (Graves et al.) | 2016 | https://www.nature.com/articles/nature20101 | Differentiable read/write memory control |
| Memory networks | Weston et al. | 2015 | https://arxiv.org/abs/1503.08895 | Early memory for QA |
| Adaptive compute | ACT (Graves) | 2016 | https://arxiv.org/abs/1603.08983 | Learn when to spend more steps |
| Budgeted inference | CALM (Schuster et al.) | 2022 | https://arxiv.org/abs/2207.07061 | Confidence-based compute allocation |
| Hierarchical control | FeUdal Networks (Vezhnevets et al.) | 2017 | https://arxiv.org/abs/1703.01161 | Planner/executor separation |
| World-model planning | MuZero (Schrittwieser et al.) | 2019 | https://www.nature.com/articles/s41586-020-03051-4 | Learn model + plan |
| World-model control | Dreamer (Hafner et al.) | 2020–21 | https://arxiv.org/abs/2010.02193 | Latent dynamics for long horizons |
| Vector compression | Product Quantization (Jégou et al.) | 2011 | https://ieeexplore.ieee.org/document/5432202 | 10–100× vector compression |
| Vector search lib | FAISS (Johnson et al.) | 2017 | https://arxiv.org/abs/1702.08734 | ANN at billion scale |
| ANN graph | HNSW (Malkov & Yashunin) | 2018 | https://arxiv.org/abs/1603.09320 | Fast sub-linear neighbor search |
| Graphics origins | Clipmaps (Tanner et al.) | 1998 | https://dl.acm.org/doi/10.1145/280814.280864 | Virtual mipmaps for large textures |
| Graphics practice | Virtual Texturing / MegaTexture | — | https://en.wikipedia.org/wiki/MegaTexture | Background on id Tech 5 technique |
| Agent memory mgmt | MemGPT | 2023 | https://arxiv.org/abs/2310.08560 | Agentic long-term memory policies |

Getting started (prototype plan)
- Run the synthetic data generator to create a lifetime stream with planted multi‑scale structure.
- Build LOD0–LOD3 latents bottom‑up (frozen encoder), store in a simple on‑disk tree with content‑addressing.
- Initialize a working context (e.g., LOD3 coverage), integrate with a small pre‑trained LM via memory‑as‑prefix adapter, and use the heuristic Lens to expand/collapse under a fixed budget.
- Evaluate vs baselines (no memory; flat RAG) on long‑horizon QA tasks that require both coarse summaries and specific leaves.

Design notes
- Parent‑present invariant enables delta/residual coding and coherent mixed‑LOD reasoning.
- Mixed‑LOD working sets amortize latency: coarse latents let you plan while detail streams in.
- Start simple (heuristic Lens, prefix adapter). Add Perceiver read, learned Lens, and PQ/HNSW later.

License
- To be determined.

Acknowledgments
- Inspired by id Software’s MegaTexture and the broader literature on retrieval‑augmented models, hierarchical memory, and efficient attention. Not affiliated with id Software.

Contributing
- Issues and PRs welcome. Ideas, counterexamples, and evaluation datasets especially appreciated.
