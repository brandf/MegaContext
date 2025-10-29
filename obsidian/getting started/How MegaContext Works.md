---
tags:
  - getting-started
summary: A complete narrative walkthrough explaining how MegaContext virtualizes LLM context memory through hierarchical compression and learned dynamic focus.
---
# How MegaContext Works

MegaContext virtualizes sequence memory for language models—enabling effectively infinite context at constant compute. This note provides a narrative walkthrough of the complete system.

---

## The Problem: Fixed Context Windows

Standard LLMs have a fundamental limitation:

![[Standard LLM Inference.png]]

**Traditional LLM context is fixed:**
- Most models support 4k–32k tokens
- Older context gets evicted when the window fills
- No way to zoom in/out on different parts
- Everything is at the same level of detail

**Problems this causes:**
- Long conversations get truncated
- Important earlier context is lost forever
- Can't distinguish between "critical details" and "background noise"
- Memory grows linearly with context length (GPU RAM limits)
- Compute grows quadratically with attention (O(n²) complexity)

---

## The MegaContext Solution: Virtual Memory for LLMs

MegaContext solves this by separating **long-term storage** from **active attention**, just like a computer's virtual memory separates disk from RAM.

![[ArchitectureDiagram.png]]

### Two-Context Architecture

MegaContext maintains **two separate contexts**:

#### 1. [[MegaContext Tree]] (Long-term Storage)
- **Location:** Disk (or RAM in [[POC Scope|POC]])
- **Size:** Unbounded—can grow to millions or billions of tokens
- **Content:** Complete interaction history stored as a hierarchical tree of [[Glossary#Gist / Gist Embedding|gists]]
- **Structure:** 32-ary tree with multiple levels of detail ([[Glossary#L0 / L1 / L2 (Level of Detail / LOD)|L0, L1, L2]], ...)
- **Role:** The "hard drive" of memory

#### 2. [[Working Context]] (Active Attention)
- **Location:** GPU memory
- **Size:** Fixed budget (W_max = 8k–32k tokens)
- **Content:** Mixed levels of detail—raw tokens where needed, [[Glossary#Gist / Gist Embedding|gists]] elsewhere
- **Structure:** Contiguous sequence of entries drawn from the tree
- **Role:** The "RAM" that the base LLM actually sees

See [[Architecture Details]] for the complete two-context design and invariants.

---

## The Core Insight: Hierarchical Compression

Instead of storing everything at the same resolution, MegaContext builds a **hierarchy of summaries**:

### Level 0 (L0): Raw Tokens
```
"The quick brown fox jumps over the lazy dog near the riverbank"
```
Every individual token at full detail—highest cost, highest fidelity.

### Level 1 (L1): 32→1 Gist
```
[gist: "narrative about fox movement near water"]
```
32 tokens compressed into a single learned [[Glossary#Embedding|embedding]] by [[GistNet]]—32× compression.

### Level 2 (L2): 32→1 Gist of Gists
```
[gist: "outdoor animal scene collection"]
```
32 L1 [[Glossary#Gist / Gist Embedding|gists]] compressed into one L2 [[Glossary#Gist / Gist Embedding|gist]]—1024× total compression.

**Key property: [[Glossary#Substitutability|Substitutability]]**
- [[Glossary#Gist / Gist Embedding|Gists]] are trained to be **drop-in replacements** for their tokens
- When a [[Glossary#Gist / Gist Embedding|gist]] replaces its tokens, the model's predictions barely change (low [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]])
- This lets the [[Working Context|working context]] swap between detail levels without breaking coherence

---

## The Four Core Components

### 1. [[GistNet]]: The Compressor

![[GistNetDiagram.png]]

**What it does:** Learns to compress 32-token blocks into single [[Glossary#Gist / Gist Embedding|gist]] [[Glossary#Embedding|embeddings]]

**Key features:**
- 32→1→32→1 refinement network with self-attention + cross-attention
- Outputs live in the same [[Glossary#Embedding|embedding]] space as tokens
- Trained to minimize [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]] (prediction error after [[Glossary#Substitutability|substitution]])
- Tiny model (~0.5M params per layer)

**Result:** Hierarchical [[Glossary#Gist / Gist Embedding|gist]] tree where each parent summarizes 32 children

See [[GistNet]] for full architecture and [[GistNet Architecture Details]] for implementation specifics.

---

### 2. [[LensNet]]: The Focus Controller

![[LensNet Diagram.png]]

**What it does:** Decides which parts of memory deserve detail vs compression

**Key features:**
- Dual cross-attention network ([[Working Context|working context]] ↔ [[tail gists]])
- Non-causal—can "look ahead" to understand what will matter
- Outputs signed [[focus score|focus scores]]: positive = [[Glossary#Expand|expand]], negative = [[Glossary#Collapse|collapse]]
- Trained via [[counterfactual labeling]]: compute [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL]] for hypothetical operations

**Why non-causal?**
Traditional LLM attention is causal (token N can't see token N+1). But to know if an old fact matters, you need to see future queries. [[LensNet]] operates on the full [[Working Context|working context]] to predict relevance.

See [[LensNet]] for overview and [[LensNet Scoring]] for detailed scoring mechanics.

---

### 3. [[Focus Allocator]]: The Action Planner

![[Focus Allocator Diagram.png]]

**What it does:** Converts [[LensNet]]'s [[focus score|focus scores]] into actual [[Glossary#Expand|expand]]/[[Glossary#Collapse|collapse]] operations

**Key features:**
- Greedy algorithm with priority queues ([[POC Scope|POC]] implementation)
- Positive scores → [[Glossary#Expand|expand]] queue, negative scores → [[Glossary#Collapse|collapse]] queue
- Applies N_diff operations (default 4) per iteration
- Maintains budget: `sum(entry_costs) ≤ W_max`
- Preserves contiguity, block alignment, and operation legality

**Result:** [[Working Context]] dynamically adjusts detail level while staying within budget

See [[Focus Allocator]] for overview and [[Focus Allocator Strategies]] for algorithm details.

---

### 4. [[Runtime Loop]]: The Orchestrator

![[CompleteSystem.png]]

**What it does:** Coordinates ingest → focus → decode cycle

**Per-block cycle (every K=32 tokens):**

1. **Ingest & Summarize** — Buffer tokens, run [[GistNet]], update [[MegaContext Tree]]
2. **Assemble [[Working Context]]** — Select spans and detail levels from tree
3. **Refocus** — [[LensNet]] scores entries, [[Focus Allocator]] applies operations
4. **Decode** — [[Glossary#Frozen Base Model|Frozen base LLM]] generates next token(s)
5. **Telemetry** — Log [[Glossary#ΔNLL / ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL]], [[swap rate|swap rates]], [[access count|access counts]]

See [[Runtime Loop]] for execution flow details.

---

## Real-World Example

Want to see how this works in practice? See [[Examples]] for a detailed walkthrough of a coding session that shows how [[LensNet]] and the [[Focus Allocator]] automatically shift detail levels as the user's attention moves between different parts of a codebase.

---

## Key System Properties

MegaContext achieves effectively infinite context at constant per-token cost with sub-linear memory growth. The system provides dynamic learned focus (not retrieval) and works with any pretrained LLM without fine-tuning. For example, per-step compute matches the base model decode with only ~1% overhead, while the [[Working Context]] stays fixed at W_max regardless of total history length.

See [[System Properties]] for complete analysis of constant compute, sub-linear memory, dynamic focus, and model-agnostic design, plus [[Performance Sketch]] for detailed compute/storage envelopes.

---

## Comparison to Alternatives

How does MegaContext differ from standard LLMs, RAG, or other approaches?

**vs. Standard LLMs:** Unbounded vs fixed context, constant vs quadratic compute, compressed vs lost history

**vs. RAG:** Inline [[Glossary#Gist / Gist Embedding|gist]] [[Glossary#Substitutability|substitution]] vs external retrieval, continuous refocusing vs query-time search, persistent evolving memory vs stateless chunks

See [[Comparisons]] for detailed comparison tables and [[MegaContext & RAG]] for RAG-specific analysis.

---

## Current Status

We're implementing the **proof-of-concept ([[POC Scope|POC]])** milestone:

- ✅ Repository & tooling setup
- ✅ Base runtime with [[Glossary#Frozen Base Model|frozen LLM]]
- 🔄 [[GistNet]] training & evaluation (Phase 2 - in progress)
- ⏳ [[LensNet]], [[Focus Allocator|focus allocator]], end-to-end [[Runtime Loop|loop]] (Phase 3)
- ⏳ Demo & benchmarks (Phase 4)

See [[POC Plan]] for full roadmap, [[POC Scope]] for constraints, and [[POC Implementation]] for technical details.

---

## Learn More

### Core Architecture
- [[Architecture Details]] — Two-context design, invariants, key terms
- [[MegaContext Tree]] — Hierarchical [[Glossary#Gist / Gist Embedding|gist]] tree structure and storage
- [[Working Context]] — Fixed-size GPU window and refocusing
- [[Invariants]] — System guarantees and constraints
- [[Storage Format]] — Serialization and disk layout

### Components Deep Dives
- [[GistNet]] — Overview and training
  - [[GistNet Architecture Details]] — Network structure
  - [[GistNet Training]] — Loss functions and optimization
- [[LensNet]] — Overview and focus control
  - [[LensNet Scoring]] — Score computation mechanics
  - [[LensNet Training]] — Counterfactual labeling
- [[Focus Allocator]] — Overview and planning
  - [[Focus Allocator Strategies]] — Algorithm details
- [[Tree Operations]] — Expand/collapse mechanics
- [[Working Context Assembly]] — Context construction
- [[Working Context Refocusing]] — Dynamic adjustment
- [[Node Metadata]] — Tree node data structure

### Operations & Training
- [[Runtime Loop]] — Ingest → focus → decode cycle
- [[Training & Operations]] — Training overview
- [[Alternating Optimization]] — GistNet/LensNet training cycles
- [[Telemetry]] — Logging and metrics
- [[Performance Sketch]] — Compute and storage analysis

### Vision & Extensions
- [[Grand Vision]] — Long-term goals and research directions
- [[MegaPrediction]] — Speculative planning in [[Glossary#Gist / Gist Embedding|gist]] space
- [[MegaCuration]] — Learned pruning strategies
- [[Cognitive Core]] — Reasoning models backed by MegaContext

### Reference
- [[Comparisons]] — Detailed comparison tables
- [[MegaContext & RAG]] — RAG-specific analysis
- [[Related Work]] — Academic context and prior art

---

## Summary

MegaContext virtualizes LLM context through three key innovations:

1. **Hierarchical compression** ([[GistNet]]) — Store history at multiple resolutions
2. **Learned dynamic focus** ([[LensNet]] + [[Focus Allocator]]) — Automatically adjust detail levels
3. **Two-context architecture** — Separate unbounded storage ([[MegaContext Tree]]) from fixed attention ([[Working Context]])

The result: **effectively infinite context at constant compute**, with automatic memory management and learned relevance detection. It's not about making context windows longer—it's about making them **smarter**.
