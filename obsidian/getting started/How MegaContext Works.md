---
tags:
  - gettingstarted
summary: A narrative walkthrough of how MegaContext virtualizes LLM context memory through hierarchical compression and learned dynamic focus, with visual diagrams illustrating the complete system flow.
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

MegaContext maintains **two separate contexts** as described in [[Architecture Details]]:

#### 1. [[MegaContext Tree]] (Long-term Storage)
- **Location:** Disk (or RAM in [[POC Scope|POC]])
- **Size:** Unbounded—can grow to millions or billions of tokens
- **Content:** Complete interaction history stored as a hierarchical tree of [[gist|gists]]
- **Structure:** 32-ary tree with multiple levels of detail (L0, L1, L2, ...)
- **Role:** The "hard drive" of memory

See [[MegaContext Tree]] for details.

#### 2. [[Working Context]] (Active Attention)
- **Location:** GPU memory
- **Size:** Fixed budget (W_max = 8k–32k tokens)
- **Content:** Mixed levels of detail—raw tokens where needed, [[gist|gists]] elsewhere
- **Structure:** Contiguous sequence of entries drawn from the [[MegaContext Tree|tree]]
- **Role:** The "RAM" that the base LLM actually sees

See [[Working Context]] for details.

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
32 tokens compressed into a single learned [[embedding]] by [[GistNet]]—32× compression.

### Level 2 (L2): 32→1 Gist of Gists
```
[gist: "outdoor animal scene collection"]
```
32 L1 [[gist|gists]] compressed into one L2 [[gist]]—1024× total compression.

**Key property: [[substitutability|Substitutability]]**
- [[gist|Gists]] are trained to be **drop-in replacements** for their tokens
- When a [[gist]] replaces its tokens, the model's predictions barely change (low [[ΔNLL@H]])
- This lets the [[Working Context|working context]] swap between detail levels without breaking coherence

---

## The Four Core Components

![[GistNetDiagram.png]]

### 1. [[GistNet]]: The Compressor

**What it does:** Learns to compress 32-token blocks into single [[gist]] [[embedding|embeddings]]

**Architecture:**
- 32→1→32→1 refinement network
- Self-attention + cross-attention with learned [[slot]] queries
- Outputs live in the same [[embedding]] space as tokens
- Tiny model (~0.5M params per layer)

**Training:**
- Minimize [[ΔNLL@H]] (prediction error after [[substitutability|substitution]])
- Optional contrastive loss to avoid [[gist collapse]]
- [[frozen base model|Frozen base model]] provides teacher signals

**Result:** Hierarchical [[gist]] tree where each parent summarizes 32 children

See [[GistNet]] for architecture details.

---

![[LensNet Diagram.png]]

### 2. [[LensNet]]: The Focus Controller

**What it does:** Decides which parts of memory deserve detail vs compression

**Architecture:**
- Dual cross-attention network ([[Working Context|working context]] ↔ [[tail gists]])
- Non-causal—can "look ahead" to understand what will matter
- Outputs signed [[focus score|focus scores]] per [[Working Context|working context]] entry:
  - **Positive score:** [[expand|Expand]] this to finer detail (L1→L0 or L2→L1)
  - **Negative score:** [[collapse|Collapse]] this to coarser summary (L0→L1 or L1→L2)

**Training:**
- [[counterfactual labeling|Counterfactual labeling]]: compute [[ΔNLL@H|ΔNLL]] for hypothetical [[expand|expands]]/[[collapse|collapses]]
- Budget regularizer: encourage zero-sum focus changes
- Legality penalties: prevent impossible operations

**Why non-causal?**
Traditional LLM attention is causal (token N can't see token N+1). But to know if an old fact matters, you need to see future queries. [[LensNet]] operates on the full [[Working Context|working context]] to predict relevance.

See [[LensNet]] for scoring details.

---

![[Focus Allocator Diagram.png]]

### 3. [[Focus Allocator]]: The Action Planner

**What it does:** Converts [[LensNet]]'s [[focus score|focus scores]] into actual [[expand]]/[[collapse]] operations

**Strategy ([[POC Scope|POC]]):**
- Greedy algorithm with priority queues
- Positive scores → [[expand]] queue (descending order)
- Negative scores → [[collapse]] queue (ascending order)
- Apply N_diff operations (default 4) per iteration
- Hysteresis/cooldowns prevent ping-ponging

**Constraints:**
- Maintain budget: `sum(entry_costs) ≤ W_max`
- Preserve contiguity: no gaps or overlaps in timeline
- Block alignment: all boundaries at 32-token multiples
- Legality: L0 can't [[expand]] further, L2 can't [[collapse]] higher (in [[POC Scope|POC]])

**Result:** [[Working Context]] dynamically adjusts detail level while staying within budget

See [[Focus Allocator]] for algorithm details.

---

![[CompleteSystem.png]]

### 4. [[Runtime Loop]]: The Orchestrator

**What it does:** Coordinates ingest → focus → decode cycle

**Per-block cycle (every K=32 tokens):**

1. **Ingest & Summarize**
   - Buffer incoming tokens into 32-token blocks
   - Run [[GistNet]] to create/update [[gist]] nodes
   - Append to [[MegaContext Tree]] storage

2. **Assemble [[Working Context]]**
   - Select which spans from the [[MegaContext Tree|tree]] to include
   - Choose detail level (L0/L1/L2) for each span
   - Concatenate into contiguous tensor for base model

3. **Refocus**
   - [[LensNet]] scores all [[Working Context|working context]] entries
   - [[Focus Allocator]] applies [[expand]]/[[collapse]] operations
   - Budget invariant maintained: expansions balanced by collapses

4. **Decode**
   - [[frozen base model|Frozen base LLM]] sees the [[Working Context|working context]]
   - Generates next token(s)
   - Feed generated tokens back to step 1

5. **Telemetry**
   - Log [[ΔNLL@H|ΔNLL]], [[swap rate|swap rates]], [[access count|access counts]], latency
   - Used for pruning decisions ([[MegaCuration]]) and training

See [[Runtime Loop]] for execution flow.

---

## An Example: Long Coding Session

Let's walk through a realistic scenario:

### Initial State
```
User: "Show me the UserAuth class"

MegaContext Tree: [empty]
Working Context: [system prompt tokens at L0]
```

### Turn 1: Loading Context
```
System loads entire codebase → 100k tokens

MegaContext Tree:
  L0: 100k raw tokens (all files)
  L1: 3,125 gists (100k ÷ 32)
  L2: 97 gists (3,125 ÷ 32)

Working Context (W_max=8k):
  - Recent tokens (UserAuth.py) at L0: 1,500 tokens
  - Related files at L1: 100 gists
  - Distant code at L2: 50 gists
  Total: 1,500 + 100 + 50 = 1,650 tokens ✓
```

### Turn 2: Specific Question
```
User: "How does the login method handle 2FA?"

LensNet scores:
  - UserAuth.py login method: +0.8 (very relevant!)
  - Other UserAuth methods: +0.2 (somewhat relevant)
  - Unrelated files: -0.5 (compress more)

Focus Allocator actions:
  ✓ Expand login method region L1→L0 (+31 tokens)
  ✓ Expand 2FA helper region L1→L0 (+31 tokens)
  ✓ Collapse distant database.py L0→L1 (-31 tokens)
  ✓ Collapse old session code L0→L1 (-31 tokens)

Working Context (updated):
  - Login & 2FA code at L0: 2,100 tokens (expanded!)
  - UserAuth methods at L1: 80 gists
  - Distant files at L2: 52 gists
  Total: 2,100 + 80 + 52 = 2,232 tokens ✓
```

The model now sees login details at full resolution while keeping unrelated code compressed.

### Turn 3: Context Shift
```
User: "Actually, show me the database schema instead"

LensNet scores:
  - Login code: -0.6 (no longer relevant!)
  - Database files: +0.9 (very relevant!)
  - Schema definitions: +0.7 (very relevant!)

Focus Allocator actions:
  ✓ Collapse login code L0→L1 (-31 tokens × 20 blocks)
  ✓ Expand database.py L1→L0 (+31 tokens × 15 blocks)
  ✓ Expand schema.sql L2→L1 (+1023 tokens)
  ✓ Expand schema L1→L0 in detail regions (+31 tokens × 10)

Working Context (updated):
  - Database & schema at L0: 1,800 tokens (refocused!)
  - Related utils at L1: 90 gists
  - Login code now at L1: 20 gists (compressed!)
  Total: 1,800 + 90 + 20 = 1,910 tokens ✓
```

**The magic:** Login code didn't disappear—it's still in the [[MegaContext Tree]] at L0 if needed later. It's just compressed to L1 in the [[Working Context|working context]]. If the conversation returns to authentication, [[LensNet]] can re-[[expand]] it.

---

## Key Properties

### 1. Constant Compute
- Per-step cost ≈ [[frozen base model]] decode
- [[GistNet]] overhead: <0.5%
- [[LensNet]] overhead: <0.5%
- Total: ~1% overhead for infinite context

### 2. Sub-linear Memory
- [[MegaContext Tree]]: O(N) where N = total tokens
- But: compressed levels add only ~3.2% overhead (32-ary tree)
- With pruning ([[MegaCuration]]): can be even more compact
- [[Working Context]]: O(W_max) = constant

### 3. Dynamic Focus
- Not retrieval: continuous refocusing, not query-time search
- Non-causal: [[LensNet]] sees full context to predict relevance
- Reversible: compressed spans can be re-[[expand|expanded]] if they become relevant
- Learned: focus policy adapts to actual [[ΔNLL@H|ΔNLL]], not heuristics

### 4. Model-Agnostic
- [[frozen base model|Frozen base model]]—no fine-tuning required
- Works with any pretrained LLM that outputs [[embedding|embeddings]]
- [[GistNet]] & [[LensNet]] are separate, lightweight networks
- Optional [[LoRA]] adapters for better [[gist]] integration

---

## Comparison to Alternatives

### vs. Standard LLMs
| Aspect | Standard LLM | MegaContext |
|--------|-------------|-------------|
| **Context length** | Fixed (4k–32k) | Unbounded |
| **Memory** | O(N²) attention | O(W_max) constant |
| **Old context** | Lost forever | Compressed, retrievable |
| **Detail control** | All same resolution | Dynamic focus |

### vs. RAG (Retrieval-Augmented Generation)
| Aspect | RAG | MegaContext |
|--------|-----|-------------|
| **Integration** | Append external docs | Inline [[gist]] [[substitutability|substitution]] |
| **Focus** | Query-time retrieval | Continuous refocusing |
| **Detail** | Text chunks (high cost) | Learned [[gist|gists]] (low cost) |
| **Memory** | Stateless | Persistent, evolving |

See [[MegaContext & RAG]] for detailed comparison.

---

## System Benefits

### For Users
- **Unbounded conversations:** Never lose context from hours or days ago
- **Automatic summarization:** System decides what to keep detailed vs compressed
- **Faster responses:** Constant-time compute regardless of total history
- **Better relevance:** Non-causal focus avoids distractors

### For Developers
- **Any base model:** Works with [[frozen base model|frozen pretrained LLMs]]
- **Predictable costs:** Fixed GPU budget, scaling via compression
- **Persistent memory:** Conversations can pause/resume without state loss
- **Telemetry-rich:** Track what the model actually attends to

### For Researchers
- **Novel architecture:** Hierarchical learned memory, not just longer context
- **Training opportunities:** Alternating [[GistNet]]/[[LensNet]] optimization (see [[Training & Operations]])
- **Future directions:** Speculative planning ([[MegaPrediction]]), pruning ([[MegaCuration]]), [[Cognitive Core|cognitive cores]]

---

## Current Status

We're implementing the **proof-of-concept ([[POC Scope|POC]])** milestone:

- ✅ Repository & tooling setup
- ✅ Base runtime with [[frozen base model|frozen LLM]]
- 🔄 [[GistNet]] training & evaluation (Phase 2 - in progress)
- ⏳ [[LensNet]], [[Focus Allocator|focus allocator]], end-to-end [[Runtime Loop|loop]] (Phase 3)
- ⏳ Demo & benchmarks (Phase 4)

See [[POC Plan]] for full roadmap and [[POC Scope]] for constraints.

---

## Learn More

### Core Architecture
- [[Architecture Details]] — Two-context design, invariants, key terms
- [[MegaContext Tree]] — Hierarchical [[gist]] tree structure and storage
- [[Working Context]] — Fixed-size GPU window and refocusing

### Components
- [[GistNet]] — 32→1 compression architecture
- [[LensNet]] — Dynamic focus controller
- [[Focus Allocator]] — Greedy [[expand]]/[[collapse]] planner

### Operations
- [[Runtime Loop]] — Ingest → focus → decode cycle
- [[Training & Operations]] — Alternating optimization, telemetry
- [[Performance Sketch]] — Compute and storage envelopes

### Vision
- [[Grand Vision]] — Long-term goals and research directions
- [[MegaPrediction]] — Speculative planning in [[gist]] space
- [[MegaCuration]] — Learned pruning strategies
- [[Cognitive Core]] — Reasoning models backed by MegaContext

---

## Summary

MegaContext virtualizes LLM context through three key innovations:

1. **Hierarchical compression** ([[GistNet]]) — Store history at multiple resolutions
2. **Learned dynamic focus** ([[LensNet]] + [[Focus Allocator]]) — Automatically adjust detail levels
3. **Two-context architecture** ([[Architecture Details]]) — Separate unbounded storage ([[MegaContext Tree]]) from fixed attention ([[Working Context]])

The result: **effectively infinite context at constant compute**, with automatic memory management and learned relevance detection. It's not about making context windows longer—it's about making them **smarter**.
