# MegaContext

MegaContext virtualizes long-context memory for frozen LLMs by mixing raw tokens with hierarchical gists inside a fixed working window. Use this vault to capture design notes, experiments, and planning updates that complement the codebase.

## Quick links
- [[Architecture Overview]]
- [[POC Architecture|POC Architecture & Interfaces]]
- [[POC Scope|POC Scope & Constraints]]
- [[Core Components]]
- [[GistNet]]
- [[LensNet]]
- [[Focus Allocator]]
- [[Runtime Loop]]
- [[Performance Sketch]]
- [[Training & Operations]]
- [[Implementation Roadmap]]
- [[Comparison - MegaContext vs RAG|Comparison — MegaContext vs. RAG]]
- [[Grand Vision]]
- [[Cognitive Core]]
- [[MegaPrediction]]
- [[Pruning MegaContext]]
- [[Related Work]]
- [[plans/POC Plan]]
- [[plans/Paper Plan]]
- [[plans/Future Plan]]

## Working ground rules
- The repo’s conceptional source of truth is the top-level `README.md` for high-level orientation; these notes hold detailed design narratives.
- Store collaborative diagrams in `assets/` and reference them from here with relative paths; keep transient sketches in this vault.
- Track roadmap adjustments in `obsidian/plans/` notes before reflecting additional context here.

## Context snapshots
- **MegaContext vs working context:** A persistent gist tree stores the full interaction history while a budgeted working window feeds the frozen base model (`[[Architecture Overview]]` dives deeper).
- **Dynamic focus:** LensNet and the focus allocator decide when to expand or collapse spans, ensuring the working window stays within the allotted token cost (`[[Runtime Loop]]` covers the loop).
- **Compression layers:** GistNet converts 32-token blocks into gists, enabling multi-level detail representations that the system can swap in on demand (`[[Core Components]]` outlines each module).
