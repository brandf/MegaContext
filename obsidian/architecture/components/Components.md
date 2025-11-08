---
tags:
  - components
summary: Navigation map for MegaContext component notes under obsidian/architecture/components/.
---

![[HighLevelMCArchitecture.png]]---

The components of MegaContext work together to enable [[Context Focus]].
- **Compression:** [[GistNet]] → feeds [[MegaContext Tree]] hierarchy (see [[POC Architecture]]).
- **Focus control:** [[LensNet]] ↔ [[Focus Allocator]] work in tandem each block.
- **Runtime orchestration:** [[Runtime Loop]] consumes outputs from all components.
- **Telemetry & tooling:** [[Training & Operations]], [[Plans]], [[Performance Sketch]]

---
## Details
* [[Architecture]] — dig into the details of the system architecture.
- [[GistNet]] — 32→1 hierarchical gists aligned with the base embedding space.
- [[LensNet]] — dual cross-attention scorer providing signed utilities per [[Working Context]] entry.
- [[Focus Allocator]] — greedy, hysteresis-aware application of [[LensNet]] utilities.
- [[Positional Encoding]] — global index and LOD-aware positional strategy for frozen and co-trained models.
- [[Multimodal MegaContext]] — image gist hierarchies, multimodal positional encoding, and mixed-LOD integration.
- [[Multi-headed Focus]] — advanced focus strategies (multi-head and staging contexts) extending the baseline loop.
- [[Runtime Loop]] — streaming ingest, focus adjustment, and base LLM decode.
- [[Training & Operations]] — counterfactual labeling, alternating optimization, and telemetry.
- [[Performance Sketch]] — expected compute/storage envelopes at various scales.


