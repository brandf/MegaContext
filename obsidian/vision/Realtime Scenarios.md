---
tags:
  - vision
summary: Estimates storage needs for a real-time MegaContext powered robot that lives for 10 years
---
## Long-term storage example: 24/7 robot (10 years)

**Assumptions**

- Sampling: 500 feature vectors / sec
- Feature size: 4,096-dim; stored as fp16 (2 bytes) unless noted
- Duration: 10 years ≈ 3.1536×10⁸ seconds ⇒ `N = 500 × 3.1536×10⁸ ≈ 1.5768×10¹¹` leaf vectors
- Tree arity: 32→1 at each level (no overlap)
- Tree depth: `log₃₂(N) ≈ 8` levels (root near level 8)
- Node payload: one vector per node (same width as leaves, different precision per scenario)

---

| Scenario | Estimated storage | Interpretation |
|---|---:|---|
| Raw leaves only (fp16) | ~1.29 PB | `N × 4096 × 2` bytes |
| Full 32-ary tree at fp16 | ~1.33 PB | 32/31 overhead across all levels |
| Full tree, 8-bit everywhere | ~667 TB | Leaves 8-bit (~646 TB) + internal (~20.8 TB) |
| Pruned: 1% leaves @ 8-bit + all internal @ 8-bit | ~27 TB | Leaves 6.46 TB + internal 20.8 TB |
| Pruned + entropy coding (approx ×0.5) | ~13–14 TB | 3.2 TB leaves + 10.4 TB internal |
| Aggressive: 0.5% leaves @ 8-bit + entropy; internal @ 4-bit + entropy | ~6–8 TB | ~1.6 TB leaves + ~5.2 TB internal |

---

**Key takeaways**

- A full 32-ary tree only adds ~3.2% storage over leaves when stored at the same precision (factor 32/31); precision and pruning dominate total footprint.
- With 8-bit quantization and reasonable pruning of raw leaves (e.g., keep only salient 0.5–1%), plus straightforward entropy coding, a decade of continuous 500 Hz, 4k-dim features compresses to single-digit TBs—practical for local SSD arrays.
- This makes a lifelong, high-bandwidth memory feasible: raw details can be recovered where preserved; elsewhere, multilevel gists maintain global context with the working context handling on-demand re-expansion.