# MegaContext Coverage Fix Plan (On Track ✅)

## Goal
Enforce the “full-window coverage” invariant for every WorkingContext (train & inference) while preserving the immutable LOD0 tail and keeping the suite green.

## Checklist
1. **Design Reference** – _Done_
   - Documented arithmetic head/tail partitioning (block-aligned head, immutable recent tail, optional max-LOD block) in code comments and notes.
2. **Implement Head Partition** – _Done_
   - `_create_variant` now slices head coverage via `tokens_per_entry` math; no boolean masks.
   - Tail absorbs partial blocks so head always starts on block boundaries.
3. **Immutable Tail & Coverage Repair** – _Done_
   - Added `_ensure_wc_full_coverage` / `_rebuild_wc_with_lod0` to auto-heal gaps after focus edits/siblings.
   - Recent tail enforcement happens after every allocator mutation.
4. **Random Span / Seed Logic** – _Done_
   - Replaced obsolete random-span variant assertions with deterministic start sampling checks.
5. **Regression Tests** – _Done_
   - `PYTHONPATH=. pytest tests/test_mc_components.py tests/test_mc_batched_rope.py` now passes (25/25).
6. **Docs & Tracking** – _Done_
   - Updated `un-suck-tests.md` (see new execution log entry) and this plan to reflect current status.

## Next Watch Items
- Add validation-path smoke test for `evaluate_bpb_with_mc` (see unsuck plan).
- Keep tail/coverage helper wired into any future allocator edits.
