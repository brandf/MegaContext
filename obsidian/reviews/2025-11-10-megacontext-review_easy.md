---
tags:
  - review
summary: Actionable “easy” tasks extracted from the 2025-11-10 MC review; sized for codex-medium with precise file edits and tests.
---

# MegaContext — Easy Action Items

This list is scoped to surgical changes with clear outcomes and minimal cross-cutting impact.

## 1) Guardrail: Disallow unsupported 2D positional modes
- Problem: `gaussian_lod2d*` doubles rotary channels and is incompatible with `nanochat.gpt.apply_rotary_emb`.
- Change:
  - Add a validation in `MCController.__init__` to reject `positional_type in {"gaussian_lod2d", "gaussian_lod2d_alibi"}` with an instructive error.
  - File: `mc/runtime.py:133` (right after positional encoder init and before use).
- Steps:
  - Insert a check that raises `ValueError` with message: “LOD-2D positional modes require a GPT rotary kernel that supports 2D; select one of {simple, gaussian, gaussian_alibi}.”
- Tests:
  - Add `tests/test_positional_guard.py` with a config using `positional_type="gaussian_lod2d"` that asserts raising in `MCController(model, cfg)`.
- Acceptance:
  - Running pytest includes the new test and it passes. The error message is clear and actionable.

## 2) Assert embedding-head divisibility in MC
- Problem: We compute `head_dim = embed_dim // num_heads` but don’t assert divisibility.
- Change:
  - Add `assert config.embed_dim % config.num_heads == 0` before computing `_head_dim`.
  - File: `mc/runtime.py:110`–`mc/runtime.py:114`.
- Tests:
  - Extend `tests/test_mc_inference.py` with a tiny config that sets `embed_dim=7, num_heads=4` and asserts `MCController` init raises an `AssertionError`.
- Acceptance:
  - New assertion triggers for invalid combos; normal configs unaffected.

## 3) Prune unused imports/knobs
- Problem: `ThreadPoolExecutor` import is unused; `MCConfig.build_workers` is unused.
- Change:
  - Remove `from concurrent.futures import ThreadPoolExecutor`.
  - Add a `# TODO(mc): remove or implement build_workers` comment near `MCConfig.build_workers` in `mc/config.py:39` to avoid silent drift.
- Tests: none required.
- Acceptance:
  - `rg -n "ThreadPoolExecutor"` returns no hits.

## 4) Optional: Add a simple LOD0 cache toggle
- Problem: LOD0 cache can become large; configurable opt-out helps memory in long contexts.
- Change (minimal wiring):
  - Add `cache_lod0: bool = True` to `MCConfig`.
  - After building the tree in `MCController._build_tree_sample`, if `not self.config.cache_lod0`, call `tree.release_lod0_cache(disable_future_cache=True)`.
  - Files: `mc/config.py` (dataclass), `mc/runtime.py:203`–`mc/runtime.py:221`.
- Tests:
  - New `tests/test_mc_append_cache.py` case: init with `cache_lod0=False` and assert `tree.get_lod0_slice(...)` triggers embedding instead of using cache (can introspect `access_counters["token_slice"]`).
- Acceptance:
  - Toggle works; default preserves current behavior.

## 5) Document the known training limitation (until we land a fix)
- Problem: For batch size > 1, `scripts/base_train.py` doesn’t apply per-sample MC RoPE (uses only `positional_cache` for single-sample batches).
- Change:
  - README note under “Runtime Requirements & Setup” → MC section to state: per-sample positional caches are returned but not applied in batched training; a fix is planned (see review).
  - Obsidian: Add the same note to `obsidian/ops/Training & Operations.md`.
- Acceptance:
  - The note appears in both docs with a link to the planned fix (see hard tasks doc).

## 6) Add a tiny LOD2 trimming test
- Purpose: Ensure `_compute_lod_losses(..., use_lod2=True)` returns a valid LOD2 when enough LOD1 blocks are available, and `None` otherwise.
- Change:
  - New test `tests/test_lod2_trimming.py` fabricates tokens to cross the `long_horizon_multiplier` boundary and asserts expected `(lod1, lod2)` tuple shapes.
- Acceptance:
  - Test passes on CPU with mocked logits.

## 7) Small code comments to clarify behavior
- Add 1–2 line comments:
  - `mc/mega_context.py:281`: clarify why `token_slice` uses cache-preferred path.
  - `mc/runtime.py:170`: explain sequential per-sample tree build for determinism/stream contention.
- Acceptance:
  - Comments added; no logic changes.

---
Owner handoff: codex-medium can tackle items 1–3 and 5–7 quickly. Item 4 is optional if time permits.

