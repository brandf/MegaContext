# MegaContext Review — Quick Wins

## Recommended Pre-run Checks (Medium Difficulty)
- [ ] Update `bash run10.sh` to pass `--allocator greedy` (or add the alias) so MC-enabled jobs do not crash before the first batch (`run10.sh:17-94`, `mc/focus_allocator.py:107-150`).
- [ ] Fix `MCController.begin_inference_session()` so it passes the `level_cache` argument and add a tiny smoke script that exercises the call once per run to catch regressions early (`mc/runtime.py:986-1008`).
- [ ] Verify WANDB + OTEL sinks ahead of ablations; today only `mc_*` span events are emitted, so double-check dashboards can ingest them even without swap-rate/residency metrics yet.

## Findings (Quick Fix Scope)

1. **Inference helper crashes immediately.** `begin_inference_session()` was copied from the training path but calls `_build_recency_variant(tree)` without the `level_cache` dict (`mc/runtime.py:986-1007`). Every decoder warm-up explodes with a `TypeError`. Thread the cache through (like `_build_sample_context` does) and cover it with the smoke test above.
2. **run10 defaults to an unsupported allocator.** The CLI passes `--allocator simple` (`run10.sh:17-94`), but `build_focus_allocator()` only implements `greedy` (`mc/focus_allocator.py:107-150`). Medium change: normalize `simple → greedy` or just change the default flag so MC toggles stop failing instantly.
3. **Docs advertise knobs that don’t exist.** README claims there’s a transformer allocator and `--mc_tree disk` option (`README.md:45-51`), yet `MCConfig.__post_init__` rejects anything except `ram` and the only allocator is greedy (`mc/config.py:67-70`). Align the docs/scripts so newcomers don’t waste time chasing phantom modes.
4. **SETUP guide references missing tooling.** `SETUP.md` still points to `scripts/setup_megacontext.sh` and `notebooks/megacontext.ipynb` (`SETUP.md:33-44`), neither of which exist. Replace that guidance with the current nanochat runbooks (`README.md`, `obsidian/index.md`).
5. **Telemetry docs are stale.** `obsidian/ops/Telemetry.md:24-139` talks about JT loops, `configs/*.yaml`, and metrics that aren’t wired up. A documentation pass (marking which metrics are live vs. planned) is a quick win before we ask others to rely on the dashboards.

## Documentation & Consistency Tasks
- Reconcile the README/Obsidian runbooks with the real CLI flags (`README.md:45-58`, `obsidian/ops/Training & Operations.md:42-60`).
- Archive or rewrite `SETUP.md:33-44` so it no longer references deleted scripts/notebooks.
- Add a short “Quick reference” blurb to `obsidian/index.md` that points people to this Quick-Wins doc plus the deeper review.

## Testing Gaps (Medium)
- Add a CPU-only test that calls `MCController.begin_inference_session()` and `inference_step()` once, ensuring the cache wiring issue doesn’t regress.
- Extend `tests/test_mc_components.py` with a minimal check that `run10`’s allocator flag resolves to `greedy` (or add a small helper test around `build_focus_allocator` if the flag map lives there).

## Next Steps for Codex-Medium
1. Land the `run10.sh` flag fix + allocator alias and update the README/runbooks accordingly.
2. Patch `begin_inference_session`, create the tiny smoke script, and add the CPU regression test.
3. Refresh `SETUP.md`, `README.md`, and `obsidian/ops/Telemetry.md` so they match the current code path and telemetry capabilities.
4. Confirm WANDB/OTEL sinks accept the existing `mc_*` events and document any manual dashboard steps for tomorrow’s ablations.
