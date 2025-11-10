# MegaContext Product Code Review

## Critical Bugs Blocking Reliable Runs
- [x] LOD construction never rolls the pooled embeddings forward, so every level above LOD0 is recomputed from the raw token grid instead of summarizing its immediate children; deeper trees end up duplicating LOD1 and the allocator/gist supervision receive incoherent structures. (`mc/mega_context.py:76`)
- [x] `_compute_lod_losses` returns a bare `None` when the horizon window is shorter than `block_size`, but the caller unconditionally expects a tuple; any short horizon (common near sequence ends) raises a `TypeError` mid-step. (`mc/runtime.py:561`, `mc/runtime.py:441`)
- [x] The inference append path feeds `[B, T, D]` tensors into `WorkingContext.append`, which only accepts `[B, D]`, and it stamps every appended chunk at a single global position; inference sessions will either crash or corrupt WC metadata as soon as you stream more tokens. (`mc/focus_allocator.py:45`, `mc/working_context.py:81`, `mc/runtime.py:821`)
- [x] The optional-import guard only wraps `mc.config`, leaving `mc.runtime` and `mc.telemetry` outside the `try`. Any environment without the MC extras still crashes during module import, defeating the `--mc_enabled=0` workflow. (`scripts/base_train.py:31`)
- [x] `mc_controller.process_batch` runs once per outer step, but the training loop keeps reusing the same `mc_result` and positional cache after `x`/`y` are swapped for the next micro-batch inside gradient accumulation. Micro-steps >1 backprop MC losses/positional overrides that belong to stale data. (`scripts/base_train.py:261`, `scripts/base_train.py:369`)
- [x] Positional overrides are derived from the first variant of the first sample only, then broadcast to every sequence in the batch; most tokens therefore see mismatched RoPE/ALiBi phases even when MC is carrying useful positions. (`mc/runtime.py:200`, `mc/runtime.py:387`, `scripts/base_train.py:372`)

## Performance & Scalability Risks
- [x] LensNet runs twice per variant: first for telemetry/sibling generation and again inside `FocusAllocator.update_focus`, immediately after `variant.lens_scores` was computed. Cache or pass the existing scores into the allocator to avoid doubling this expensive transformer. (`mc/runtime.py:320`, `mc/focus_allocator.py:63`)
- [x] Every sample is embedded twice: once when the tree is built (`self.embed(tokens)`) and again inside the main GPT forward pass, roughly doubling embedding FLOPs and SRAM footprint for long contexts. Sharing the embedding lookup (or reusing MC’s cached LOD0 tensor) would free ~15–20% wall-clock. (`mc/runtime.py:144`, `scripts/base_train.py:272`)
- [x] `_reshape_for_pool` and the mean baseline treat zero padding as real tokens, so tail blocks with <`block_size` tokens get diluted gists and inaccurate access stats; masking out padded positions before pooling would keep higher LODs faithful. (`mc/mega_context.py:94`)
- [x] Tree+variant construction is entirely serial over the batch, with Python `random` sampling and repeated tensor clones on the default stream; large device batches stall waiting for MC bookkeeping. Consider vectorizing by batching token embeddings and doing variant sampling in Torch. (`mc/runtime.py:140`)
- [x] Horizon LOD losses materialize full vocab probabilities (`torch.matmul(pred_probs, self.embed.weight)`) for every block. This scales poorly with large vocabularies; projecting via a smaller learned head or reusing the model’s internal value projections could cut both compute and memory. (`mc/runtime.py:573`)

## Architecture & Feature Suggestions
- [ ] Provide per-sample positional caches (or a map from session→cos/sin) so the model can opt into MC-conditioned RoPE on a subset of sequences instead of forcing a single cache across the whole batch. (`mc/runtime.py:200`)
- [ ] Expose a reusable embedding buffer from `MegaContextTree` so the training loop can pass `inputs_embeds` into GPT and avoid re-embedding when experimenting with counterfactual working contexts. (`mc/mega_context.py:48`, `scripts/base_train.py:386`)
- [ ] The `mc_tree_type="disk"` and non-greedy allocator modes are advertised via CLI/config but currently raise `NotImplementedError` or silently alias to the greedy policy; document the roadmap or gate the flags so experiment scripts do not reference unsupported combinations. (`mc/mega_context.py:134`, `mc/focus_allocator.py:183`, `scripts/base_train.py:116`)
- [x] Randomized span sampling makes ablation diffs hard to reproduce. Threading a deterministic generator (or exposing a seed knob) around `_build_random_span_variant` would make MC behaviors traceable in sweeps. (`mc/runtime.py:226`)
- [x] Telemetry currently logs snapshots but does not expose structured counters for horizon triggers, sibling success rates, or allocator thrash. Surfacing those aggregates would make the upcoming dashboards actionable. (`mc/runtime.py:623`)

## Consistency, Comments, and Documentation Gaps
- [x] CLI defaults still mention `allocator_type="transformer"`/`"simple"`, but the factory always returns `GreedyFocusAllocator`; add a note in the config help or prune the unused options to avoid confusion during ablations. (`scripts/base_train.py:116`, `mc/focus_allocator.py:183`)
- [x] `WorkingContext.append`’s docstring never states that it only handles a single time-step (`[B, D]`), which hid the inference bug above. Updating the docstring (and adding an assert) would save future readers time. (`mc/working_context.py:81`)
- [x] The telemetry provider section in `scripts/base_train.py` assumes OTLP deps are present once `mc_enabled=1`, but the setup docs never call this out. A short note in the ops README about required `opentelemetry-sdk` wheels would prevent runtime import errors. (`scripts/base_train.py:186`)
- [x] Comments still describe MC as “Phase 1 instrumentation” but the code now drives auxiliary losses into training; aligning the README/ops docs with the current behavior would set expectations for anyone comparing vanilla nanochat baselines. (`mc/runtime.py:84`, `obsidian/ops/Training & Operations.md:47`)

## Recommended Pre-run Checks
- [x] Fix the critical tree construction/positional/controller bugs above, then add unit tests that cover short horizons, multi-sample batches, and inference appends. (See `tests/test_mc_components.py`.)
- [x] Extend CPU-only coverage to allocator expand/collapse paths, padding-aware pooling, and WorkingContext replace semantics so regressions are caught before GPU runs. (`tests/test_mc_components.py`)
- [ ] Benchmark MC on a single GPU with and without the LensNet caching to quantify the current overhead before launching wide sweeps.
- [ ] Update the Obsidian runbook with the clarified flag support and OTLP dependency note so tomorrow’s runs do not stall on missing packages.
