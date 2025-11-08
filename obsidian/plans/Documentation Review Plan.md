---
tags:
  - plans
summary: Action plan for auditing and updating the MegaContext documentation set without executing the edits yet.
---
# Documentation Review Plan

> **Goal:** Provide another agent with a concrete punch list for bringing the documentation to a 10/10 state while we hold off on actually making the edits right now.

---

## Objectives
- Align environment/tooling guidance (README, `AGENTS.md`, `pyproject.toml`) with the versions that nanochat enforces so we can switch cleanly once the fork lands.
- Make it clear that the notebook is the *current* workflow and the nanochat migration is planned work so nobody runs non-existent commands.
- Treat the PRDs as the plan-of-record and mark the older Phase/POC docs as historical context.
- Preserve the curated pacing on `obsidian/index.md` while removing redundant text from the README and other intros so information still unfolds once the reader clicks deeper.
- Backfill coverage for mid/post-training operations and runtime workflows beyond the dated decode demo.
- Fix quality issues (typos, outdated bullets, broken links) before we start the nanochat import.

---

## Workstreams & Tasks

### 1. Environment & Tooling Consistency ✅
1. Cross-reference the nanochat repo to capture its Python/CUDA/PyTorch matrix, then mirror those versions in `README.md`, `AGENTS.md`, and `pyproject.toml` so we share a baseline before the code move.
2. If nanochat already targets Python 3.11, bump our formatter/test configs; otherwise, call out that we temporarily remain on 3.10 until the import.
3. Consolidate the CUDA + driver guidance from SETUP, README, and ops notes into one checklist that explicitly states “matches nanochat requirements.”

### 2. Runtime & Onboarding Path ✅
1. Make the notebook flow explicit as the active workflow today and push the `nanochat` steps into “Planned migration” callouts until implementation begins.
2. Update README, SETUP, `obsidian/ops/Training & Operations.md`, and `obsidian/ops/Base Runtime.md` so they only describe commands/configs that exist in this repo, with a separate section summarizing the upcoming nanochat plan plus links to the PRDs.
3. In `obsidian/ops/Nanochat Integration Guide.md`, label every command as pre-work (e.g., “To be enabled once nanochat fork is imported”) and reference the migration PRD for status.
4. Verify all surviving commands point to actual configs under `configs/`; add placeholder configs if needed so smoke instructions are runnable.

### 3. Architecture ↔ Code Sync (Legacy vs. Incoming) ✅
1. Annotate architectural notes with “legacy implementation under `src/megacontext/...`” vs. “to be rehomed in nanochat” so readers know the current code is transitional.
2. For components that already exist, link to the current module *and* explain that the file will be superseded during the nanochat import; for components not yet implemented, clearly mark them as design-only.
3. Capture any missing diagrams/assets referenced in the docs or temporarily remove the callouts until replacements are ready. *(diagram TODO still open)*

### 4. Roadmap & Phase Harmonization ✅
1. Treat the PRDs as the plan-of-record: add pointers from README, the landing page, and ops docs to `obsidian/plans/PRDs/index.md`.
2. Label the older Phase/POC docs (including the `_old` folder, Migration Plan, Research Paper Plan) as historical reference and summarize how they relate to the active PRDs.
3. Add a “Program taxonomy” snippet (e.g., inside `obsidian/plans/Plans.md`) that distinguishes PRDs (active) vs. Migration Plan (upcoming nanochat work) vs. legacy POC phases.

### 5. Content Deduplication & Navigation ✅
1. Leave `obsidian/index.md` as the narrative landing page with limited outbound links, per the current design, but trim duplicate “two-context” prose from the README and other intros so they simply point readers to the index for the story.
2. Restructure README to focus on repo logistics/onboarding and add a short “Start reading here → `obsidian/index.md`” pointer. ✅
3. Ensure deeper docs (e.g., `How MegaContext Works`) summarize components + link out to the canonical notes instead of re-teaching every concept. *(Reading map + component quick reference added; revisit after nanochat migration if new sections are added.)*

### 6. Coverage Gaps (Mid/Post-Training & Runtime Ops) 🔄
1. Add a “Lifecycle” page summarizing how MegaContext affects pre-training, mid-training refreshes (e.g., re-gisting after GistNet updates), post-training deployment, and inference telemetry. ✅
2. Expand `obsidian/ops/Base Runtime.md` (or a new “Runtime Ops” page) with concrete steps for running inference today: commands, expected artifacts, troubleshooting, and how to roll new checkpoints into the tree. *(need more troubleshooting + validation detail)*
3. Document how to validate migrations (smoke tests, parity checks) once the nanochat plan actually lands; for now, mark those sections as TODO so they don’t look complete.

### 7. Vision & Plans Cleanup 🔄
1. Fix typos/duplication in `obsidian/vision/Vision.md`. ✅
2. For each PRD, add a “Status: POR” label so it is obvious these govern the work, and add “Legacy” notes to the older phase documents. *(applied to some PRDs; need to finish the set + add badges to remaining ones, e.g., KV caching, PRD tracker).*
3. Cross-link future-looking pages (`Future Plan`, `Realtime Scenarios`) back to the PRD index so readers can see how speculative ideas relate to the active roadmap.

### 8. Documentation QA / Polish (New)
1. Run a vault-wide proofreading/link lint pass to catch typos, dead wikilinks, outdated TODO markers (e.g., missing diagrams such as `assets/module_stack.png`). ✅ (`tools/check_links.py` added + vault is currently clean; retain script for future runs.)
2. Verify every command/config referenced in docs exists (or is flagged as planned) and add troubleshooting snippets where appropriate. *(Ops/Lifecycle/Base Runtime enriched; remaining commands tied to nanochat will be updated post-migration.)*
3. Summarize recent changes or “last updated” metadata on key pages so readers can quickly gauge freshness. ❌ *Decision: skip to avoid unnecessary maintenance until the nanochat migration stabilizes the doc set.*

---

## Deliverables
- Updated README + Obsidian landing pages with consistent onboarding guidance.
- Revised architecture/component notes that reference the actual code base.
- A canonical roadmap index (or taxonomy) that clarifies the various phase systems.
- Runtime ops + lifecycle documentation covering training, mid-run maintenance, and inference.
- Cleaned-up vision/plan pages with corrected bullets and status labels.

---

## Suggested Sequence
1. **Week 1 (done):** Environment/tooling alignment + onboarding story (Workstreams 1–2).
2. **Week 2 (done):** Architecture annotations + phase taxonomy (Workstreams 3–4).
3. **Week 3 (in progress):** Deeper dedup/navigation audit plus lifecycle/runtime Ops expansion (Workstreams 5–6).
4. **Week 4 (pending):** Finish vision/plan cleanup, add POR badges everywhere, and perform the QA/polish sweep (Workstreams 7–8).

Update this plan as decisions land (e.g., once we officially migrate to nanochat or Python 3.11). For now, treat it as the authoritative to-do list for documentation remediation.
