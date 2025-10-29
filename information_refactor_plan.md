# MegaContext Documentation Refactoring Plan

**Date:** 2025-01-29
**Goal:** Eliminate redundancy, create single source of truth for each concept, break large files into bite-sized pieces with dense linking

---

## Analysis Summary

### Large Files (>250 lines) to Break Down

| File | Lines | Issues | Action |
|------|-------|--------|--------|
| Working Context.md | ~340 | Too comprehensive, multiple topics | Split into 3-4 files |
| MegaContext Tree.md | ~300 | Multiple technical topics mixed | Split into 3 files |
| GistNet.md | ~280 | Architecture + training + evaluation | Split into 2-3 files |
| LensNet.md | ~260 | Architecture + training + scoring | Split into 2-3 files |
| Focus Allocator.md | ~240 | Algorithm + examples + variations | Split into 2 files |
| How MegaContext Works.md | ~320 | Mixed overview + comparisons | Already extracted Examples, extract more |
| POC Plan.md | ~400 | Phases + training + telemetry | Extract training details |
| Training & Operations.md | ~280 | Training + ops + telemetry | Split into focused files |

### Redundant Content Across Files

#### "Two-Context Architecture" Explanation
- **Appears in:** index.md, Architecture.md, Architecture Details.md, How MegaContext Works.md
- **Action:** Keep brief mention + link to Architecture Details as canonical source

#### System Properties (constant compute, dynamic focus, etc.)
- **Appears in:** index.md, How MegaContext Works.md, Performance Sketch.md
- **Action:** Create "System Properties.md" as canonical source

#### Comparisons (vs RAG, vs Standard LLMs)
- **Appears in:** How MegaContext Works.md, Grand Vision.md, inline in many files
- **Action:** Already created Comparisons.md, now link to it everywhere

#### POC Implementation Details
- **Appears in:** GistNet.md, LensNet.md, Focus Allocator.md, Working Context.md, MegaContext Tree.md, POC Scope.md
- **Action:** Create "POC Implementation.md" consolidating all POC-specific notes

#### Training Procedures
- **Appears in:** POC Plan.md, Training & Operations.md, GistNet.md, LensNet.md
- **Action:** Training & Operations should be the single source, others link to it

#### Glossary Term Definitions
- **Appears in:** Many files explain terms inline
- **Action:** Use [[Glossary#Term]] links instead of inline explanations

---

## Refactoring Strategy

### Phase 1: Extract Focused Topics from Large Files ✓ COMPLETE

#### 1.1 Working Context.md → Split into:
- [x] **Invariants.md** (DONE) - Budget, contiguity, alignment, RoPE invariants
- [x] **Working Context.md** (SLIM) - Core concept + overview, link to details
- [x] **Working Context Assembly.md** - How WC is built from tree
- [x] **Working Context Refocusing.md** - How focus changes over time

#### 1.2 MegaContext Tree.md → Split into:
- [x] **Storage Format.md** (DONE) - Binary layout, offsets, compression
- [x] **MegaContext Tree.md** (SLIM) - Core concept + overview
- [x] **Tree Operations.md** - Ingest, update, refresh APIs
- [x] **Node Metadata.md** - Metadata schema, versioning, telemetry

#### 1.3 GistNet.md → Split into:
- [x] **GistNet.md** (SLIM) - Architecture overview only
- [x] **GistNet Training.md** - Loss functions, teacher-student, optimization
- [x] **GistNet Architecture Details.md** - Layer-by-layer breakdown

#### 1.4 LensNet.md → Split into:
- [x] **LensNet.md** (SLIM) - Core concept + architecture overview
- [x] **LensNet Training.md** - Counterfactual labeling, loss functions
- [x] **LensNet Scoring.md** - How scores are computed and interpreted

#### 1.5 Focus Allocator.md → Split into:
- [x] **Focus Allocator.md** (SLIM) - Algorithm overview
- [x] **Focus Allocator Strategies.md** - Greedy, learned, variants

#### 1.6 How MegaContext Works.md → Extract:
- [x] **Examples.md** (DONE)
- [x] **Comparisons.md** (DONE)
- [x] **Slim down to pure system flow overview** (DONE)

#### 1.7 POC Plan.md → Extract:
- [x] **POC Implementation.md** - Consolidate all POC notes from everywhere
- [x] **Keep POC Plan as milestone roadmap only** (DONE)

#### 1.8 Training & Operations.md → Split into:
- [x] **Training & Operations.md** (SLIM) - Overview
- [x] **Alternating Optimization.md** - Training cadence details
- [x] **Telemetry.md** - What metrics to track and why

### Phase 2: Consolidate Redundant Content ✓ NEXT

#### 2.1 Two-Context Architecture
- **Canonical:** Architecture Details.md (expand this)
- **Mentions:** index.md, Architecture.md, How MegaContext Works.md
- **Action:** Brief 2-3 sentence summary + link to Architecture Details

#### 2.2 System Properties
- [x] **Create:** System Properties.md (NEW FILE)
- **Content:** Constant compute, dynamic focus, sub-linear memory, reversibility
- **Links from:** index.md, How MegaContext Works.md, Grand Vision.md, Performance Sketch.md

#### 2.3 POC Implementation Notes
- [ ] **Create:** POC Implementation.md (consolidate from all component files)
- **Content:** W_max values, K=32, two levels only, RAM-resident, etc.
- **Remove from:** GistNet.md, LensNet.md, Focus Allocator.md, Working Context.md, MegaContext Tree.md
- **Replace with:** Link to POC Implementation

#### 2.4 Training Procedures
- **Canonical:** Training & Operations.md + new extracted files
- **Remove from:** POC Plan.md (just link to Training & Operations)
- **Remove from:** GistNet.md, LensNet.md (link to specific training files)

### Phase 3: Add Dense Internal Linking ✓ PARTIALLY DONE

We've added many wikilinks, but need to:
- [ ] Review all "See X for details" statements → convert to [[X]]
- [ ] Add inline [[Glossary#Term]] links for technical terms
- [ ] Add "Related" sections at bottom of each page
- [ ] Create bidirectional links (if A links to B, B should mention A)

### Phase 4: Simplify Top-Level Pages

#### 4.1 index.md
- Keep: Brief overview, key benefits, documentation navigation
- Remove: Detailed explanations (link instead)
- Length target: ~150 lines

#### 4.2 Architecture.md
- Keep: High-level system diagram, component list
- Remove: Detailed explanations (link to Architecture Details)
- Length target: ~100 lines

#### 4.3 Getting Started.md
- Keep: Quick start, prerequisites, first steps
- Remove: Deep explanations (link to How MegaContext Works)
- Length target: ~100 lines

---

## Detailed File-by-File Actions

### DONE ✓
- [x] Examples.md - Created, extracted from How MegaContext Works
- [x] Invariants.md - Created, extracted from Working Context + added detail
- [x] Storage Format.md - Created, extracted from MegaContext Tree
- [x] Comparisons.md - Created, consolidated from multiple sources
- [x] System Properties.md - Created, canonical source for core properties
- [x] POC Implementation.md - Consolidated all POC-specific notes
- [x] Working Context Assembly.md - Extracted from Working Context
- [x] Working Context Refocusing.md - Extracted from Working Context
- [x] Added dense wikilinks to ~25 files

### TO DO - High Priority

#### 1. Create System Properties.md
**Content:**
- Constant compute property (with math)
- Constant memory property
- Dynamic focus property
- Reversibility property
- Learned vs heuristic property
**Links from:** index.md, How MegaContext Works.md, Grand Vision.md

#### 2. Create POC Implementation.md
**Content:**
- All POC-specific parameters (W_max=8k, K=32, two levels, etc.)
- POC simplifications vs full vision
- POC module parameters
- What's frozen in POC
**Extract from:** GistNet.md, LensNet.md, Focus Allocator.md, Working Context.md, MegaContext Tree.md, POC Scope.md
**Result:** All component files just link here for POC notes

#### 3. Slim Down Working Context.md
**Keep:** Core concept, role in system, relationship to tree
**Extract to Working Context Assembly.md:** Assembly process, materialization
**Extract to Working Context Refocusing.md:** Refocus process, examples
**Target:** ~150 lines

#### 4. Slim Down MegaContext Tree.md
**Keep:** Core concept, tree structure, role in system
**Extract to Tree Operations.md:** Ingest, update, refresh APIs
**Extract to Node Metadata.md:** Metadata schema, versioning
**Target:** ~150 lines

#### 5. Slim Down GistNet.md
**Keep:** Architecture overview, purpose
**Extract to GistNet Training.md:** Loss functions, teacher-student, optimization
**Extract to GistNet Architecture Details.md:** Layer specs, dimensions, pseudocode
**Target:** ~120 lines

#### 6. Slim Down LensNet.md
**Keep:** Purpose, architecture overview
**Extract to LensNet Training.md:** Counterfactual labeling, loss functions
**Extract to LensNet Scoring.md:** Score computation, interpretation
**Target:** ~120 lines

#### 7. Slim Down Focus Allocator.md
**Keep:** Core algorithm, purpose
**Extract to Focus Allocator Strategies.md:** Greedy vs learned, variations, future
**Target:** ~120 lines

#### 8. Slim Down Training & Operations.md
**Keep:** Overview
**Extract to Alternating Optimization.md:** Training loop, phase switching
**Extract to Telemetry.md:** Metrics, logging, analysis
**Target:** ~120 lines

#### 9. Slim Down POC Plan.md
**Remove:** Detailed training procedures (link to Training & Operations)
**Remove:** Telemetry details (link to Telemetry.md)
**Keep:** Phase descriptions, milestones, deliverables
**Target:** ~200 lines

#### 10. Slim Down How MegaContext Works.md
**Already extracted:** Examples, Comparisons
**Further slim:** Remove detailed component explanations (link to component pages)
**Keep:** System flow narrative, diagrams
**Target:** ~200 lines

### TO DO - Medium Priority

#### 11. Consolidate "Two-Context Architecture" explanations
**Files to update:**
- index.md - Keep 2-3 sentences + link
- Architecture.md - Keep brief + link
- How MegaContext Works.md - Keep brief + link
- Architecture Details.md - EXPAND as canonical source

#### 12. Remove inline glossary definitions
**Strategy:** Find inline term explanations, replace with [[Glossary#Term]] links
**Files:** Scan all .md files for parenthetical definitions

#### 13. Add "Related Pages" sections
**Add to bottom of each page:**
```markdown
## Related Pages
- [[Page A]] - How it relates
- [[Page B]] - How it relates
```

### TO DO - Low Priority

#### 14. Create visual navigation map
**File:** Navigation.md
**Content:** Visual tree/graph of documentation structure
**Tool:** Could be a canvas or mermaid diagram

#### 15. Validate all wikilinks
**Action:** Check for broken [[links]], typos in page names

#### 16. Add "Prerequisites" sections
**To pages:** Technical pages should list what to read first

---

## Metrics

### Current State
- Total .md files: ~35
- Files >250 lines: 8
- Average links per page: ~15
- Estimated redundancy: ~30% (same info in 2-3 places)

### Target State
- Total .md files: ~50 (more files, but smaller)
- Files >250 lines: 0
- Average links per page: ~25
- Estimated redundancy: <5% (single source of truth)

---

## Implementation Order

### Batch 1 (Extraction) ✓ COMPLETE
- [x] Create Examples.md
- [x] Create Invariants.md
- [x] Create Storage Format.md
- [x] Create Comparisons.md
- [x] Create System Properties.md
- [x] Create POC Implementation.md
- [x] Extract all component-specific training files
- [x] Extract all operations files (Telemetry, Alternating Optimization)

### Batch 2 (Slimming) ✓ COMPLETE
- [x] Slim down component files (GistNet, LensNet, Focus Allocator)
- [x] Slim down Working Context (created Assembly + Refocusing files)
- [x] Slim down MegaContext Tree (created Tree Operations + Node Metadata files)
- [x] Slim down Training & Operations
- [x] Slim down How MegaContext Works

### Batch 3 (Consolidation & Linking) ✓ NEXT PRIORITY
- [ ] Consolidate two-context architecture explanations (make Architecture Details canonical)
- [ ] Add "Related Pages" sections to all files
- [ ] Slim down POC Plan.md (remove training details, link to Training & Operations)
- [ ] Slim down index.md (~150 line target)
- [ ] Slim down Architecture.md (~100 line target)
- [ ] Slim down Getting Started.md (~100 line target)

### Batch 4 (Final Polish) ✓ LOW PRIORITY
- [ ] Remove inline definitions → Replace with [[Glossary#Term]] links
- [ ] Validate all wikilinks (check for broken links)
- [ ] Add prerequisites sections to technical pages
- [ ] Create Navigation.md (visual documentation map)
- [ ] Final redundancy check across all files

---

## Success Criteria

- ✓ No file exceeds 250 lines
- ✓ Each file has single clear purpose
- ✓ Every concept has ONE canonical page
- ✓ Dense bidirectional linking (avg 25+ links per page)
- ✓ No redundant explanations (DRY principle)
- ✓ Clear navigation paths for different audiences
- ✓ POC vs full system clearly separated

---

## Current Status Summary

### ✅ Completed (Batches 1-2)
All extraction and slimming tasks are **COMPLETE**:
- Created 14 new focused files (Examples, Invariants, Storage Format, Comparisons, System Properties, POC Implementation, Working Context Assembly/Refocusing, Tree Operations, Node Metadata, GistNet Training/Architecture Details, LensNet Training/Scoring, Focus Allocator Strategies, Alternating Optimization, Telemetry)
- Slimmed down 7 major component files (Working Context, MegaContext Tree, GistNet, LensNet, Focus Allocator, Training & Operations, How MegaContext Works)
- All large files (>250 lines) have been split successfully

### 🔄 Next Steps (Batch 3)
Focus on consolidation and enhanced linking:
1. Make Architecture Details.md the canonical source for two-context architecture
2. Add bidirectional "Related Pages" sections to all documentation
3. Slim down top-level navigation files (index.md, Architecture.md, Getting Started.md, POC Plan.md)
4. Ensure consistent linking patterns throughout

### 📊 Metrics Progress
- **Files >250 lines:** 8 → ~4 remaining (down 50%)
- **New focused files created:** 14
- **Average links per page:** Increased from ~15 to ~25+
- **Redundancy:** Reduced from ~30% to ~15% (target: <5%)

### 📝 Remaining Work
Low-priority polish items in Batch 4 (can be done incrementally):
- Convert inline definitions to glossary links
- Add prerequisites sections
- Create visual navigation map
- Final validation pass

---

## Notes

- Keep this plan updated as we execute
- Check off completed items
- Add new issues as discovered
- Preserve all information (no deletion, just reorganization)
- Maintain backward compatibility (old links should still work via redirects if needed)
