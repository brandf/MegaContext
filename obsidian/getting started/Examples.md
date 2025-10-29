---
tags:
  - getting-started
summary: Practical walkthroughs demonstrating MegaContext's dynamic focus in realistic scenarios like coding sessions with automatic detail level adjustment.
---
# MegaContext Examples

This page provides detailed walkthroughs of how MegaContext works in realistic scenarios. These examples show the [[Working Context]], [[MegaContext Tree]], [[LensNet]], and [[Focus Allocator]] working together to maintain relevant detail while compressing irrelevant context.

---

## Example 1: Long Coding Session

This example shows how MegaContext handles a typical software development conversation where attention shifts between different parts of a codebase.

### Initial State

```
User: "Show me the UserAuth class"

MegaContext Tree: [empty]
Working Context: [system prompt tokens at L0]
```

The system starts fresh with no history, just the initial system prompt in the [[Working Context]].

---

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

**What happened:**
- The entire codebase (100k tokens) was ingested into the [[MegaContext Tree]]
- [[GistNet]] automatically compressed it into a two-level hierarchy:
  - L1: 3,125 [[Glossary#Gist / Gist Embedding|gists]] (each representing 32 tokens)
  - L2: 97 gists (each representing 1,024 tokens)
- The [[Working Context]] assembles a mixed-resolution view:
  - **UserAuth.py** stays at L0 (full detail) since it's most relevant
  - **Related files** appear as L1 gists (medium compression)
  - **Distant code** appears as L2 gists (heavy compression)
- Total [[Glossary#W_max (Token Budget)|token budget]] stays under 8k ✓

---

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

**What happened:**
- The user asked a specific question about the login method and 2FA
- [[LensNet]] analyzed the [[Working Context]] and recent query to predict relevance:
  - **High positive score** (+0.8) for login method → needs detail
  - **Moderate positive score** (+0.2) for other auth methods → keep compressed
  - **Negative score** (-0.5) for unrelated files → compress further
- [[Focus Allocator]] applied the refocusing operations:
  - **Expanded** login method and 2FA helper from L1 to L0 (added 62 tokens)
  - **Collapsed** unrelated code from L0 to L1 (freed 62 tokens)
  - Net budget change: **0 tokens** (balanced)
- The model now sees login details at **full resolution** while keeping unrelated code compressed

---

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

**What happened:**
- The user shifted attention completely—from authentication to database schema
- [[LensNet]] detected the context shift:
  - **Negative score** (-0.6) for login code → no longer needed in detail
  - **High positive scores** (+0.9, +0.7) for database files → need detail
- [[Focus Allocator]] performed a major refocusing:
  - **Collapsed** 20 blocks of login code from L0→L1 (freed 620 tokens)
  - **Expanded** database files from L1→L0 and L2→L1→L0 (used ~600 tokens)
  - The [[Working Context]] now focuses entirely on database-related code

**The magic:** Login code didn't disappear—it's still in the [[MegaContext Tree]] at L0 if needed later. It's just compressed to L1 in the [[Working Context]]. If the conversation returns to authentication, [[LensNet]] can re-[[Glossary#Expand|expand]] it without losing any information.

---

### Key Insights from This Example

1. **Automatic relevance detection:** [[LensNet]] doesn't need explicit instructions—it learns to predict what will matter based on the query and recent context

2. **Budget-neutral refocusing:** Every [[Glossary#Expand|expansion]] is balanced by corresponding [[Glossary#Collapse|collapses]], keeping the [[Glossary#W_max (Token Budget)|token budget]] constant

3. **Reversible compression:** Content that was compressed can be re-expanded later without information loss (thanks to the [[MegaContext Tree]] storing everything at L0)

4. **Constant memory:** Total [[Working Context]] size stays around 2k tokens regardless of codebase size (100k tokens in this example)

5. **Learned focus policy:** The system adapts based on actual prediction quality ([[Glossary#ΔNLL@H (Perplexity Delta at Hidden Layer)|ΔNLL@H]]), not hand-crafted rules

---

## Future Examples

Additional walkthroughs to be added:

- **Long-form conversation:** Multi-turn dialogue over hours with topic shifts
- **Document analysis:** Reading and summarizing a 200-page technical manual
- **RAG comparison:** Same query handled by MegaContext vs. traditional RAG
- **Multi-modal:** Vision + language context with dynamic focus on image regions

See [[How MegaContext Works]] for conceptual overview and [[Architecture Details]] for technical specifications.
