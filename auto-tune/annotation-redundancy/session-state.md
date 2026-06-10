# Auto-Tune Session State

## Goal
Tune NLP annotation (comment) decisions: suppress inline annotations that would bind a character to the same word it's already bound to in the scope chain. The previous session (nlp-first-annotations) was too eager — e.g., lesson 5 of mhall-svo-equivalence.md has `S(ubject) = M(ary)` where M(ary) redundantly re-binds M→"Mary" which was already bound in lesson 1. Add `is_bound_to(char, word)` to NLPSymbolTable, update binding resolver, fix curriculum, update spec.

## Done Criteria
- NLPSymbolTable has `is_bound_to(char, word)` method that checks if char is already bound to that word in the scope chain
- Binding resolver uses the check to skip redundant inline annotations (emit a dev-mode log warning instead)
- Lesson 5 of mhall-svo-equivalence.md has redundant annotations removed
- nlp-first-curriculum-annotations.md spec updated with NA-9: redundancy rule
- All existing tests pass
- Training run with mhall-svo-equivalence curriculum shows clean results

## Session
- **Name:** annotation-redundancy
- **Curriculum:** curricula/mhall-svo-equivalence.md
- **Branch:** auto-tune/annotation-redundancy
- **Worktree:** .worktrees/auto-tune/annotation-redundancy
- **Started:** 2026-06-10

## Current Phase
editing

## Next Action
Add is_bound_to() to NLPSymbolTable, update binding resolver, fix curriculum, run 2

## Run Log

### Run 1 — baseline ✅ (baseline)
- **Code changes:** baseline (no changes)
- **Observation:** Lessons 1-4 pass clean (13/13 satisfied). Lesson 5 compiles 18 entries but L > M triggers massive expansion noise (hundreds of S3 proposals at sig 200), causing low_confidence escalation then budget_exhaustion. Reactive scaffolding generates 3 rounds of LLM calls. Eventually completes 24/24 but with heavy noise.
- **Verdict:** baseline captured — redundant annotations in lesson 5 cause expansion noise

## Run 0 (not run)

## Patterns & Notes
- Issue: `S(ubject) = M(ary)` in lesson 5 re-binds M→"Mary" when M was already bound in lesson 1
- Similarly `V(erb) = H(ad)` re-binds H→"had", `O(bject) = A(ll)` re-binds A→"All"
- Node-side inline comments on the right of `=` operators are the main source of redundancy
- The binding resolver's `_resolve_primary` handles `pc.node_inline_comment` — needs the redundancy check there
- Also check sig-side: `S(ubject) = M(ary)` — S is bound in lesson 5 subscript scope, which is a NEW scope so may not be redundant. Need to check carefully.

## Files Modified
(none yet)
