# Auto-Tune Session State

## Goal
Make the rationaliser resolve undersign entries like `{M: S}` through graph expansion without relying on the LLM cogitator. Currently, when a knowledge node has an undersign (e.g., a Morpheme typed as Stem but not yet grounded), the rationaliser should expand the concept graph to find the missing definitions rather than escalating to the LLM.

## Done Criteria
Open session — observe how the rationaliser currently handles undersign entries, identify where graph expansion could replace LLM escalation, and iterate until rationaliser resolves `{M: S}` entries via graph traversal alone.

## Session
- **Name:** rationaliser-graph-expansion
- **Curriculum:** curricula/mhall-svo-single.md
- **Branch:** auto-tune/rationaliser-graph-expansion
- **Started:** 2026-06-06

## Current Phase
complete

## Next Action
session complete — ready for cascade documentation if desired

## Run Log

### Run 3 (latest)
- **Code changes:** (same as run 2)
- **Observation:** 18/18 satisfied, 0 LLM calls. Stable — same result as run 2.
- **Verdict:** confirmed stable

### Run 2
- **Code changes:** Added Phase 3b graph expansion and `_resolve_unknown_via_graph()` method.
- **Observation:** 18/18 satisfied, 0 LLM calls, 0 escalations. `A > D` resolved via fast path.
- **Verdict:** improved — goal met

### Run 1
- **Code changes:** baseline — 17/18 satisfied, LLM cogitator called for `A > D`
- **Verdict:** baseline

## Patterns & Notes
- `A > D` compiles as connotate (S3, `>` operator), not undersign (S1, `=` operator). D is undefined in the model.
- Phase 3b strategy: when sig participates in a grounded structure (e.g., A is in ALL's nodes, and ALL is S1), ground the unknown node by creating a frame for it, then accept the connotate as S1.
- The `is_s1` check on the containing kline is the key gate — it ensures we only ground unknowns when there's structural evidence.
- This works because ALL => A L L is canonical (S1) and A is in its nodes.

## Files Modified
- `src/kalvin/agent.py` — added Phase 3b graph expansion logic and `_resolve_unknown_via_graph()` method
