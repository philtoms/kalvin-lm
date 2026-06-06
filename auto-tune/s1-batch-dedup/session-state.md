# Auto-Tune Session State

## Goal
Stop the cogitator from processing entries that already resolved as S1 during the same rationalise batch.

## Done Criteria
Open session — criteria will emerge from observations.

## Session
- **Name:** s1-batch-dedup
- **Curriculum:** curricula/mhall-svo-single.md
- **Branch:** auto-tune/s1-batch-dedup
- **Started:** 2026-06-06

## Current Phase
complete

## Next Action
session complete

## Run Log

### Run 2
- **Code changes:** deferred cogitator submission in Phase 5 + break-on-S1 in Cogitator._run_work_item
- **Observation:** Same behavior as run 1 — mhall-svo-single curriculum doesn't exercise the fix directly. Fix verified via targeted unit tests.
- **Verdict:** improved

### Run 1
- **Code changes:** baseline
- **Observation:** 18 entries. 17 S1 fast path. `A > D` cogitation with dual misfit. Reactive scaffolding submitted after session ended.
- **Verdict:** baseline

## Patterns & Notes
- Root cause: Phase 5 inline submission of S2/S3 work items before S1 scan completed.
- Fix 1 (Phase 5): Collect slow candidates, submit only if no S1 found.
- Fix 2 (Cogitator): Break after S1 in `_run_work_item`.
- mhall-svo-single doesn't exercise Phase 5 ordering. Curriculum with mixed candidates needed to observe fix in training runs.
- 2 new tests: `test_s2_before_s1_no_cogitator_submit` (AGT-38), `test_cogitator_stops_on_s1` (AGT-39).

## Files Modified
- `src/kalvin/agent.py` — Phase 5 deferred cogitator submission + Cogitator break-on-S1
- `tests/test_agent.py` — 2 new tests
- `specs/agent.md` — updated Phase 5 and Cogitator descriptions, added AGT-38/39
- `plans/impl/agent.md` — updated AGT-19, added AGT-38/39 with test mapping
