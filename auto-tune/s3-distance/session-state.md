# Auto-Tune Session State

## Goal
Replace the quadratic `_pack()` distance function (`distance * distance`) in `expand.py` with a sub-quadratic or linear function that prevents S3 significance explosion while preserving the fine-grained ordering invariant: S1 > S2 > S3(1-hop) > S3(2-hop) > S3(3-hop) > ... > S4.

## Done Criteria
Significance values for S3 candidates at various hop depths remain well-separated and bounded, with no quadratic explosion. Observable via harness.log significance values across runs. The ordering invariant holds at every hop depth.

## Session
- **Name:** s3-distance
- **Curriculum:** curricula/conflict-drill.md (then cascade-pressure for comparison)
- **Branch:** auto-tune/s3-distance
- **Worktree:** .worktrees/auto-tune/s3-distance
- **Started:** 2026-06-07

## Current Phase
complete

## Next Action
Goal met. All changes verified. Consider merging to main.

## Run Log

### Run 5 (latest) — cascade-pressure with linear distance
- **Code changes:** same as run 3 (linear S3 distance + raw distance logging)
- **Observations:**
  - Lessons 1-2 clean: 10/10, 20/20 satisfied (identities and countersigns)
  - Lesson 3: `FRAME AB > C → 100` — distance visible (was `0.00` in quadratic)
  - LLM cogitation still needed (expected — no auto-countersign for S3 proposals)
  - Log format: `→ 100` raw distance instead of `→ 0.00` normalized
- **Verdict:** improved — distances visible, no regression

### Run 3 — conflict-drill with linear distance + raw distance logging
- Linear S3 distance verified: S3(1hop)=101, S3(20hop)=120 (vs quadratic: 121, 841)
- Raw distance in log: `FRAME AB > D C → 200` (was `→ 0.00`)
- Zero regressions vs baseline

- Run 2: discarded (port conflict, no useful data)
- Run 1: baseline — all S2/S3 significance values show as `0.00`

## Patterns & Notes

### Key Finding: Linear Distance Preserves Ordering
- S2(max=100) > S3(1hop=101) > S3(2hop=102) > ... > S3(20hop=120) > S4(0)
- Growth: +1 per hop (linear) vs +2n+1 per hop (quadratic)
- 20-hop S3: distance 120 (linear) vs 841 (quadratic) — 7x reduction

### Key Finding: Raw Distance Logging > Normalized
- Previous: `sig_norm = max(0.0, 1.0 - distance / 100)` → 0.00 for anything ≥100
- Now: raw distance → `→ 200` means 200 hops of distance, `→ 100` at S2|S3 boundary
- S1 fast-path unchanged: `→ S1 (fast path)`

### Test Results
- 0 regressions (same 40 pre-existing failures, all async/infrastructure)
- 3 test updates needed: `test_expand_connotation_bridging`, `test_s23_sits_between_max_hop_and_min_s3`, `test_s2_significance_log`

## Files Modified
- `src/kalvin/expand.py` — `_pack()` now linear (returns distance unchanged), `_S3_BIAS=1`, S3 connotation path uses `S2_S3_DISTANCE + s3_hop + _S3_BIAS - 1`
- `src/trainer/trainer.py` — raw distance logging, import simplified to just `D_MAX`
- `tests/test_expand.py` — updated 2 tests for linear distance expectations
- `tests/test_training_log.py` — updated `_S2_SIGNIFICANCE` to proper inverted value, assertion for raw distance
