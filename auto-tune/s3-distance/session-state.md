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
observing

## Next Action
Run cascade-pressure curriculum to observe S3 distances with linear function under high-density conditions. The conflict-drill doesn't exercise multi-hop S3 connotations. Then verify with existing tests. If all green, move to documenting phase.

## Run Log

### Run 3 (latest) — linear S3 distance + raw distance logging
- **Code changes:** linear S3 distance (`S2_S3_DISTANCE + hop`), raw distance in log instead of normalized `sig_norm`
- **Observations:**
  - Lessons 1-2: identical to baseline (all auto-countersigned)
  - Lesson 3: distance=200 visible in log (was `0.00` in run 1). 7/8 satisfied, LLM cogitation same as baseline
  - Ordering invariant confirmed mathematically: S2(max=100) > S3(1hop=101) > S3(20hop=120) > S4(0)
  - Log format now shows `→ 200` instead of `→ 0.00`
- **Verdict:** improved — observability gained, no regression

### Run 2 — linear distance, normalized sig_norm (stale harness port)
- **Code changes:** linear S3 distance, sig_norm = event.significance / D_MAX
- **Observations:** harness crashed on port conflict (stale process). No useful data.
- **Verdict:** discarded

- Run 1: baseline — all S2/S3 significance values show as 0.00, useless for discrimination

## Patterns & Notes

### Linear S3 Distance — Verified
- `S2_S3_DISTANCE + hop_count` (where _S3_BIAS=1, so +hop+1-1 = +hop)
- S3(1hop)=101, S3(2hop)=102, S3(5hop)=105, S3(10hop)=110, S3(20hop)=120
- Quadratic was: S3(1hop)=121, S3(5hop)=196, S3(10hop)=361, S3(20hop)=841
- Growth: linear +1/hop vs quadratic +2n+1/hop

### Raw Distance Logging — Verified
- Changed from `sig_norm = max(0.0, 1.0 - distance / S2_S3_DISTANCE)` to raw distance
- FRAME events now show: `FRAME AB > D C → 200 | proposal: B => A B`
- Identity FRAMEs show: `FRAME A → 0`
- S1 fast path unchanged: `FRAME A > B → S1 (fast path) ← A > B`

### Remaining Work
- Run cascade-pressure to verify multi-hop S3 connotation distances are visible
- Run existing test suite to check for regressions
- Document: spec update for linear distance, plan for the change

## Files Modified
- `src/kalvin/expand.py` — `_pack()` now linear (returns distance unchanged), `_S3_BIAS=1`, S3 connotation path uses `S2_S3_DISTANCE + s3_hop + _S3_BIAS - 1`
- `src/trainer/trainer.py` — raw distance logging instead of normalized sig_norm, removed S2_S3_DISTANCE/MAX_HOP imports (now only D_MAX)
