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
Design and implement linear S3 distance. Proposed approach: S3 distance = `S2_S3_DISTANCE + hop_count` instead of `_pack(hop + _S3_BIAS)`. Must also fix the normalization formula in trainer.py so S3 values are visible in logs. Verify ordering invariant mathematically (note: S1 and S2(1-hop) share the same raw significance `D_MAX-1` — this is OK because S1 is detected by the fast path, not by significance). Then run 2 with the fix.

## Run Log

### Run 1 (latest) — baseline conflict-drill
- **Code changes:** none (baseline)
- **Observations:**
  - 4/4 lessons complete: 4/4, 6/6, 7/8, 11/11 satisfied
  - Budget exhaustion on lesson 3 (expected for conflict-drill)
  - **842 FRAME events at significance 0.00** — all S2/S3 invisible in log
  - Only 44 S1 fast-path events show non-zero significance
  - LLM calls: 5, log lines: 1094
- **Verdict:** baseline captured. Significance display completely broken for S2/S3.

## Patterns & Notes

### Critical Finding: Significance Display Broken
- `sig_norm = max(0.0, 1.0 - distance / S2_S3_DISTANCE)` with `S2_S3_DISTANCE=100`
- ALL S3 packed distances ≥ 100, so norm = 0.0 for every S3 event
- S2 events with distance > 100 also show 0.00
- 842/886 events = 95% show as 0.00 — log is useless for S2/S3 discrimination

### Critical Finding: S1 vs S2(1-hop) Overlap
- S1 significance = `D_MAX - 1` = `0xfffffffffffffffe`
- S2(1-hop) significance = `(~1) & MASK64` = `0xfffffffffffffffe` — IDENTICAL
- This is OK in practice because S1 is detected by the fast path in rationalise(), not by significance value
- The significance overlap doesn't cause misclassification

### Current Distance Mechanics (Detailed)
- `_pack(d) = d * d` (quadratic) — ONLY used for S3 connotation hops
- `_S3_BIAS = 9` — S3 hops biased by +9 before packing
- `S2_S3_DISTANCE = 100` — S2|S3 boundary threshold
- S2 direct hops are NOT packed (distance = raw hop count)
- S3 connotation hops are packed: `_pack(hop + _S3_BIAS)` = `(hop+9)²`

### Quadratic Growth Pattern
- S3(1-hop): packed=100, sig=0xffffff9b
- S3(5-hop): packed=196, sig=0xffffff3b
- S3(10-hop): packed=361, sig=0xfffffe96
- S3(20-hop): packed=841, sig=0xfffffcb6
- Explosion: each additional hop adds ~2x more distance than the previous

### Proposed Fix: Linear S3 Distance
- Replace `_pack(hop + _S3_BIAS)` with `S2_S3_DISTANCE + hop_count`
- S3(1hop) = 101, S3(2hop) = 102, S3(10hop) = 110, S3(20hop) = 120
- Ordering invariant verified: S2(max=100) > S3(1hop=101) > S3(2hop=102) > ... > S4
- Normalized values discriminable: S3(1h)=0.495, S3(10h)=0.45, S3(20h)=0.40
- Must also update `sig_norm` formula in trainer.py (use `full_scale = S2_S3_DISTANCE + MAX_HOP`)

### Files to Modify
- `src/kalvin/expand.py` — remove `_pack()` from S3 path, change to linear distance
- `src/trainer/trainer.py` — update `sig_norm` normalization denominator
- Possibly remove `_S3_BIAS` constant entirely (linear distance doesn't need it)

## Files Modified
(none yet — baseline run)
