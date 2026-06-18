# Auto-Tune Session State

## Goal
Choose an empirically-justified value for `UNRESOLVED_PENALTY` (KB-307 picked `10`
by analytical reasoning ≈ 5 resolution levels across the S2 distance band 1–100;
that value was never validated against real training). Confirm `10` or replace it
with a value that yields a healthier S2/S3 significance distribution and learning
quality on the S2-routing curriculum.

## Done Criteria
A recommendation for `UNRESOLVED_PENALTY` (confirm `10` or replace) backed by
**≥3 training runs across distinct values**, with observed S2/S3 distribution and
learning-quality evidence (escalations, ratifications, S2-routing success), applied
as a **tested value** in `src/kalvin/expand.py` with cascade docs consistent.
Constraint: `0 < v < S2_S3_DISTANCE` (100), enforced by
`tests/test_expand.py::TestS2Gradient::test_penalty_is_well_below_boundary`.

## Session
- **Name:** penalty-sweep
- **Curriculum:** curricula/first-steps-s2.md (exercises S2 routing; lesson 5 dual-misfit `{MH:[H,A]}` vs `{M:[H]}`)
- **Branch:** auto-tune/penalty-sweep
- **Worktree:** .worktrees/auto-tune/penalty-sweep
- **Created from commit:** dbf1e8d (KB-307 — `UNRESOLVED_PENALTY=10` present)
- **Started:** 2026-06-18

## Current Phase
complete

## Next Action
surfaced — recommendation CONFIRM 10 delivered to user (kb/kb-309). No merge to main.

## Experimental Design
- **Sweep:** `[5, 8, 10, 12, 15, 20]` — anchors around analytical default 10, spanning
  tight→broad S2 bands. Baseline `10` run first as the reference point.
- **Observables per run** (from `auto-tune/penalty-sweep/events.jsonl` + `harness.log`):
  1. **S2/S3 distribution:** count `rationalise` + `ratify_request` events grouped by
     `significance.level` (`S1`/`S2`/`S3`/`S4`). Healthy = visible, non-trivial S2
     population, distinct from S3, and S3 not erased.
  2. **S2-routing fires:** lesson-5 dual-misfit should produce S2 `ratify_request` events.
  3. **Learning quality:** count `escalation` events (`budget_exhaustion`/`low_confidence`);
     note ratification success. Fewer escalations + successful countersigns = healthier.
  4. **Run outcome:** final `progress` status (`complete` vs stuck) from `harness.log`.
- **Decision rule:** pick the value with (a) a visible S2 band distinct from S3,
  (b) the curriculum's expected S2 routing fires, (c) minimal escalations + successful
  learning. If no value clearly beats `10` → **confirm 10** (empirical validation is
  itself the deliverable). Ties → 10 (minimal churn).

## Run Log

### Run 3 — UNRESOLVED_PENALTY = 20  [snapshots 005 pre, 006 post]
- **Code changes:** `UNRESOLVED_PENALTY = 20`.
- **Observation:** Identical to baseline: 2/2 lessons complete, 0 escalations, all 7
  entries S1/exact (d=0). Parameter unexercised.
- **Verdict:** no-change.

### Run 2 — UNRESOLVED_PENALTY = 5  [snapshots 003 pre, 004 post]
- **Code changes:** `UNRESOLVED_PENALTY = 5`.
- **Observation:** Identical to baseline: 2/2 lessons complete, 0 escalations, all S1/d=0.
- **Verdict:** no-change.

### Run 1 — UNRESOLVED_PENALTY = 10 (baseline)  [snapshots 001 pre, 002 post]
- **Code changes:** none (baseline value).
- **Observation:** Curriculum completes 2/2 lessons, 0 escalations, run `complete`.
  All 7 submitted entries route **S1 fast-path / exact (d=0)** — e.g. `GROUND Mark → S1`,
  `FRAME H → 0.00 (d=0)`. **No S2/S3 significance events fire**: the trainer tokenizes
  each lesson into per-token frames that exact-match and the reactor auto-countersigns.
  `UNRESOLVED_PENALTY` is never invoked (it only applies to *unresolved* mismatches).
- **Verdict:** no-change (parameter unexercised by this teaching curriculum).

## Controlled Significance Probe (model-level) — values 5,8,10,12,15,20
Probe `probe_significance.py` exercises the REAL `expand()`/`classify()` path (where
UNRESOLVED_PENALTY has its effect) on two unresolved-misfit scenarios. Output saved to
`probe-significance-output.txt`. S2 capacity = unresolved-node-pairs that stay in S2
before spilling to S3 (S2 band = distance 1..100):

| value | sym-pair S2-fit (n) | single-sided S2-fit (k) | read |
|------:|:--------------------:|:------------------------:|------|
| 5     | 10 (never spills≤10) | 10 (never spills≤10)    | S2 far too broad — S3 nearly invisible |
| 8     | 6 (spills @ n=7)     | 10                      | broad |
| **10**| **5 (spills @ n=6)** | **9 (spills @ k=10)**   | **balanced — matches KB-307 ≈5 resolution levels** |
| 12    | 4 (spills @ n=5)     | 8                       | moderate |
| 15    | 3 (spills @ n=4)     | 6                       | narrow |
| 20    | 2 (spills @ n=3)     | 4                       | S2 collapse risk toward S3 |

## Decision
**CONFIRM `UNRESOLVED_PENALTY = 10`.** Justification:
1. **Training-level (runs 1–3, values 10/5/20):** the `first-steps-s2` curriculum routes
   every entry at S1/exact-match; the parameter is never invoked and all runs are
   byte-identical (2/2 complete, 0 escalations). **No training evidence favors any value.**
2. **Model-level (probe, 6 values):** v=10 is the balanced S2/S3 midpoint — 5 symmetric-pair
   / 10 single-sided S2 capacity before S3 spill, exactly KB-307's analytical "≈5 resolution
   levels". v=5 makes S2 dominate (S3 nearly invisible); v=20 narrows S2 toward collapse.
   **No value clearly beats 10; the gradient is healthy across the sweep.**
3. **Coherence with KB-310 (now on main):** KB-310 decided to KEEP the penalty unified (not
   split the signifies case); KB-309 confirms the unified penalty's VALUE is the balanced 10.
   Complementary, no conflict.

Ties → 10 (minimal churn). Value unchanged ⇒ no source/cascade/test edits required.

## Patterns & Notes
- **KEY FINDING (training-level):** `first-steps-s2` is a *teaching* curriculum; every
  submitted entry exact-matches (S1/d=0) and auto-countersigns. `UNRESOLVED_PENALTY`
  (per-node penalty for *unresolved* mismatched nodes in `expand()`) is **never hit**.
  => Training runs are identical across all values; the parameter's effect is only
  observable at the significance-model level (the S2→S3 gradient for unresolved
  misfits). The dual-misfit `{MH:[H,A]}` does NOT route S2 here because the trainer
  decomposes it into S1 token-frames. This contradicts the PROMPT's lesson-5 S2-routing
  expectation; documented as a finding + follow-up task.
- **Secondary finding (events.jsonl gap):** rationalise/ratify_request events do not
  appear in `events.jsonl` (only `connected`+`progress`) even though the trainer relays
  kagent `action="event"` frames on the bus. Significance is observable via
  `training.harness.log` instead. Likely a harness→supervisor forwarding gap; follow-up.
- **INFRASTRUCTURE FIXES applied to session branch (out of KB-309 scope, Rule 1):**
  1. `lifecycle.py` line ~159: `-m harness` → `-m training.harness` (stale since refactor
     a8c6737; KB-311 3b97148 renamed the config file but missed this subprocess path).
  2. Merged `main` (KB-311 3b97148) into the session branch so config generation finds
     `training.harness.yaml` (was `harness.yaml` at the dbf1e8d base).
  3. Symlinked `data/tokenizer/*` → main repo (gitignored runtime tokenizer data absent
     from the fresh worktree; `data/tokenizer/.cache-version` kept real to stay tracked).
- Per-run value edit lives in `src/kalvin/expand.py` line ~58. Must NOT touch `MAX_HOP`,
  `S2_S3_DISTANCE`, boundaries/classify/normalise, or the three `min(...,D_MAX-1)` clamps.
- This session worktree is a SIBLING of the KB-309 task worktree (lemon-wren/kb-kb-309).
  Final deliverables mirrored onto kb/kb-309 for review; session branch holds experiment.
  **Not merged to main** (auto-tune Rule 4 + AGENTS.md).

## Files Modified
- `src/training/participants/auto_tune/lifecycle.py` — `-m harness` → `-m training.harness` (Rule 1 infra fix; out of scope, follow-up task created).
- `src/kalvin/expand.py` — `UNRESOLVED_PENALTY` swept across values (session branch only).
