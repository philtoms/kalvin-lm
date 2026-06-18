# KB-309 Findings — `UNRESOLVED_PENALTY` Tuning (Auto-Tune Session `penalty-sweep`)

**Date:** 2026-06-18 · **Session branch:** `auto-tune/penalty-sweep` · **Task branch:** `kb/kb-309`
**Recommendation: CONFIRM `UNRESOLVED_PENALTY = 10`** (the KB-307 analytical default).

KB-307 decoupled the per-node significance penalty into `UNRESOLVED_PENALTY` and chose
`10` by analytical reasoning (≈5 resolution levels across the S2 distance band 1–100).
This session validated that value empirically. The conclusion: **no value in the sweep
clearly beats 10**, so 10 is confirmed. Minimal churn; the analytical reasoning holds up.

## Method

Auto-tune session `penalty-sweep`, curriculum `curricula/first-steps-s2.md`, branched from
`dbf1e8d` (KB-307). Two independent lines of evidence:

1. **Training runs** (auto-tune run loop) — values **10, 5, 20** (baseline + extremes).
   Per run: harness/supervisor started, curriculum stepped to completion, snapshotted,
   observed from `training.harness.log` + `events.jsonl`.
2. **Controlled significance probe** (`probe_significance.py`) — exercises the *real*
   `expand()`/`classify()` code path (the only place `UNRESOLVED_PENALTY` has an effect)
   across the full sweep **5, 8, 10, 12, 15, 20** on two unresolved-misfit scenarios.

## Evidence

### 1. Training runs — the parameter is NOT exercised by this curriculum

> **Training-level re-run (KB-332):** the parameter is now exercised. Against the fixed
> curriculum, the 5/10/20 sweep produces **distinct significance distributions per value**
> (S2 normalised 0.95/0.90/0.80; distances 10/20/40). See `auto-tune/penalty-sweep-v2/findings.md`.
> The "parameter unexercised" conclusion below is superseded at the training level; this v1 §1
> stands as the historical record of the blocked original sweep.

All three training runs (values 10, 5, 20) are **byte-identical**:

| value | lessons | escalations | outcome | significance of submitted entries |
|------:|:-------:|:-----------:|:-------:|:----------------------------------|
| 10    | 2/2     | 0           | complete | all **S1 fast-path / exact (d=0)** |
| 5     | 2/2     | 0           | complete | all **S1 fast-path / exact (d=0)** |
| 20    | 2/2     | 0           | complete | all **S1 fast-path / exact (d=0)** |

`first-steps-s2` is a *teaching* curriculum: the trainer tokenizes each lesson into
per-token frames that **exact-match** the model (e.g. `GROUND Mark → S1`, `FRAME H → 0.00 (d=0)`),
and the reactor **auto-countersigns** them. `UNRESOLVED_PENALTY` only applies to
*unresolved* mismatched nodes in `expand()`, so it is **never invoked**. Consequently no
training-time observable differentiates the values. (The dual-misfit `{MH:[H,A]}` does
**not** route S2 under this trainer because it is decomposed into S1 token-frames — see
Follow-ups.) **No training evidence favors any value.**

### 2. Controlled significance probe — v=10 is the balanced S2/S3 midpoint

S2 band = distance 1..100 (`S2_S3_DISTANCE = 100`). "S2 capacity" = how many unresolved
nodes/pairs stay in S2 before spilling into S3.

| value | sym-pair S2-fit (n, dist=2·n·v) | single-sided S2-fit (k, dist=k·v) | read |
|------:|:-------------------------------:|:---------------------------------:|:-----|
| 5     | 10 (never spills ≤10)           | 10 (never spills ≤10)             | S2 far too broad — S3 nearly invisible |
| 8     | 6 (spills @ n=7)                | 10                                | broad |
| **10**| **5 (spills @ n=6)**            | **9 (spills @ k=10)**             | **balanced — matches KB-307 ≈5 resolution levels** |
| 12    | 4 (spills @ n=5)                | 8                                 | moderate |
| 15    | 3 (spills @ n=4)                | 6                                 | narrow |
| 20    | 2 (spills @ n=3)                | 4                                 | S2 collapse risk toward S3 |

`v=10` sits at the balanced midpoint: a clearly-visible, non-trivial S2 band (≈5 symmetric
resolution levels) **and** S3 still reachable for moderate misfits. `v=5` makes S2 so broad
that S3 effectively vanishes for realistic misfits; `v=20` narrows S2 toward the original
collapse. Full machine output: `probe-significance-output.txt`.

## Decision

**CONFIRM `UNRESOLVED_PENALTY = 10`.**

1. Training-level: parameter unexercised by the teaching curriculum; runs identical across
   10/5/20 → **no training evidence for change**.
2. Model-level: `v=10` is the balanced S2/S3 midpoint matching KB-307's analytical
   reasoning; the gradient is healthy across the entire sweep; **no value clearly beats 10**.
3. Coherent with **KB-310** (now on main), which decided to KEEP the penalty unified (not
   split the S2 "signifies" case). KB-309 confirms the *value* of that unified penalty.
4. Decision rule: ties → 10 (minimal churn).

**Value unchanged → no `src`, cascade, or test edits required.** `expand.py`,
`specs/model.md`, `plans/implement-kalvin.md` already read `10`; the symbolic
`TestS2Gradient` regression (`2 * n * UNRESOLVED_PENALTY`, `len(s2) >= 1`, `"S3" in bands`,
`UNRESOLVED_PENALTY < S2_S3_DISTANCE`) passes for 10.

## Evidence pointers
- Session directory: `.worktrees/auto-tune/penalty-sweep/auto-tune/penalty-sweep/`
- `session-state.md` — run log + decision record
- `runs/001..006/` — per-run snapshots (pre/post for values 10, 5, 20)
- `probe_significance.py` + `probe-significance-output.txt` — reproducible model-level probe
- `training.harness.log` (per run, in snapshots) — per-entry significance

## Infrastructure fixes applied to the session branch (out of KB-309 scope — Rule 1)
The auto-tune `start-harness` was **completely broken on `main`**; fixing it was required to
run the session (auto-tune Rule 1: fix crashes, never work around):
1. `lifecycle.py` ~line 159: subprocess `-m harness` → `-m training.harness` (stale since
   refactor `a8c6737`; KB-311 `3b97148` renamed the config file but missed this; KB-312
   `ff933e3` fixed only docstrings/help-text — **still broken on main HEAD `ff933e3`**).
2. Merged `main` (KB-311 `3b97148`) into the session branch so config generation finds
   `training.harness.yaml`.
3. Symlinked gitignored tokenizer data into the fresh worktree.

## Follow-ups (out of scope — surfaced for separate tasks)
- **Fix `lifecycle.py` `-m harness` → `-m training.harness` on `main`** (auto-tune
  `start-harness` unusable; KB-311/KB-312 missed it).
- **`events.jsonl` records only `progress` events, not `rationalise`/`ratify_request`** —
  the trainer relays kagent `action="event"` frames on the bus but they do not reach the
  CLI supervisor's event log. Significance is observable via `training.harness.log` instead.
- **`first-steps-s2` lesson-5 S2-routing expectation does not fire** — the trainer
  decomposes the dual-misfit into S1 token-frames; the curriculum does not exercise the S2
  band as intended. A curriculum/trainer investigation (not a `UNRESOLVED_PENALTY` issue).

## Status
Work lives on branch `auto-tune/penalty-sweep` (experiment) and `kb/kb-309` (this report).
**Not merged to `main`** (auto-tune Rule 4 + AGENTS.md). Propagation: none required for the
value (10 is already canonical); the infra fixes need a separate, user-confirmed task.
