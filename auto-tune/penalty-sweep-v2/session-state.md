# Auto-Tune Session State

## Goal
Re-run KB-309's `UNRESOLVED_PENALTY` sweep (values 10/5/20) against the **fixed**
`curricula/first-steps-s2.md` curriculum to obtain the **training-level** evidence
KB-309 could never reach — distinct, non-degenerate significance-level distributions
per `UNRESOLVED_PENALTY` value — and issue a value recommendation (confirm `10` or
replace). KB-309's runs were byte-identical S1-only because the curriculum never
exercised the S2 band (root causes A: lessons 2–4 deleted; B: `=>` CANONIZED kline
canonical-by-construction → AGT-14 S1 short-circuit) and `events.jsonl` did not carry
significance (wire-serialisation gap). **Both blockers are now fixed on `main`**:
KB-320 (`2ecefa7`) restored the lessons + rewrote lesson 5 as a genuine `>` CONNOTED
misfit; KB-319 (`4362c0d`) serialises `RationaliseEvent`/`KLine` at the WebSocket
boundary so `rationalise`/`ratify_request` frames (carrying the `{raw, normalised,
level}` significance object) reach `events.jsonl`.

## Done Criteria
A `findings.md` report demonstrating that the 10/5/20 sweep against the fixed
curriculum produces **distinct, non-degenerate significance-level distributions** per
value (resolving KB-309's "byte-identical S1-only runs" finding), with a
recommendation for `UNRESOLVED_PENALTY` (confirm `10` or replace) backed by ≥3 training
runs. The primary success criterion is met as long as the three runs produce DISTINCT
significance distributions — **even if the value recommendation is unchanged (10)**,
because the training-level corroboration is itself the deliverable.

## Session
- **Name:** penalty-sweep-v2
- **Curriculum:** curricula/first-steps-s2.md (KB-320 fixed: 5 lessons; lesson 5 = `(Mark Halo Alpha)\nMH > H A` → genuine misfits `{MH:[Halo]}` (underfit, S2) + `{MH:[Alpha]}` (dual misfit, escapes auto-countersign → `ratify_request`))
- **Branch:** auto-tune/penalty-sweep-v2
- **Worktree:** .worktrees/auto-tune/penalty-sweep-v2
- **Created from commit:** 874c8f2 (KB-320 `2ecefa7` + KB-319 `4362c0d` both present)
- **Started:** 2026-06-18

## Current Phase
complete

## Next Action
complete — recommendation (CONFIRM 10) surfaced to user. Deliverables: findings.md (this session)
+ mirrored to kb/kb-332; KB-309 v1 findings cross-referenced. Out-of-scope follow-ups KB-337
(ratification crash) + KB-340 (cogitator test failures) created. Not merged to main (auto-tune
Rule 4 + AGENTS.md).

## Experimental Design
- **Sweep:** `[10, 5, 20]` — baseline `10` first (KB-309/KB-307 default = reference point),
  then the two extremes from KB-309's original sweep. (KB-309's model-level probe already
  covered the intermediate 8/12/15; this task's scope is the 5/10/20 training re-run that
  matches KB-309's original goal.) Constraint: `0 < v < S2_S3_DISTANCE` (100), enforced by
  `tests/test_expand.py::TestS2Gradient::test_penalty_is_well_below_boundary`.
- **Observables per run** (from `auto-tune/penalty-sweep-v2/events.jsonl` — now the PRIMARY
  source per KB-319 — with `training.harness.log` as the KB-309 fallback):
  1. **Significance distribution (PRIMARY):** count of `rationalise` + `ratify_request`
     events grouped by `significance.level` (`S1`/`S2`/`S3`/`S4`). **Success criterion:**
     the three runs must NOT be byte-identical — at least one run yields a different S2/S3
     event count or a different set of `significance.normalised` values than another.
  2. **Per-event significance detail:** the multiset of `significance.raw` /
     `significance.normalised` for the lesson-5 misfit events (`{MH:[Halo]}` and
     `{MH:[Alpha]}`). KB-320's validation predicts `{MH:[Halo]}` (2 unresolved nodes) →
     terminal distance `2 * UNRESOLVED_PENALTY` → d=10/20/40 for v=5/10/20, all in S2 but
     at distinct normalised values.
  3. **Learning quality:** count of `escalation` events; count of `ratify_request`; whether
     ratifications succeed.
  4. **Run outcome:** final `progress` status (`complete` vs stalled) from
     `events.jsonl`/`harness.log`.

## Ratification Policy
**Leave pending proposals (drive with `{"action":"continue"}` only — do NOT ratify)**,
applied **identically** across all three runs. Lesson 5's dual-misfit then reaches the
documented "no-supervisor stall" terminal state (KB-320-notes §Step 3: the `{MH:[Alpha]}`
misfit escapes auto-countersign and emits `ratify_request`s that legitimately stall without
ratification). **This is a switch from the Step-1 recommended "ratify" policy**, forced by
an out-of-scope crash discovered during the baseline ratify attempt — see KB-337 below.

**Why the switch is sound for the PRIMARY observable:** the Step-1 design (and KB-320)
note that `rationalise`/`ratify_request` significance events fire during rationalisation,
BEFORE/independent of ratification. **Empirically verified:** the continue-only baseline
run reproduced the IDENTICAL significance burst (50 rationalise + 12 ratify_request, same
levels/distances) as the ratify-policy attempt — confirming the observable is policy-robust.

**KB-337 (out-of-scope finding → follow-up task):** sending `{"action":"ratify"}` crashes
the harness-bus thread: `adapter._handle_countersign` → `agent.countersign(kline)` receives
a plain **dict** (from the supervisor's countersign WebSocket frame) where a **KLine** is
expected → `AttributeError: 'dict' object has no attribute 'nodes'` (`agent.py:309`). The
thread dies and the run stalls. `src/kalvin/agent.py` + `src/training/harness/adapter.py`
are settled/out-of-KB-332-scope, so KB-332 uses leave-pending and surfaced KB-337. The
significance observable is unaffected; only the (optional) drive-to-completion path is
blocked. The "run outcome" observable is therefore `stalled (lesson 5, no-supervisor stall,
per KB-320)` for all three runs — a consistent, comparable terminal state.

## Decision Rule
Pick the value that best satisfies (a) a visible, non-trivial S2 band (S2 events present,
distinct from S3, S3 not erased), (b) the curriculum's S2 routing fires (KB-320's
`TestFirstStepsS2Routing` already proves this at the unit level), (c) minimal escalations +
successful learning. If no value clearly beats `10`, **confirm `10`** — KB-309's
probe-level decision is then corroborated at the training level, and that corroboration is
itself the deliverable. Ties go to `10` (minimal churn). **Crucially:** even if the value
recommendation is unchanged, the task succeeds as long as the three runs produce DISTINCT
significance distributions (resolving KB-309's blocked goal).

## Run Log

### Run 3 — UNRESOLVED_PENALTY = 20 (continue-only)  [snapshots 006 pre, 007 post]
- **Code changes:** `UNRESOLVED_PENALTY = 20` (transient; canonical default 10).
- **Observation:** Same structure: lessons 1–4 complete, lesson 5 stalls at seq 68
  (no-supervisor stall, not a crash; 0 AttributeError). Significance levels: **S1=19,
  S2=36, S3=7** — identical LEVEL COUNTS to v=5/v=10 (2×20=40 < 100, stays in S2). The
  24 S2 `rationalise` + 12 `ratify_request` carry raw `0xffffffffffffffd7` = **distance 40
  = 2 × 20** → normalised **0.80**. KB-320's "20→0xff..d7 (d=40)" prediction confirmed. 0 escalations.
- **Verdict:** DISTINCT from v=5 and v=10. **PRIMARY SUCCESS CRITERION MET** across all 3 runs.

### Cross-run significance comparison (lesson-5 misfit band)
| value | S1 | S2 | S3 | S2 normalised | S2 raw | distance | ratify_request |
|------:|---:|---:|---:|:-------------:|:------:|:--------:|---------------:|
| 5     | 19 | 36 | 7  | **0.95**      | 0xff..f5 | **10** | 12 (all S2) |
| 10    | 19 | 36 | 7  | **0.90**      | 0xff..eb | **20** | 12 (all S2) |
| 20    | 19 | 36 | 7  | **0.80**      | 0xff..d7 | **40** | 12 (all S2) |

Three DISTINCT within-band significance values (0.95/0.90/0.80) and distances (10/20/40) —
exactly KB-320's direct-replay prediction (5→d=10, 10→d=20, 20→d=40), now confirmed at the
FULL TRAINING LEVEL. The level COUNTS are identical because all three 2×v distances stay in
S2 (<100): the penalty moves the misfit's WITHIN-BAND position, not its band COUNT — which
is the correct, expected behaviour for a 2-unresolved-node underfit. (KB-309 had 0 S2/S3
events per run — byte-identical S1-only; this run has 36 S2 + 7 S3 per run, distinct per value.)

### Run 2 — UNRESOLVED_PENALTY = 5 (continue-only)  [snapshots 004 pre, 005 post]
- **Code changes:** `UNRESOLVED_PENALTY = 5` (transient; canonical default 10).
- **Observation:** Same structure as baseline: lessons 1–4 complete, lesson 5 stalls at
  seq 68 (no-supervisor stall, not a crash; 0 AttributeError). Significance levels:
  **S1=19, S2=36, S3=7** — identical LEVEL COUNTS to v=10 (same 2-unresolved-node misfit
  structure; 2×5=10 < 100 so it stays in S2). BUT the within-band values DIFFER: the 24 S2
  `rationalise` + 12 `ratify_request` carry raw `0xfffffffffffffff5` = **distance 10 =
  2 × 5** → normalised **0.95** (vs v=10's 0.90 / d=20). KB-320's "5→0xff..f5 (d=10)"
  prediction confirmed exactly. 0 escalations.
- **Verdict:** DISTINCT from baseline (v=10): same level counts, different S2 normalised
  value (0.95 vs 0.90) and raw distance (10 vs 20). Primary success criterion emerging.

### Run 1 — UNRESOLVED_PENALTY = 10 (baseline, continue-only)  [snapshots 002 pre, 003 post]
- **Code changes:** none (canonical default value; the per-sweep value edit is skipped for
  the baseline reference point).
- **Observation:** Lessons 1–4 complete (`progress: lesson_complete` ×4); lesson 5 emits the
  significance burst then stalls at seq 68 (no-supervisor stall, per KB-320 — NOT a crash;
  the ratification-path crash KB-337 is avoided by the leave-pending policy). Significance
  levels (rationalise + ratify_request): **S1=19, S2=36, S3=7**. The 24 S2 `rationalise`
  events + all 12 `ratify_request` events carry raw `0xffffffffffffffeb` = **distance 20 =
  2 × UNRESOLVED_PENALTY** → normalised **0.90** — exactly KB-320's prediction for the
  `{MH:[Halo]}` 2-unresolved-node underfit. The 7 S3 events are raw `0x0` (identity frames).
  0 escalations. Harness log clean (only benign websocket-handshake-retry tracebacks +
  `Auto-countersign` INFO lines; no `AttributeError`).
- **Verdict:** DISTINCT from KB-309 (S2/S3 events present; parameter IS exercised). Baseline
  reference established. S2 normalised value 0.90 (d=20) to compare against v=5 and v=20.

## Decision
**CONFIRM `UNRESOLVED_PENALTY = 10`** — KB-309's probe-level confirmation is now
**corroborated at the training level** (the deliverable). Justification:

1. **Primary success criterion MET (runs 1–3, values 10/5/20):** the three runs produce
   DISTINCT significance distributions — S2 normalised values **0.95 / 0.90 / 0.80** and raw
   distances **10 / 20 / 40** (`2 × UNRESOLVED_PENALTY`) for the lesson-5 misfit band.
   KB-309's "byte-identical S1-only runs" blocker is resolved (KB-309 had 0 S2/S3 events;
   these runs have 36 S2 + 7 S3 per run, distinct per value). KB-320's direct-replay
   prediction (5→d=10, 10→d=20, 20→d=40) is confirmed at the full training level.
2. **Cross-check with KB-309's probe — AGREES (complementary, no disagreement):** the probe
   measured S2 *capacity* (how many unresolved-node pairs fit in S2 before spilling to S3):
   v=5 broad / v=10 balanced (5 sym-pairs) / v=20 narrow (2 sym-pairs). The training runs
   measure S2 *position* (where a given misfit lands): for this curriculum's specific
   2-unresolved-node misfit, all three `2×v` distances (10/20/40) stay in S2 (<100), so the
   band COUNTS are identical (S1=19/S2=36/S3=7) — the penalty moves the misfit's WITHIN-BAND
   position, not its band. Both views describe the same healthy gradient; the probe's
   "balanced midpoint" verdict (v=10) is exactly where the training misfit sits (d=20,
   norm 0.90, mid-S2).
3. **Decision rule applied:** no value clearly beats 10 — all three produce a healthy,
   non-trivial S2 band (36 S2 events, distinct from the 7 S3); 0 escalations across all
   runs; the difference is purely the within-band position. v=10 places the 2-node misfit at
   mid-S2 (0.90) — the balanced position, consistent with KB-307's analytical "≈5 resolution
   levels". v=5 crowds the misfit toward the S2 top (0.95); v=20 pushes it toward S2
   mid-lower (0.80) and (per the probe) risks S2-band narrowing for larger misfits. **Ties →
   10 (minimal churn).**

**Value unchanged (10) ⇒ no `src`, cascade, or test edits required.** `expand.py` swept value
must be REVERTED to 10 on the experiment branch. `specs/model.md`, `plans/implement-kalvin.md`,
`tests/test_expand.py` already read 10; the symbolic `TestS2Gradient` regression passes for 10.

## Patterns & Notes
- **Operational setup (NOT a code change):** the worktree lacks gitignored tokenizer data
  (`data/` is in `.gitignore`; only `.cache-version` is tracked). Symlinked
  `data/tokenizer/*` → main repo (keeping `.cache-version` real/tracked), mirroring KB-309.
  After symlinking, `NLPTokenizer.from_files()` loads and `TestFirstStepsS2Routing` passes
  (2 passed, not skipped) inside the worktree.
- **Both KB-320 + KB-319 fixes re-verified in the worktree** (init branches from HEAD
  `874c8f2`): `grep -c "^### " curricula/first-steps-s2.md` = 5;
  `_domain_json_default` present in `src/training/harness/protocol.py`.
- **Module path:** use `python -m training.participants.auto_tune` (the skill's documented
  `participants.auto_tune` path is stale — KB-309 documented this; do NOT "fix" the skill).
- **Config filename:** `training.harness.yaml` (renamed by KB-311/KB-316/KB-317).
- **AT_PYTHON:** absolute `/Users/phil/dev/ai/kalvin/.venv/bin/python` (the worktree has no
  `.venv`; the venv lives at the main repo root).
- **KB-309 failure mode to watch for:** zero S2/S3 events ⇒ a fix regressed; STOP and
  re-verify per the init preflight. Do NOT accept an S1-only run as a result.
- Per-run value edit lives in `src/kalvin/expand.py` (~line 73). Must NOT touch `MAX_HOP`,
  `S2_S3_DISTANCE`, boundaries/classify/normalise, the three `min(...,D_MAX-1)` clamps, or
  the KB-310 penalty-unification rationale comment.

## Files Modified
- `src/kalvin/expand.py` — `UNRESOLVED_PENALTY` swept across 5 / 20 then REVERTED to canonical 10
  (byte-identical to `main` `874c8f2`); no net change. Cascade value sites (`specs/model.md`,
  `plans/implement-kalvin.md`) unchanged — they already read 10.
- `auto-tune/penalty-sweep-v2/session-state.md` — run log + decision record (this session).
- `auto-tune/penalty-sweep-v2/findings.md` — empirical findings report (primary deliverable).
- `auto-tune/penalty-sweep/findings.md` — one-line cross-reference to this v2 report (on kb/kb-332).
- (Out-of-scope, surfaced as follow-ups, NOT edited here) KB-337: `src/kalvin/agent.py` +
  `src/training/harness/adapter.py` ratification crash; KB-340: `tests/test_agent.py` latent failures.
