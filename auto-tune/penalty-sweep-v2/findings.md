# KB-332 Findings — `UNRESOLVED_PENALTY` Training-Level Sweep v2 (Auto-Tune Session `penalty-sweep-v2`)

> **Denser-curriculum re-run (KB-341):** the penalty observable is corroborated
> against `cascade-pressure`'s ~113k-event L3–L5 band (5 distinct within-band
> positions per value vs the single position here); `UNRESOLVED_PENALTY = 10`
> confirmed. See `auto-tune/penalty-sweep-cascade/findings.md`.

**Date:** 2026-06-18 · **Session branch:** `auto-tune/penalty-sweep-v2` · **Task branch:** `kb/kb-332`
**Recommendation: CONFIRM `UNRESOLVED_PENALTY = 10`** — KB-309's probe-level confirmation is now
**corroborated at the training level** (this task's primary deliverable).

KB-309 swept `UNRESOLVED_PENALTY` across 5/10/20 against `curricula/first-steps-s2.md` and got
**byte-identical S1-only runs** for every value: the curriculum never exercised the S2 band
(lessons 2–4 deleted + the `=>` CANONIZED kline was canonical-by-construction → AGT-14 S1
short-circuit) and `events.jsonl` did not carry significance (a WebSocket serialisation gap).
KB-309 fell back to a model-level probe and confirmed `10` from the significance-gradient
geometry alone, with the explicit caveat that the training-level sweep was blocked. **Both
blockers are now fixed on `main`:** KB-320 (`2ecefa7`) restored the lessons and rewrote lesson 5
as a genuine `>` CONNOTED misfit; KB-319 (`4362c0d`) serialises `RationaliseEvent`/`KLine` at the
WebSocket boundary so `rationalise`/`ratify_request` frames (carrying the `{raw, normalised,
level}` significance object) reach `events.jsonl`. **This session re-runs KB-309's 10/5/20 sweep
against the fixed curriculum and delivers the training-level evidence KB-309 could never reach:
distinct, non-degenerate significance-level distributions per `UNRESOLVED_PENALTY` value.**

## Method

Auto-tune session `penalty-sweep-v2`, curriculum `curricula/first-steps-s2.md` (the KB-320 fix),
branched from `874c8f2` (KB-320 + KB-319 both present). Three training runs, values **10 (baseline),
5, 20** — baseline first, then the two extremes from KB-309's original sweep. Per run: harness +
supervisor started, curriculum stepped, snapshotted (pre/post), observed from
`events.jsonl` (the KB-319 primary observable) with `training.harness.log` as the KB-309 fallback.

**Ratification policy:** lesson 5's dual-misfit `{MH:[Alpha]}` escapes auto-countersign and emits
`ratify_request`s that, per KB-320, legitimately stall without a supervisor. All three runs were
driven with `{"action":"continue"}` only (leave-pending) so the significance observable is
captured identically and the terminal state is comparable. (The ratify policy was attempted first
but hit an out-of-scope crash — `agent.countersign()` receiving a `dict` where a `KLine` is
expected — surfaced as follow-up **KB-337**. The significance observable is policy-robust: the
continue-only baseline reproduced the identical significance burst as the ratify attempt, and
rationalise/ratify_request events fire during rationalisation, before/independent of ratification.)

## Evidence — the parameter is NOW exercised; distributions are DISTINCT per value

### Per-run significance distribution (the primary success criterion)

For each run: the count of `rationalise` + `ratify_request` events grouped by `significance.level`,
plus the within-band detail for the lesson-5 `{MH:[Halo]}` underfit (2 unresolved nodes →
terminal distance `2 × UNRESOLVED_PENALTY`):

| value | S1 | S2 | S3 | `{MH:[Halo]}` S2 normalised | S2 raw | distance | `ratify_request` | escalations | outcome |
|------:|---:|---:|---:|:----------------------------:|:------:|:--------:|:----------------:|:-----------:|:-------:|
| 5     | 19 | 36 | 7  | **0.95**                     | `0xff..f5` | **10** | 12 (all S2)   | 0           | stall (lesson 5) |
| 10    | 19 | 36 | 7  | **0.90**                     | `0xff..eb` | **20** | 12 (all S2)   | 0           | stall (lesson 5) |
| 20    | 19 | 36 | 7  | **0.80**                     | `0xff..d7` | **40** | 12 (all S2)   | 0           | stall (lesson 5) |

**Three DISTINCT within-band significance values (0.95 / 0.90 / 0.80) and raw distances
(10 / 20 / 40 = `2 × UNRESOLVED_PENALTY`)** for the lesson-5 S2 band — exactly KB-320's
direct-replay prediction (5→d=10, 10→d=20, 20→d=40), now confirmed at the **full training level**,
not just direct replay. KB-309's "byte-identical S1-only runs" blocker is resolved: where KB-309
saw **0 S2/S3 events per run**, this sweep sees **36 S2 + 7 S3 events per run**, and the S2 band's
within-band position scales correctly with the penalty.

**Why the level COUNTS are identical (and why that is correct):** this curriculum's lesson-5
misfit is a 2-unresolved-node underfit. For all three swept penalties, `2 × v` (10/20/40) stays
below `S2_S3_DISTANCE` (100), so the misfit lands in S2 for every value — the band POPULATION
(S1=19/S2=36/S3=7) is invariant, and the penalty moves the misfit's WITHIN-BAND position rather
than its band. This is the expected, correct behaviour: the S2→S3 band COUNT only differentiates
the values for *larger* misfits (more unresolved nodes), which is precisely what KB-309's
model-level probe measured (see cross-check below). The two observables are complementary, not
contradictory.

**Run outcome:** all three runs complete lessons 1–4 (`progress: lesson_complete` ×4) and stall at
lesson 5 — the documented no-supervisor stall (KB-320 §Step 3: `{MH:[Alpha]}` escapes
auto-countersign and requires ratification). This is the comparable terminal state; it is NOT a
crash (0 `AttributeError`s in the continue-only runs).

### Cross-check with KB-309's model-level probe — AGREES (complementary)

KB-309's probe (`auto-tune/penalty-sweep/probe-significance-output.txt`) measured S2 *capacity* —
how many unresolved-node pairs stay in S2 before spilling to S3:

| value | symmetric-pair S2-fit (n) | single-sided S2-fit (k) | read |
|------:|:-------------------------:|:------------------------:|:-----|
| 5     | 10 (never spills ≤10)     | 10                       | S2 far too broad — S3 nearly invisible |
| **10**| **5 (spills @ n=6)**      | **9 (spills @ k=10)**    | **balanced — matches KB-307 ≈5 resolution levels** |
| 20    | 2 (spills @ n=3)          | 4                        | S2 collapse risk toward S3 |

The training runs measure the *complementary* S2 *position* for a fixed (small) misfit: a 2-node
underfit lands at distance `2 × v` (10/20/40), all in S2, at distinct normalised values
(0.95/0.90/0.80). **No disagreement:** the probe's "balanced midpoint = 10" verdict is exactly
where the training misfit sits (d=20, normalised 0.90, mid-S2). v=5 crowds the 2-node misfit toward
the S2 top (0.95); v=20 pushes it toward S2 mid-lower (0.80) and (per the probe) narrows the S2
band for larger misfits. Both views describe the same healthy gradient; the training run supplies
the per-value within-band evidence the probe could never reach because the curriculum was broken.

## Decision

**CONFIRM `UNRESOLVED_PENALTY = 10`.** KB-309's probe-level decision is corroborated at the
training level.

1. **Primary success criterion MET (runs 1–3):** distinct significance distributions per value
   (S2 normalised 0.95/0.90/0.80; distances 10/20/40). KB-309's byte-identical S1-only blocker is
   resolved. **Even though the value recommendation is unchanged, the task succeeds** — the
   training-level corroboration is itself the deliverable.
2. **No value clearly beats 10:** all three produce a healthy, non-trivial S2 band (36 S2 events,
   distinct from the 7 S3) with 0 escalations; the only difference is the within-band position.
   v=10 places the 2-node misfit at mid-S2 (0.90), the balanced position consistent with KB-307's
   analytical "≈5 resolution levels" and KB-309's probe "balanced midpoint".
3. **Coherence with KB-310 (on main):** KB-310 decided to KEEP the penalty unified (not split the
   S2 "signifies" case). This sweep corroborates the unified penalty's VALUE at the training level.
4. **Decision rule:** ties → 10 (minimal churn).

**Value unchanged (10) ⇒ no `src`, cascade, or test edits required.** `src/kalvin/expand.py`
reverted to its canonical `UNRESOLVED_PENALTY = 10` (byte-identical to `main` `874c8f2`);
`specs/model.md` (§Constants, line ~602) and `plans/implement-kalvin.md` (line ~164) already read
`10`; the symbolic `TestS2Gradient` regression (`2 * n * UNRESOLVED_PENALTY`, `len(s2) >= 1`,
`"S3" in bands`, `UNRESOLVED_PENALTY < S2_S3_DISTANCE`) passes for 10.

## Verification

- **Apples-to-apples full suite** (kb/kb-332, `expand.py = 10`, no tokenizer data — identical
  config to the Step-0 baseline): **1189 passed, 403 skipped, 0 failures** — zero new failures.
- **Zero `src`/`tests` diff vs `main` `874c8f2`** in the experiment worktree (KB-332 touched no
  code; `expand.py` is byte-identical to main). `ruff check`/`ruff format --check`/`mypy` clean on
  `expand.py`.
- **Targeted regression guards green:** `TestS2Gradient` (the KB-307 gradient guard),
  `TestFirstStepsS2Routing` (the KB-320 curriculum-routing guard),
  `TestDomainObjectPayloadSerialisation` (the KB-319 wire guard).

## Out-of-scope findings surfaced during the runs

- **KB-337 — supervisor ratification path crashes.** Sending `{"action":"ratify"}` crashes the
  harness-bus thread: `adapter._handle_countersign` → `agent.countersign(kline)` receives a plain
  **dict** (from the supervisor's countersign WebSocket frame) where a **KLine** is expected →
  `AttributeError: 'dict' object has no attribute 'nodes'` (`agent.py:309`). The thread dies and
  the run stalls. `src/kalvin/agent.py` + `src/training/harness/adapter.py` are settled/out-of-scope,
  so KB-332 used the leave-pending policy (which captures the identical significance observable)
  and surfaced KB-337. KB-320's `test_emits_ratify_request_full_stack` only asserts a
  `ratify_request` fires — never that ratification completes — so this regression was uncaught.
- **KB-340 — pre-existing latent cogitator test failures** revealed when NLP tokenizer data is
  present (`tests/test_agent.py::TestCogitatorWithFakeHandler`). Unrelated to KB-332
  (`git diff 874c8f2 -- src tests` is empty); the whole `test_agent.py` module is
  `requires_nlp_data`-gated, so these are skipped in the standard no-data CI config.

## Evidence pointers

- Session directory (experiment branch `auto-tune/penalty-sweep-v2`):
  `.worktrees/auto-tune/penalty-sweep-v2/auto-tune/penalty-sweep-v2/`
- `session-state.md` — run log + decision record (the single source of truth for the session)
- `runs/001..007/` — per-run snapshots (pre/post for values 10, 5, 20)
- `events.jsonl` + `training.harness.log` (per run, in snapshots) — the per-event significance
  observable and the KB-309 fallback
- KB-309's prior report + probe: `auto-tune/penalty-sweep/findings.md`,
  `auto-tune/penalty-sweep/probe_significance.py`, `auto-tune/penalty-sweep/probe-significance-output.txt`

## Status

Experiment work lives on branch `auto-tune/penalty-sweep-v2` (the session) and `kb/kb-332` (this
report + the mirrored session-state). **Not merged to `main`** (auto-tune skill Rule 4 +
`AGENTS.md`). Propagation: **none required for the value** — `UNRESOLVED_PENALTY = 10` is already
canonical; the deliverable is the training-level corroboration, not a value change. KB-337
(ratification crash) and KB-340 (cogitator test failures) are separate, user-confirmed follow-ups.
