# Auto-Tune Session State

## Goal
Re-run the `UNRESOLVED_PENALTY` sweep (5/10/20) against the **KB-334-fixed**
`curricula/cascade-pressure.md` curriculum and deliver a **denser-curriculum
corroboration (or refutation) of `10`** than KB-332's first-steps-s2 sweep.
The primary success criterion is that the three values produce **DISTINCT,
RICHER significance distributions per value** (many within-band data points per
value, not KB-332's single one), resolving whether the denser curriculum
discriminates the values more sharply.

## Done Criteria
A `findings.md` demonstrating that the 5/10/20 sweep against cascade-pressure
produces distinct, non-degenerate, richer significance-level distributions per
`UNRESOLVED_PENALTY` value than KB-332's first-steps-s2 evidence — specifically:
(a) the S3-spillover event count **differs** across values (the discriminator
first-steps-s2 could not reach), and (b) the distinct `significance.normalised`
value sets differ and are far more numerous than KB-332's one-per-kline — backed
by the in-process probe's full L1–L5 evidence PLUS ≥1 harness training run, with
a recommendation for `UNRESOLVED_PENALTY` (confirm `10` or replace).

## Session
- **Name:** penalty-sweep-cascade
- **Curriculum:** curricula/cascade-pressure.md (KB-334 fix — CONNOTED `>` misfits)
- **Branch:** auto-tune/penalty-sweep-cascade
- **Worktree:** /Users/phil/dev/ai/kalvin/.worktrees/auto-tune/penalty-sweep-cascade
- **Base HEAD:** 7c67486 (main; contains KB-334 `ce0a81b` + KB-337 `7c67486`)
- **Started:** 2026-06-18
- **Task:** KB-341 (parent branch `kb/kb-341`)

## Current Phase
complete

## Next Action
surface recommendation to user (CONFIRM 10; the denser-corriculum corroboration is
the deliverable; not merged to main). Stale-snapshot follow-up task created.

## Experimental Design
- **Curriculum:** `curricula/cascade-pressure.md` — the dense S2-exercising
  curriculum (KB-334 fix; 5 lessons; L3/L4/L5 are single-node CONNOTED misfits).
- **Value sweep:** `[10, 5, 20]` — baseline `10` first (the KB-309/KB-307/KB-332
  default), then the two KB-309 extremes. Constraint `0 < v < S2_S3_DISTANCE`
  (100), enforced by `test_penalty_is_well_below_boundary`.
- **Two instruments (both part of the KB-309/KB-332 methodology):**
  - **(A) In-process `_replay_curriculum` probe — PRIMARY.** Calls `rationalise`
    directly + drains the cogitator per lesson, does NOT go through the
    reactor/ratification gate, so it captures the **FULL L1–L5 distribution**
    (all ~113k mid-band events at L5) per value, cheaply. KB-309
    `probe_significance.py` analogue, built on KB-334's generalised helper.
  - **(B) Harness session — CORROBORATING.** The KB-332 run loop; captures the
    training-level observable (`events.jsonl`) for the **L1–L3 portion** (it
    stalls at L3 under leave-pending — the first misfit lesson).

## Budget Analysis (the central design decision)
1. **Density:** KB-334's audit measured cascade-pressure emitting **113 486**
   candidate-bearing mid-band events across L3–L5 (L3=113, L4=7060, L5=113486)
   via `_replay_curriculum` (default `max_candidates=8`). A full harness training
   run that reached L5 would be far more expensive than KB-332's near-instant
   first-steps-s2 runs.
2. **The harness-stall reality:** cascade-pressure's L3–L5 misfit proposals
   (`{AB:[Charlie]}` etc.) are NEW klines that escape `Reactor._auto_countersign`,
   so they emit `ratify_request`s. Under the leave-pending policy the harness run
   stalls at the first un-resolved `ratify_request` — **L3** (the first misfit
   lesson). So a harness run captures only **L1–L3** significance events, never
   L4/L5. (NOTE: KB-337's ratification crash is now FIXED on this base
   `7c67486`, but the leave-pending policy is retained — it is the documented
   KB-332 policy, policy-robust for the significance observable, and the stall
   occurs because we choose not to ratify, not because ratify crashes.)
3. **The instrument split:** the in-process probe captures the full L1–L5
   distribution cheaply (seconds-to-minutes per value); the harness session
   corroborates the L1–L3 subset at the training level. The "expensive 113k-event
   L5 run" is reachable ONLY via the probe, never the harness — reconciled
   explicitly in the findings report.

## Observables Per Value
- **Significance distribution (primary):** count of mid-band
  (`0 < significance < D_MAX`, level ∈ {S2,S3}) events grouped by level AND the
  multiset of `significance.normalised` values. **The success criterion** — the
  three values must produce DISTINCT distributions, far richer than KB-332's
  one-per-kline (0.95/0.90/0.80).
- **S2→S3 spillover per value (the discriminator KB-332 could NOT reach):** for
  multi-node-density misfits, `total_distance ≈ 2 * n * UNRESOLVED_PENALTY`. At
  v=5, n up to ~10 stays in S2 (`2*10*5=100` borderline); at v=20, n≥3 spills to
  S3 (`2*3*20=120>100`). Tally S3 event count per value; expect it to DIFFER.
- **Per-event significance detail (probe):** distinct `significance.raw` /
  `significance.normalised` values + frequencies, per lesson (L1/L2 all-S1;
  L3/L4/L5 mid-band).
- **Learning quality (harness):** `escalation` count (`budget_exhaustion` after
  `max_reactive_rounds=5` — EXPECT on L3 misfits under leave-pending);
  `ratify_request` count; run outcome (L3 stall EXPECTED, not a crash).

## Ratification Policy
**Leave-pending:** `{"action":"continue"}` for every event, NEVER
`{"action":"ratify"}`. The spec's design forces this (KB-337 crash at writing
time); it is policy-robust for the significance observable (`rationalise` events
fire during rationalisation, before/independent of ratification). Applied
identically across all harness runs. KB-332 used the same policy and reproduced
the identical significance burst as the ratify attempt.

## Decision Rule
Pick the value that best satisfies (a) a visible, non-trivial, well-separated S2
band (S2 events present, distinct from S3; S3 not erased nor dominant), (b) the
curriculum's S2/S3 routing fires (the KB-334 regression already proves this), (c)
the probe's S2-capacity cross-check (KB-309's table) is consistent. If no value
clearly beats 10, **confirm 10** (KB-332's training-level + KB-309's probe-level
decisions are now corroborated against a denser curriculum — that denser
corroboration is itself the deliverable). Ties go to 10 (minimal churn).
**Crucially:** even if the value recommendation is unchanged, the task succeeds
as long as the three values produce DISTINCT, richer distributions than KB-332.

## Run Log

### Probe sweep (latest) — full L1–L5 via `_replay_curriculum` (PRIMARY evidence)
- **Instrument:** `probe_cascade_significance.py` (KB-309 analogue on the REAL dense
  curriculum; reuses KB-334's `_RoutingAdapter` + `CURRICULA_DIR`; mirrors the
  helper body for per-lesson slicing). Drift cross-check at v=10 → **MATCH** (113486).
- **Per-value aggregate (full L1–L5):** mid-band population INVARIANT across values
  (113486 candidate-bearing events; all S2; **S3=0 for every value**) — exactly
  KB-332's "penalty moves within-band POSITION not band POPULATION", now measurable
  across a 1000× denser stream. Per-lesson: L1/L2 all-S1; L3=113, L4=7060,
  L5=106313 (sum 113486, matching KB-334's headline; the spec's "L5=113486" was a
  mislabel — 113486 is the L3–L5 TOTAL).

| value | mid_band |    S2 |   S3 | distinct normalised positions |
|------:|---------:|------:|-----:|:------------------------------|
| 5     |   113486 | 113486 |   0  | 0.9950, 0.9750, 0.9700, 0.9500, 0.9250 |
| 10    |   113486 | 113486 |   0  | 0.9950, 0.9500, 0.9450, 0.9000, 0.8500 |
| 20    |   113486 | 113486 |   0  | 0.9950, 0.9000, 0.8950, 0.8000, 0.7000 |

- **Distance groups (why S3=0):** 5 distances per value scaling as
  `1` (matched-ungrounded, common, ×90495), `1·v` (×12365), `1·v+1` (×2244),
  `2·v` (×1036), `3·v` (×7346). Max distance = 15 / 30 / 60 for v=5/10/20 — ALL
  `< S2_S3_DISTANCE` (100) → **S3 never reached**. Cascade-pressure's KB-334
  single-node misfits contribute only 1–3 unresolved nodes per kline, so even v=20
  peaks at distance 60.
- **PRIMARY SUCCESS CRITERION — nuanced verdict:**
  - (b) distinct normalised value SETS differ + far more numerous than KB-332's
    one-per-kline → **MET**: 5 distinct within-band positions per value (vs
    KB-332's single 0.95/0.90/0.80); 4 of 5 differ across values (only the
    distance-1 0.9950 anchor is common). The penalty's within-S2 POSITION profile
    is far richer and clearly discriminates the values.
  - (a) S3-spillover count differs across values → **NOT MET**: S3=0 for all. The
    spec's predicted S3 discriminator (`2·n·v ≥ 100` at n≥3 for v=20) did NOT
    materialise — cascade-pressure's misfits are single-node (max ~3 unresolved
    nodes), so even v=20 stays in S2. This REFUTES the spec's specific S3
    prediction; it is itself a finding (the denser REAL curriculum differs from
    KB-309's synthetic symmetric-pair scenario that could reach high n).
- **Cross-check KB-309 probe:** CONSISTENT. KB-309's synthetic symmetric-pair S2
  capacity (v=5→10, v=10→5 spills@n=6, v=20→2 spills@n=3) describes the S2→S3
  boundary for LARGE n; cascade-pressure's REAL misfits sit at small n (1–3),
  comfortably in S2 for all values — exactly the regime KB-309's single-sided
  scenario B describes (k·v with k≤10). No disagreement.
- **Value read:** v=10 places positions at balanced mid-S2 (0.950/0.900/0.850);
  v=5 crowds high (toward S1); v=20 spreads toward the S2 floor (0.700). v=10 is
  the balanced midpoint — consistent with KB-309/KB-332. **Tentative decision:
  CONFIRM 10.**
- **Verdict:** distinct-from-baseline + richer-than-KB-332 (criterion b met);
  S3-spillover discriminator absent (criterion a refuted by the single-node
  misfit structure). No probe crash.

(no harness runs yet — Step 4 baseline run at 10 next)

### Harness sweep (3 runs, v=10/5/20) — training-level corroboration
- **Instrument:** auto-tune harness session (KB-332 run loop), leave-pending policy
  (`{"action":"continue"}` only). Driven via an in-process loop over the same
  `send_command`/`read_events` protocol (one ack per event; the dense L1–L3 stream
  is ~470 events/run). Snapshots runs/001–006 (before/after per value).
- **All 3 runs STRUCTURALLY IDENTICAL:** L1 + L2 `lesson_complete`, then **stall at
  L3** (the first misfit lesson) under leave-pending — exactly the budget-analysis
  prediction. Per run: 285 `rationalise`, 181 `ratify_request`, 0 `escalation`,
  1 `disconnected`. (0 `AttributeError`s — the leave-pending path is clean; KB-337's
  ratify crash was never triggered.)
- **Harness significance distribution is BYTE-IDENTICAL across all 3 values:**
  `{S1:77, S2:368, S3:21}`, distinct normalised `{0.0:21, 0.995:368, 1.0:77}`. The
  harness L1–L3 observable is **value-invariant** — dominated by value-independent
  distance-1 events (matched-but-ungrounded `+1`, normalised 0.995) and S1
  resolutions (identities/countersigns, 1.0); the 21 raw-0 events are proposal
  edges the reactor marks. **The value-discrimination lives entirely in the probe's
  L4/L5 evidence, which the harness cannot reach.**
- **Instrument reconciliation (HONEST):** the harness corroborates the
  **STRUCTURE** (L1/L2 complete, L3 stall, ratify_requests fire for L3's
  auto-countersign-escaping misfits). Its significance observable is a COARSER,
  reactor-mediated view than the probe: harness `ratify_request`s carry the
  proposal-resolution significance (distance-1, value-independent), NOT expand()'s
  per-candidate terminal distance the probe tallies. So the probe is the
  AUTHORITATIVE significance instrument (by design — the spec chose it primary
  because the harness stalls at L3). The harness run is the training-level proof
  the curriculum routes S2/S3 at the training level and that the stall is at L3.
- **Verdict:** corroborates-probe-structure; harness-significance value-invariant
  (expected — confirms the budget analysis that only the probe reaches L4/L5).

## Patterns & Notes
- **Environment:** worktree `data/tokenizer` is git-ignored (empty); set
  `KALVIN_DATA_DIR=/Users/phil/dev/ai/kalvin/data` for every probe/harness
  command (documented worktree approach in `src/kalvin/paths.py`; auto-tune
  `init` also points subprocesses there).
- **Base discrepancy vs spec:** spec assumed `main` at `c877890` (broken) +
  KB-337 in-progress. Empirically `main` is at `7c67486` and contains BOTH
  KB-334 (`ce0a81b`) and KB-337 (`7c67486`) — so KB-341 ran from a fully-fixed
  base. KB-334's commit hash is `ce0a81b` (not the spec's `668d55c` — rebased).
- **Pre-existing baseline failures (4, must NOT regress):**
  1. `test_agent.py::TestCogitatorWithFakeHandler::test_fake_handler_receives_s1` [KB-340]
  2. `test_agent.py::TestCogitatorWithFakeHandler::test_cogitator_stops_on_s1` [KB-340]
  3. `test_nlp_curriculum_compat.py::TestCurriculumLessonCountGuard::test_normalized_structure_unchanged[cascade-pressure.md]` [KB-334 stale `_EXPECTED_STRUCTURE` snapshot — KB-334 rewrote the curriculum but did not update the baseline]
  4. `test_nlp_curriculum_compat.py::TestCurriculumLessonCountGuard::test_normalized_structure_unchanged[conflict-drill.md]` [same KB-334 stale-snapshot class]
- **Module path:** use `python -m training.participants.auto_tune` (the skill's
  `participants.auto_tune` path is stale). Config is `training.harness.yaml`.

## Decision
**CONFIRM `UNRESOLVED_PENALTY = 10`.** KB-309's probe-level + KB-332's
first-steps-s2 training-level decisions are now corroborated against the ~1000×
denser cascade-pressure curriculum.

**Justification (Step-1 decision rule applied to the evidence):**
- **(a) Visible, well-separated S2 band:** all three values produce a rich S2 band
  (5 distinct within-band normalised positions, 113 486 mid-band events). v=10
  places them at the **balanced mid-S2** positions (0.950 / 0.900 / 0.850); v=5
  crowds the S2 top (0.975 / 0.970 / 0.950, too close to S1); v=20 spreads toward
  the S2 floor (0.900 / 0.800 / 0.700, KB-309's "S2-collapse-risk" regime for
  larger misfits). v=10 is the balanced midpoint.
- **(b) S2/S3 routing fires:** the KB-334 `test_curriculum_reaches_s2_s3_band`
  guard passes; the probe confirms 113 486 candidate-bearing mid-band events
  (vs first-steps-s2's 15). ✓
- **(c) Consistent with KB-309's S2-capacity table:** KB-309's symmetric-pair S2
  capacity (v=5→10, v=10→5 spills@n=6, v=20→2) describes the S2→S3 boundary for
  large n. Cascade-pressure's KB-334 single-node misfits sit at small n (1–3),
  comfortably in S2 for v=10 — the same regime KB-309's single-sided scenario-B
  (k·v) describes. No disagreement.
- **No value clearly beats 10; ties → 10 (minimal churn).**

**The primary deliverable is the denser corroboration, not a value change.** The
three values produce DISTINCT, RICHER distributions than KB-332 (5 distinct
within-band positions per value vs KB-332's single 0.95/0.90/0.80; the position
SETS differ across values), confirming the penalty moves within-S2 POSITION (now
measurable across 5 position-groups and 113k events) — a 1000× denser confirmation
of KB-332's mechanism. The spec's predicted S3-spillover discriminator did NOT
materialise (S3=0 for all values: cascade-pressure's single-node misfits peak at
distance 3·v = 60 for v=20, under the 100 boundary) — itself a finding about the
curriculum's misfit structure that distinguishes the REAL dense curriculum from
KB-309's synthetic high-n symmetric-pair scenario.

**Supporting rows:** probe `probe-output.txt` (per-value aggregate + distance
groups + drift cross-check MATCH); harness `runs/002,004,006` (L1–L3 value-invariant
significance, L3 stall).

## Files Modified
- `auto-tune/penalty-sweep-cascade/probe_cascade_significance.py` (new — in-process value-sweep probe; reuses KB-334's `_replay_curriculum` recording contract; sweeps `kalvin.expand.UNRESOLVED_PENALTY` at runtime)
- `auto-tune/penalty-sweep-cascade/probe-output.txt` (new — per-value tally + distance groups + drift cross-check)
- `auto-tune/penalty-sweep-cascade/_drive_run.py` (new — transient in-process harness-run driver; same `send_command`/`read_events` protocol)
- `auto-tune/penalty-sweep-cascade/runs/001..006/` (per-run snapshots: v=10/5/20 before+after)
- `src/kalvin/expand.py` — swept 10→5→20→10 across the harness runs; **restored to canonical `UNRESOLVED_PENALTY = 10`** (byte-identical to the base `7c67486`). No net change.
