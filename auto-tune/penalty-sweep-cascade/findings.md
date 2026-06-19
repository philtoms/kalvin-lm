# KB-341 Findings — `UNRESOLVED_PENALTY` Denser-Curriculum Sweep (cascade-pressure)

**Date:** 2026-06-18 · **Session branch:** `auto-tune/penalty-sweep-cascade` · **Task branch:** `kb/kb-341`
**Recommendation: CONFIRM `UNRESOLVED_PENALTY = 10`.** KB-309's probe-level and
KB-332's first-steps-s2 training-level confirmations are now corroborated against
cascade-pressure — a curriculum **~1000× denser** than first-steps-s2 (113 486
candidate-bearing mid-band events across L3–L5 vs first-steps-s2's 15). The denser
curriculum yields **5 distinct within-band significance positions per value** (vs
KB-332's single 0.95/0.90/0.80), and the position **sets differ across values** —
a far richer, sharper discrimination of the penalty than first-steps-s2 could
reach.

## Mission recap

KB-332 re-ran KB-309's `UNRESOLVED_PENALTY` sweep (5/10/20) against the fixed
`first-steps-s2.md` and delivered training-level evidence KB-309 could never reach
— but first-steps-s2 is **thin**: it emits only ~15 candidate-bearing mid-band
events (lesson 5 only), so the within-band evidence was a single per-value data
point per kline. KB-334 (`ce0a81b`) rewrote `cascade-pressure.md` L3–L5 from the
broken `=>` CANONIZED operator to `>` CONNOTED single-node misfits, and its audit
measured the curriculum emitting **113 486 candidate-bearing mid-band events
across L3–L5**. **This session sweeps 5/10/20 against that dense curriculum and
delivers the denser-curriculum corroboration (or refutation) of `10`.**

## Method

Two complementary instruments (both part of the KB-309/KB-332 methodology):

- **(A) In-process `_replay_curriculum` probe — PRIMARY.** `probe_cascade_significance.py`
  reuses KB-334's generalised replay (the `_RoutingAdapter` recorder + `CURRICULA_DIR`),
  mirrors the helper's body for per-lesson slicing, and sweeps the value by setting
  `kalvin.expand.UNRESOLVED_PENALTY` at runtime (the module global is read at
  call-time by `expand()` — **no source edit**). It calls `rationalise` directly and
  drains the cogitator per lesson, bypassing the reactor/ratification gate, so it
  captures the **FULL L1–L5 distribution** (all 113 486 mid-band events) per value in
  ~3 s. A drift cross-check at v=10 confirms the inline loop matches the imported
  helper's aggregate exactly (113 486 = 113 486 → MATCH).
- **(B) Harness session — CORROBORATING.** The KB-332 run loop against the dense
  curriculum, driven with the leave-pending policy (`{"action":"continue"}` only).

**Ratification policy:** leave-pending (never `ratify`). cascade-pressure's L3–L5
misfit proposals escape `Reactor._auto_countersign` and emit `ratify_request`s; under
leave-pending the harness run **stalls at L3** (the first misfit lesson), so it
captures only the **L1–L3 portion**. (The spec assumed KB-337's ratification crash
forced this; KB-337 is in fact fixed on this base `7c67486`, but leave-pending is the
documented KB-332 policy and is policy-robust for the significance observable — the
stall occurs because we choose not to ratify, not because ratify crashes. The
leave-pending runs had 0 `AttributeError`s.)

## Evidence — the parameter is exercised on a 1000× denser stream; distributions are DISTINCT and RICHER per value

### Per-value probe aggregate (the primary success criterion) — FULL L1–L5

| value | total | mid_band |    S2 |   S3 | distinct within-band positions | top normalised (count) |
|------:|------:|---------:|------:|-----:|:-------------------------------|:-----------------------|
| 5     | 113671 | 113486 | 113486 | 0 | 0.9950, 0.9750, 0.9700, 0.9500, 0.9250 | 0.9950×90495, 0.9750×12365, 0.9250×7346, 0.9700×2244, 0.9500×1036 |
| 10    | 113671 | 113486 | 113486 | 0 | 0.9950, 0.9500, 0.9450, 0.9000, 0.8500 | 0.9950×90495, 0.9500×12365, 0.8500×7346, 0.9450×2244, 0.9000×1036 |
| 20    | 113671 | 113486 | 113486 | 0 | 0.9950, 0.9000, 0.8950, 0.8000, 0.7000 | 0.9950×90495, 0.9000×12365, 0.7000×7346, 0.8950×2244, 0.8000×1036 |

**Three DISTINCT within-band normalised value SETS** (4 of 5 positions differ per
value; only the distance-1 `0.9950` anchor is common). KB-332's first-steps-s2 had
**ONE** within-band normalised value per value (0.95/0.90/0.80, for a single 2-node
misfit across ~15 events); cascade-pressure yields **5 distinct within-band
positions** across **113 486 events** — a 1000× denser, 5× richer confirmation.

**Why the band POPULATION is invariant (and why that is correct):** the mid-band
event count (113 486) and S3 count (0) are identical for every value, because the
penalty moves the misfits' **within-S2 POSITION**, not their **band POPULATION**.
This is the same mechanism KB-332 identified for first-steps-s2 — now measurable
across 5 position-groups and a 1000× denser stream. Per-lesson breakdown:
L1/L2 are all-S1 (identities + countersigns); L3=113, L4=7060, L5=106313 mid-band
events (sum 113 486, matching KB-334's headline — the spec's "L5=113486" was a
mislabel; 113 486 is the L3–L5 total).

### The within-S2 position groups scale with the penalty (why S3=0)

| value | distance groups (count) | max distance |
|------:|:------------------------|-------------:|
| 5     | 1×90495, **5**×12365, 6×2244, **10**×1036, **15**×7346 | 15 |
| 10    | 1×90495, **10**×12365, 11×2244, **20**×1036, **30**×7346 | 30 |
| 20    | 1×90495, **20**×12365, 21×2244, **40**×1036, **60**×7346 | 60 |

The four penalty-dependent groups scale as `1·v`, `1·v+1`, `2·v`, `3·v` (the `1`
group is the value-independent matched-but-ungrounded `+1`). The **max distance** is
15 / 30 / 60 for v=5/10/20 — **all `< S2_S3_DISTANCE` (100), so S3=0 for every
value.**

### The S3-spillover discriminator did NOT materialise — a finding

The spec predicted the S3-spillover count would **differ** across values (v=5 ≈ 0,
v=10 some, v=20 more, because `2·n·v ≥ 100` crosses the boundary at different n).
**Empirically S3 = 0 for all three values.** Reason: cascade-pressure's KB-334
misfits are **single-node** (decomposed per rule 47), so each kline contributes only
**1–3 unresolved nodes** to the terminal distance. Even at v=20 the largest group is
`3·v = 60 < 100`. The spec's `2·n·v` formula came from KB-309's *synthetic*
symmetric-disjoint-pair scenario (which can reach high n); the **REAL** dense
curriculum's misfits sit at small n. **This refutes the spec's specific S3
prediction; it is itself a finding** — the denser real curriculum differs from the
synthetic probe in exactly the dimension (n) that governs S3 spillover.

The broader goal is nonetheless met: the distributions ARE distinct and richer than
KB-332 (criterion b), via **within-S2 multi-position** discrimination rather than
S3 spillover.

## Instrument reconciliation (honest)

The harness run corroborates the **STRUCTURE**, not the probe's significance
distribution:

- **Harness (3 runs, v=10/5/20):** all structurally identical — L1+L2
  `lesson_complete`, then **stall at L3** under leave-pending; per run 285
  `rationalise`, 181 `ratify_request`, 0 `escalation`, 0 `AttributeError`. The
  harness **cannot reach L4/L5** (the value-discriminating lessons) — confirming the
  budget analysis.
- **Harness significance is BYTE-IDENTICAL across all three values:**
  `{S1:77, S2:368, S3:21}`, distinct normalised `{0.0:21, 0.995:368, 1.0:77}`. The
  harness L1–L3 observable is **value-invariant** — dominated by value-independent
  distance-1 events (matched-but-ungrounded, 0.995) and S1 resolutions. The
  value-discrimination lives **entirely in the probe's L4/L5 evidence**.
- **Why the harness ≠ the probe at the significance level:** the probe tallies
  `expand()`'s per-candidate terminal distance (the faithful `UNRESOLVED_PENALTY`
  effect). The harness significance is a **coarser, reactor-mediated** view: harness
  `ratify_request`s carry the proposal-resolution significance (distance-1,
  value-independent), and `KAgentAdapter` filters events through its sender-map. So
  the **probe is the authoritative significance instrument** (chosen primary by
  design); the harness is the training-level proof the curriculum routes S2/S3 at
  the training level and that the stall is at L3.
- **Budget reconciliation:** the "expensive 113 486-event L5 run" is reachable **only
  via the in-process probe** (~3 s), never the harness (which stalls at L3). Each
  harness run is bounded by L1–L3 (~470 events) and is affordable.

## Cross-check with KB-309 + KB-332 — CONSISTENT

- **KB-309 probe** (`auto-tune/penalty-sweep/probe-significance-output.txt`): the
  symmetric-pair S2-capacity table (v=5→10, v=10→5 spills@n=6, v=20→2) describes the
  S2→S3 boundary for LARGE n. Cascade-pressure's real single-node misfits sit at
  small n (1–3), comfortably in S2 for all values — exactly KB-309's single-sided
  scenario-B regime (k·v). **No disagreement**; the two describe the same healthy
  gradient at different n.
- **KB-332** (`auto-tune/penalty-sweep-v2/findings.md`): cascade-pressure's
  within-S2 positions agree with first-steps-s2's for the equivalent misfit sizes —
  v=10 sits at the balanced mid-S2 (KB-332's 0.90 for the 2-node misfit; here
  0.950/0.900/0.850 across the position groups). The denser evidence **extends** the
  first-steps-s2 corroboration; it does not contradict it.

## Decision

**CONFIRM `UNRESOLVED_PENALTY = 10`.**

1. **Primary success criterion MET (richer, distinct distributions):** 5 distinct
   within-band positions per value (vs KB-332's 1), position sets differing across
   values, across a 1000× denser stream. The denser-corriculum corroboration is
   itself the deliverable.
2. **No value clearly beats 10:** v=10 places the positions at balanced mid-S2
   (0.950/0.900/0.850); v=5 crowds the S2 top (toward S1); v=20 spreads toward the
   S2 floor (KB-309's S2-collapse-risk regime). v=10 is the balanced midpoint,
   consistent with KB-307's "≈5 resolution levels" and KB-309's "balanced midpoint".
3. **Coherence with KB-310:** the unified penalty's VALUE is corroborated against
   the denser curriculum.
4. **Decision rule:** ties → 10 (minimal churn). The S3-spillover discriminator the
   spec hoped for did not materialise, but it does not indicate a better value — it
   indicates the curriculum's misfits stay in S2 for all three values, and v=10 is
   the balanced within-S2 position.

**Value unchanged (10) ⇒ no `src`, cascade, or test edits required.**
`src/kalvin/expand.py` is byte-identical to the base `7c67486`
(`UNRESOLVED_PENALTY = 10`, KB-310 rationale comment intact); `specs/model.md` and
`plans/implement-kalvin.md` already read `10`; the symbolic `TestS2Gradient`
regression passes for 10.

## Verification

- **Targeted guards green:** `TestS2Gradient` (KB-307 gradient guard —
  `0 < UNRESOLVED_PENALTY < S2_S3_DISTANCE`), `TestCurriculumMisfitRouting` (KB-334
  routing guard — cascade-pressure reaches S2/S3), `TestDomainObjectPayloadSerialisation`
  (KB-319 wire guard).
- **Full suite:** 4 failed / 1615 passed — **identical to the Step-0 baseline, zero
  new failures.** The 4 are pre-existing: 2 × `test_agent.py::TestCogitatorWithFakeHandler`
  [KB-340] and 2 × `test_nlp_curriculum_compat.py::TestCurriculumLessonCountGuard::test_normalized_structure_unchanged`
  for `cascade-pressure.md` + `conflict-drill.md` [KB-334 stale `_EXPECTED_STRUCTURE`
  snapshot — KB-334 rewrote both curricula but did not update the baselines; surfaced
  as a follow-up].
- **No source diff vs base** (`git diff 7c67486 -- src/kalvin/expand.py` empty);
  `docs/kalvin-vision.md`, `curricula/cascade-pressure.md`, and
  `tests/test_nlp_curriculum_compat.py` byte-for-byte unchanged.

## Out-of-scope findings surfaced

- **KB-334 stale structure-snapshot baseline** (follow-up task created): KB-334
  rewrote `cascade-pressure.md` (`=>`→`>`, L5 nodes `A B C D E F G H I J`→`A C E G`)
  AND `conflict-drill.md` but did **not** update `_EXPECTED_STRUCTURE` for either —
  so `test_normalized_structure_unchanged[cascade-pressure.md]` and
  `[conflict-drill.md]` fail. Out of scope for KB-341 (the test file is explicitly
  not in scope); surfaced for a dedicated fix.
- **KB-337 (ratification crash) is FIXED** on this base (`7c67486`) — did not block
  KB-341 (leave-pending was retained as the documented policy regardless).
- **KB-340 (cogitator tests)** — pre-existing, did not block; re-confirmed unchanged.

## Evidence pointers

- Session directory (experiment branch `auto-tune/penalty-sweep-cascade`):
  `.worktrees/auto-tune/penalty-sweep-cascade/auto-tune/penalty-sweep-cascade/`
- `session-state.md` — run log + decision record (single source of truth)
- `probe_cascade_significance.py` + `probe-output.txt` — the primary full-L1–L5 evidence
- `runs/001..006/` — per-run harness snapshots (v=10/5/20 before+after)
- KB-309 prior probe: `auto-tune/penalty-sweep/probe_significance.py` + `probe-significance-output.txt`
- KB-332 prior findings: `auto-tune/penalty-sweep-v2/findings.md`

## Status

Experiment work lives on branch `auto-tune/penalty-sweep-cascade` (session) and
`kb/kb-341` (this report + mirrored session-state). **Not merged to `main`**
(auto-tune skill Rule 4 + `AGENTS.md`). **Propagation: none required for the value**
— `UNRESOLVED_PENALTY = 10` is already canonical; the deliverable is the
denser-curriculum corroboration. The stale-snapshot follow-up is the only actionable
item.
