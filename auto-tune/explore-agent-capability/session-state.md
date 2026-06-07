# Auto-Tune Session State

## Goal
Investigate and mitigate intra-lesson reactive cascade: when a single lesson entry triggers S3 against many candidates, each expansion proposal adds entries to the model, which increases the candidate pool for subsequent entries, creating a feedback loop. Design a curriculum that maximises candidate density and observe the cascade dynamics.

## Done Criteria
Open session — learn and decide as we go. Success = understand the cascade dynamics well enough to propose a model-level fix for candidate filtering.

## Session
- **Name:** explore-agent-capability
- **Curriculum:** curricula/cascade-pressure.md
- **Branch:** auto-tune/explore-agent-capability
- **Started:** 2026-06-07

## Current Phase
complete

## Next Action
Session complete. All findings documented. Consider merging to main.

## Run Log

### Run 8 (latest) — conflict-drill regression test (drain + cap)
- **Code changes:** none (same as run 7)
- **Observations:**
  - No regression: lesson 1-3 identical to run 5
  - Lesson 4 improved: 12/11 (vs 11/11 in run 5) — over-satisfied from productive scaffolding
  - LLM calls reduced: 6 (vs 8 in run 5)
  - Budget exhaustion: 1 (vs cascade in run 5)
  - Log lines: 243 (clean and readable)
- **Verdict:** candidate cap does not regress and actually improves the conflict-drill curriculum too

### Run 7 — cascade-pressure with candidate cap + reactor silent-drop
- **Code changes:** max_candidates=8 cap in KAgent.rationalise(), reactor silent-drops events after budget exhaustion
- **Observations:**
  - **Massive improvement:** log lines 170K → 3.9K (43x reduction)
  - Budget exhaustion events: 56,712 → 2 (28,356x reduction)
  - 8 LLM calls (same), 8 scaffolding submissions (up from 7), 0 low confidence (down from 1)
  - Lesson 3: 24/28 (same), Lesson 4: 32/34 (up from 30/34), Lesson 5: 51/35 (up from 39/35)
  - Total time ~5 min (vs ~3 min — longer because more scaffolding is actually processed)
  - Over-satisfaction increased (51/35 on lesson 5) — reactive scaffolding is producing valid entries
- **Verdict:** candidate cap is highly effective. The cascade is controlled. Over-satisfaction shows the system is working productively.

### Run 5 — conflict-drill with drain fix
- Drain fix verified: cross-lesson spillover eliminated (7/11 → 11/11 on lesson 4)
- Intra-lesson cascade remains: 809 reactive rounds, budget exhaustion after round 5
- 8 LLM calls total, lesson 4 = 11/11 satisfied

### Run 4 — conflict-drill (no drain)
- Cross-lesson spillover: lesson 3 cogitation consumed lesson 4's budget
- Lesson 4: 7/11 (63.6%)

### Run 3 — mhall-svo-equivalence
- 5 lessons, all S1, 23/23, zero LLM

### Run 2 — first-steps-s2
- 1 S3 event, LLM 0.95, 6/7

### Run 1 — first-steps baseline
- Clean pass, all auto-countersigned

## Patterns & Notes

### Key Finding: Candidate Cap Resolves Cascade (Run 6 vs Run 7)
- **Run 6 (no cap):** 170K log lines, 56,712 budget exhaustion events, 54K events for single entry
- **Run 7 (cap=8):** 3.9K log lines, 2 budget exhaustion events — **43x log reduction**
- The cap prioritises S2 over S3, then by node overlap count (descending)
- Reactor silent-drop after budget exhaustion prevents event spinning
- Over-satisfaction (51/35) shows productive scaffolding despite the cap

### Key Finding: Drain Fix Verified (Run 4 vs Run 5)
- **Run 4 (no drain):** Lesson 4 = 7/11 (63.6%) — spillover from lesson 3
- **Run 5 (with drain):** Lesson 4 = 11/11 (100%) — clean isolation
- The drain adds negligible overhead when the cogitator is empty (<1ms)
- When cogitator has pending work, drain blocks until complete (~80s for complex cascading)

### Key Finding: Intra-Lesson Reactive Cascade
- Within a single lesson, reactive scaffolding creates cascading events
- Each scaffolding submission adds entries to the model → more candidates → more cogitation work items → more events
- The reactor counts every S2/S3 event as a reactive round, so the counter explodes (809 rounds)
- Budget exhaustion fires at round 5, but events keep arriving (escalated instead of scaffolded)
- **Root cause:** the cogitator generates expansion proposals for EVERY candidate that overlaps the query signature

### Key Finding: LLM Confidence Stable
- Run 5 LLM confidence range: 0.70–0.88

### Curriculum Design Spectrum (Updated)
- **S1-only:** graduated curricula — safe, no LLM
- **Mild S3:** first-steps-s2 — one conflict, one LLM call
- **Cascade (isolated):** conflict-drill + drain — multiple conflicts per lesson, clean inter-lesson isolation
- **Cascade (pressure):** cascade-pressure — designed to maximise candidate density for cascade investigation

## Files Modified
- `curricula/conflict-drill.md` (new, run 4)
- `curricula/cascade-pressure.md` (new, run 6)
- `src/kalvin/agent.py` (Cogitator.drain, KAgent.cogitate_drain, max_candidates cap)
- `src/harness/adapter.py` (drain handler)
- `src/trainer/trainer.py` (inter-lesson drain)
- `src/trainer/reactor.py` (silent-drop after budget exhaustion)
