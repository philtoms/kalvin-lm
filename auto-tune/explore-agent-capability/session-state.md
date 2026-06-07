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
running

## Next Action
Reset session and start run 6 with cascade-pressure curriculum

## Run Log

### Run 6 (latest) — cascade-pressure (pending)
- **Code changes:** none yet
- **Observation:** pending
- **Verdict:** pending

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
- `src/kalvin/agent.py` (Cogitator.drain, KAgent.cogitate_drain)
- `src/harness/adapter.py` (drain handler)
- `src/trainer/trainer.py` (inter-lesson drain)
