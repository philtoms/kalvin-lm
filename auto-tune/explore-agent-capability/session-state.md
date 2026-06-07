# Auto-Tune Session State

## Goal
Investigate LLM-agent capability using curricula as a basis for exploratory lessons. Understand what triggers rationalisation, escalation, and reactive scaffolding, then exercise those paths.

## Done Criteria
Open session — learn and decide as we go.

## Session
- **Name:** explore-agent-capability
- **Curriculum:** curricula/conflict-drill.md
- **Branch:** auto-tune/explore-agent-capability
- **Started:** 2026-06-07

## Current Phase
observing

## Next Action
Document findings and close session. The drain fix resolved cross-lesson spillover. Remaining issue (intra-lesson reactive cascade) is a separate, deeper problem requiring model-level changes to candidate filtering.

## Run Log

### Run 5 (latest) — conflict-drill with drain fix
- **Code changes:** implemented Cogitator.drain(), KAgent.cogitate_drain(), adapter drain handler, Trainer inter-lesson drain
- **Observation:**
  - **Drain working correctly:** drain called before each lesson, blocks until cogitator empty
  - **Lesson 1:** 4/4 auto-countersigned → drain → instant
  - **Lesson 2:** A==B S1 fast path, 6/6 → drain → instant
  - **Lesson 3:** AB=>D C, S3 misfit → 4 reactive scaffolding rounds (LLM calls at 0.70, 0.78, 0.75, 0.82 confidence) → drain took ~80s to flush cascading cogitation → 7/8 satisfied
  - **Lesson 4:** 11/11 satisfied! (vs 7/11 in run 4 without drain). ACE=>B D properly tested independently. 4 reactive scaffolding rounds (0.72, 0.88, 0.70, 0.75 confidence). 809 reactive rounds total (budget exhaustion after round 5, remaining 804 just escalate).
  - Total LLM calls: 8 (4 lesson 3 + 4 lesson 4)
- **Verdict:** drain fix successful — cross-lesson spillover eliminated, lesson 4 now properly tested

### Run 4 — conflict-drill (no drain)
- Cross-lesson spillover: lesson 3 cogitation consumed lesson 4's budget
- Lesson 4: 7/11 (63.6%) — entries lost to spillover
- Budget exhaustion, 5 LLM calls, ~122s total latency

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
- Even with inter-lesson drain, within a single lesson the reactive scaffolding creates cascading events
- Each scaffolding submission adds entries to the model → more candidates → more cogitation work items → more events
- The reactor counts every S2/S3 event as a reactive round, so the counter explodes (809 rounds)
- Budget exhaustion fires at round 5, but events keep arriving (they're just escalated instead of scaffolded)
- **Root cause:** the cogitator generates expansion proposals for EVERY candidate that overlaps the query signature. In a populated model, a single query can match dozens of candidates.

### Key Finding: LLM Confidence Stable
- Run 5 LLM confidence range: 0.70–0.88 (narrower than run 4's 0.65–0.85)
- The drain may have helped by giving a cleaner model state to the LLM

### Curriculum Design Spectrum (Updated)
- **S1-only:** graduated curricula — safe, no LLM
- **Mild S3:** first-steps-s2 — one conflict, one LLM call
- **Cascade (isolated):** conflict-drill + drain — multiple conflicts per lesson, clean inter-lesson isolation, but intra-lesson cascading remains

## Files Modified
- `curricula/conflict-drill.md` (new, run 4)
- `src/kalvin/agent.py` (Cogitator.drain, KAgent.cogitate_drain)
- `src/harness/adapter.py` (drain handler)
- `src/trainer/trainer.py` (inter-lesson drain)
