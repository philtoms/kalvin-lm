# Auto-Tune Session State

## Goal
Replace agent routing (Phase 5 in `agent.py`) with direct push to cogitation for all candidates. Eliminate the S1 short-circuit so every new kline proposal flows through the cogitator.

## Done Criteria
- Phase 5 no longer has S1 short-circuit logic; all candidates submitted to cogitator
- Training run completes with zero LLM, zero supervisor interaction
- harness.log clean (no errors, warnings, or escalations)
- All lessons satisfied (18/18, not 17/18)
- All tests updated and passing

## Session
- **Name:** direct-cogitation-push
- **Curriculum:** curricula/mhall-svo-single.md
- **Branch:** auto-tune/direct-cogitation-push
- **Worktree:** .worktrees/auto-tune/direct-cogitation-push
- **Started:** 2026-06-08

## Current Phase
complete — merged to main

## Next Action
Session already merged and complete. Current work is on branch `nlp-tokeniser` doing NLP-BPE pipeline improvements (unrelated to this auto-tune session).

## Run Log

### Run 4 (latest) — satisfaction-based completion + re-fire guard ✅
- **Code changes:** (1) `_check_lesson_complete` uses `len(satisfied) >= len(submitted)` instead of reactor event count. (2) Guards against re-fire: skips if current lesson already in `lesson_satisfied` or no current lesson.
- **Observations:** 18/18 satisfied, zero LLM, zero supervisor, zero escalations. Clean completion in 4 events. No "Lesson ? complete" repeats. 152 log lines.
- **Verdict:** All done criteria met.

### Run 3 — S1 skip + satisfaction guard, 17/18 (event-count completion bug)
- Zero LLM but 17/18 due to event-count-based completion firing too early.

### Run 2 — direct push, LLM called (S3 expansions triggered reactive scaffolding)
### Run 1 — baseline (18/18, zero LLM, all S1)

## Patterns & Notes

### Architecture of the change (5 commits)
1. **Phase 5 rewrite** — all candidates to cogitator, S1-first sort
2. **S1 fast-path** — cogitator skips expand() for S1 items
3. **Satisfaction guard** — skip S2/S3 events for already-satisfied entries
4. **Satisfaction-based completion** — `len(satisfied) >= len(submitted)` instead of event count
5. **Re-fire guard** — block duplicate lesson completion

### Key insight: event-counting vs satisfaction
- Cogitation generates multiple events per entry; event counting fires too early
- Satisfaction counting waits until every entry is resolved
- Re-fire guard prevents duplicate completion from post-completion events

### Documentation updated (5 files)
- `specs/agent.md` — Phase 5, S1 fast-path, test matrix AGT-18/19/22/40/41/42
- `specs/cascade-control.md` — CC-3/CC-4, rules 7-9, tests CC-9 to CC-11
- `specs/curriculum.md` — Lesson completion rules 39-40
- `specs/harness-server.md` — HRNS-24 satisfaction semantics
- `plans/impl/agent.md` — Phase 5 test mapping

## Files Modified
- `src/kalvin/agent.py` — Phase 5: direct push, S1-first sort, S1 skip in cogitator
- `src/trainer/trainer.py` — Satisfaction guard, satisfaction-based completion, re-fire guard
- `tests/test_agent.py` — Updated short-circuit and agt18 tests
- `tests/test_cascade_control.py` — Updated priority assertions
- `specs/agent.md`, `specs/cascade-control.md`, `specs/curriculum.md`, `specs/harness-server.md`, `plans/impl/agent.md` — documentation
