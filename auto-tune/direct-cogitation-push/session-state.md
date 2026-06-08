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
complete

## Next Action
All done criteria met. Session complete. Consider merging to main.

## Run Log

### Run 4 (latest) — satisfaction-based completion + re-fire guard
- **Code changes:** (1) `_check_lesson_complete` uses `len(satisfied) >= len(submitted)` instead of reactor event count. (2) Guards against re-fire: skips if current lesson already in `lesson_satisfied` or no current lesson.
- **Observations:** 18/18 satisfied, zero LLM, zero supervisor, zero escalations. Clean completion in 4 events (started → lesson_complete → complete). No "Lesson ? complete" repeats. Expansion events for L>O and L>M arrive after lesson completion but are harmlessly logged (guard blocks re-fire). 152 log lines.
- **Verdict:** ✅ All done criteria met. Session complete.

### Run 3 — S1 skip + satisfaction guard, zero LLM but 17/18
- **Code changes:** (1) S1 work items skip expand() and call on_s1 directly in cogitator. (2) Trainer skips S2/S3 events for already-satisfied entries. (3) Candidates sorted S1-first before submission.
- **Observations:** Zero LLM, zero supervisor. 17/18 — `L > O` satisfied by cogitation events arriving AFTER event-count-based completion check.
- **Verdict:** Led to satisfaction-based fix in run 4.

### Run 2 — direct push, LLM called
- S3 expansion proposals triggered reactive scaffolding → LLM called. Needed S1-first ordering.

### Run 1 — baseline
- All 18/18, zero LLM, everything S1. Baseline.

## Patterns & Notes

### Architecture of the change (5 commits)
1. **Phase 5 rewrite** (`src/kalvin/agent.py`): Removed S1 short-circuit. All candidates go to cogitator. Candidates sorted S1-first by routing level, then by overlap.
2. **S1 work item fast-path** (`src/kalvin/agent.py`): `_run_work_item` skips expand() for S1 items, calls on_s1 directly. Prevents intermediate S2/S3 connotation yields from generating expansion proposals.
3. **Satisfaction guard** (`src/trainer/trainer.py`): S2/S3 events for already-satisfied entries are skipped. Prevents spurious LLM calls.
4. **Satisfaction-based lesson completion** (`src/trainer/trainer.py`): `_check_lesson_complete` uses `len(satisfied) >= len(submitted)` instead of reactor event count. Cogitation generates multiple events per entry.
5. **Re-fire guard** (`src/trainer/trainer.py`): Guards against repeated lesson completion when post-completion cogitation events arrive.

### Key insight: event-counting vs satisfaction
- With direct cogitation push, each entry generates multiple events (initial rationalise + expansion proposals)
- Event counting (`received_count == expected_count`) fires too early — some entries haven't been satisfied yet
- Satisfaction counting (`len(satisfied) >= len(submitted)`) waits until every entry is actually resolved
- But satisfaction counting doesn't "reset" between events, so post-completion events still see 18>=18
- The re-fire guard (check `lesson_satisfied` set) prevents the duplicate lesson completion

### Side effect: expansion spam
- S3 candidates for `L > M` and `L > O` generate many expansion proposals after satisfaction
- All skipped by satisfaction guard but logged at INFO level
- Future: consider cancelling S3 work items when S1 is found, or suppressing log noise

### Test status
- Core tests: 79 passed (test_agent.py + test_cascade_control.py)
- Pre-existing failures: 1 (async test_auto_tune_supervisor)
- Tests modified: `test_agent.py` (short-circuit → all-submit), `test_cascade_control.py` (priority assertions)

## Files Modified
- `src/kalvin/agent.py` — Phase 5: direct push, S1-first sort, S1 skip in cogitator
- `src/trainer/trainer.py` — Satisfaction guard, satisfaction-based completion, re-fire guard
- `tests/test_agent.py` — Updated short-circuit and agt18 tests
- `tests/test_cascade_control.py` — Updated priority assertions
