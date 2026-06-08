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
editing

## Next Action
Fix lesson completion counting for the direct-cogitation-push world. Run 3 completed with zero LLM/zero supervisor but 17/18 satisfied — `L > O` is satisfied by cogitation events that arrive AFTER the lesson completion check fires (at received_count=18). The reactor's `is_lesson_complete` uses `received_count == expected_count`, but cogitation generates multiple events per entry, so the count hits 18 before all entries are satisfied. Need to change completion to be based on `len(state.satisfied) == len(state.submitted)` instead of event counting. Then re-run.

## Run Log

### Run 3 (latest) — S1 skip + satisfaction guard, zero LLM but 17/18
- **Code changes:** (1) S1 work items skip expand() and call on_s1 directly in cogitator. (2) Trainer skips S2/S3 events for already-satisfied entries. (3) Candidates sorted S1-first before submission.
- **Observations:** Zero LLM, zero supervisor, zero escalations! `L > M` got S1 via cogitation (`FRAME L > M → S1 ← MHALL => M H A L L`). But `L > O` is satisfied by cogitation events arriving AFTER lesson completion check. 17/18 satisfied. Many S3 expansion events generated but all skipped (already satisfied). The expansion events for `M > S` also fire (cross-contamination from L>M's candidates sharing nodes with M>S).
- **Verdict:** Nearly there. Zero LLM achieved. Need to fix lesson completion to be satisfaction-based, not event-count-based.

### Run 2 — direct push, LLM called
- **Code changes:** Phase 5 submits all candidates to cogitator (no S1 short-circuit).
- **Observations:** `L > M` went through cogitation. S3 expansion proposals triggered reactive scaffolding → LLM called. S1 candidates processed but AFTER S3 (FIFO ordering).
- **Verdict:** Phase 5 change works. Need S1-first ordering and S1 skip in cogitator.

### Run 1 — baseline
- **Code changes:** none
- **Observations:** All 18/18 entries satisfied. Zero LLM, zero supervisor. Everything S1.
- **Verdict:** Baseline.

## Patterns & Notes

### Architecture of the change (3 commits)
1. **Phase 5 rewrite** (`src/kalvin/agent.py`): Removed S1 short-circuit. All candidates go to cogitator. Candidates sorted S1-first by routing level, then by overlap.
2. **S1 work item fast-path** (`src/kalvin/agent.py`): `_run_work_item` skips expand() for S1 items, calls on_s1 directly. This prevents intermediate S2/S3 connotation yields from generating expansion proposals.
3. **Satisfaction guard** (`src/trainer/trainer.py`): S2/S3 events for already-satisfied entries are skipped (no reactor/ratify). Prevents spurious LLM calls when S1 events arrive before S3 expansions.

### Remaining issue: lesson completion counting
- Reactor counts `received_count` (every frame/ground event). Expected = number of entries (18).
- With cogitation, each Phase-5 entry generates multiple events (initial + expansions).
- `received_count` hits 18 before all 18 entries are satisfied.
- `L > O` is the 18th entry — its S1 event arrives from cogitation ~5ms after the 18th received_count triggers lesson completion.
- **Fix:** Change `_check_lesson_complete` to use `len(state.satisfied) == len(state.submitted)` instead of reactor's event count.

### Side effect: expansion spam
- The S3 candidates for `L > M` and `L > O` generate many expansion proposals (AL=>O..., L=>AHLM..., etc.)
- These are all skipped by the satisfaction guard but still logged at INFO level
- Consider: should the cogitator skip S3 work items for queries that already have pending S1 items? Or should we cancel S3 work items when S1 is found?

### Test status
- Core tests: 123 passed, 0 new failures
- Pre-existing failures: 32 (trainer drain flow, async tests, countersign resolution)
- Tests modified: `test_agent.py` (short-circuit → all-submit), `test_cascade_control.py` (priority assertions)

## Files Modified
- `src/kalvin/agent.py` — Phase 5: direct push, S1-first sort, S1 skip in cogitator
- `src/trainer/trainer.py` — Skip S2/S3 events for satisfied entries
- `tests/test_agent.py` — Updated short-circuit and agt18 tests
- `tests/test_cascade_control.py` — Updated priority assertions
