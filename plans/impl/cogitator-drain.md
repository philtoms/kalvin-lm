# Plan: Cogitator Inter-Lesson Drain

## Summary

Implementation of inter-lesson cogitator draining to prevent cross-lesson
spillover of S2/S3 events.

## Design Decisions

1. **Async bus-based drain** — the Trainer sends a `drain` message to the adapter
   and defers lesson submission until the `drained` response arrives. This avoids
   deadlock (the bus thread is not blocked waiting for itself).

2. **Two-phase submit** — `_submit_next_lesson()` now sends a drain request;
   `_do_submit_lesson()` performs the actual compilation and submission after drain.

3. **Processing flag** — `Cogitator._processing` is set/cleared around work item
   execution, allowing `drain()` to distinguish between "idle" and "processing
   the last item."

## Implementation Tasks

| Task | Description | Files |
|------|-------------|-------|
| T1 | Add `drain()` method to Cogitator with `_processing` flag | `src/kalvin/agent.py` |
| T2 | Expose `cogitate_drain()` on KAgent | `src/kalvin/agent.py` |
| T3 | Add `_handle_drain()` to KAgentAdapter with bus response | `src/training/harness/adapter.py` |
| T4 | Add `_drain_pending` state and `_handle_drained()` to Trainer | `src/training/trainer/trainer.py` |
| T5 | Refactor `_submit_next_lesson` to drain-first, defer to `_do_submit_lesson` | `src/training/trainer/trainer.py` |

## Test Mapping

| Spec ID | Test File | Test Function |
|---------|-----------|---------------|
| DRN-1 | `tests/test_cogitator_drain.py` | `test_drain_before_each_lesson` | ✅ |
| DRN-2 | `tests/test_cogitator_drain.py` | `test_lesson_deferred_until_drained` | ✅ |
| DRN-3 | `tests/test_cogitator_drain.py` | `test_empty_backlog_drain_fast` | ✅ |
| DRN-4 | `tests/test_cogitator_drain.py` | `test_drain_timeout_returns_false` | ✅ |
| DRN-5 | `tests/test_cogitator_drain.py` | `test_processing_flag_guards_drain` | ✅ |
| DRN-6 | `tests/test_cogitator_drain.py` | `test_no_cross_lesson_spillover` | ✅ |

## Evidence

- Auto-tune session: `auto-tune/explore-agent-capability/`
- Spec: `specs/cogitator-drain.md`
