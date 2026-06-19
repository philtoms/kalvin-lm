# Cascade Control — Implementation Plan

**Status:** Done — implementation and tests complete.
**Spec refs:**
- `@specs/reactive-delegation.md` §Reactive-Round Budget (Default Mode) (RD-8a, RD-8b)

## Changes

### 1. `src/training/trainer/reactor.py` — silent drop past budget

- `_handle_reactive` uses a two-stage check against `max_reactive_rounds` (default 5):
  - `_reactive_rounds > max_reactive_rounds`: silent drop (return immediately, no escalation/log/bus message).
  - `_reactive_rounds == max_reactive_rounds`: escalate `budget_exhaustion` (one escalation per lesson).
- The reactive-round counter resets at the start of each lesson.

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| RD-8a | `tests/test_cascade_control.py` | `test_first_budget_exhaustion_escalates` | ✅ |
| RD-8b | `tests/test_cascade_control.py` | `test_subsequent_events_silently_dropped` | ✅ |

## Design Decisions

1. **Silent drop rather than stopping the Cogitator.** The Cogitator may still have in-flight WorkItems; stopping it would lose those results. Silent-drop prevents the Reactor from spinning on the event stream while the Cogitator drains, without side effects.
